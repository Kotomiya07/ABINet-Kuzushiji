"""
日本古典籍データから ABINet 学習用の LMDB と charset を生成する。

対応形式:
  1. ndl-minhon v2:
     datasets/
       v2/<corpus>/<doc_id>/<page>.json
       img_v2/<corpus>/<doc_id>/<page>.jpg
  2. ndl-minhon v1:
     datasets/
       v1/<doc_id>/<page>.json
       img_v1/<doc_id>/<page>.jpg
  3. 汎用 split 形式:
     datasets/<name>/
       train/images + labels.csv|tsv|jsonl|json|labels/
       val/images   + ...

ndl-minhon の json は、ページ画像と textline bbox から列画像をクロップして LMDB 化する。
"""

import argparse
import csv
import json
import os
import random
from concurrent.futures import ThreadPoolExecutor
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Iterable, Iterator, List, Optional, Sequence, Tuple

import cv2
import lmdb
import numpy as np
from tqdm import tqdm


IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".tif", ".tiff"}
IMAGE_KEYS = ("image", "image_path", "file_name", "filename", "path")
TEXT_KEYS = ("text", "label", "transcription", "gt", "value")


@dataclass
class Record:
    image_path: Path
    text: str
    group_key: str
    sample_id: str
    bbox: Optional[Tuple[int, int, int, int]] = None


@dataclass
class BuildStats:
    pages_seen: int = 0
    pages_with_image: int = 0
    missing_img: int = 0
    invalid_img: int = 0
    textlines_seen: int = 0
    empty_text: int = 0
    records_collected: int = 0
    too_long: int = 0
    invalid_crop: int = 0
    written: int = 0
    by_corpus: Dict[str, Dict[str, int]] = field(default_factory=lambda: defaultdict(lambda: defaultdict(int)))

    def inc(self, key: str, count: int = 1, corpus: Optional[str] = None) -> None:
        setattr(self, key, getattr(self, key) + count)
        if corpus is not None:
            self.by_corpus[corpus][key] += count

    def merge_counter_map(self, counter_map: Dict[str, Dict[str, int]]) -> None:
        for corpus, counters in counter_map.items():
            for key, count in counters.items():
                self.inc(key, count=count, corpus=corpus)


def normalize_text(text: str) -> str:
    return text.replace("\n", "").replace("\r", "").strip()


def list_images(images_dir: Path) -> List[Path]:
    return sorted([p for p in images_dir.rglob("*") if p.is_file() and p.suffix.lower() in IMAGE_EXTENSIONS])


def write_cache(env, cache: Dict[bytes, bytes]) -> None:
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def check_image_valid(image_bin: bytes) -> bool:
    if image_bin is None:
        return False
    buf = np.frombuffer(image_bin, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape[:2]
    return h * w > 0


def infer_key(record: Dict[str, object], candidates: Sequence[str], kind: str) -> str:
    lowered = {str(k).lower(): k for k in record.keys()}
    for candidate in candidates:
        if candidate in lowered:
            return lowered[candidate]
    raise ValueError(f"{kind}列が見つかりません: keys={list(record.keys())}")


def load_table_records(annotation_path: Path) -> List[Dict[str, object]]:
    suffix = annotation_path.suffix.lower()
    if suffix in {".csv", ".tsv"}:
        delimiter = "\t" if suffix == ".tsv" else ","
        with annotation_path.open("r", encoding="utf-8-sig", newline="") as f:
            return list(csv.DictReader(f, delimiter=delimiter))
    if suffix == ".jsonl":
        records = []
        with annotation_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    records.append(json.loads(line))
        return records
    if suffix == ".json":
        with annotation_path.open("r", encoding="utf-8") as f:
            data = json.load(f)
        if isinstance(data, list):
            return data
        if isinstance(data, dict):
            for key in ("records", "annotations", "items", "data"):
                value = data.get(key)
                if isinstance(value, list):
                    return value
        raise ValueError(f"JSON annotation format is unsupported: {annotation_path}")
    raise ValueError(f"Unsupported annotation file: {annotation_path}")


def resolve_image(images_dir: Path, image_name: str) -> Path:
    candidate = Path(image_name)
    if candidate.is_absolute() and candidate.exists():
        return candidate

    joined = images_dir / candidate
    if joined.exists():
        return joined

    stem = candidate.stem or candidate.name
    matches = [p for p in images_dir.rglob("*") if p.is_file() and p.stem == stem and p.suffix.lower() in IMAGE_EXTENSIONS]
    if len(matches) == 1:
        return matches[0]
    if len(matches) > 1:
        raise ValueError(f"同名画像が複数見つかりました: {image_name}")
    raise FileNotFoundError(f"画像が見つかりません: {image_name}")


def iter_label_dir(images_dir: Path, labels_dir: Path) -> Iterator[Record]:
    images = list_images(images_dir)
    if not images:
        raise ValueError(f"画像が見つかりません: {images_dir}")
    for image_path in images:
        label_path = labels_dir / f"{image_path.stem}.txt"
        if not label_path.exists():
            continue
        text = normalize_text(label_path.read_text(encoding="utf-8"))
        if text:
            rel = image_path.relative_to(images_dir)
            yield Record(image_path=image_path, text=text, group_key=str(rel.parent), sample_id=str(rel))


def iter_table_annotations(images_dir: Path, annotation_path: Path) -> Iterator[Record]:
    records = load_table_records(annotation_path)
    if not records:
        return

    text_key = infer_key(records[0], TEXT_KEYS, "text")
    image_key = None
    try:
        image_key = infer_key(records[0], IMAGE_KEYS, "image")
    except ValueError:
        pass

    if image_key is None:
        images = list_images(images_dir)
        if len(images) != len(records):
            raise ValueError(
                f"画像列が無く、画像数とレコード数も一致しません: images={len(images)} records={len(records)}"
            )
        for image_path, record in zip(images, records):
            text = normalize_text(str(record[text_key]))
            if text:
                rel = image_path.relative_to(images_dir)
                yield Record(image_path=image_path, text=text, group_key=str(rel.parent), sample_id=str(rel))
        return

    for record in records:
        image_name = str(record[image_key])
        image_path = resolve_image(images_dir, image_name)
        text = normalize_text(str(record[text_key]))
        if text:
            rel = image_path.relative_to(images_dir)
            yield Record(image_path=image_path, text=text, group_key=str(rel.parent), sample_id=str(rel))


def iter_generic_split_records(split_dir: Path) -> Iterator[Record]:
    images_dir = split_dir / "images"
    if not images_dir.is_dir():
        raise ValueError(f"images ディレクトリがありません: {images_dir}")

    labels_dir = split_dir / "labels"
    if labels_dir.is_dir():
        yield from iter_label_dir(images_dir, labels_dir)
        return

    for name in (
        "labels.csv",
        "labels.tsv",
        "labels.jsonl",
        "labels.json",
        "annotations.csv",
        "annotations.tsv",
        "annotations.jsonl",
        "annotations.json",
    ):
        annotation_path = split_dir / name
        if annotation_path.exists():
            yield from iter_table_annotations(images_dir, annotation_path)
            return

    raise ValueError(f"対応するアノテーションが見つかりません: {split_dir}")


def detect_format(dataset_root: Path) -> str:
    if (dataset_root / "v2").is_dir() and (dataset_root / "img_v2").is_dir():
        return "ndl-minhon-v2"
    if (dataset_root / "v1").is_dir() and (dataset_root / "img_v1").is_dir():
        return "ndl-minhon-v1"
    if any((dataset_root / split).is_dir() for split in ("train", "val", "test")):
        return "generic"
    raise ValueError(f"未対応のデータ構成です: {dataset_root}")


def parse_bbox(points: Sequence[Sequence[float]], width: int, height: int, margin: int) -> Tuple[int, int, int, int]:
    xs = [int(p[0]) for p in points]
    ys = [int(p[1]) for p in points]
    x1 = max(min(xs) - margin, 0)
    y1 = max(min(ys) - margin, 0)
    x2 = min(max(xs) + margin, width)
    y2 = min(max(ys) + margin, height)
    return x1, y1, x2, y2


def crop_record_image(record: Record) -> bytes:
    if record.bbox is None:
        image_bin = record.image_path.read_bytes()
        if not check_image_valid(image_bin):
            raise ValueError(f"invalid image: {record.image_path}")
        return image_bin

    img = cv2.imread(str(record.image_path))
    if img is None:
        raise ValueError(f"failed to read {record.image_path}")
    x1, y1, x2, y2 = record.bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid bbox: {record.sample_id}")
    crop = img[y1:y2, x1:x2]
    ok, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError(f"imencode failed: {record.sample_id}")
    return enc.tobytes()


def crop_record_from_loaded_image(record: Record, img: np.ndarray) -> bytes:
    if record.bbox is None:
        raise ValueError("bbox is required for loaded-image crop path")
    x1, y1, x2, y2 = record.bbox
    if x2 <= x1 or y2 <= y1:
        raise ValueError(f"invalid bbox: {record.sample_id}")
    crop = img[y1:y2, x1:x2]
    ok, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError(f"imencode failed: {record.sample_id}")
    return enc.tobytes()


def _init_counter_map() -> Dict[str, Dict[str, int]]:
    return defaultdict(lambda: defaultdict(int))


def _inc_counter(counter_map: Dict[str, Dict[str, int]], corpus: str, key: str, count: int = 1) -> None:
    counter_map[corpus][key] += count


def process_image_records(image_path: Path, image_records: List[Record], max_length: int) -> Tuple[List[Tuple[bytes, str]], Dict[str, Dict[str, int]]]:
    samples: List[Tuple[bytes, str]] = []
    counter_map = _init_counter_map()
    valid_records = []
    for record in image_records:
        corpus = record.sample_id.split("/")[0] if "/" in record.sample_id else "generic"
        if len(record.text) > max_length:
            _inc_counter(counter_map, corpus, "too_long")
            continue
        valid_records.append(record)
    if not valid_records:
        return samples, counter_map

    if any(record.bbox is not None for record in valid_records):
        img = cv2.imread(str(image_path))
        if img is None:
            for record in valid_records:
                corpus = record.sample_id.split("/")[0] if "/" in record.sample_id else "generic"
                _inc_counter(counter_map, corpus, "invalid_crop")
            return samples, counter_map
        for record in valid_records:
            corpus = record.sample_id.split("/")[0] if "/" in record.sample_id else "generic"
            try:
                if record.bbox is None:
                    image_bin = record.image_path.read_bytes()
                else:
                    image_bin = crop_record_from_loaded_image(record, img)
            except Exception:
                _inc_counter(counter_map, corpus, "invalid_crop")
                continue
            if not check_image_valid(image_bin):
                _inc_counter(counter_map, corpus, "invalid_crop")
                continue
            samples.append((image_bin, record.text))
            _inc_counter(counter_map, corpus, "written")
        return samples, counter_map

    for record in valid_records:
        corpus = record.sample_id.split("/")[0] if "/" in record.sample_id else "generic"
        try:
            image_bin = crop_record_image(record)
        except Exception:
            _inc_counter(counter_map, corpus, "invalid_crop")
            continue
        if not check_image_valid(image_bin):
            _inc_counter(counter_map, corpus, "invalid_crop")
            continue
        samples.append((image_bin, record.text))
        _inc_counter(counter_map, corpus, "written")
    return samples, counter_map


def create_lmdb(samples: List[Tuple[bytes, str]], output_dir: Path, map_size: int, split_name: str) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_dir), map_size=map_size)
    cache: Dict[bytes, bytes] = {}
    cnt = 1
    for image_bin, label in tqdm(samples, desc=f"lmdb:{split_name}", ncols=80):
        image_key = f"image-{cnt:09d}".encode()
        label_key = f"label-{cnt:09d}".encode()
        cache[image_key] = image_bin
        cache[label_key] = label.encode("utf-8")
        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
        cnt += 1
    cache[b"num-samples"] = str(cnt - 1).encode()
    write_cache(env, cache)
    env.close()


def build_charset(labels: Iterable[str], charset_path: Path) -> None:
    charset = sorted(set("".join(labels)))
    charset_path.parent.mkdir(parents=True, exist_ok=True)
    with charset_path.open("w", encoding="utf-8") as f:
        for idx, ch in enumerate(charset):
            f.write(f"{idx}\t{ch}\n")


def estimate_map_size(num_samples: int, avg_size: int = 250_000) -> int:
    return int(max(num_samples, 1) * avg_size * 1.5)


def collect_generic_records(dataset_root: Path) -> Dict[str, List[Record]]:
    split_to_records: Dict[str, List[Record]] = {}
    for split in ("train", "val", "test"):
        split_dir = dataset_root / split
        if split_dir.is_dir():
            split_to_records[split] = list(iter_generic_split_records(split_dir))
    return split_to_records


def iter_ndl_v1_records(
    dataset_root: Path,
    margin: int,
    max_docs_per_corpus: Optional[int],
    stats: BuildStats,
) -> Iterator[Record]:
    ann_root = dataset_root / "v1"
    img_root = dataset_root / "img_v1"
    doc_dirs = sorted([p for p in ann_root.iterdir() if p.is_dir()])
    if max_docs_per_corpus is not None:
        doc_dirs = doc_dirs[:max_docs_per_corpus]

    for doc_dir in tqdm(doc_dirs, desc="docs:v1", ncols=80):
        image_dir = img_root / doc_dir.name
        if not image_dir.is_dir():
            continue
        json_paths = sorted(doc_dir.glob("*.json"))
        for json_path in tqdm(json_paths, desc=f"pages:v1/{doc_dir.name}", ncols=80, leave=False):
            stats.inc("pages_seen", corpus="v1")
            image_path = image_dir / f"{json_path.stem}.jpg"
            if not image_path.exists():
                stats.inc("missing_img", corpus="v1")
                continue
            image = cv2.imread(str(image_path))
            if image is None:
                stats.inc("invalid_img", corpus="v1")
                continue
            stats.inc("pages_with_image", corpus="v1")
            h, w = image.shape[:2]
            items = json.loads(json_path.read_text(encoding="utf-8"))
            if not isinstance(items, list):
                continue
            for idx, item in enumerate(items):
                if str(item.get("isTextline", "false")).lower() != "true":
                    continue
                stats.inc("textlines_seen", corpus="v1")
                text = normalize_text(str(item.get("text", "")))
                if not text:
                    stats.inc("empty_text", corpus="v1")
                    continue
                bbox = parse_bbox(item["boundingBox"], w, h, margin=margin)
                sample_id = f"{doc_dir.name}/{json_path.stem}#{idx}"
                stats.inc("records_collected", corpus="v1")
                yield Record(image_path=image_path, text=text, group_key=doc_dir.name, sample_id=sample_id, bbox=bbox)


def iter_ndl_v2_records(
    dataset_root: Path,
    margin: int,
    corpora: Optional[Sequence[str]],
    max_docs_per_corpus: Optional[int],
    max_pages_per_doc: Optional[int],
    stats: BuildStats,
) -> Iterator[Record]:
    ann_root = dataset_root / "v2"
    img_root = dataset_root / "img_v2"
    target_corpora = set(corpora) if corpora else None

    for corpus_dir in sorted([p for p in ann_root.iterdir() if p.is_dir()]):
        if target_corpora and corpus_dir.name not in target_corpora:
            continue
        img_corpus_dir = img_root / corpus_dir.name
        if not img_corpus_dir.is_dir():
            continue
        doc_dirs = sorted([p for p in corpus_dir.iterdir() if p.is_dir()])
        if max_docs_per_corpus is not None:
            doc_dirs = doc_dirs[:max_docs_per_corpus]
        for doc_dir in tqdm(doc_dirs, desc=f"docs:v2/{corpus_dir.name}", ncols=80, leave=False):
            img_doc_dir = img_corpus_dir / doc_dir.name
            if not img_doc_dir.is_dir():
                continue
            json_paths = sorted(doc_dir.glob("*.json"))
            if max_pages_per_doc is not None:
                json_paths = json_paths[:max_pages_per_doc]
            for json_path in tqdm(json_paths, desc=f"pages:{corpus_dir.name}/{doc_dir.name}", ncols=80, leave=False):
                stats.inc("pages_seen", corpus=corpus_dir.name)
                image_path = img_doc_dir / f"{json_path.stem}.jpg"
                if not image_path.exists():
                    stats.inc("missing_img", corpus=corpus_dir.name)
                    continue
                image = cv2.imread(str(image_path))
                if image is None:
                    stats.inc("invalid_img", corpus=corpus_dir.name)
                    continue
                stats.inc("pages_with_image", corpus=corpus_dir.name)
                h, w = image.shape[:2]
                obj = json.loads(json_path.read_text(encoding="utf-8"))
                items = obj["words"] if isinstance(obj, dict) else obj
                if not isinstance(items, list):
                    continue
                for idx, item in enumerate(items):
                    if str(item.get("isTextline", "false")).lower() != "true":
                        continue
                    stats.inc("textlines_seen", corpus=corpus_dir.name)
                    text = normalize_text(str(item.get("text", "")))
                    if not text:
                        stats.inc("empty_text", corpus=corpus_dir.name)
                        continue
                    bbox = parse_bbox(item["boundingBox"], w, h, margin=margin)
                    sample_id = f"{corpus_dir.name}/{doc_dir.name}/{json_path.stem}#{idx}"
                    group_key = f"{corpus_dir.name}/{doc_dir.name}"
                    stats.inc("records_collected", corpus=corpus_dir.name)
                    yield Record(image_path=image_path, text=text, group_key=group_key, sample_id=sample_id, bbox=bbox)


def collect_ndl_records(
    dataset_root: Path,
    source_format: str,
    margin: int,
    corpora: Optional[Sequence[str]],
    max_docs_per_corpus: Optional[int],
    max_pages_per_doc: Optional[int],
    stats: BuildStats,
) -> List[Record]:
    if source_format == "ndl-minhon-v1":
        return list(iter_ndl_v1_records(dataset_root, margin=margin, max_docs_per_corpus=max_docs_per_corpus, stats=stats))
    return list(
        iter_ndl_v2_records(
            dataset_root,
            margin=margin,
            corpora=corpora,
            max_docs_per_corpus=max_docs_per_corpus,
            max_pages_per_doc=max_pages_per_doc,
            stats=stats,
        )
    )


def split_records_by_group(records: List[Record], val_ratio: float, test_ratio: float, seed: int) -> Dict[str, List[Record]]:
    groups: Dict[str, List[Record]] = {}
    for record in records:
        groups.setdefault(record.group_key, []).append(record)

    group_names = sorted(groups)
    rng = random.Random(seed)
    rng.shuffle(group_names)
    total_groups = len(group_names)
    if total_groups == 1:
        counts = {"train": 1, "val": 0, "test": 0}
    else:
        test_group_count = int(round(total_groups * test_ratio))
        val_group_count = int(round(total_groups * val_ratio))
        if test_ratio > 0 and test_group_count == 0:
            test_group_count = 1
        remaining_after_test = total_groups - test_group_count
        if val_ratio > 0 and remaining_after_test > 1 and val_group_count == 0:
            val_group_count = 1
        if test_group_count + val_group_count >= total_groups:
            overflow = test_group_count + val_group_count - (total_groups - 1)
            if overflow > 0:
                reduce_test = min(overflow, max(test_group_count - 1, 0))
                test_group_count -= reduce_test
                overflow -= reduce_test
            if overflow > 0:
                val_group_count = max(val_group_count - overflow, 0)
        counts = {
            "test": test_group_count,
            "val": val_group_count,
            "train": total_groups - test_group_count - val_group_count,
        }
        if counts["train"] <= 0:
            counts["train"] = 1
            if counts["val"] >= counts["test"] and counts["val"] > 0:
                counts["val"] -= 1
            elif counts["test"] > 0:
                counts["test"] -= 1

    test_groups = set(group_names[: counts["test"]])
    val_groups = set(group_names[counts["test"] : counts["test"] + counts["val"]])

    split_to_records = {"train": [], "val": [], "test": []}
    for group_name, group_records in groups.items():
        if group_name in test_groups:
            split = "test"
        elif group_name in val_groups:
            split = "val"
        else:
            split = "train"
        split_to_records[split].extend(group_records)

    # 少量データのデバッグ用フォールバック:
    # group 単位分割で split が空になった場合は sample 単位で分ける
    required_splits = ["train"]
    if val_ratio > 0:
        required_splits.append("val")
    if test_ratio > 0:
        required_splits.append("test")
    if records and any(not split_to_records[name] for name in required_splits):
        ordered = list(records)
        rng.shuffle(ordered)
        total = len(ordered)
        test_count = min(max(1, int(round(total * test_ratio))) if test_ratio > 0 else 0, max(total - 2, 0))
        remaining = total - test_count
        val_count = min(max(1, int(round(total * val_ratio))) if val_ratio > 0 else 0, max(remaining - 1, 0))
        split_to_records["test"] = ordered[:test_count]
        split_to_records["val"] = ordered[test_count : test_count + val_count]
        split_to_records["train"] = ordered[test_count + val_count :]
    return split_to_records


def records_to_samples(
    records: List[Record],
    max_length: int,
    split_name: str,
    stats: BuildStats,
    num_workers: int,
) -> List[Tuple[bytes, str]]:
    samples: List[Tuple[bytes, str]] = []
    records_by_image: Dict[Path, List[Record]] = defaultdict(list)
    for record in records:
        records_by_image[record.image_path].append(record)

    items = list(records_by_image.items())
    if num_workers <= 1:
        iterator = (
            process_image_records(image_path, image_records, max_length)
            for image_path, image_records in items
        )
        for sample_batch, counter_map in tqdm(iterator, total=len(items), desc=f"scan:{split_name}", ncols=80):
            samples.extend(sample_batch)
            stats.merge_counter_map(counter_map)
        return samples

    with ThreadPoolExecutor(max_workers=num_workers) as executor:
        futures = [
            executor.submit(process_image_records, image_path, image_records, max_length)
            for image_path, image_records in items
        ]
        for future in tqdm(futures, desc=f"scan:{split_name}", ncols=80):
            sample_batch, counter_map = future.result()
            samples.extend(sample_batch)
            stats.merge_counter_map(counter_map)
    return samples


def print_stats(stats: BuildStats) -> None:
    print("summary:")
    print(f"  pages_seen={stats.pages_seen}")
    print(f"  pages_with_image={stats.pages_with_image}")
    print(f"  missing_img={stats.missing_img}")
    print(f"  invalid_img={stats.invalid_img}")
    print(f"  textlines_seen={stats.textlines_seen}")
    print(f"  empty_text={stats.empty_text}")
    print(f"  records_collected={stats.records_collected}")
    print(f"  too_long={stats.too_long}")
    print(f"  invalid_crop={stats.invalid_crop}")
    print(f"  written={stats.written}")
    if stats.by_corpus:
        print("by_corpus:")
        print("  corpus\tpages_seen\tmissing_img\ttextlines_seen\trecords_collected\ttoo_long\tinvalid_crop\twritten")
        for corpus in sorted(stats.by_corpus):
            row = stats.by_corpus[corpus]
            values = [corpus] + [
                str(row.get(key, 0))
                for key in ("pages_seen", "missing_img", "textlines_seen", "records_collected", "too_long", "invalid_crop", "written")
            ]
            print("  " + "\t".join(values))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset-root", type=Path, default=Path("datasets"))
    parser.add_argument("--output-root", type=Path, default=Path("data/japanese_classical_column_lmdb"))
    parser.add_argument("--source-format", choices=["auto", "ndl-minhon-v1", "ndl-minhon-v2", "generic"], default="auto")
    parser.add_argument("--max-length", type=int, default=40)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--test-ratio", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--margin", type=int, default=4)
    parser.add_argument("--num-workers", type=int, default=max((os.cpu_count() or 1) - 1, 1))
    parser.add_argument("--corpora", nargs="*", default=None, help="v2 用。対象 corpus 名を限定")
    parser.add_argument("--max-docs-per-corpus", type=int, default=None, help="デバッグ用に corpus ごとの文書数を制限")
    parser.add_argument("--max-pages-per-doc", type=int, default=None, help="デバッグ用に文書ごとのページ数を制限")
    args = parser.parse_args()

    source_format = detect_format(args.dataset_root) if args.source_format == "auto" else args.source_format
    print(f"detected source format: {source_format}")
    stats = BuildStats()

    if source_format == "generic":
        split_to_records = collect_generic_records(args.dataset_root)
    else:
        records = collect_ndl_records(
            args.dataset_root,
            source_format=source_format,
            margin=args.margin,
            corpora=args.corpora,
            max_docs_per_corpus=args.max_docs_per_corpus,
            max_pages_per_doc=args.max_pages_per_doc,
            stats=stats,
        )
        if not records:
            print_stats(stats)
            raise SystemExit("No valid textline records found.")
        split_to_records = split_records_by_group(records, val_ratio=args.val_ratio, test_ratio=args.test_ratio, seed=args.seed)

    split_to_samples: Dict[str, List[Tuple[bytes, str]]] = {}
    all_labels: List[str] = []
    for split, records in split_to_records.items():
        samples = records_to_samples(
            records,
            max_length=args.max_length,
            split_name=split,
            stats=stats,
            num_workers=args.num_workers,
        )
        if not samples:
            print_stats(stats)
            raise SystemExit(f"No valid samples found in split={split}")
        split_to_samples[split] = samples
        all_labels.extend([label for _, label in samples])

    charset_path = args.output_root / "charset_japanese_classical_column.txt"
    build_charset(all_labels, charset_path)

    total_samples = sum(len(samples) for samples in split_to_samples.values())
    map_size = estimate_map_size(total_samples)
    for split, samples in split_to_samples.items():
        create_lmdb(samples, args.output_root / split, map_size=map_size, split_name=split)
        print(f"{split}: {len(samples)} samples")

    print_stats(stats)
    print(f"charset saved to {charset_path}")


if __name__ == "__main__":
    main()
