"""
datasets/kuzushiji-column の列（行）検出アノテーションを、
datasets/raw/dataset の文字座標CSVと突き合わせて文字列ラベル付きLMDBを生成するスクリプト。

手順:
  - kuzushiji-column/{train,val}/images にあるページ画像は raw/dataset/{vol}/images/*.jpg と同名（末尾に Roboflow の rf ハッシュ付き）で対応。
  - kuzushiji-column/{split}/labels/*.txt の YOLO フォーマット bbox（列領域）内に含まれる文字を、
    raw/dataset/{vol}/{vol}_coordinate.csv の文字ボックスから拾い、上から順に並べてテキストを作る。
  - 列領域をクロップして LMDB に格納し、ラベルは連結文字列。
  - charset は全ラベルから自動生成。

使い方例:
  uv run python tools/build_kuzushiji_column_lmdb.py \
    --column-root datasets/kuzushiji-column \
    --raw-root datasets/raw/dataset \
    --output-root data/kuzushiji_column_lmdb \
    --margin 4
"""

import argparse
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Tuple

import cv2
import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm


def codepoint_to_char(codepoint: str) -> str:
    if not codepoint.startswith("U+"):
        return codepoint
    return chr(int(codepoint[2:], 16))


def load_coord_csv(volume_dir: Path) -> pd.DataFrame:
    csv_path = volume_dir / f"{volume_dir.name}_coordinate.csv"
    if not csv_path.exists():
        raise FileNotFoundError(csv_path)
    cols = ["Unicode", "Image", "X", "Y", "Block ID", "Char ID", "Width", "Height"]
    df = pd.read_csv(csv_path, dtype={"Unicode": str, "Image": str})
    return df[cols]


@dataclass
class ColumnBox:
    x1: float
    y1: float
    x2: float
    y2: float


def yolo_to_box(line: str, w: int, h: int) -> ColumnBox:
    # YOLO format: cls cx cy bw bh (all normalized)
    parts = line.strip().split()
    if len(parts) < 5:
        raise ValueError(f"Invalid label line: {line}")
    _, cx, cy, bw, bh = map(float, parts[:5])
    x_c, y_c = cx * w, cy * h
    bw_pix, bh_pix = bw * w, bh * h
    x1, y1 = x_c - bw_pix / 2, y_c - bh_pix / 2
    x2, y2 = x_c + bw_pix / 2, y_c + bh_pix / 2
    return ColumnBox(x1, y1, x2, y2)


def char_rows_in_box(rows: pd.DataFrame, box: ColumnBox) -> pd.DataFrame:
    x1, y1, x2, y2 = box.x1, box.y1, box.x2, box.y2
    sel = rows[
        (rows["X"] >= x1)
        & (rows["Y"] >= y1)
        & (rows["X"] + rows["Width"] <= x2)
        & (rows["Y"] + rows["Height"] <= y2)
    ]
    if "Char ID" in sel.columns:
        sel = sel.sort_values("Char ID")
    else:
        sel = sel.sort_values(["Y", "X"])
    return sel


def crop_box(img: np.ndarray, box: ColumnBox, margin: int) -> np.ndarray:
    h, w = img.shape[:2]
    x1 = max(int(box.x1) - margin, 0)
    y1 = max(int(box.y1) - margin, 0)
    x2 = min(int(box.x2) + margin, w)
    y2 = min(int(box.y2) + margin, h)
    return img[y1:y2, x1:x2]


def encode_jpg(img: np.ndarray) -> bytes:
    ok, enc = cv2.imencode(".jpg", img, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError("imencode failed")
    return enc.tobytes()


def create_lmdb(split: str, samples: List[Tuple[bytes, str]], out_dir: Path, map_size: int):
    out_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(out_dir), map_size=map_size)
    cache: Dict[bytes, bytes] = {}
    cnt = 1
    for image_bin, label in tqdm(samples, desc=f"lmdb:{split}", ncols=80):
        key_i = f"image-{cnt:09d}".encode()
        key_l = f"label-{cnt:09d}".encode()
        cache[key_i] = image_bin
        cache[key_l] = label.encode("utf-8")
        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
        cnt += 1
    cache[b"num-samples"] = str(cnt - 1).encode()
    write_cache(env, cache)
    env.close()


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def estimate_map_size(num_samples: int, avg_size: int = 200_000) -> int:
    return int(num_samples * avg_size * 1.5)


def build_charset(labels: List[str], charset_path: Path):
    charset = sorted(set("".join(labels)))
    charset_path.parent.mkdir(parents=True, exist_ok=True)
    with charset_path.open("w", encoding="utf-8") as f:
        for idx, ch in enumerate(charset):
            f.write(f"{idx}\t{ch}\n")


def process_split(split_dir: Path, raw_root: Path, margin: int) -> List[Tuple[bytes, str]]:
    images_dir = split_dir / "images"
    labels_dir = split_dir / "labels"
    samples: List[Tuple[bytes, str]] = []
    for label_file in sorted(labels_dir.glob("*.txt")):
        base = label_file.stem  # e.g., 100241706_00006_2_jpg.rf.xxxxx
        orig = base.split("_jpg.rf")[0]  # -> 100241706_00006_2
        volume = orig.split("_")[0]
        raw_vol_dir = raw_root / volume
        raw_img_path = raw_vol_dir / "images" / f"{orig}.jpg"
        coord_df = load_coord_csv(raw_vol_dir)

        img = cv2.imread(str(raw_img_path))
        if img is None:
            continue
        h, w = img.shape[:2]
        rows_img = coord_df[coord_df["Image"] == orig]
        if rows_img.empty:
            continue

        with label_file.open() as f:
            label_lines = [ln.strip() for ln in f if ln.strip()]

        for ln in label_lines:
            box = yolo_to_box(ln, w, h)
            rows = char_rows_in_box(rows_img, box)
            if rows.empty:
                continue
            text = "".join(codepoint_to_char(u) for u in rows["Unicode"].tolist())
            crop = crop_box(img, box, margin=margin)
            try:
                image_bin = encode_jpg(crop)
            except Exception:
                continue
            samples.append((image_bin, text))
    return samples


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--column-root", type=Path, default=Path("datasets/kuzushiji-column"))
    parser.add_argument("--raw-root", type=Path, default=Path("datasets/raw/dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("data/kuzushiji_column_lmdb"))
    parser.add_argument("--margin", type=int, default=4)
    args = parser.parse_args()

    all_labels: List[str] = []
    for split in ["train", "val"]:
        split_dir = args.column_root / split
        samples = process_split(split_dir, args.raw_root, margin=args.margin)
        all_labels.extend([lbl for _, lbl in samples])
        map_size = estimate_map_size(len(samples))
        create_lmdb(split, samples, args.output_root / split, map_size)
        print(f"{split}: {len(samples)} samples")

    charset_path = args.output_root / "charset_kuzushiji_column.txt"
    build_charset(all_labels, charset_path)
    print(f"charset saved to {charset_path}")


if __name__ == "__main__":
    main()
