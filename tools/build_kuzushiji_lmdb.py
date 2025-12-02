"""
日本古典籍くずし字データセット(datasets/raw/dataset)の「ページ画像 + 座標CSV」から
行(ブロック)単位のクロップ画像を作り、ABINet用のLMDBを生成する。

前提:
  datasets/raw/dataset/{volume}/
    - images/*.jpg          : ページ画像
    - {volume}_coordinate.csv : 列 Unicode,Image,X,Y,Block ID,Char ID,Width,Height

出力:
  - {output_root}/train, {output_root}/val に LMDB (image-NNNNNN.jpg / label-NNNNNN)
  - {output_root}/charset_kuzushiji.txt に charset (index\tchar 形式)

実行例:
  uv run python tools/build_kuzushiji_lmdb.py \
    --source-root datasets/raw/dataset \
    --output-root data/kuzushiji_blocks \
    --val-ratio 0.1 \
    --margin 4 \
    --seed 42
"""

import argparse
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import cv2
import lmdb
import numpy as np
import pandas as pd
from tqdm import tqdm


def check_image_valid(image_bin: bytes) -> bool:
    """簡易に画像が壊れていないか確認する。"""
    if image_bin is None:
        return False
    buf = np.frombuffer(image_bin, dtype=np.uint8)
    img = cv2.imdecode(buf, cv2.IMREAD_GRAYSCALE)
    if img is None:
        return False
    h, w = img.shape[:2]
    return h * w > 0


def write_cache(env, cache):
    with env.begin(write=True) as txn:
        for k, v in cache.items():
            txn.put(k, v)


def build_charset(samples: Iterable[str], charset_path: Path):
    charset = sorted(set(samples))
    charset_path.parent.mkdir(parents=True, exist_ok=True)
    with charset_path.open("w", encoding="utf-8") as f:
        for idx, ch in enumerate(charset):
            f.write(f"{idx}\t{ch}\n")
    return charset


def codepoint_to_char(codepoint: str) -> str:
    """'U+3042' -> 'あ'"""
    if not codepoint.startswith("U+"):
        return codepoint
    return chr(int(codepoint[2:], 16))


def load_annotations(coord_csv: Path) -> pd.DataFrame:
    cols = ["Unicode", "Image", "X", "Y", "Block ID", "Char ID", "Width", "Height"]
    df = pd.read_csv(coord_csv, dtype={"Unicode": str, "Image": str})
    df = df[cols]
    return df


def group_blocks(df: pd.DataFrame) -> Dict[Tuple[str, str], pd.DataFrame]:
    """(image, block_id) ごとにまとめる。"""
    grouped = {}
    for (img, block_id), g in df.groupby(["Image", "Block ID"]):
        # 文字順: Char ID があればそれ、無ければ X, Y で並べる
        if "Char ID" in g.columns:
            g = g.sort_values("Char ID")
        else:
            g = g.sort_values(["Y", "X"])
        grouped[(img, block_id)] = g
    return grouped


def crop_block(image_path: Path, rows: pd.DataFrame, margin: int) -> Tuple[bytes, str]:
    img = cv2.imread(str(image_path))
    if img is None:
        raise ValueError(f"failed to read {image_path}")
    xs = rows["X"].values
    ys = rows["Y"].values
    ws = rows["Width"].values
    hs = rows["Height"].values
    x1 = max(int(xs.min()) - margin, 0)
    y1 = max(int(ys.min()) - margin, 0)
    x2 = int((xs + ws).max()) + margin
    y2 = int((ys + hs).max()) + margin
    crop = img[y1:y2, x1:x2]
    ok, enc = cv2.imencode(".jpg", crop, [int(cv2.IMWRITE_JPEG_QUALITY), 95])
    if not ok:
        raise ValueError(f"imencode failed for {image_path}")
    text = "".join(codepoint_to_char(u) for u in rows["Unicode"].values)
    return enc.tobytes(), text


def create_lmdb(split_name: str, samples: List[Tuple[bytes, str]], output_dir: Path, map_size: int):
    output_dir.mkdir(parents=True, exist_ok=True)
    env = lmdb.open(str(output_dir), map_size=map_size)
    cache = {}
    cnt = 1
    for image_bin, label in tqdm(samples, desc=f"lmdb:{split_name}", ncols=80):
        if not check_image_valid(image_bin):
            continue
        image_key = f"image-{cnt:09d}".encode()
        label_key = f"label-{cnt:09d}".encode()
        cache[image_key] = image_bin
        cache[label_key] = label.encode("utf-8")
        if cnt % 1000 == 0:
            write_cache(env, cache)
            cache = {}
        cnt += 1
    cache["num-samples".encode()] = str(cnt - 1).encode()
    write_cache(env, cache)
    env.close()


def estimate_map_size(num_samples: int, avg_size: int = 120_000) -> int:
    """LMDBのmap_sizeをざっくり推定 (ページクロップは大きめなので1枚 ~120KB と仮定)。"""
    return int(num_samples * avg_size * 1.5)


def collect_block_crops(source_root: Path, margin: int) -> List[Tuple[bytes, str]]:
    crops: List[Tuple[bytes, str]] = []
    for volume_dir in sorted(source_root.iterdir()):
        coord_csv = volume_dir / f"{volume_dir.name}_coordinate.csv"
        images_dir = volume_dir / "images"
        if not coord_csv.exists() or not images_dir.is_dir():
            continue
        df = load_annotations(coord_csv)
        for (img_name, _block_id), rows in group_blocks(df).items():
            image_path = images_dir / f"{img_name}.jpg"
            if not image_path.exists():
                continue
            try:
                image_bin, text = crop_block(image_path, rows, margin=margin)
            except Exception:
                continue
            # 長すぎるテキストは ABINet の max_length (デフォ25) で切る想定なのでここでは残す
            crops.append((image_bin, text))
    return crops


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-root", type=Path, default=Path("datasets/raw/dataset"))
    parser.add_argument("--output-root", type=Path, default=Path("data/kuzushiji_blocks"))
    parser.add_argument("--val-ratio", type=float, default=0.1)
    parser.add_argument("--margin", type=int, default=4, help="各ブロック bbox への上下左右マージン(px)")
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    crops = collect_block_crops(args.source_root, margin=args.margin)
    if len(crops) == 0:
        raise SystemExit(f"No crops found under {args.source_root}")

    random.seed(args.seed)
    random.shuffle(crops)
    split = int(len(crops) * (1 - args.val_ratio))
    train_samples = crops[:split]
    val_samples = crops[split:]

    charset_path = args.output_root / "charset_kuzushiji.txt"
    all_text = "".join([t for _, t in crops])
    build_charset(all_text, charset_path)

    map_size = estimate_map_size(len(crops))
    create_lmdb("train", train_samples, args.output_root / "train", map_size)
    create_lmdb("val", val_samples, args.output_root / "val", map_size)

    print(f"Done. Train: {len(train_samples)} Val: {len(val_samples)}")
    print(f"Charset saved to {charset_path}")


if __name__ == "__main__":
    main()
