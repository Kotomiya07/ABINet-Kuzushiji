"""
ABINet 用 LMDB から Language pretrain 用の TSV を生成する。

出力形式:
  inp<TAB>gt

BCN の事前学習では入力と正解に同じ文字列を使うため、
各サンプルのラベル文字列をそのまま 2 列に複製して保存する。
"""

import argparse
from pathlib import Path

import lmdb


def iter_labels(lmdb_dir: Path):
    env = lmdb.open(str(lmdb_dir), readonly=True, lock=False, readahead=False, meminit=False)
    if env is None:
        raise RuntimeError(f"Cannot open LMDB: {lmdb_dir}")
    with env.begin(write=False) as txn:
        num_samples = int(txn.get(b"num-samples"))
        for idx in range(1, num_samples + 1):
            label_key = f"label-{idx:09d}".encode()
            label = txn.get(label_key)
            if label is None:
                continue
            text = label.decode("utf-8").replace("\n", "").replace("\r", "").strip()
            if text:
                yield text
    env.close()


def write_tsv(input_dir: Path, output_path: Path, dedupe: bool) -> int:
    seen = set()
    count = 0
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        f.write("inp\tgt\n")
        for text in iter_labels(input_dir):
            if dedupe and text in seen:
                continue
            if dedupe:
                seen.add(text)
            f.write(f"{text}\t{text}\n")
            count += 1
    return count


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-lmdb", type=Path, required=True)
    parser.add_argument("--val-lmdb", type=Path, required=True)
    parser.add_argument("--output-train", type=Path, default=Path("data/japanese_classical_language_train.tsv"))
    parser.add_argument("--output-val", type=Path, default=Path("data/japanese_classical_language_val.tsv"))
    parser.add_argument("--dedupe", action="store_true", help="同一文字列を重複除去する")
    args = parser.parse_args()

    train_count = write_tsv(args.train_lmdb, args.output_train, dedupe=args.dedupe)
    val_count = write_tsv(args.val_lmdb, args.output_val, dedupe=args.dedupe)

    print(f"train: {train_count} rows -> {args.output_train}")
    print(f"val: {val_count} rows -> {args.output_val}")


if __name__ == "__main__":
    main()
