#!/usr/bin/env python3
"""
Hugging Faceデータセットから文字セットを生成するスクリプト

使用方法:
    uv run python tools/create_charset_from_huggingface.py \
        --dataset-names Kotomiya07/honkoku-hq Kotomiya07/honkoku-v3.0 \
        --text-column text \
        --output data/kuzushiji_column_lmdb/charset_kuzushiji_column.txt \
        --splits train validation
"""

import argparse
import logging
import unicodedata
from pathlib import Path
from collections import OrderedDict

from datasets import load_dataset

logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def is_valid_char(char):
    """
    文字が文字セットに含めるべき有効な文字かどうかを判定

    Args:
        char: 判定する文字

    Returns:
        bool: 有効な文字の場合True
    """
    # 制御文字を除外（\x00-\x1F, \x7F-\x9F）
    if unicodedata.category(char).startswith("C"):
        return False

    # 改行、タブ、復帰などの制御文字を明示的に除外
    if char in "\n\r\t\f\v":
        return False

    # スペース文字も除外（文字認識では通常不要）
    if char == " ":
        return False

    # 印刷可能な文字のみを含める
    return char.isprintable()


def collect_chars_from_dataset(dataset_names, text_column="text", splits=None):
    """
    Hugging Faceデータセットからすべての文字を収集

    Args:
        dataset_names: データセット名のリスト
        text_column: テキストが含まれるカラム名
        splits: 使用するsplitのリスト（Noneの場合はすべてのsplitを使用）

    Returns:
        set: 収集された文字のセット
    """
    all_chars = set()

    for dataset_name in dataset_names:
        logging.info(f"Loading dataset: {dataset_name}")
        try:
            if splits:
                # 指定されたsplitのみを使用
                for split in splits:
                    try:
                        ds = load_dataset(dataset_name, split=split)
                        logging.info(f"  Processing split: {split} ({len(ds)} samples)")
                        for item in ds:
                            text = item.get(text_column, "")
                            if isinstance(text, str):
                                # 有効な文字のみを追加
                                all_chars.update(c for c in text if is_valid_char(c))
                    except Exception as e:
                        logging.warning(f"  Failed to load split {split}: {e}")
            else:
                # すべてのsplitを使用
                ds_info = load_dataset(dataset_name, split=None)
                available_splits = list(ds_info.keys())
                logging.info(f"  Available splits: {available_splits}")
                for split in available_splits:
                    ds = ds_info[split]
                    logging.info(f"  Processing split: {split} ({len(ds)} samples)")
                    for item in ds:
                        text = item.get(text_column, "")
                        if isinstance(text, str):
                            # 有効な文字のみを追加
                            all_chars.update(c for c in text if is_valid_char(c))
        except Exception as e:
            logging.error(f"Failed to load dataset {dataset_name}: {e}")
            raise

    return all_chars


def create_charset_file(chars, output_path, null_char="░"):
    """
    文字セットファイルを作成

    Args:
        chars: 文字のセットまたはリスト
        output_path: 出力ファイルのパス
        null_char: null文字として使用する文字（デフォルト: '░'）
    """
    # null文字を除外してUnicodeコードポイント順にソート
    chars_without_null = [c for c in chars if c != null_char]
    sorted_chars = sorted(chars_without_null, key=ord)

    # null文字を最初に配置（Unicode順の先頭）
    unique_chars = [null_char] + sorted_chars

    # 重複を除去（順序を保持）
    unique_chars = list(OrderedDict.fromkeys(unique_chars))

    # ファイルに書き込み
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as f:
        for i, char in enumerate(unique_chars):
            f.write(f"{i}\t{char}\n")

    logging.info(f"Created charset file: {output_path}")
    logging.info(f"Total characters: {len(unique_chars)}")
    logging.info(f"Null character (index 0): {unique_chars[0]}")

    return unique_chars


def main():
    parser = argparse.ArgumentParser(
        description="Hugging Faceデータセットから文字セットを生成",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument(
        "--dataset-names",
        nargs="+",
        required=True,
        help="Hugging Faceデータセット名のリスト（例: Kotomiya07/honkoku-hq Kotomiya07/honkoku-v3.0）",
    )
    parser.add_argument("--text-column", default="text", help="テキストが含まれるカラム名（デフォルト: text）")
    parser.add_argument(
        "--output", required=True, help="出力ファイルのパス（例: data/kuzushiji_column_lmdb/charset_kuzushiji_column.txt）"
    )
    parser.add_argument("--splits", nargs="+", default=None, help="使用するsplitのリスト（デフォルト: すべてのsplit）")
    parser.add_argument("--null-char", default="░", help="null文字として使用する文字（デフォルト: ░）")
    parser.add_argument("--min-count", type=int, default=1, help="文字の最小出現回数（デフォルト: 1、すべての文字を含む）")

    args = parser.parse_args()

    # 文字を収集
    logging.info("Collecting characters from datasets...")
    all_chars = collect_chars_from_dataset(args.dataset_names, text_column=args.text_column, splits=args.splits)

    logging.info(f"Collected {len(all_chars)} unique characters")

    # 文字セットファイルを作成
    logging.info("Creating charset file...")
    create_charset_file(all_chars, args.output, null_char=args.null_char)

    logging.info("Done!")


if __name__ == "__main__":
    main()
