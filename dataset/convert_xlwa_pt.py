#!/usr/bin/env python3
"""Converte os arquivos XL-WA em arquivos separados para o BinaryAlign."""

import argparse
from pathlib import Path


SPLITS = ("train", "dev", "test")


def convert_split(input_file: Path, output_dir: Path, split: str) -> None:
    sources = []
    targets = []
    alignments = []

    with input_file.open(encoding="utf-8") as input_stream:
        for line_number, line in enumerate(input_stream, start=1):
            columns = line.rstrip("\n").split("\t")
            if len(columns) != 3:
                raise ValueError(
                    f"{input_file}:{line_number}: esperado 3 colunas, "
                    f"encontradas {len(columns)}"
                )

            source, target, alignment = columns
            sources.append(source)
            targets.append(target)
            alignments.append(alignment)

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        f"{split}.src": sources,
        f"{split}.tgt": targets,
        f"{split}.talp": alignments,
    }
    for filename, lines in files.items():
        with (output_dir / filename).open("w", encoding="utf-8") as output_stream:
            output_stream.write("\n".join(lines))
            output_stream.write("\n")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Converte dataset/XL-WA/pt para o formato do BinaryAlign."
    )
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("XL-WA/pt"),
        help="diretório que contém train.tsv, dev.tsv e test.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("portugues-ingles"),
        help="diretório onde serão criadas as divisões convertidas",
    )
    args = parser.parse_args()

    for split in SPLITS:
        input_file = args.input_dir / f"{split}.tsv"
        if not input_file.is_file():
            raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")
        convert_split(input_file, args.output_dir / split, split)


if __name__ == "__main__":
    main()
