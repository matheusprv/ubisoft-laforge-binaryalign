#!/usr/bin/env python3
"""Converte os arquivos XL-WA em arquivos separados para o BinaryAlign."""

import argparse
import re
from pathlib import Path


SPLITS = ("train", "dev", "test")
ALIGNMENT_PATTERN = re.compile(r"(\d+)([-p])(\d+)")


def convert_alignment(alignment: str, one_indexed: bool) -> str:
    """Retorna o alinhamento, opcionalmente com índices iniciando em 1."""
    if not one_indexed:
        return alignment

    def increment(match: re.Match[str]) -> str:
        source, separator, target = match.groups()
        return f"{int(source) + 1}{separator}{int(target) + 1}"

    return ALIGNMENT_PATTERN.sub(increment, alignment)


def convert_split(input_file: Path, output_dir: Path, one_indexed: bool) -> None:
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
            alignments.append(convert_alignment(alignment, one_indexed))

    output_dir.mkdir(parents=True, exist_ok=True)
    files = {
        "source.src": sources,
        "target.tgt": targets,
        "align.talp": alignments,
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
        default=Path("dataset/XL-WA/pt"),
        help="diretório que contém train.tsv, dev.tsv e test.tsv",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("dataset/portugues-ingles"),
        help="diretório onde serão criadas as divisões convertidas",
    )
    parser.add_argument(
        "--one-indexed",
        action="store_true",
        help="incrementa em 1 ambos os índices dos alinhamentos",
    )
    args = parser.parse_args()

    for split in SPLITS:
        input_file = args.input_dir / f"{split}.tsv"
        if not input_file.is_file():
            raise FileNotFoundError(f"Arquivo não encontrado: {input_file}")
        convert_split(input_file, args.output_dir / split, args.one_indexed)


if __name__ == "__main__":
    main()
