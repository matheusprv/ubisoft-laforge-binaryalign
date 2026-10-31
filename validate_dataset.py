#!/usr/bin/env python3
"""Validate BinaryAlign .src/.tgt/.talp files. Usage: python validate_dataset.py meu_dataset/train/pten"""
import argparse
import re
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("prefix", help="File prefix without .src/.tgt/.talp extension")
parser.add_argument("--zero-indexed", action="store_true", help="Alignments start at index 0, not 1")
args = parser.parse_args()
files = [Path(args.prefix + ext) for ext in (".src", ".tgt", ".talp")]
for file in files:
    if not file.is_file():
        parser.error(f"Missing: {file}")
contents = [file.read_text(encoding="utf-8-sig").splitlines() for file in files]
if len({len(x) for x in contents}) != 1:
    parser.error(f"Different line counts: {[len(x) for x in contents]}")
errors = []
for line_no, (src, tgt, talp) in enumerate(zip(*contents), 1):
    nsrc, ntgt = len(src.split()), len(tgt.split())
    if nsrc == 0 or ntgt == 0:
        errors.append(f"Line {line_no}: empty sentence")
    seen = set()
    for link in talp.split():
        if not re.fullmatch(r"[0-9]+(?:-|p)[0-9]+", link):
            errors.append(f"Line {line_no}: invalid link {link!r}")
            continue
        sep = 'p' if 'p' in link else '-'
        a, b = map(int, link.split(sep))
        lower = 0 if args.zero_indexed else 1
        if not (lower <= a < nsrc + lower and lower <= b < ntgt + lower):
            errors.append(f"Line {line_no}: {link} outside {nsrc}x{ntgt} words")
        if link in seen:
            errors.append(f"Line {line_no}: repeated alignment {link}")
        seen.add(link)
    if len(errors) >= 30:
        break
print(f"Pairs: {len(contents[0])}; problems: {len(errors)}")
if errors:
    print('\n'.join(errors[:30]))
    raise SystemExit(1)
print("OK: sentence counts and alignment indices are consistent")
