#!/usr/bin/env python3
"""Re-tile the run's example image into one row for the paper (pixels copied unchanged).

Source: runs/pixel_baseline_v1_20260926T161957Z/dev_examples.png, which stacks the first four
development groups as rows of [member 0 | member 1] (224 px tiles). Output:
paper/figures/dev_examples_row.png with the same eight tiles in one row, groups left to right.
"""

from pathlib import Path

from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "runs/pixel_baseline_v1_20260926T161957Z/dev_examples.png"
DST = ROOT / "paper/figures/dev_examples_row.png"
T = 224


def main():
    src = Image.open(SRC).convert("RGB")
    rows = src.height // T
    out = Image.new("RGB", (2 * rows * T + (rows - 1) * 16, T), "white")
    for r in range(rows):
        for m in (0, 1):
            tile = src.crop((m * T, r * T, (m + 1) * T, (r + 1) * T))
            out.paste(tile, (r * (2 * T + 16) + m * T, 0))
    out.save(DST)
    print(DST, out.size)


if __name__ == "__main__":
    main()
