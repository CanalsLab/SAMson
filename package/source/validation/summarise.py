"""Markdown summary of validation result files: python summarise.py results_*.csv"""
import sys

import pandas as pd


def summary(path):
    table = pd.read_csv(path)
    grouped = table.groupby(["dataset", "mode"], sort=False).agg(
        subjects=("identical", "size"), identical=("identical", "sum"), max_differing_voxels=("differing_voxels", "max"),
        min_jaccard=("jaccard_vs_published", "min"))
    lines = [f"**{path}**: {int(table.identical.sum())}/{len(table)} masks identical", "",
             "| dataset | mode | subjects | identical | max. differing voxels | min. Jaccard vs published |", "|---|---|---|---|---|---|"]
    for (dataset, mode), row in grouped.iterrows():
        lines.append(f"| {dataset} | {mode} | {int(row.subjects)} | {int(row.identical)} | {int(row.max_differing_voxels)} | {row.min_jaccard:.6f} |")
    return "\n".join(lines)


if __name__ == "__main__":
    print("\n\n".join(summary(p) for p in sys.argv[1:]))
