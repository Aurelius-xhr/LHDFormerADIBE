#!/usr/bin/env python
"""Build the LHDFormer ABIDE npy file from ABIDE1 csv and mat files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import scipy.io as sio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--csv", default="data/ABIDE1.csv", help="ABIDE1 phenotype csv path.")
    parser.add_argument(
        "--mat-dir",
        default="data/ABIDE1_ts/sub_ts",
        help="Directory containing sub-*.mat files with a ts variable.",
    )
    parser.add_argument("--output", default="data/abide.npy", help="Output npy path.")
    parser.add_argument(
        "--summary",
        default="data/abide_preprocess_summary.json",
        help="Summary json path.",
    )
    parser.add_argument(
        "--timepoints",
        type=int,
        default=100,
        help="Number of leading time points kept for LHDFormer's five 20-step windows.",
    )
    parser.add_argument(
        "--min-site-count",
        type=int,
        default=0,
        help="Optionally drop sites with fewer subjects than this after timepoint filtering. Use 0 to keep all sites.",
    )
    return parser.parse_args()


def load_ts(path: Path) -> np.ndarray:
    mat = sio.loadmat(path)
    if "ts" not in mat:
        raise KeyError(f"{path} does not contain a 'ts' variable")
    ts = np.asarray(mat["ts"], dtype=np.float32)
    if ts.ndim != 2:
        raise ValueError(f"{path} has ts shape {ts.shape}, expected 2 dimensions")
    return ts


def pearson_corr(ts: np.ndarray) -> np.ndarray:
    corr = np.corrcoef(ts)
    corr = np.nan_to_num(corr, nan=0.0, posinf=0.0, neginf=0.0)
    return corr.astype(np.float32)


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    mat_dir = Path(args.mat_dir)
    output_path = Path(args.output)
    summary_path = Path(args.summary)

    phenotype = pd.read_csv(csv_path)
    phenotype["SUB_ID"] = phenotype["SUB_ID"].astype(int)

    time_series = []
    correlations = []
    labels = []
    sites = []
    subjects = []
    original_timepoints = []
    skipped_short = []

    node_count = None
    for row in phenotype.sort_values("SUB_ID").itertuples(index=False):
        subject_id = int(row.SUB_ID)
        mat_path = mat_dir / f"sub-{subject_id:07d}.mat"
        if not mat_path.exists():
            raise FileNotFoundError(f"Missing time series file: {mat_path}")

        ts = load_ts(mat_path)
        if node_count is None:
            node_count = ts.shape[0]
        elif ts.shape[0] != node_count:
            raise ValueError(
                f"{mat_path} has {ts.shape[0]} nodes, expected {node_count}"
            )

        raw_timepoints = int(ts.shape[1])
        if raw_timepoints < args.timepoints:
            skipped_short.append(
                {
                    "subject": subject_id,
                    "site": row.SITE_ID,
                    "timepoints": raw_timepoints,
                }
            )
            continue

        ts = ts[:, : args.timepoints]
        time_series.append(ts)
        correlations.append(pearson_corr(ts))
        labels.append(1 if int(row.DX_GROUP) == 1 else 0)
        sites.append(str(row.SITE_ID))
        subjects.append(subject_id)
        original_timepoints.append(raw_timepoints)

    keep_indices = []
    skipped_rare_site = []
    if args.min_site_count > 0:
        site_counts = {}
        for site in sites:
            site_counts[site] = site_counts.get(site, 0) + 1

        for idx, site in enumerate(sites):
            if site_counts[site] < args.min_site_count:
                skipped_rare_site.append(
                    {
                        "subject": int(subjects[idx]),
                        "site": site,
                        "site_count": int(site_counts[site]),
                    }
                )
            else:
                keep_indices.append(idx)
    else:
        keep_indices = list(range(len(subjects)))

    time_series = [time_series[idx] for idx in keep_indices]
    correlations = [correlations[idx] for idx in keep_indices]
    labels = [labels[idx] for idx in keep_indices]
    sites = [sites[idx] for idx in keep_indices]
    subjects = [subjects[idx] for idx in keep_indices]
    original_timepoints = [original_timepoints[idx] for idx in keep_indices]

    data = {
        "timeseires": np.stack(time_series).astype(np.float32),
        "corr": np.stack(correlations).astype(np.float32),
        "label": np.asarray(labels, dtype=np.int64),
        "site": np.asarray(sites),
        "subject": np.asarray(subjects, dtype=np.int64),
        "original_timepoints": np.asarray(original_timepoints, dtype=np.int64),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_path, data, allow_pickle=True)

    summary = {
        "source_csv": str(csv_path),
        "source_mat_dir": str(mat_dir),
        "output": str(output_path),
        "n_source_subjects": int(len(phenotype)),
        "n_kept_subjects": int(len(subjects)),
        "n_skipped_short_subjects": int(len(skipped_short)),
        "n_skipped_rare_site_subjects": int(len(skipped_rare_site)),
        "skipped_short_subjects": skipped_short,
        "skipped_rare_site_subjects": skipped_rare_site,
        "node_count": int(node_count or 0),
        "kept_timepoints": int(args.timepoints),
        "min_site_count": int(args.min_site_count),
        "label_mapping": {"DX_GROUP=1": 1, "DX_GROUP=2": 0},
        "keys": list(data.keys()),
        "shapes": {
            "timeseires": list(data["timeseires"].shape),
            "corr": list(data["corr"].shape),
            "label": list(data["label"].shape),
            "site": list(data["site"].shape),
        },
    }
    summary_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")

    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
