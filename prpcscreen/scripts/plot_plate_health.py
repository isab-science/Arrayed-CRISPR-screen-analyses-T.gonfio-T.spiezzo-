from __future__ import annotations

import os
import sys
import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from prpcscreen.visualization.plate_level_qc import plot_plate_qualities
from prpcscreen.visualization.plotly_exports import write_plotly_interactive_html
from prpcscreen.analysis.calculating_scores import calculate_ssmd_ctrls

DEBUG_ENV_DEFAULT = os.environ.get("PRPCSCREEN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def debug_log(message: str, enabled: bool) -> None:
    """Emit debug lines for plate QC plotting."""
    if enabled:
        print(f"[04_plot_plate_health] {message}", file=sys.stderr)


def _sorted_plate_qc_arrays(rep1: list[float], rep2: list[float], labels: list[str]) -> tuple[np.ndarray, np.ndarray, list[str]]:
    rep1_vals = np.asarray(rep1, dtype=float)
    rep2_vals = np.asarray(rep2, dtype=float)
    pair_mean = np.nanmean(np.vstack([rep1_vals, rep2_vals]), axis=0)
    sort_key = np.where(np.isfinite(pair_mean), pair_mean, np.inf)
    order = np.argsort(sort_key, kind="stable")
    sorted_labels = [str(labels[i]) for i in order]
    return rep1_vals[order], rep2_vals[order], sorted_labels


def _write_interactive_qc_html(rep1: list[float], rep2: list[float], labels: list[str], output_png: str) -> Path:
    rep1_vals, rep2_vals, sorted_labels = _sorted_plate_qc_arrays(rep1, rep2, labels)
    n = len(sorted_labels)
    x_vals = list(range(1, n + 1))

    traces: list[dict] = []
    all_values: list[float] = []
    for idx, x in enumerate(x_vals):
        y1 = float(rep1_vals[idx]) if np.isfinite(rep1_vals[idx]) else None
        y2 = float(rep2_vals[idx]) if np.isfinite(rep2_vals[idx]) else None
        color = "#d62728" if idx % 2 == 0 else "#1f77b4"
        x_pair: list[float] = []
        y_pair: list[float] = []
        if y1 is not None:
            x_pair.append(float(x))
            y_pair.append(y1)
            all_values.append(y1)
        if y2 is not None:
            x_pair.append(float(x))
            y_pair.append(y2)
            all_values.append(y2)
        if not x_pair:
            continue
        traces.append(
            {
                "x": x_pair,
                "y": y_pair,
                "mode": "lines+markers",
                "type": "scatter",
                "line": {"color": color, "width": 1.3},
                "marker": {"color": color, "size": 7},
                "name": f"Plate {sorted_labels[idx]}",
                "showlegend": False,
                "hovertemplate": f"Plate {sorted_labels[idx]}<br>SSMD: %{{y:.3f}}<extra></extra>",
            }
        )

    if all_values:
        y_lo = min(-0.2, min(all_values) - 0.05)
        y_hi = max(1.0, max(all_values) + 0.05)
    else:
        y_lo, y_hi = -0.2, 1.0

    layout = {
        "paper_bgcolor": "#f6f6f6",
        "plot_bgcolor": "#ffffff",
        "margin": {"l": 70, "r": 30, "t": 35, "b": 110},
        "xaxis": {
            "title": "Plates",
            "tickmode": "array",
            "tickvals": x_vals,
            "ticktext": sorted_labels,
            "tickangle": -45,
            "range": [0.4, n + 0.6],
        },
        "yaxis": {"title": "SSMD controls", "range": [y_lo, y_hi]},
    }

    output_html = Path(output_png).with_name(Path(output_png).stem + "_interactive.html")
    return write_plotly_interactive_html(
        output_html=output_html,
        traces=traces,
        layout=layout,
        title="Plate quality controls (SSMD)",
        filename_base="plate_qc_ssmd_controls",
    )


def run_qc_cli() -> None:
    # Parse file paths for analyzed input and QC figure output.
    parser = argparse.ArgumentParser(description="Generate plate-level QC plots.")
    parser.add_argument("input_csv")
    parser.add_argument("output_png")
    parser.add_argument(
        "--interactive-only",
        action="store_true",
        help="Only export interactive HTML (skip static PNG output).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()
    debug_enabled = DEBUG_ENV_DEFAULT or args.debug

    # Compute control separation metrics plate-by-plate for both replicates.
    df = pd.read_csv(args.input_csv)
    debug_log(f"Loaded analyzed data: {args.input_csv} ({len(df)} rows)", debug_enabled)
    rep1, rep2, labels = [], [], []
    for plate_label, sub in df.groupby("Plate_number_384", sort=True):
        rep1.append(calculate_ssmd_ctrls(sub, "Raw_rep1", filter_nt=True))
        rep2.append(calculate_ssmd_ctrls(sub, "Raw_rep2", filter_nt=True))
        labels.append(str(plate_label))
    debug_log(f"Computed QC metrics for {len(rep1)} plates", debug_enabled)

    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    if not args.interactive_only:
        # Render and persist the static QC panel when requested.
        fig, _ = plot_plate_qualities(rep1, rep2, ylabel="SSMD controls", plate_labels=labels)
        fig.savefig(args.output_png, dpi=600, bbox_inches="tight")
        plt.close(fig)
        debug_log(f"Wrote QC figure: {args.output_png}", debug_enabled)
    else:
        debug_log("Skipping static QC PNG (--interactive-only).", debug_enabled)

    html_path = _write_interactive_qc_html(rep1, rep2, labels, args.output_png)
    debug_log(f"Wrote interactive QC HTML: {html_path}", debug_enabled)


if __name__ == "__main__":
    run_qc_cli()
