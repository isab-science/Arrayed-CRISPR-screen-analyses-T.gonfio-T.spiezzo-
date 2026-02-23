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

from prpcscreen.visualization.plate_well_series_plots import plate_well_plot
from prpcscreen.visualization.plotly_exports import write_plotly_interactive_html

DEBUG_ENV_DEFAULT = os.environ.get("PRPCSCREEN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def debug_log(message: str, enabled: bool) -> None:
    """Emit debug lines for plate/well trend plotting."""
    if enabled:
        print(f"[05_plot_well_trajectories] {message}", file=sys.stderr)


def _write_interactive_trajectory_html(df: pd.DataFrame, use_column: str, output_png: str) -> Path:
    if "Well_number_384" in df.columns:
        sub = df.sort_values("Well_number_384").copy()
        x_title = "Well order"
    else:
        sub = df.reset_index(drop=True).copy()
        x_title = "Feature order"
    values = pd.to_numeric(sub[use_column], errors="coerce")
    idx = np.arange(1, len(values) + 1, dtype=float)
    valid = values.notna()
    traces = [
        {
            "x": idx[valid].tolist(),
            "y": values.loc[valid].astype(float).tolist(),
            "mode": "lines",
            "type": "scatter",
            "line": {"color": "#1f77b4", "width": 1.0},
            "name": use_column,
            "hovertemplate": "Ordered well: %{x}<br>Value: %{y:.4f}<extra></extra>",
        }
    ]
    layout = {
        "paper_bgcolor": "#f6f6f6",
        "plot_bgcolor": "#ffffff",
        "margin": {"l": 70, "r": 30, "t": 35, "b": 60},
        "xaxis": {"title": x_title},
        "yaxis": {"title": use_column},
    }
    output_html = Path(output_png).with_name(Path(output_png).stem + "_interactive.html")
    file_base = f"plate_well_series_{use_column.lower()}"
    return write_plotly_interactive_html(
        output_html=output_html,
        traces=traces,
        layout=layout,
        title=f"Plate-well series ({use_column})",
        filename_base=file_base,
    )


def run_trajectory_cli() -> None:
    # Parse source table, destination image, and metric selection.
    parser = argparse.ArgumentParser(description="Generate plate-well series plots.")
    parser.add_argument("input_csv")
    parser.add_argument("output_png")
    parser.add_argument("--column", default="Raw_rep1")
    parser.add_argument(
        "--interactive-only",
        action="store_true",
        help="Only export interactive HTML (skip static PNG output).",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()
    debug_enabled = DEBUG_ENV_DEFAULT or args.debug

    # Load analyzed data and render requested trajectory panel.
    df = pd.read_csv(args.input_csv)
    debug_log(f"Loaded analyzed data: {args.input_csv} ({len(df)} rows)", debug_enabled)
    debug_log(f"Using column for trajectories: {args.column}", debug_enabled)
    Path(args.output_png).parent.mkdir(parents=True, exist_ok=True)
    if not args.interactive_only:
        fig, _ = plate_well_plot(df, use_column=args.column)
        fig.savefig(args.output_png, dpi=180, bbox_inches="tight")
        plt.close(fig)
        debug_log(f"Wrote trajectory figure: {args.output_png}", debug_enabled)
    else:
        debug_log("Skipping static trajectory PNG (--interactive-only).", debug_enabled)

    html_path = _write_interactive_trajectory_html(df, args.column, args.output_png)
    debug_log(f"Wrote interactive trajectory HTML: {html_path}", debug_enabled)


if __name__ == "__main__":
    run_trajectory_cli()

