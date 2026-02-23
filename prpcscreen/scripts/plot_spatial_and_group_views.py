from __future__ import annotations

import os
import sys
import argparse
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from prpcscreen.visualization.box_plots import beebox_plates
from prpcscreen.visualization.plotly_exports import write_plotly_interactive_html

DEBUG_ENV_DEFAULT = os.environ.get("PRPCSCREEN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
HEATMAP_REPLICATE_COLUMNS = [("Raw_rep1", "Replicate 1"), ("Raw_rep2", "Replicate 2")]


def debug_log(message: str, enabled: bool) -> None:
    """Emit debug lines for spatial/group summary plotting."""
    if enabled:
        print(f"[plot_spatial_and_group_views] {message}", file=sys.stderr)


def _write_interactive_heatmap_html(df: pd.DataFrame, plate: str, output_png: Path) -> Path:
    matrix_rep1 = _plate_matrix(df, plate, "Raw_rep1")
    matrix_rep2 = _plate_matrix(df, plate, "Raw_rep2")
    matrix_diff = matrix_rep1 - matrix_rep2
    finite_values = np.concatenate(
        [arr[np.isfinite(arr)] for arr in (matrix_rep1, matrix_rep2) if np.isfinite(arr).any()]
    )
    if finite_values.size:
        vmin = float(np.nanmin(finite_values))
        vmax = float(np.nanmax(finite_values))
    else:
        vmin, vmax = 0.0, 1.0
    diff_finite = matrix_diff[np.isfinite(matrix_diff)]
    if diff_finite.size:
        diff_abs_max = float(np.nanmax(np.abs(diff_finite)))
        if diff_abs_max == 0.0:
            diff_abs_max = 1.0
    else:
        diff_abs_max = 1.0

    y_labels = [chr(ord("A") + i) for i in range(16)]
    x_labels = [str(i + 1) for i in range(24)]
    traces = [
        {
            "type": "heatmap",
            "z": matrix_rep1.tolist(),
            "xaxis": "x",
            "yaxis": "y",
            "x": x_labels,
            "y": y_labels,
            "colorscale": "Viridis",
            "zmin": vmin,
            "zmax": vmax,
            "colorbar": {"title": "Raw value", "x": 1.01},
            "hovertemplate": "Row %{y} / Col %{x}<br>Value: %{z:.4f}<extra></extra>",
        },
        {
            "type": "heatmap",
            "z": matrix_rep2.tolist(),
            "xaxis": "x2",
            "yaxis": "y2",
            "x": x_labels,
            "y": y_labels,
            "colorscale": "Viridis",
            "zmin": vmin,
            "zmax": vmax,
            "showscale": False,
            "hovertemplate": "Row %{y} / Col %{x}<br>Value: %{z:.4f}<extra></extra>",
        },
        {
            "type": "heatmap",
            "z": matrix_diff.tolist(),
            "xaxis": "x3",
            "yaxis": "y3",
            "x": x_labels,
            "y": y_labels,
            "colorscale": "RdBu",
            "zmid": 0.0,
            "zmin": -diff_abs_max,
            "zmax": diff_abs_max,
            "colorbar": {"title": "Rep1 - Rep2", "x": 1.10},
            "hovertemplate": "Row %{y} / Col %{x}<br>Diff: %{z:.4f}<extra></extra>",
        }
    ]
    layout = {
        "paper_bgcolor": "#f6f6f6",
        "plot_bgcolor": "#ffffff",
        "margin": {"l": 70, "r": 140, "t": 54, "b": 70},
        "xaxis": {"title": "Column", "domain": [0.0, 0.28], "constrain": "domain"},
        "xaxis2": {"title": "Column", "domain": [0.34, 0.62], "constrain": "domain"},
        "xaxis3": {"title": "Column", "domain": [0.68, 0.96], "constrain": "domain"},
        "yaxis": {"title": "Row", "autorange": "reversed", "scaleanchor": "x", "scaleratio": 1, "constrain": "domain"},
        "yaxis2": {"title": "Row", "autorange": "reversed", "anchor": "x2", "scaleanchor": "x2", "scaleratio": 1, "constrain": "domain"},
        "yaxis3": {"title": "Row", "autorange": "reversed", "anchor": "x3", "scaleanchor": "x3", "scaleratio": 1, "constrain": "domain"},
        "annotations": [
            {"text": "Replicate 1", "xref": "paper", "yref": "paper", "x": 0.14, "y": 1.08, "showarrow": False},
            {"text": "Replicate 2", "xref": "paper", "yref": "paper", "x": 0.48, "y": 1.08, "showarrow": False},
            {"text": "Rep1 - Rep2", "xref": "paper", "yref": "paper", "x": 0.82, "y": 1.08, "showarrow": False},
        ],
    }
    output_html = output_png.with_name(output_png.stem + "_interactive.html")
    return write_plotly_interactive_html(
        output_html=output_html,
        traces=traces,
        layout=layout,
        title=f"Plate {plate} heatmap (replicates + difference)",
        filename_base=f"plate_heatmap_replicates_plate{plate}",
    )


def _enforce_square_wells(axis: plt.Axes) -> None:
    """Keep each heatmap well square (1:1 width/height)."""
    axis.set_aspect("equal", adjustable="box")
    axis.set_box_aspect(16 / 24)


def _plate_matrix(df: pd.DataFrame, plate: str, column: str) -> np.ndarray:
    sub = df[df["Plate_number_384"].astype(str) == str(plate)].copy().sort_values("Well_number_384")
    values = pd.to_numeric(sub[column], errors="coerce").to_numpy(dtype=float)
    if values.size != 384:
        raise ValueError(f"Expected 384 wells for plate {plate}, got {values.size}")
    return values.reshape(16, 24)


def _cleanup_heatmap_outputs(base_png: Path) -> None:
    prefix = base_png.stem
    parent = base_png.parent
    for candidate in parent.glob(f"{prefix}*"):
        if candidate.is_file() and candidate.suffix.lower() in {".png", ".html", ".svg", ".gz"}:
            candidate.unlink(missing_ok=True)


def _write_single_plate_heatmap_png(df: pd.DataFrame, plate: str, output_png: Path) -> None:
    matrices = [(label, _plate_matrix(df, plate, col)) for col, label in HEATMAP_REPLICATE_COLUMNS]
    matrix_diff = matrices[0][1] - matrices[1][1]
    finite_values = np.concatenate([m[np.isfinite(m)] for _, m in matrices if np.isfinite(m).any()])
    if finite_values.size:
        vmin = float(np.nanmin(finite_values))
        vmax = float(np.nanmax(finite_values))
    else:
        vmin, vmax = 0.0, 1.0
    diff_finite = matrix_diff[np.isfinite(matrix_diff)]
    if diff_finite.size:
        diff_abs_max = float(np.nanmax(np.abs(diff_finite)))
        if diff_abs_max == 0.0:
            diff_abs_max = 1.0
    else:
        diff_abs_max = 1.0

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.6), constrained_layout=True)
    x_ticks = [0, 5, 11, 17, 23]
    x_tick_labels = ["1", "6", "12", "18", "24"]
    y_ticks = [0, 3, 7, 11, 15]
    y_tick_labels = ["A", "D", "H", "L", "P"]
    image_value = None
    flat_axes = np.atleast_1d(axes).ravel()
    for axis, (label, matrix) in zip(flat_axes[:2], matrices):
        image_value = axis.imshow(matrix, cmap="viridis", origin="upper", aspect="equal", vmin=vmin, vmax=vmax)
        _enforce_square_wells(axis)
        axis.set_title(f"{label}", fontsize=10)
        axis.set_xticks(x_ticks)
        axis.set_xticklabels(x_tick_labels, fontsize=8)
        axis.set_yticks(y_ticks)
        axis.set_yticklabels(y_tick_labels, fontsize=8)
        axis.set_xlabel("Column", fontsize=8)
        axis.set_ylabel("Row", fontsize=8)

    axis_diff = flat_axes[2]
    image_diff = axis_diff.imshow(
        matrix_diff,
        cmap="RdBu",
        origin="upper",
        aspect="equal",
        vmin=-diff_abs_max,
        vmax=diff_abs_max,
    )
    _enforce_square_wells(axis_diff)
    axis_diff.set_title("Rep1 - Rep2", fontsize=10)
    axis_diff.set_xticks(x_ticks)
    axis_diff.set_xticklabels(x_tick_labels, fontsize=8)
    axis_diff.set_yticks(y_ticks)
    axis_diff.set_yticklabels(y_tick_labels, fontsize=8)
    axis_diff.set_xlabel("Column", fontsize=8)
    axis_diff.set_ylabel("Row", fontsize=8)

    if image_value is not None:
        fig.colorbar(image_value, ax=flat_axes[:2].tolist(), shrink=0.9, label="Raw value")
    fig.colorbar(image_diff, ax=[axis_diff], shrink=0.9, label="Rep1 - Rep2")
    fig.suptitle(f"Plate {plate}: replicates and difference", fontsize=11)
    fig.savefig(output_png, dpi=180, bbox_inches="tight")
    plt.close(fig)


def _write_collection_heatmap_svg(df: pd.DataFrame, selected_plates: list[str], output_png: Path) -> Path:
    plate_data: dict[str, dict[str, np.ndarray | float]] = {}
    for plate in selected_plates:
        rep1 = _plate_matrix(df, plate, "Raw_rep1")
        rep2 = _plate_matrix(df, plate, "Raw_rep2")
        diff = rep1 - rep2

        rep_values = np.concatenate([m[np.isfinite(m)] for m in (rep1, rep2) if np.isfinite(m).any()])
        if rep_values.size:
            vmin = float(np.nanmin(rep_values))
            vmax = float(np.nanmax(rep_values))
        else:
            vmin, vmax = 0.0, 1.0

        diff_values = diff[np.isfinite(diff)]
        if diff_values.size:
            diff_abs_max = float(np.nanmax(np.abs(diff_values)))
            if diff_abs_max == 0.0:
                diff_abs_max = 1.0
        else:
            diff_abs_max = 1.0

        plate_data[plate] = {
            "rep1": rep1,
            "rep2": rep2,
            "diff": diff,
            "vmin": vmin,
            "vmax": vmax,
            "diff_abs_max": diff_abs_max,
        }

    rows = len(selected_plates)
    cols = len(HEATMAP_REPLICATE_COLUMNS) + 1
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 4.1, rows * 3.0), constrained_layout=True)
    axes_arr = np.atleast_2d(axes)

    x_ticks = [0, 5, 11, 17, 23]
    x_tick_labels = ["1", "6", "12", "18", "24"]
    y_ticks = [0, 3, 7, 11, 15]
    y_tick_labels = ["A", "D", "H", "L", "P"]
    for r, plate in enumerate(selected_plates):
        row_data = plate_data[plate]
        vmin = float(row_data["vmin"])
        vmax = float(row_data["vmax"])
        diff_abs_max = float(row_data["diff_abs_max"])
        row_raw_image = None
        row_diff_image = None
        for c, (_, label) in enumerate(HEATMAP_REPLICATE_COLUMNS):
            matrix = row_data["rep1"] if c == 0 else row_data["rep2"]
            axis = axes_arr[r, c]
            row_raw_image = axis.imshow(matrix, cmap="viridis", origin="upper", aspect="equal", vmin=vmin, vmax=vmax)
            _enforce_square_wells(axis)
            axis.set_title(f"Plate {plate} - {label}", fontsize=10)
            axis.set_xticks(x_ticks)
            axis.set_xticklabels(x_tick_labels, fontsize=8)
            axis.set_yticks(y_ticks)
            axis.set_yticklabels(y_tick_labels, fontsize=8)
            axis.set_xlabel("Column", fontsize=8)
            axis.set_ylabel("Row", fontsize=8)
        diff_axis = axes_arr[r, 2]
        image_diff = diff_axis.imshow(
            row_data["diff"],
            cmap="RdBu",
            origin="upper",
            aspect="equal",
            vmin=-diff_abs_max,
            vmax=diff_abs_max,
        )
        row_diff_image = image_diff
        _enforce_square_wells(diff_axis)
        diff_axis.set_title(f"Plate {plate} - Rep1 - Rep2", fontsize=10)
        diff_axis.set_xticks(x_ticks)
        diff_axis.set_xticklabels(x_tick_labels, fontsize=8)
        diff_axis.set_yticks(y_ticks)
        diff_axis.set_yticklabels(y_tick_labels, fontsize=8)
        diff_axis.set_xlabel("Column", fontsize=8)
        diff_axis.set_ylabel("Row", fontsize=8)

        if row_raw_image is not None:
            fig.colorbar(row_raw_image, ax=axes_arr[r, :2].ravel().tolist(), shrink=0.92, label="Raw value")
        if row_diff_image is not None:
            fig.colorbar(row_diff_image, ax=[axes_arr[r, 2]], shrink=0.92, label="Rep1 - Rep2")

    output_svg = output_png.with_name(f"{output_png.stem}_collection.svg")
    fig.savefig(output_svg, format="svg", bbox_inches="tight")
    plt.close(fig)
    return output_svg


def _write_interactive_grouped_html(df: pd.DataFrame, column: str, output_png: str) -> Path:
    plot_df = df.copy()
    if "Target_flag" in plot_df:
        flag = plot_df["Target_flag"].fillna("").astype(str).str.strip().str.lower()
        own_non_targeting = flag.isin({"own non-targeting control", "own non targeting control", "own nt control"})
        plot_df["group"] = "Gene"
        plot_df.loc[plot_df["Is_pos_ctrl"].astype(bool), "group"] = "Positive"
        plot_df.loc[plot_df["Is_NT_ctrl"].astype(bool), "group"] = "Non-targeting"
        plot_df.loc[own_non_targeting, "group"] = "Own Non-targeting"
    else:
        plot_df["group"] = "Gene"
        plot_df.loc[plot_df["Is_pos_ctrl"].astype(bool), "group"] = "Positive"
        plot_df.loc[plot_df["Is_NT_ctrl"].astype(bool), "group"] = "Non-targeting"

    order = [g for g in ["Gene", "Non-targeting", "Own Non-targeting", "Positive"] if g in plot_df["group"].unique()]
    palette = {
        "Gene": "#90caf9",
        "Non-targeting": "#3b82f6",
        "Own Non-targeting": "#0ea5e9",
        "Positive": "#ef4444",
    }
    traces: list[dict] = []
    for group in order:
        vals = pd.to_numeric(plot_df.loc[plot_df["group"] == group, column], errors="coerce").dropna().astype(float)
        vals_list = vals.tolist()
        traces.append(
            {
                "type": "violin",
                "name": group,
                "y": vals_list,
                "box": {"visible": False},
                "meanline": {"visible": False},
                "points": False,
                "line": {"color": palette.get(group, "#1f1f1f"), "width": 1.0},
                "fillcolor": palette.get(group, "#90caf9"),
                "opacity": 0.6,
                "hovertemplate": f"{group}<br>{column}: %{{y:.4f}}<extra></extra>",
                "legendgroup": group,
                "showlegend": True,
            }
        )
        traces.append(
            {
                "type": "box",
                "name": f"{group} box",
                "x": [group] * len(vals_list),
                "y": vals_list,
                "boxpoints": False,
                "marker": {"opacity": 0},
                "line": {"color": "#111111", "width": 1.4},
                "fillcolor": "rgba(255,255,255,0.35)",
                "whiskerwidth": 0.6,
                "width": 0.22,
                "hovertemplate": f"{group} box<br>{column}: %{{y:.4f}}<extra></extra>",
                "legendgroup": group,
                "showlegend": False,
            }
        )

    layout = {
        "paper_bgcolor": "#f6f6f6",
        "plot_bgcolor": "#ffffff",
        "margin": {"l": 70, "r": 30, "t": 35, "b": 70},
        "xaxis": {"title": "Group"},
        "yaxis": {"title": column},
        "violinmode": "group",
        "boxmode": "overlay",
    }
    output_html = Path(output_png).with_name(Path(output_png).stem + "_interactive.html")
    return write_plotly_interactive_html(
        output_html=output_html,
        traces=traces,
        layout=layout,
        title=f"Grouped violin/box ({column})",
        filename_base=f"grouped_boxplot_{column.lower()}",
    )


def plate_sort_key(plate: str) -> tuple[int, int | str]:
    # Keep numeric plate labels in natural order, then non-numeric labels.
    try:
        return (0, int(str(plate)))
    except ValueError:
        return (1, str(plate))


def parse_plate_selector(selector: str, available_complete_plates: list[str]) -> list[str]:
    """
    Parse a heatmap plate selector.

    Supported formats:
    - single number: "1"
    - numeric range: "1-4"
    - numeric series: "1,2,6"
    - all plates: "all"
    """
    requested = str(selector).strip().lower()
    if not requested:
        raise ValueError("Heatmap plate selector is required.")

    values: list[str]
    sorted_available = sorted(available_complete_plates, key=plate_sort_key)
    has_numeric_labels = any(re.fullmatch(r"\d+", p) for p in available_complete_plates)
    ordinal_index = {str(i + 1): plate for i, plate in enumerate(sorted_available)}
    if requested == "all":
        values = sorted_available
    else:
        m_range = re.fullmatch(r"(\d+)\s*-\s*(\d+)", requested)
        if m_range:
            start = int(m_range.group(1))
            end = int(m_range.group(2))
            if start > end:
                raise ValueError("Heatmap plate range must be ascending (for example, '1-4').")
            values = [str(v) for v in range(start, end + 1)]
        elif re.fullmatch(r"\d+(?:\s*,\s*\d+)+", requested):
            values = [v.strip() for v in requested.split(",")]
        elif re.fullmatch(r"\d+", requested):
            values = [requested]
        else:
            raise ValueError(
                "Invalid heatmap plate selector. Use one number (for example '1'), "
                "a range ('1-4'), a series ('1,2,6'), or 'all'."
            )

    selected: list[str] = []
    missing: list[str] = []
    available_set = set(available_complete_plates)
    for value in values:
        if value in available_set:
            if value not in selected:
                selected.append(value)
        elif (not has_numeric_labels) and (value in ordinal_index):
            mapped_plate = ordinal_index[value]
            if mapped_plate not in selected:
                selected.append(mapped_plate)
            print(
                f"[plot_spatial_and_group_views] INFO: Interpreting selector '{value}' as plate position -> '{mapped_plate}'.",
                file=sys.stderr,
            )
        else:
            missing.append(value)

    if missing:
        print(
            f"[plot_spatial_and_group_views] WARNING: Requested plate(s) not found as complete 384-well sets: {', '.join(missing)}",
            file=sys.stderr,
        )

    if not selected:
        available_preview = ", ".join(sorted(available_complete_plates, key=plate_sort_key)[:20])
        raise ValueError(
            "No requested plate was available as a complete 384-well set. "
            f"Requested='{selector}'. Available (first 20): {available_preview}"
        )

    return sorted(selected, key=plate_sort_key)


def build_heatmap_outputs(path: str, selected_plates: list[str]) -> list[Path]:
    # Keep the exact output path for one plate; fan out with suffixes for multi-plate runs.
    base = Path(path)
    if len(selected_plates) == 1:
        return [base]
    suffix = base.suffix or ".png"
    stem = base.stem if base.suffix else base.name
    return [base.with_name(f"{stem}_plate{plate}{suffix}") for plate in selected_plates]


def run_spatial_cli() -> None:
    # Parse analyzed input, heatmap target, combined violin/box target, and plate selector.
    parser = argparse.ArgumentParser(description="Generate heatmap and combined violin/box plots.")
    parser.add_argument("input_csv")
    parser.add_argument("heatmap_png")
    parser.add_argument("boxplot_png")
    parser.add_argument("--plate", default="1")
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()
    debug_enabled = DEBUG_ENV_DEFAULT or args.debug

    # Load analyzed dataset once and create both spatial and grouped summaries.
    df = pd.read_csv(args.input_csv)
    debug_log(f"Loaded analyzed data: {args.input_csv} ({len(df)} rows)", debug_enabled)
    if "Plate_number_384" not in df.columns:
        raise ValueError("Missing required column: Plate_number_384")

    plate_series = df["Plate_number_384"].astype(str)
    counts = plate_series.value_counts()
    complete_plates = [str(p) for p, c in counts.items() if int(c) == 384]
    if not complete_plates:
        available = ", ".join(sorted(counts.index.astype(str).tolist())[:20])
        raise ValueError(f"No complete 384-well plates found. Available (first 20): {available}")

    selected_plates = sorted(complete_plates, key=plate_sort_key)
    debug_log(f"Heatmap plate selector: {args.plate}", debug_enabled)
    debug_log(
        "Heatmap output forced to all complete plates (selector retained for compatibility only).",
        debug_enabled,
    )
    debug_log(f"Heatmap plates resolved: {selected_plates}", debug_enabled)

    base_heatmap_path = Path(args.heatmap_png)
    base_heatmap_path.parent.mkdir(parents=True, exist_ok=True)
    _cleanup_heatmap_outputs(base_heatmap_path)
    if len(selected_plates) == 1:
        output_paths = build_heatmap_outputs(args.heatmap_png, selected_plates)
        for plate_for_heatmap, output_path in zip(selected_plates, output_paths):
            _write_single_plate_heatmap_png(df, plate_for_heatmap, output_path)
            debug_log(f"Wrote heatmap figure for plate {plate_for_heatmap}: {output_path}", debug_enabled)
            heatmap_html = _write_interactive_heatmap_html(df, plate_for_heatmap, output_path)
            debug_log(f"Wrote interactive heatmap HTML for plate {plate_for_heatmap}: {heatmap_html}", debug_enabled)
    else:
        output_svg = _write_collection_heatmap_svg(df, selected_plates, base_heatmap_path)
        debug_log(f"Wrote combined multi-plate heatmap SVG: {output_svg}", debug_enabled)

    fig_b, _ = beebox_plates(df, "Raw_rep1")
    Path(args.boxplot_png).parent.mkdir(parents=True, exist_ok=True)
    fig_b.savefig(args.boxplot_png, dpi=180, bbox_inches="tight")
    plt.close(fig_b)
    debug_log(f"Wrote violin/box figure: {args.boxplot_png}", debug_enabled)
    grouped_html = _write_interactive_grouped_html(df, "Raw_rep1", args.boxplot_png)
    debug_log(f"Wrote interactive grouped violin/box HTML: {grouped_html}", debug_enabled)


if __name__ == "__main__":
    run_spatial_cli()
