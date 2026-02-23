from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Sequence

import matplotlib.pyplot as plt
import pandas as pd

ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

from prpcscreen.visualization.plotly_exports import write_plotly_interactive_html
from prpcscreen.visualization.volcano_and_flashlight_plots import _build_sublibrary_series

DEBUG_ENV_DEFAULT = os.environ.get("PRPCSCREEN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}
REQUIRED_SKYLINE_COLUMNS = ("gene", "log2fc", "chrom", "pos")
REQUIRED_SKYLINE_COLUMN_LABELS = {
    "gene": "Gene_symbol",
    "log2fc": "Mean_log2FC",
    "chrom": "Chromosome",
    "pos": "Start_Position",
}
SKYLINE_COLUMN_ALIASES = {
    "gene": ("Gene_symbol", "Gene symbol", "GeneSymbol", "Gene", "Symbol", "gene"),
    "log2fc": ("Mean_log2FC", "Mean_log2", "mean_log2fc", "mean_log2", "log2fc"),
    "chrom": ("Chromosome", "chromosome", "Chrom", "Chr", "chrom"),
    "pos": ("Start_Position", "Start position", "StartPosition", "Start", "pos"),
}
SUBLIBRARY_COLUMN_ALIASES = (
    "Sublibrary",
    "SubLibrary",
    "Sub-Library",
    "sub_library",
    "library",
    "Library",
)


def debug_log(message: str, enabled: bool) -> None:
    """Emit debug lines for genomic skyline plotting."""
    if enabled:
        print(f"[genomic_signal_skyline] {message}", file=sys.stderr)


def _resolve_column_mapping(columns: Sequence[object]) -> tuple[dict[str, str], list[str]]:
    lookup: dict[str, str] = {}
    for col in columns:
        text = str(col).strip()
        key = text.lower()
        if key and key not in lookup:
            lookup[key] = text

    mapping: dict[str, str] = {}
    missing: list[str] = []
    for canonical in REQUIRED_SKYLINE_COLUMNS:
        matched = None
        for candidate in SKYLINE_COLUMN_ALIASES[canonical]:
            matched = lookup.get(candidate.strip().lower())
            if matched:
                break
        if matched:
            mapping[canonical] = matched
        else:
            missing.append(canonical)
    return mapping, missing


def _required_column_names(missing: Sequence[str] | None = None) -> str:
    names = (
        [REQUIRED_SKYLINE_COLUMN_LABELS[c] for c in missing]
        if missing is not None
        else [REQUIRED_SKYLINE_COLUMN_LABELS[c] for c in REQUIRED_SKYLINE_COLUMNS]
    )
    return ", ".join(names)


def _canonicalize_skyline_columns(data: pd.DataFrame, sheet_name: str, input_excel: str) -> pd.DataFrame:
    mapping, missing = _resolve_column_mapping(list(data.columns))
    if missing:
        expected = _required_column_names(missing)
        actual = ", ".join(map(str, data.columns.tolist()[:12]))
        raise ValueError(
            f"Worksheet '{sheet_name}' in workbook '{input_excel}' is missing required skyline column(s): {expected}. "
            f"Available columns (first 12): {actual}"
        )
    rename = {mapping["gene"]: "gene", mapping["log2fc"]: "log2fc", mapping["chrom"]: "chrom", mapping["pos"]: "pos"}
    return data.rename(columns=rename)


def _chromosome_rank_token(chrom: str) -> tuple[int, str]:
    # Keep numeric chromosomes in natural order and push non-numeric labels after them.
    c = str(chrom).replace("chr", "")
    if c.isdigit():
        return (0, f"{int(c):02d}")
    return (1, c)


def _normalize_chromosome_label(chrom: object) -> str | None:
    raw = str(chrom).strip()
    if not raw or raw.lower() == "nan":
        return None
    c = raw.upper().replace("CHR", "")
    if c == "MT":
        c = "M"
    canonical = {str(i) for i in range(1, 23)} | {"X", "Y"}
    return c if c in canonical else None


def _resolve_sublibrary_column(columns: Sequence[object]) -> str | None:
    lookup: dict[str, str] = {}
    for col in columns:
        text = str(col).strip()
        key = text.lower()
        if key and key not in lookup:
            lookup[key] = text
    for alias in SUBLIBRARY_COLUMN_ALIASES:
        found = lookup.get(alias.strip().lower())
        if found:
            return found
    return None


def _resolve_sheet_name(input_excel: str, requested_sheet: str, debug_enabled: bool = False) -> str:
    # Resolve sheet robustly: prefer requested sheet if schema-compatible, then Skyline-like, then any compatible sheet.
    xls = pd.ExcelFile(input_excel)
    available: Sequence[str] = xls.sheet_names
    if not available:
        raise ValueError(f"No worksheets found in workbook: {input_excel}")

    requested_norm = requested_sheet.strip().lower()
    header_cache: dict[str, tuple[bool, list[str], str]] = {}

    def inspect_sheet(sheet_name: str) -> tuple[bool, list[str], str]:
        cached = header_cache.get(sheet_name)
        if cached is not None:
            return cached
        try:
            header = pd.read_excel(input_excel, sheet_name=sheet_name, nrows=0)
        except Exception as exc:  # noqa: BLE001
            info = (False, list(REQUIRED_SKYLINE_COLUMNS), f"Header read failed: {exc}")
            header_cache[sheet_name] = info
            return info

        _, missing = _resolve_column_mapping(list(header.columns))
        if missing:
            info = (False, missing, "")
        else:
            info = (True, [], "")
        header_cache[sheet_name] = info
        return info

    requested_candidates: list[str] = []
    if requested_sheet in available:
        requested_candidates.append(requested_sheet)
    requested_candidates.extend(
        [s for s in available if s.strip().lower() == requested_norm and s not in requested_candidates]
    )
    for sheet_name in requested_candidates:
        ok, _, _ = inspect_sheet(sheet_name)
        if ok:
            return sheet_name

    if requested_candidates:
        first_bad = requested_candidates[0]
        _, missing, err = inspect_sheet(first_bad)
        if err:
            print(
                f"[genomic_signal_skyline] WARNING: Worksheet '{first_bad}' could not be used ({err}).",
                file=sys.stderr,
            )
        else:
            print(
                f"[genomic_signal_skyline] WARNING: Worksheet '{first_bad}' is missing required columns: {_required_column_names(missing)}.",
                file=sys.stderr,
            )

    skyline_like = [s for s in available if "skyline" in s.strip().lower()]
    if skyline_like:
        compatible = [s for s in skyline_like if inspect_sheet(s)[0]]
        if compatible:
            fallback = compatible[0]
            print(
                f"[genomic_signal_skyline] WARNING: Worksheet '{requested_sheet}' not found/usable; using Skyline-like sheet '{fallback}'.",
                file=sys.stderr,
            )
            return fallback

    compatible_any = [s for s in available if inspect_sheet(s)[0]]
    if compatible_any:
        fallback = compatible_any[0]
        print(
            f"[genomic_signal_skyline] WARNING: Worksheet '{requested_sheet}' not found/usable; using compatible sheet '{fallback}'.",
            file=sys.stderr,
        )
        return fallback

    checked: list[str] = []
    max_report = 12
    for sheet_name in available[:max_report]:
        _, missing, err = inspect_sheet(sheet_name)
        if err:
            checked.append(f"'{sheet_name}' ({err})")
        else:
            checked.append(f"'{sheet_name}' missing: {_required_column_names(missing)}")
    if len(available) > max_report:
        checked.append(f"... {len(available) - max_report} additional sheet(s) omitted")
    debug_log(f"Available sheets: {available}", debug_enabled)
    raise ValueError(
        f"No worksheet in workbook '{input_excel}' contains the required skyline columns "
        f"({_required_column_names()}). Checked: {'; '.join(checked)}. "
        "This usually means the selected workbook is a layout/raw workbook rather than the genomics workbook."
    )


def render_chromosome_signal_map(
    input_excel: str,
    sheet: str,
    output_png: str,
    interactive_only: bool = False,
    debug_enabled: bool = False,
) -> None:
    # Load and normalize expected input column names.
    resolved_sheet = _resolve_sheet_name(input_excel, sheet, debug_enabled=debug_enabled)
    data = pd.read_excel(input_excel, sheet_name=resolved_sheet)
    data = _canonicalize_skyline_columns(data, resolved_sheet, input_excel)
    sublibrary_input = pd.DataFrame({"Gene_symbol": data["gene"]})
    mapped_sublibrary, sublibrary_options = _build_sublibrary_series(sublibrary_input, input_excel)
    sublibrary_options = [str(opt) for opt in sublibrary_options if str(opt).strip().lower() != "unknown"]
    data["_sublibrary"] = mapped_sublibrary.fillna("").astype(str).str.strip()
    data["_sublibrary"] = data["_sublibrary"].where(data["_sublibrary"] != "", "Unknown")
    data["log2fc"] = pd.to_numeric(data["log2fc"], errors="coerce")
    data["pos"] = pd.to_numeric(data["pos"], errors="coerce")
    data = data.dropna(subset=["chrom", "pos", "log2fc"]).copy()
    data["chrom"] = data["chrom"].map(_normalize_chromosome_label)
    dropped_noncanonical = int(data["chrom"].isna().sum())
    data = data.dropna(subset=["chrom"]).copy()
    if dropped_noncanonical:
        print(
            f"[genomic_signal_skyline] WARNING: Dropped {dropped_noncanonical} rows with non-canonical chromosome labels.",
            file=sys.stderr,
        )
    debug_log(f"Loaded {len(data)} rows from sheet '{resolved_sheet}'", debug_enabled)

    # Build chromosome ordering and x-axis offsets so each chromosome occupies a contiguous block.
    chrom_order = sorted(data["chrom"].astype(str).unique(), key=_chromosome_rank_token)
    offsets: dict[str, float] = {}
    acc = 0.0
    xticks = []
    xticklabels = []

    for chrom in chrom_order:
        sub = data[data["chrom"].astype(str) == chrom]
        min_pos, max_pos = sub["pos"].min(), sub["pos"].max()
        offsets[chrom] = acc - min_pos
        xticks.append(acc + (max_pos - min_pos) / 2)
        xticklabels.append(chrom.replace("chr", ""))
        acc += (max_pos - min_pos) + 1e7
    debug_log(f"Chromosome blocks: {len(chrom_order)}", debug_enabled)

    # Translate genomic positions to one continuous x-axis.
    data["x"] = data.apply(lambda r: float(r["pos"]) + offsets[str(r["chrom"])], axis=1)

    # Draw alternating-color scatter for visual separation between adjacent chromosomes.
    fig, ax = plt.subplots(figsize=(14, 7))
    palette = ["#9a7e6f", "#caa18d"]
    for i, chrom in enumerate(chrom_order):
        sub = data[data["chrom"].astype(str) == chrom]
        ax.scatter(sub["x"], sub["log2fc"], s=10, c=palette[i % 2], linewidths=0)

    # Final axis styling and figure persistence.
    ax.set_title("Skyline Plot")
    ax.set_ylabel("Mean log2FC")
    ax.set_xlabel("Chromosome")
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticklabels)
    ax.grid(axis="y", alpha=0.2)

    Path(output_png).parent.mkdir(parents=True, exist_ok=True)
    if not interactive_only:
        fig.savefig(output_png, dpi=200, bbox_inches="tight")
        debug_log(f"Wrote skyline figure: {output_png}", debug_enabled)
    else:
        Path(output_png).unlink(missing_ok=True)
    plt.close(fig)

    color_by_chrom = {chrom: palette[i % 2] for i, chrom in enumerate(chrom_order)}
    plot_source = data[["gene", "x", "log2fc", "chrom", "_sublibrary"]].copy()
    plot_source["gene"] = plot_source["gene"].fillna("").astype(str).str.strip()
    plot_source["sublibrary"] = plot_source["_sublibrary"].fillna("Unknown").astype(str)
    plot_source["color"] = plot_source["chrom"].astype(str).map(color_by_chrom).fillna(palette[0])

    point_payload = {
        "gene": plot_source["gene"].tolist(),
        "x": plot_source["x"].astype(float).tolist(),
        "log2fc": plot_source["log2fc"].astype(float).tolist(),
        "sublibrary": plot_source["sublibrary"].tolist(),
        "color": plot_source["color"].tolist(),
    }

    traces: list[dict] = [
        {
            "type": "scattergl",
            "mode": "markers",
            "name": "All genes",
            "x": point_payload["x"],
            "y": point_payload["log2fc"],
            "text": point_payload["gene"],
            "customdata": point_payload["sublibrary"],
            "marker": {"color": point_payload["color"], "size": 4, "opacity": 0.9},
            "hovertemplate": (
                "Gene: %{text}<br>Sublibrary: %{customdata}<br>x: %{x:.0f}<br>Mean log2FC: %{y:.4f}<extra></extra>"
            ),
            "showlegend": False,
        }
    ]

    label_trace_index = len(traces)
    traces.append(
        {
            "type": "scattergl",
            "mode": "markers+text",
            "name": "Labeled genes",
            "x": [],
            "y": [],
            "text": [],
            "textposition": "top center",
            "marker": {"color": "#d62728", "size": 8, "line": {"color": "#000000", "width": 1}},
            "textfont": {"color": "#111111", "size": 10},
            "showlegend": False,
            "hovertemplate": "Gene: %{text}<br>x: %{x:.0f}<br>Mean log2FC: %{y:.4f}<extra></extra>",
        }
    )

    layout = {
        "paper_bgcolor": "#f6f6f6",
        "plot_bgcolor": "#ffffff",
        "margin": {"l": 70, "r": 30, "t": 35, "b": 70},
        "xaxis": {"title": "Chromosome", "tickmode": "array", "tickvals": xticks, "ticktext": xticklabels},
        "yaxis": {"title": "Mean log2FC"},
    }
    extra_controls_html = (
        "    <div class=\"controls\">\n"
        "      <label for=\"skyline-sublibrary-filter\">Sublibrary</label>\n"
        "      <select id=\"skyline-sublibrary-filter\"></select>\n"
        "      <span id=\"sublibrary-status\"></span>\n"
        "      <label for=\"label-genes\">Label genes (comma separated)</label>\n"
        "      <input id=\"label-genes\" type=\"text\" value=\"PRNP\" placeholder=\"PRNP, APP, SNCA\" "
        "style=\"min-width: 300px; border: 1px solid #d0d7de; border-radius: 6px; padding: 6px 8px; font-size: 13px; background: #ffffff;\">\n"
        "      <label for=\"label-top-n\">Top +/- strongest</label>\n"
        "      <select id=\"label-top-n\">\n"
        "        <option value=\"0\">Off</option>\n"
        "        <option value=\"10\">10</option>\n"
        "        <option value=\"20\" selected>20</option>\n"
        "        <option value=\"50\">50</option>\n"
        "      </select>\n"
        "      <button id=\"labels-apply\" type=\"button\">Apply labels</button>\n"
        "      <button id=\"labels-clear\" type=\"button\">Clear labels</button>\n"
        "      <span id=\"labels-status\"></span>\n"
        "    </div>\n"
    )
    extra_script = (
        "    const SKYLINE_POINT_SOURCE = "
        + json.dumps(point_payload, separators=(",", ":"))
        + ";\n"
        "    const SKYLINE_SUBLIBRARY_OPTIONS = "
        + json.dumps([str(opt) for opt in sublibrary_options], separators=(",", ":"))
        + ";\n"
        "    const SKYLINE_MAIN_TRACE_IDX = 0;\n"
        f"    const SKYLINE_LABEL_TRACE_IDX = {label_trace_index};\n"
        "    function buildPointRows() {\n"
        "      const rows = [];\n"
        "      const g = SKYLINE_POINT_SOURCE.gene || [];\n"
        "      const x = SKYLINE_POINT_SOURCE.x || [];\n"
        "      const y = SKYLINE_POINT_SOURCE.log2fc || [];\n"
        "      const s = SKYLINE_POINT_SOURCE.sublibrary || [];\n"
        "      const c = SKYLINE_POINT_SOURCE.color || [];\n"
        "      const n = Math.min(g.length, x.length, y.length, s.length, c.length);\n"
        "      for (let i = 0; i < n; i += 1) {\n"
        "        const gene = (g[i] || '').toString().trim();\n"
        "        const xVal = Number(x[i]);\n"
        "        const yVal = Number(y[i]);\n"
        "        const sub = (s[i] || 'Unknown').toString().trim() || 'Unknown';\n"
        "        const color = (c[i] || '#9a7e6f').toString();\n"
        "        if (!Number.isFinite(xVal) || !Number.isFinite(yVal)) continue;\n"
        "        rows.push({ gene, key: gene.toUpperCase(), x: xVal, y: yVal, sublibrary: sub, color });\n"
        "      }\n"
        "      return rows;\n"
        "    }\n"
        "    const SKYLINE_ROWS = buildPointRows();\n"
        "    function normalizedSub(value) {\n"
        "      return (value || '').toString().trim().toLowerCase();\n"
        "    }\n"
        "    function populateSublibraryFilter() {\n"
        "      const sel = document.getElementById('skyline-sublibrary-filter');\n"
        "      sel.innerHTML = '';\n"
        "      const all = document.createElement('option');\n"
        "      all.value = '__all__';\n"
        "      all.textContent = 'All sublibraries';\n"
        "      sel.appendChild(all);\n"
        "      for (const sub of SKYLINE_SUBLIBRARY_OPTIONS) {\n"
        "        if (!sub) continue;\n"
        "        const opt = document.createElement('option');\n"
        "        opt.value = sub;\n"
        "        opt.textContent = sub;\n"
        "        sel.appendChild(opt);\n"
        "      }\n"
        "    }\n"
        "    function activeSublibrary() {\n"
        "      const sel = document.getElementById('skyline-sublibrary-filter');\n"
        "      const value = sel ? sel.value : '__all__';\n"
        "      return value && value !== '__all__' ? value : '';\n"
        "    }\n"
        "    function filteredRows() {\n"
        "      const sub = activeSublibrary();\n"
        "      if (!sub) return SKYLINE_ROWS.slice();\n"
        "      const key = normalizedSub(sub);\n"
        "      return SKYLINE_ROWS.filter((r) => normalizedSub(r.sublibrary) === key);\n"
        "    }\n"
        "    function applySublibraryFilter() {\n"
        "      const rows = filteredRows();\n"
        "      const xVals = rows.map((r) => r.x);\n"
        "      const yVals = rows.map((r) => r.y);\n"
        "      const textVals = rows.map((r) => r.gene);\n"
        "      const subVals = rows.map((r) => r.sublibrary);\n"
        "      const colorVals = rows.map((r) => r.color);\n"
        "      Plotly.restyle(\n"
        "        PLOT_DIV_ID,\n"
        "        { x: [xVals], y: [yVals], text: [textVals], customdata: [subVals], 'marker.color': [colorVals] },\n"
        "        [SKYLINE_MAIN_TRACE_IDX]\n"
        "      );\n"
        "      Plotly.restyle(PLOT_DIV_ID, {x: [[]], y: [[]], text: [[]]}, [SKYLINE_LABEL_TRACE_IDX]);\n"
        "      document.getElementById('labels-status').textContent = '';\n"
        "      const sub = activeSublibrary();\n"
        "      const label = sub ? `Sublibrary: ${sub}` : 'All sublibraries';\n"
        "      document.getElementById('sublibrary-status').textContent = `${label} (${rows.length.toLocaleString()} points)`;\n"
        "    }\n"
        "    function parseGeneTokens(raw) {\n"
        "      if (!raw) return [];\n"
        "      const tokens = raw\n"
        "        .split(/[\\s,;]+/)\n"
        "        .map((t) => t.trim())\n"
        "        .filter(Boolean)\n"
        "        .map((t) => t.toUpperCase());\n"
        "      return [...new Set(tokens)];\n"
        "    }\n"
        "    function buildRows() {\n"
        "      return filteredRows().filter((r) => !!(r.gene || '').trim());\n"
        "    }\n"
        "    function strongestPerGene(rows, positive) {\n"
        "      const best = new Map();\n"
        "      for (const r of rows) {\n"
        "        if (positive && r.y < 0) continue;\n"
        "        if (!positive && r.y > 0) continue;\n"
        "        const prev = best.get(r.key);\n"
        "        if (!prev) {\n"
        "          best.set(r.key, r);\n"
        "          continue;\n"
        "        }\n"
        "        if (positive) {\n"
        "          if (r.y > prev.y) best.set(r.key, r);\n"
        "        } else {\n"
        "          if (r.y < prev.y) best.set(r.key, r);\n"
        "        }\n"
        "      }\n"
        "      const arr = Array.from(best.values());\n"
        "      arr.sort((a, b) => positive ? (b.y - a.y) : (a.y - b.y));\n"
        "      return arr;\n"
        "    }\n"
        "    function findManualLabels(rows, keys) {\n"
        "      const wanted = new Set(keys);\n"
        "      const best = new Map();\n"
        "      for (const r of rows) {\n"
        "        if (!wanted.has(r.key)) continue;\n"
        "        const prev = best.get(r.key);\n"
        "        if (!prev || Math.abs(r.y) > Math.abs(prev.y)) best.set(r.key, r);\n"
        "      }\n"
        "      return best;\n"
        "    }\n"
        "    function applySkylineLabels() {\n"
        "      const rows = buildRows();\n"
        "      const topN = Number(document.getElementById('label-top-n').value || '0');\n"
        "      const tokens = parseGeneTokens(document.getElementById('label-genes').value);\n"
        "      const selected = new Map();\n"
        "      const manualMap = findManualLabels(rows, tokens);\n"
        "      for (const [k, v] of manualMap.entries()) selected.set(k, v);\n"
        "      if (topN > 0) {\n"
        "        const topPos = strongestPerGene(rows, true).slice(0, topN);\n"
        "        const topNeg = strongestPerGene(rows, false).slice(0, topN);\n"
        "        for (const r of topPos) selected.set(r.key, r);\n"
        "        for (const r of topNeg) selected.set(r.key, r);\n"
        "      }\n"
        "      const labeled = Array.from(selected.values());\n"
        "      labeled.sort((a, b) => Math.abs(b.y) - Math.abs(a.y));\n"
        "      const xVals = labeled.map((r) => r.x);\n"
        "      const yVals = labeled.map((r) => r.y);\n"
        "      const textVals = labeled.map((r) => r.gene);\n"
        "      Plotly.restyle(PLOT_DIV_ID, {x: [xVals], y: [yVals], text: [textVals]}, [SKYLINE_LABEL_TRACE_IDX]);\n"
        "      const missing = tokens.filter((k) => !manualMap.has(k));\n"
        "      let msg = `Labeled ${labeled.length} gene(s).`;\n"
        "      if (topN > 0) msg += ` Top +${topN}/-${topN} included.`;\n"
        "      if (missing.length > 0) msg += ` Not found: ${missing.join(', ')}`;\n"
        "      document.getElementById('labels-status').textContent = msg;\n"
        "    }\n"
        "    function clearSkylineLabels() {\n"
        "      Plotly.restyle(PLOT_DIV_ID, {x: [[]], y: [[]], text: [[]]}, [SKYLINE_LABEL_TRACE_IDX]);\n"
        "      document.getElementById('labels-status').textContent = '';\n"
        "    }\n"
        "    document.getElementById('labels-apply').addEventListener('click', applySkylineLabels);\n"
        "    document.getElementById('labels-clear').addEventListener('click', clearSkylineLabels);\n"
        "    document.getElementById('skyline-sublibrary-filter').addEventListener('change', applySublibraryFilter);\n"
        "    document.getElementById('label-genes').addEventListener('keydown', (ev) => {\n"
        "      if (ev.key === 'Enter') {\n"
        "        ev.preventDefault();\n"
        "        applySkylineLabels();\n"
        "      }\n"
        "    });\n"
        "    document.getElementById('label-top-n').addEventListener('change', applySkylineLabels);\n"
        "    populateSublibraryFilter();\n"
        "    applySublibraryFilter();\n"
        "    applySkylineLabels();\n"
    )
    output_html = Path(output_png).with_name(Path(output_png).stem + "_interactive.html")
    html_path = write_plotly_interactive_html(
        output_html=output_html,
        traces=traces,
        layout=layout,
        title="Genomic skyline (Mean log2FC)",
        filename_base=Path(output_png).stem,
        extra_controls_html=extra_controls_html,
        extra_script=extra_script,
    )
    debug_log(f"Wrote interactive skyline HTML: {html_path}", debug_enabled)


def run_skyline_cli() -> None:
    # Parse source workbook/sheet and output location.
    parser = argparse.ArgumentParser(description="Generate Skyline plot from screen results.")
    parser.add_argument("input_excel")
    parser.add_argument("output_png", nargs="?", help="Output PNG path (required unless --validate-only is set).")
    parser.add_argument("--sheet", default="skylineplot2")
    parser.add_argument(
        "--interactive-only",
        action="store_true",
        help="Generate interactive HTML only and skip static PNG creation.",
    )
    parser.add_argument(
        "--validate-only",
        action="store_true",
        help="Validate workbook/sheet compatibility for skyline plotting and exit.",
    )
    parser.add_argument("--debug", action="store_true", help="Enable verbose debug logging.")
    args = parser.parse_args()
    debug_enabled = DEBUG_ENV_DEFAULT or args.debug
    try:
        if args.validate_only:
            resolved_sheet = _resolve_sheet_name(args.input_excel, args.sheet, debug_enabled=debug_enabled)
            header = pd.read_excel(args.input_excel, sheet_name=resolved_sheet, nrows=0)
            _canonicalize_skyline_columns(header, resolved_sheet, args.input_excel)
            print(
                f"[genomic_signal_skyline] OK: workbook '{args.input_excel}' is skyline-compatible via sheet '{resolved_sheet}'.",
                file=sys.stderr,
            )
            debug_log(
                f"Workbook '{args.input_excel}' is skyline-compatible via sheet '{resolved_sheet}'.",
                debug_enabled,
            )
            return
        if not args.output_png:
            parser.error("output_png is required unless --validate-only is set.")
        render_chromosome_signal_map(
            args.input_excel,
            args.sheet,
            args.output_png,
            interactive_only=args.interactive_only,
            debug_enabled=debug_enabled,
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[genomic_signal_skyline] ERROR: {exc}", file=sys.stderr)
        if debug_enabled:
            raise
        raise SystemExit(1) from exc


if __name__ == "__main__":
    run_skyline_cli()
