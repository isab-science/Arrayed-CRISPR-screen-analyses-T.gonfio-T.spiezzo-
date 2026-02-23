from __future__ import annotations

import argparse
import os
import subprocess
import sys
from pathlib import Path

DEBUG_ENV_DEFAULT = os.environ.get("PRPCSCREEN_DEBUG", "").strip().lower() in {"1", "true", "yes", "on"}


def debug_log(message: str, enabled: bool) -> None:
    if enabled:
        print(f"[run_pooled_pipeline] {message}", file=sys.stderr)


def run_python_step(name: str, script_path: Path, args: list[str], debug_enabled: bool) -> None:
    cmd = [sys.executable, str(script_path), *args]
    print(f"==> {name}")
    print("    " + " ".join(f'"{part}"' if " " in part else part for part in cmd))
    completed = subprocess.run(cmd, check=False, text=True, capture_output=True)
    if completed.stdout:
        for line in completed.stdout.splitlines():
            print(f"    {line}")
    if completed.stderr:
        for line in completed.stderr.splitlines():
            print(f"    {line}")
    if completed.returncode != 0:
        raise RuntimeError(f"Step failed ({name}) with exit code {completed.returncode}.")
    debug_log(f"Completed: {name}", debug_enabled)


def run_pipeline_cli() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Run pooled-screen analysis and reuse existing figure scripts "
            "(replicate agreement, distributions, volcano/flashlight, optional trajectory and skyline)."
        )
    )
    parser.add_argument("input_table", help="CSV/TSV/XLSX pooled table.")
    parser.add_argument("--output-dir", default="results_pooled")
    parser.add_argument("--sheet", default=None, help="Excel sheet to read when input is XLSX.")

    parser.add_argument("--reference-cols", nargs="+", default=None)
    parser.add_argument("--treatment-cols", nargs="+", default=None)
    parser.add_argument("--reference-regex", default=r"(?i)^Negative_R\d+$")
    parser.add_argument("--treatment-regex", default=r"(?i)^Positive_R\d+$")
    parser.add_argument("--pseudocount", type=float, default=1.0)
    parser.add_argument("--pvalue-method", choices=("welch", "student", "paired"), default="welch")
    parser.add_argument("--p-cutoff", type=float, default=0.05)
    parser.add_argument("--log2fc-cutoff", type=float, default=0.3)
    parser.add_argument("--nt-regex", default=r"(?i)(?:^control(?:_|$)|^nt(?:_|$)|non[-_ ]?target|negative[_ ]?control|nt[_ -]?ctrl)")
    parser.add_argument("--pos-regex", default=r"(?i)(?:^prnp$|positive[_ ]?control|pos[_ -]?ctrl)")

    parser.add_argument("--skip-trajectory", action="store_true", help="Skip the generic feature-series trajectory plot.")
    parser.add_argument("--trajectory-column", default="Log2FC_rep1")
    parser.add_argument("--genomics-excel", default=None, help="Optional genomics workbook for sublibrary mapping and skyline.")
    parser.add_argument("--skyline-sheet", default="skylineplot2")
    parser.add_argument("--debug", action="store_true")
    args = parser.parse_args()

    debug_enabled = DEBUG_ENV_DEFAULT or args.debug
    root = Path(__file__).resolve().parents[2]
    scripts_dir = root / "prpcscreen" / "scripts"

    output_dir = Path(args.output_dir)
    figures_dir = output_dir / "figures"
    output_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    integrated_csv = output_dir / "01_integrated.csv"
    analyzed_csv = output_dir / "02_analyzed.csv"
    hits_csv = output_dir / "03_hits.csv"

    compute_args = [
        str(args.input_table),
        str(analyzed_csv),
        "--integrated_csv",
        str(integrated_csv),
        "--hits_csv",
        str(hits_csv),
        "--reference-regex",
        args.reference_regex,
        "--treatment-regex",
        args.treatment_regex,
        "--pseudocount",
        str(args.pseudocount),
        "--pvalue-method",
        args.pvalue_method,
        "--p-cutoff",
        str(args.p_cutoff),
        "--log2fc-cutoff",
        str(args.log2fc_cutoff),
        "--nt-regex",
        args.nt_regex,
        "--pos-regex",
        args.pos_regex,
    ]
    if args.sheet:
        compute_args.extend(["--sheet", args.sheet])
    if args.reference_cols:
        compute_args.extend(["--reference-cols", *args.reference_cols])
    if args.treatment_cols:
        compute_args.extend(["--treatment-cols", *args.treatment_cols])
    if debug_enabled:
        compute_args.append("--debug")

    steps: list[tuple[str, Path, list[str]]] = [
        ("Compute pooled metrics", scripts_dir / "compute_pooled_metrics.py", compute_args),
        (
            "Replicate agreement diagnostics",
            scripts_dir / "plot_replicate_agreement.py",
            [str(analyzed_csv), str(figures_dir / "replicate_agreement_log2fc.png"), "--stem", "Log2FC"] + (["--debug"] if debug_enabled else []),
        ),
        (
            "Signal distribution histogram",
            scripts_dir / "plot_signal_distributions.py",
            [
                str(analyzed_csv),
                "--output_html",
                str(figures_dir / "distribution_log2fc_rep1_interactive.html"),
                "--column",
                "Log2FC_rep1",
            ]
            + (["--genomics_excel", str(args.genomics_excel)] if args.genomics_excel else [])
            + (["--debug"] if debug_enabled else []),
        ),
    ]

    volcano_args = [
        str(analyzed_csv),
        str(figures_dir / "candidate_flashlight_ranked_meanlog2.png"),
        "--volcano_html",
        str(figures_dir / "candidate_volcano_interactive.html"),
    ]
    if args.genomics_excel:
        volcano_args.extend(["--genomics_excel", str(args.genomics_excel)])
    if debug_enabled:
        volcano_args.append("--debug")
    steps.append(("Candidate landscape plots", scripts_dir / "plot_candidate_landscape.py", volcano_args))

    if not args.skip_trajectory:
        steps.append(
            (
                "Feature-series trajectory",
                scripts_dir / "plot_well_trajectories.py",
                [
                    str(analyzed_csv),
                    str(figures_dir / f"feature_series_{args.trajectory_column.lower()}.png"),
                    "--column",
                    args.trajectory_column,
                ]
                + (["--debug"] if debug_enabled else []),
            )
        )

    if args.genomics_excel:
        steps.append(
            (
                "Genomic skyline plot",
                scripts_dir / "plot_genomic_signal_skyline.py",
                [
                    str(args.genomics_excel),
                    str(figures_dir / "genomic_skyline_meanlog2fc.png"),
                    "--sheet",
                    args.skyline_sheet,
                    "--interactive-only",
                ]
                + (["--debug"] if debug_enabled else []),
            )
        )

    for name, script_path, step_args in steps:
        run_python_step(name, script_path, step_args, debug_enabled=debug_enabled)

    print("")
    print("Pooled pipeline completed.")
    print("Outputs:")
    print(f"  {integrated_csv}")
    print(f"  {analyzed_csv}")
    print(f"  {hits_csv}")
    print(f"  {figures_dir}")


if __name__ == "__main__":
    run_pipeline_cli()
