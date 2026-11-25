#!/usr/bin/env python3
"""Build Fair GP comparison tables from saved dataset experiment results."""

from __future__ import annotations

import argparse
import base64
import hashlib
import html
import io
import json
from pathlib import Path
import re
from textwrap import fill
from zipfile import ZipFile

import numpy as np
import pandas as pd


CELL_KEYS = ["dataset", "method", "split", "model_key", "lambda_index"]
MODEL_KEYS = CELL_KEYS[:-1]
METHODS = ("MMD", "Sinkhorn")
SPLITS = ("train", "test")
DISPARITIES = {
    "disp": "mean squared prediction difference at sensitive labels 0 and 1",
    "cf_max_abs": "maximum absolute prediction difference at sensitive labels 0 and 1",
    "cf_mean_abs": "mean absolute prediction difference at sensitive labels 0 and 1",
}
COLORS = {"mse": "#0072B2", "ks": "#B2182B", "disp": "#8A6500"}
SPREAD_NOTE = "Mean ± sample SD across split seeds (ddof=1), rounded to 0.001."


def validate_records(frame, path, disparity, *, seed=None, dataset=None):
    """Validate both standalone exports and per-split inputs."""
    numeric = ["rho", "lambda_index", "lambda", "lambda_relative", "MSE", "KS", "disp"]
    if disparity not in numeric:
        numeric.append(disparity)
    required = set(CELL_KEYS) | {"run", "rho_estimated", *numeric}

    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"{path} is missing columns {sorted(missing)}.")
    if frame.empty or frame[list(required)].isna().any().any():
        raise ValueError(f"{path} has empty metric records.")
    frame[numeric + ["run"]] = frame[numeric + ["run"]].apply(pd.to_numeric, errors="raise")
    if not np.isfinite(frame[numeric].to_numpy()).all():
        raise ValueError(f"{path} has non-finite metrics or parameters.")
    if seed is not None and not frame["run"].eq(seed).all():
        raise ValueError(f"{path} has run IDs that differ from the split directory seed.")
    if dataset is not None and not frame["dataset"].eq(dataset).all():
        raise ValueError(f"{path} has a dataset label that differs from its directory name.")
    if not frame["method"].isin(METHODS).all() or not frame["split"].isin(SPLITS).all():
        raise ValueError(f"{path} has an unknown method or evaluation split.")
    indices = frame["lambda_index"]
    if not (indices.ge(0) & indices.eq(indices.round())).all():
        raise ValueError(f"{path} has invalid lambda indices.")
    frame["lambda_index"] = indices.astype(int)
    if not frame["rho"].between(-1, 1).all() or (frame[["lambda", "lambda_relative"]] < 0).any().any():
        raise ValueError(f"{path} has rho or lambda outside the allowed range.")
    estimated = frame["rho_estimated"].astype(str).str.strip().str.lower()
    flags = estimated.map({"true": True, "false": False, "1": True, "0": False})
    if flags.isna().any():
        raise ValueError(f"{path} has a non-Boolean rho_estimated value.")
    frame["rho_estimated"] = flags.astype(bool)
    runs = frame["run"]
    if not (np.isfinite(runs).all() and runs.ge(0).all() and runs.eq(runs.round()).all()):
        raise ValueError(f"{path} has invalid repetition IDs.")
    frame["run"] = runs.astype(int)
    return frame


def read_csv_results(path, disparity, experiment):
    frame = pd.read_csv(path, dtype={"model_key": str})
    if "experiment" in frame:
        labels = frame["experiment"].dropna().unique()
        if experiment is None and (len(labels) != 1 or frame["experiment"].isna().any()):
            raise ValueError(f"Choose one experiment with --experiment. Available labels: {list(labels)}")
        if experiment is not None:
            frame = frame.loc[frame["experiment"].eq(experiment)].copy()
            if frame.empty:
                raise ValueError(f"No rows match --experiment {experiment!r}. Available labels: {list(labels)}")
    elif experiment is not None:
        raise ValueError("--experiment requires a CSV with an experiment column.")
    frame = validate_records(frame, path, disparity)
    label = frame["experiment"].iloc[0] if "experiment" in frame else None
    return validate_grid(frame), {"source_csv": str(path), "experiment": label}


def read_results(root: Path, disparity: str) -> tuple[pd.DataFrame, dict]:
    """Read each completed dataset and split exactly once."""
    metadata_path = root / "manifest.json"
    manifest = json.loads(metadata_path.read_text(encoding="utf-8")) if metadata_path.is_file() else {}
    frames = []

    for marker in sorted(root.glob("*/split_*/complete.json")):
        match = re.fullmatch(r"split_(\d+)", marker.parent.name)
        if match is None:
            continue
        seed = int(match[1])
        path = marker.with_name("metrics.csv")
        if not path.is_file():
            raise ValueError(f"{marker.parent} is marked complete but has no metrics.csv.")
        complete = json.loads(marker.read_text(encoding="utf-8"))
        experiment_id = manifest.get("experiment_id")
        if experiment_id and complete.get("experiment_id") != experiment_id:
            raise ValueError(f"{marker.parent} belongs to a different experiment.")
        raw = path.read_bytes()
        checksum = complete.get("metrics_sha256")
        if checksum and hashlib.sha256(raw).hexdigest() != checksum:
            raise ValueError(f"{path} does not match the checksum recorded at completion.")
        frame = pd.read_csv(io.BytesIO(raw), dtype={"model_key": str})
        if len(frame) != complete.get("metric_rows", len(frame)):
            raise ValueError(f"{path} does not match the recorded metric row count.")
        frames.append(validate_records(frame, path, disparity, seed=seed, dataset=marker.parent.parent.name))

    if not frames:
        raise ValueError(f"No completed dataset/split directories found in {root}.")
    return validate_grid(pd.concat(frames, ignore_index=True)), manifest


def validate_grid(data):
    if data.duplicated(CELL_KEYS + ["run"]).any():
        raise ValueError("Duplicate metric cells for the same dataset, split seed, model, and penalty.")

    rho_roles = data.groupby(["dataset", "model_key"])["rho_estimated"].nunique()
    if rho_roles.gt(1).any():
        raise ValueError("A model key switches between fitted and fixed rho across runs.")
    fixed_rhos = data.loc[~data.rho_estimated].groupby(["dataset", "model_key"])["rho"].nunique()
    if fixed_rhos.gt(1).any():
        raise ValueError("A fixed model key has different rho values across runs.")
    repeated_rhos = data.groupby(MODEL_KEYS + ["run"])["rho"].nunique()
    if repeated_rhos.gt(1).any():
        raise ValueError("A model has different rho values along one penalty path.")
    weights = data.groupby(["dataset", "method", "lambda_index"])["lambda_relative"].agg(["min", "max"])
    if not np.allclose(weights["min"], weights["max"], rtol=1e-9, atol=1e-12):
        raise ValueError("Relative weights differ at the same lambda index. Use a separate report for each grid.")
    return data


def summarize(data: pd.DataFrame) -> pd.DataFrame:
    """Aggregate each cell across split seeds, using sample variance and SD."""
    metrics = ["MSE", "KS", *DISPARITIES]
    measures = {metric: ["mean", "std", "var"] for metric in metrics if metric in data}
    measures.update({"rho": ["mean", "std"], "lambda": ["min", "max"], "lambda_relative": ["mean"]})
    summary = data.groupby(CELL_KEYS).agg(measures)
    summary.columns = ["_".join(parts) for parts in summary.columns]
    summary["n_splits"] = data.groupby(CELL_KEYS)["run"].nunique()
    summary = summary.reset_index()

    # Count a fitted rho once per split even if some paths have different lengths.
    fitted = data.drop_duplicates(MODEL_KEYS + ["run"])
    rows = fitted.groupby(MODEL_KEYS).agg(
        row_rho=("rho", "mean"),
        rho_min=("rho", "min"),
        rho_max=("rho", "max"),
        rho_estimated=("rho_estimated", "first"),
    )
    summary = summary.merge(rows.reset_index(), on=MODEL_KEYS, validate="many_to_one")
    return summary.sort_values(
        ["dataset", "method", "split", "row_rho", "rho_estimated", "model_key", "lambda_index"]
    ).reset_index(drop=True)


def number(value):
    if pd.isna(value):
        return "—"
    return f"{value:.3f}"


def metric_lines(row, disparity):
    for metric, label, symbol, style in (
        ("MSE", "MSE", "■", "mse"),
        ("KS", "KS", "●", "ks"),
        (disparity, "disp", "▲", "disp"),
    ):
        mean = number(row[f"{metric}_mean"])
        sd = number(row[f"{metric}_std"])
        yield style, f"{symbol} {label} {mean} ± {sd}"


def rho_label(row):
    return f"ρ* = {row.row_rho:.3f}" if row.rho_estimated else f"ρ = {row.row_rho:g}"


def grouped_tables(summary, manifest):
    datasets = list(dict.fromkeys([*manifest.get("datasets", []), *summary["dataset"].unique()]))
    for dataset in datasets:
        for method in METHODS:
            for split in SPLITS:
                rows = summary.loc[summary.dataset.eq(dataset) & summary.method.eq(method) & summary.split.eq(split)]
                yield dataset, method, split, rows


def table_html(rows: pd.DataFrame, disparity: str) -> str:
    """Render a method and evaluation split with rho rows and lambda columns."""
    weights = rows.groupby("lambda_index")["lambda_relative_mean"].first()
    header = ["<th scope='col'>ρ / λ</th>"]
    for index, factor in weights.items():
        header.append(f"<th scope='col'>λ<sub>{index}</sub><small>c = {factor:.4g}</small></th>")

    body = []
    for key, path in rows.groupby("model_key", sort=False):
        first = path.iloc[0]
        label = rho_label(first)
        rho_detail = f"<small>range {first.rho_min:.3f}–{first.rho_max:.3f}</small>" if first.rho_estimated else ""
        header_cell = f"<th scope='row' title='{html.escape(str(key), quote=True)}'>{label}{rho_detail}</th>"
        path = path.set_index("lambda_index")
        cells = [header_cell]
        for index in weights.index:
            if index not in path.index:
                cells.append("<td>—</td>")
                continue
            row = path.loc[index]
            cell = "".join(f"<div class='{style}'>{value}</div>"
                           for style, value in metric_lines(row, disparity))
            title = f"Absolute lambda range {row.lambda_min:.6g} to {row.lambda_max:.6g}"
            cells.append(f"<td title='{title}'>{cell}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    return "<table><thead><tr>" + "".join(header) + "</tr></thead><tbody>" + "".join(body) + "</tbody></table>"


def render_png(rows, title, disparity, path):
    """Draw a table with the same means, SDs, and colors as the HTML."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle

    weights = rows.groupby("lambda_index")["lambda_relative_mean"].first()
    model_rows = list(rows.groupby("model_key", sort=False))
    columns = len(weights)
    fig, ax = plt.subplots(figsize=(max(7, 2.0 + 2.0 * columns), 2.1 + 0.88 * len(model_rows)))
    fig.subplots_adjust(left=0.015, right=0.985, bottom=1.12 / fig.get_figheight(), top=1 - 0.72 / fig.get_figheight())
    ax.set(xlim=(0, columns + 1), ylim=(len(model_rows) + 1, 0))
    ax.axis("off")
    ax.add_patch(Rectangle((0, 0), columns + 1, 1, facecolor="#EAF0F6"))
    for ri in range(1, len(model_rows) + 1):
        if ri % 2 == 0:
            ax.add_patch(Rectangle((1, ri), columns, 1, facecolor="#FAFBFD"))
        ax.add_patch(Rectangle((0, ri), 1, 1, facecolor="#F0F3F6"))
    for x in range(columns + 2):
        ax.vlines(x, 0, len(model_rows) + 1, color="#CAD2DC", linewidth=0.65, clip_on=False)
    for y in range(len(model_rows) + 2):
        ax.hlines(y, 0, columns + 1, color="#CAD2DC", linewidth=0.65, clip_on=False)

    ax.text(0.5, 0.5, "ρ / λ", ha="center", va="center", fontsize=12, weight="bold")
    for ci, (index, factor) in enumerate(weights.items(), start=1):
        ax.text(ci + 0.5, 0.5, f"λ{index}\nc = {factor:.4g}", ha="center", va="center",
                fontsize=11, weight="bold")
    for ri, (_, model) in enumerate(model_rows, start=1):
        first = model.iloc[0]
        label = rho_label(first)
        if first.rho_estimated:
            ax.text(0.5, ri + 0.36, label, ha="center", va="center", fontsize=11, weight="bold")
            ax.text(0.5, ri + 0.69, f"range {first.rho_min:.3f}–{first.rho_max:.3f}",
                    ha="center", va="center", fontsize=8.5, color="#59636E")
        else:
            ax.text(0.5, ri + 0.5, label, ha="center", va="center", fontsize=11, weight="bold")
        by_index = model.set_index("lambda_index")
        for ci, index in enumerate(weights.index, start=1):
            if index not in by_index.index:
                ax.text(ci + 0.5, ri + 0.5, "—", ha="center", va="center", color="#59636E")
                continue
            for y, (style, value) in zip((0.22, 0.5, 0.78), metric_lines(by_index.loc[index], disparity)):
                ax.text(ci + 0.08, ri + y, value, va="center", fontsize=10.5, color=COLORS[style])

    fig.text(0.5, 1 - 0.34 / fig.get_figheight(), title, ha="center", va="center", fontsize=17, weight="bold")
    fig.text(0.5, 0.23 / fig.get_figheight(),
             f"{SPREAD_NOTE}\n"
             f"{fill(f'■ MSE  ● KS  ▲ disp = {DISPARITIES[disparity]}.', width=90)}\n"
             "ρ* is the mean fitted rho. c is the relative penalty.\n"
             "— indicates unavailable data or SD.",
             ha="center", va="bottom", fontsize=9.5, color="#253140")
    fig.savefig(path, dpi=200, facecolor="white")
    plt.close(fig)


def make_images(summary, manifest, disparity, output):
    """Write individual PNGs and return an archive for the portable report."""
    images = output / "images"
    images.mkdir(exist_ok=True)
    archive_buffer = io.BytesIO()
    image_count = 0
    with ZipFile(archive_buffer, "w") as archive:
        for dataset, method, split, rows in grouped_tables(summary, manifest):
            if rows.empty:
                continue
            image_count += 1
            safe_name = re.sub(r"[^A-Za-z0-9_.-]+", "_", str(dataset)).strip("._") or "dataset"
            path = images / f"{image_count:02d}_{safe_name}_{method}_{split}.png"
            render_png(rows, f"{dataset} · {method} · {split.title()}", disparity, path)
            archive.write(path, arcname=path.name)
    archive_bytes = archive_buffer.getvalue()
    (output / "fair_gp_table_images.zip").write_bytes(archive_bytes)
    return base64.b64encode(archive_bytes).decode("ascii"), image_count


def report_html(summary: pd.DataFrame, manifest: dict, disparity: str, images_archive: str) -> str:
    datasets = list(dict.fromkeys([*manifest.get("datasets", []), *summary["dataset"].unique()]))
    filters = []
    for field, values in [("dataset", datasets), ("method", METHODS), ("split", SPLITS)]:
        options = ["<option value=''>All</option>"]
        for value in values:
            safe = html.escape(str(value), quote=True)
            options.append(f"<option value='{safe}'>{safe}</option>")
        filters.append(f"<label>{field.title()} <select data-filter='{field}'>" + "".join(options) + "</select></label>")

    panels = []
    for dataset, method, split, rows in grouped_tables(summary, manifest):
        safe = html.escape(str(dataset), quote=True)
        table = table_html(rows, disparity) if not rows.empty else "<p>No completed observations.</p>"
        panels.append(
            f"<section data-dataset='{safe}' data-method='{method}' data-split='{split}'>"
            f"<h2>{safe} · {method} · {split.title()}</h2><div class='scroll'>{table}</div></section>"
        )

    explanation = html.escape(DISPARITIES[disparity])
    details = html.escape(json.dumps(manifest, indent=2, ensure_ascii=False)) if manifest else "No manifest supplied."
    return f"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Fair GP experiment results</title>
<style>
body {{font: 15px system-ui, sans-serif; margin: 24px; color: #202124}}
h1 {{font-size: 25px}} h2 {{font-size: 19px; margin-top: 26px}}
label {{display: inline-block; margin: 3px 12px 3px 0}}
select {{font: inherit; padding: 4px}}
.scroll {{overflow-x: auto}}
table {{border-collapse: collapse; white-space: nowrap; font-variant-numeric: tabular-nums}}
th, td {{border: 1px solid #cad2dc; padding: 7px 10px; text-align: left; vertical-align: top}}
thead th {{background: #eaf0f6}}
tbody th {{background: #f0f3f6}}
tbody tr:nth-child(even) td {{background: #fafbfd}}
th:first-child {{position: sticky; left: 0}}
small {{display: block; color: #59636e}}
.download {{display: inline-block; padding: 6px 12px; margin: 3px 12px 3px 0;
  border-radius: 5px; background: #1e4b73; color: white; text-decoration: none}}
.download:focus-visible {{outline: 2px solid #163b62; outline-offset: 3px}}
.mse {{color: #0072B2}} .ks {{color: #B2182B}} .disp {{color: #8A6500}}
pre {{white-space: pre-wrap}}
details {{margin-top: 26px}}
@media print {{
  .filters {{display: none}}
  th:first-child {{position: static}}
  section {{break-inside: avoid}}
}}
</style>
</head>
<body>
<h1>Fair GP experiment results</h1>
<p>{SPREAD_NOTE} The ± term measures variation across splits.
SD is unavailable when only one split contributes. Counts and sample variance are included in summary.csv.</p>
<p><span class="mse">■ MSE</span>, <span class="ks">● KS</span>, and
<span class="disp">▲ disparate treatment</span> appear in every cell.
Disparate treatment is the {explanation} for paired predictions on the same inputs.</p>
<p>The fitted model stays in one row across split seeds. Its label gives the mean fitted rho.
Columns show the relative penalty c. Actual lambda ranges are in the cell tooltip and summary CSV.</p>
<div class="filters">
<a class="download" href="data:application/zip;base64,{images_archive}"
   download="fair_gp_table_images.zip" title="Download all available table images">Download images</a>
<span>ZIP of all available PNG tables</span>
{"".join(filters)}
</div>
{"".join(panels)}
<details><summary>Experiment settings</summary><pre>{details}</pre></details>
<script>
const controls = [...document.querySelectorAll('[data-filter]')];
const panels = [...document.querySelectorAll('section')];
for (const control of controls) {{
  control.addEventListener('change', () => {{
    for (const panel of panels) {{
      panel.hidden = controls.some(input =>
        input.value && panel.dataset[input.dataset.filter] !== input.value);
    }}
  }});
}}
</script>
</body>
</html>
"""


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("root", type=Path, help="Experiment directory or a CSV of per-run metrics.")
    parser.add_argument("--output", type=Path, help="Report directory (default: report/ alongside the CSV or inside the experiment directory).")
    parser.add_argument("--disparity", choices=DISPARITIES, default="disp",
                        help="Disparate treatment measure shown in the tables (default: disp).")
    parser.add_argument("--experiment", help="Experiment label when a combined CSV contains several experiments.")
    args = parser.parse_args()
    root = args.root.resolve()
    try:
        if root.is_file() and root.suffix.lower() == ".csv":
            output = args.output or root.parent / "report"
            data, manifest = read_csv_results(root, args.disparity, args.experiment)
        elif root.is_dir():
            if args.experiment is not None:
                parser.error("--experiment applies to a combined CSV.")
            output = args.output or root / "report"
            data, manifest = read_results(root, args.disparity)
        else:
            parser.error("Supply an experiment directory or a CSV with one row per repetition and grid cell.")
    except (ValueError, OSError) as error:
        parser.error(str(error))
    summary = summarize(data)
    output.mkdir(parents=True, exist_ok=True)
    summary.to_csv(output / "summary.csv", index=False)
    images_archive, image_count = make_images(summary, manifest, args.disparity, output)
    document = report_html(summary, manifest, args.disparity, images_archive)
    report = output / "comparison.html"
    report.write_text(document, encoding="utf-8")
    count = len(data[["dataset", "run"]].drop_duplicates())
    print(f"Summarized {count} completed dataset splits into {len(summary)} metric cells.")
    print(f"Saved {image_count} PNG tables to {output / 'images'}.")
    print(f"Download archive: {output / 'fair_gp_table_images.zip'}")
    print(f"Open {report} in a browser.")


if __name__ == "__main__":
    main()
