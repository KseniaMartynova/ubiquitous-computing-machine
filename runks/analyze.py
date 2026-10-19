#!/usr/bin/env python3
import argparse
import csv
import json
import math
import pathlib
import sys

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_MPL = True
except ImportError:
    HAS_MPL = False
    plt = None

import numpy as np
import scipy.stats


def _import_run_grid():
    runks_dir = pathlib.Path(__file__).resolve().parent
    if str(runks_dir) not in sys.path:
        sys.path.insert(0, str(runks_dir))
    import run_grid
    return run_grid


RUN_GRID = _import_run_grid()
HEADER = RUN_GRID.HEADER
COMBOS = RUN_GRID.COMBOS

OPERATIONS = ["cholesky", "lu", "multiplication", "svd"]


def _f_2n3(n):
    return 2 * n ** 3


GFLOPS_FORMULAS = {
    ("cholesky", "mkl"): lambda n: n ** 3,
    ("cholesky", "openblas"): lambda n: n ** 3,
    ("cholesky", "numpy"): lambda n: 7 * n ** 3 / 3,
    ("lu", "mkl"): _f_2n3,
    ("lu", "openblas"): _f_2n3,
    ("lu", "numpy"): _f_2n3,
    ("multiplication", "mkl"): _f_2n3,
    ("multiplication", "openblas"): _f_2n3,
    ("multiplication", "numpy"): _f_2n3,
    ("svd", "mkl"): lambda n: 11 * n ** 3,
    ("svd", "openblas"): lambda n: 11 * n ** 3,
    ("svd", "numpy"): lambda n: 11 * n ** 3,
}

SUMMARY_FIELDS = [
    "operation", "implementation", "thread_mode", "n", "count",
    "median_seconds", "q1_seconds", "q3_seconds", "gflops",
    "peak_rss_kb_max",
]
FITS_FIELDS = [
    "operation", "implementation", "thread_mode", "n_points",
    "a", "c", "c_ci95_low", "c_ci95_high", "status",
]


class DataError(Exception):
    pass

def read_csv(path):
    try:
        with open(path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))
    except FileNotFoundError:
        raise DataError(f"{path}: файл не найден")
    except OSError as exc:
        raise DataError(f"{path}: ошибка чтения: {exc}")
    if not rows:
        raise DataError(f"{path}: пустой файл")
    return rows[0], rows[1:]


def read_manifest(path):
    try:
        with open(path, encoding="utf-8") as f:
            return json.load(f)
    except FileNotFoundError:
        raise DataError(f"{path}: manifest не найден")
    except (OSError, json.JSONDecodeError) as exc:
        raise DataError(f"{path}: ошибка чтения manifest: {exc}")


def _parse_row(csv_path, idx, row):
    if len(row) != len(HEADER):
        raise DataError(
            f"{csv_path}: строка {idx}: {len(row)} полей вместо {len(HEADER)}"
        )
    try:
        parsed = {
            "run_order": int(row[0]),
            "implementation": row[2],
            "operation": row[3],
            "n": int(row[4]),
            "rep": int(row[5]),
            "seconds": float(row[6]),
            "peak_rss_kb": int(row[8]),
            "thread_mode": row[12],
            "image": row[13],
            "image_id": row[14],
            "commit": row[15],
        }
    except (ValueError, IndexError) as exc:
        raise DataError(f"{csv_path}: строка {idx}: ошибка разбора: {exc}")
    if not math.isfinite(parsed["seconds"]) or parsed["seconds"] <= 0:
        raise DataError(
            f"{csv_path}: строка {idx}: seconds={row[6]!r} "
            f"не положительное конечное"
        )
    if parsed["peak_rss_kb"] <= 0:
        raise DataError(
            f"{csv_path}: строка {idx}: peak_rss_kb={row[8]!r} "
            f"не положительное"
        )
    return parsed


def validate_pair(csv_path, manifest_path):
    header, rows = read_csv(csv_path)
    manifest = read_manifest(manifest_path)

    if header != HEADER:
        raise DataError(
            f"{csv_path}: заголовок не совпадает с run_grid.HEADER"
        )

    block = manifest.get("block")
    if not isinstance(block, dict):
        raise DataError(f"{manifest_path}: нет block")
    block_sizes = block.get("sizes")
    block_reps = block.get("repetitions")
    block_mode = block.get("thread_mode")

    if (not isinstance(block_sizes, list) or not block_sizes
            or not all(isinstance(x, int) and x > 0 for x in block_sizes)):
        raise DataError(f"{manifest_path}: block.sizes некорректны")
    if not isinstance(block_reps, int) or block_reps <= 0:
        raise DataError(f"{manifest_path}: block.repetitions некорректны")
    if block_mode not in ("default", "single"):
        raise DataError(f"{manifest_path}: block.thread_mode некорректен")

    manifest_commit = manifest.get("commit")
    if not isinstance(manifest_commit, str) or not manifest_commit:
        raise DataError(f"{manifest_path}: отсутствует commit")
    images = manifest.get("images")
    if not isinstance(images, dict):
        raise DataError(f"{manifest_path}: отсутствует images")

    parsed = []
    seen = set()
    for idx, row in enumerate(rows, start=2):
        p = _parse_row(csv_path, idx, row)

        if p["thread_mode"] != block_mode:
            raise DataError(
                f"{csv_path}: строка {idx}: thread_mode={p['thread_mode']!r} "
                f"!= block.thread_mode={block_mode!r}"
            )
        if p["commit"] != manifest_commit:
            raise DataError(
                f"{csv_path}: строка {idx}: commit={p['commit']!r} "
                f"!= manifest.commit={manifest_commit!r}"
            )
        image_info = images.get(p["image"])
        expected_image_id = image_info.get("id") if isinstance(image_info, dict) else None
        if expected_image_id is None:
            raise DataError(
                f"{csv_path}: строка {idx}: image={p['image']!r} "
                f"нет в manifest.images"
            )
        if p["image_id"] != expected_image_id:
            raise DataError(
                f"{csv_path}: строка {idx}: image_id={p['image_id']!r} "
                f"!= manifest {expected_image_id!r}"
            )

        key = (p["implementation"], p["operation"], p["n"], p["rep"])
        if key in seen:
            raise DataError(f"{csv_path}: дубль запуска {key}")
        seen.add(key)
        parsed.append(p)

    expected_combos = {(impl, op) for (_, impl, op) in COMBOS}
    for p in parsed:
        if (p["implementation"], p["operation"]) not in expected_combos:
            raise DataError(
                f"{csv_path}: строка run_order={p['run_order']}: "
                f"неизвестная комбинация "
                f"({p['implementation']}, {p['operation']})"
            )

    for n in block_sizes:
        for rep in range(1, block_reps + 1):
            for (impl, op) in expected_combos:
                if (impl, op, n, rep) not in seen:
                    raise DataError(
                        f"{csv_path}: отсутствует запуск "
                        f"impl={impl} op={op} n={n} rep={rep}"
                    )

    return {
        "csv_path": str(csv_path),
        "manifest_path": str(manifest_path),
        "manifest": manifest,
        "parsed": parsed,
        "commit": manifest_commit,
        "thread_mode": block_mode,
        "sizes": list(block_sizes),
    }


def validate_all(csv_paths, expected_sizes):
    pairs = []
    for csv_path in csv_paths:
        p = pathlib.Path(csv_path)
        manifest_path = p.with_suffix(".manifest.json")
        pairs.append(validate_pair(p, manifest_path))

    if len(pairs) > 1:
        commits = {pair["commit"] for pair in pairs}
        if len(commits) != 1:
            raise DataError(f"разные commit в файлах: {sorted(commits)}")
        modes = {pair["thread_mode"] for pair in pairs}
        if len(modes) != 1:
            raise DataError(f"разные thread_mode: {sorted(modes)}")
        by_image = {}
        for pair in pairs:
            for img, info in pair["manifest"].get("images", {}).items():
                if isinstance(info, dict) and "id" in info:
                    by_image.setdefault(img, set()).add(info["id"])
        for img, ids in by_image.items():
            if len(ids) != 1:
                raise DataError(
                    f"разные image_id для {img!r}: {sorted(ids)}"
                )

    combined = []
    seen = set()
    all_sizes = set()
    for pair in pairs:
        for p in pair["parsed"]:
            key = (p["implementation"], p["operation"], p["n"], p["rep"])
            if key in seen:
                raise DataError(f"запуск {key} встречается дважды")
            seen.add(key)
            combined.append(p)
            all_sizes.add(p["n"])

    if sorted(all_sizes) != sorted(expected_sizes):
        raise DataError(
            f"размеры в CSV {sorted(all_sizes)} "
            f"не совпадают с --sizes {sorted(expected_sizes)}"
        )
    return combined, pairs[0]["thread_mode"]

def compute_summary(parsed, thread_mode):
    groups = {}
    for p in parsed:
        key = (p["operation"], p["implementation"], p["n"])
        groups.setdefault(key, []).append(p)

    rows = []
    for (op, impl, n), items in sorted(groups.items()):
        times = np.array([it["seconds"] for it in items], dtype=float)
        q1, median, q3 = np.quantile(
            times, [0.25, 0.5, 0.75], method="linear"
        )
        peak_max = max(it["peak_rss_kb"] for it in items)
        f = GFLOPS_FORMULAS[(op, impl)](n)
        gflops = f / float(median) / 1e9
        rows.append({
            "operation": op,
            "implementation": impl,
            "thread_mode": thread_mode,
            "n": n,
            "count": len(items),
            "median_seconds": float(median),
            "q1_seconds": float(q1),
            "q3_seconds": float(q3),
            "gflops": float(gflops),
            "peak_rss_kb_max": int(peak_max),
        })
    return rows


def compute_fits(summary_rows):
    groups = {}
    for row in summary_rows:
        key = (row["operation"], row["implementation"], row["thread_mode"])
        groups.setdefault(key, []).append(row)

    fits = []
    for (op, impl, mode), items in sorted(groups.items()):
        items_sorted = sorted(items, key=lambda r: r["n"])
        ns = [r["n"] for r in items_sorted]
        medians = [r["median_seconds"] for r in items_sorted]
        n_points = len(ns)
        row = {
            "operation": op,
            "implementation": impl,
            "thread_mode": mode,
            "n_points": n_points,
            "a": "",
            "c": "",
            "c_ci95_low": "",
            "c_ci95_high": "",
            "status": "insufficient_sizes",
        }
        if n_points >= 3:
            x = np.log(np.array(ns, dtype=float))
            y = np.log(np.array(medians, dtype=float))
            if np.all(np.isfinite(x)) and np.all(np.isfinite(y)):
                fit = scipy.stats.linregress(x, y)
                a = math.exp(fit.intercept)
                c = float(fit.slope)
                if math.isfinite(a) and a > 0 and math.isfinite(c):
                    half = scipy.stats.t.ppf(0.975, df=n_points - 2) * fit.stderr
                    row["a"] = a
                    row["c"] = c
                    row["c_ci95_low"] = c - half
                    row["c_ci95_high"] = c + half
                    row["status"] = "ok"
        fits.append(row)
    return fits

def write_csv(path, fields, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields, lineterminator="\n")
        writer.writeheader()
        for row in rows:
            writer.writerow({k: row[k] for k in fields})

def _prepare_operation_rows(operation, summary_rows):
    result = {}
    for row in summary_rows:
        if row["operation"] != operation:
            continue
        result.setdefault(row["implementation"], []).append(row)
    for impl in result:
        result[impl].sort(key=lambda r: r["n"])
    return result


def plot_time(operation, summary_rows, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    groups = _prepare_operation_rows(operation, summary_rows)
    for impl in sorted(groups):
        items = groups[impl]
        ns = [r["n"] for r in items]
        meds = [r["median_seconds"] for r in items]
        q1 = [r["q1_seconds"] for r in items]
        q3 = [r["q3_seconds"] for r in items]
        ax.plot(ns, meds, marker="o", label=impl)
        ax.fill_between(ns, q1, q3, alpha=0.2)
    ax.set_xlabel("Размер матрицы n")
    ax.set_ylabel("Время (с)")
    ax.set_yscale("log")
    ax.set_title(f"{operation}: медиана и Q1–Q3")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)


def plot_gflops(operation, summary_rows, path):
    fig, ax = plt.subplots(figsize=(8, 6))
    groups = _prepare_operation_rows(operation, summary_rows)
    for impl in sorted(groups):
        items = groups[impl]
        ns = [r["n"] for r in items]
        values = [r["gflops"] for r in items]
        ax.plot(ns, values, marker="o", label=impl)
    ax.set_xlabel("Размер матрицы n")
    ylabel = "GFLOP/s"
    if operation == "svd":
        ylabel += " (≈11n³)"
    ax.set_ylabel(ylabel)
    ax.set_title(f"{operation}: нормированная производительность")
    ax.legend()
    ax.grid(True, which="both", alpha=0.3)
    fig.tight_layout()
    fig.savefig(path, dpi=120)
    plt.close(fig)

def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Единый анализ результатов run_grid"
    )
    parser.add_argument("--csv", nargs="+", required=True)
    parser.add_argument("--sizes", required=True)
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--all", action="store_true")
    group.add_argument("--operation", choices=OPERATIONS)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args(argv)

    try:
        sizes = RUN_GRID.parse_sizes(args.sizes)
    except ValueError as exc:
        parser.error(str(exc))

    try:
        parsed, thread_mode = validate_all(args.csv, sizes)
    except DataError as exc:
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1

    if not HAS_MPL:
        print("Ошибка: matplotlib недоступен", file=sys.stderr)
        return 1

    summary_rows = compute_summary(parsed, thread_mode)
    fits = compute_fits(summary_rows)

    output_dir = pathlib.Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    write_csv(output_dir / "summary.csv", SUMMARY_FIELDS, summary_rows)
    write_csv(output_dir / "fits.csv", FITS_FIELDS, fits)

    operations = OPERATIONS if args.all else [args.operation]
    for op in operations:
        plot_time(op, summary_rows, output_dir / f"{op}_time.png")
        plot_gflops(op, summary_rows, output_dir / f"{op}_gflops.png")

    print(
        f"Готово: {output_dir} "
        f"({len(summary_rows)} строк summary, {len(fits)} строк fits)",
        file=sys.stderr,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
