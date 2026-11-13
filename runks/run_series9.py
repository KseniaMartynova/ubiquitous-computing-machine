#!/usr/bin/env python3

import argparse
import csv
import json
import pathlib
import subprocess
import sys
import time
from collections import Counter


DEFAULT_SERIES = [
    (2500, 10), (5000, 10), (7500, 10), (10000, 10),
    (12500, 3), (15000, 3), (17500, 3), (20000, 3),
]
SHUFFLE_SEED = 20260818


def run(cmd, allow_dirty=False):
    label = f"[{time.strftime('%H:%M:%S')}] " if False else ""
    print(f"+ {' '.join(str(c) for c in cmd)}", file=sys.stderr, flush=True)
    r = subprocess.run(cmd)
    if r.returncode != 0:
        raise SystemExit(
            f"команда завершилась с кодом {r.returncode}: {' '.join(str(c) for c in cmd)}"
        )


def repo_root():
    return pathlib.Path(__file__).resolve().parent.parent


def load_run_grid(root):
    runks = root / "runks"
    if str(runks) not in sys.path:
        sys.path.insert(0, str(runks))
    import run_grid
    return run_grid


def check_block(csv_path, manifest_path, expected_n, expected_reps,
                expected_head, expected_ids, expected_pairs):
    with open(manifest_path) as f:
        m = json.load(f)

    if m["commit"] != expected_head:
        raise SystemExit(
            f"{manifest_path}: commit={m['commit']!r} != HEAD={expected_head!r}"
        )
    if m["block"]["thread_mode"] != "default":
        raise SystemExit(
            f"{manifest_path}: thread_mode={m['block']['thread_mode']!r} != default"
        )
    if m["block"]["sizes"] != [expected_n]:
        raise SystemExit(
            f"{manifest_path}: sizes={m['block']['sizes']!r} != [{expected_n}]"
        )
    if m["block"]["repetitions"] != expected_reps:
        raise SystemExit(
            f"{manifest_path}: repetitions={m['block']['repetitions']!r} "
            f"!= {expected_reps}"
        )

    ids = {img: info["id"] for img, info in m["images"].items()}
    if expected_ids is not None and ids != expected_ids:
        raise SystemExit(
            f"{manifest_path}: image ID отличаются от первого блока"
        )

    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    expected_rows = 12 * expected_reps
    if len(rows) != expected_rows:
        raise SystemExit(
            f"{csv_path} (n={expected_n}): {len(rows)} строк, "
            f"ожидалось {expected_rows}"
        )

    seen = set()
    for i, row in enumerate(rows, start=2):
        if row["commit"] != expected_head:
            raise SystemExit(
                f"{csv_path}: строка {i}: commit={row['commit']!r} != HEAD"
            )
        if row["thread_mode"] != "default":
            raise SystemExit(
                f"{csv_path}: строка {i}: thread_mode != default"
            )
        if row["image"] not in ids:
            raise SystemExit(
                f"{csv_path}: строка {i}: image={row['image']!r} нет в manifest"
            )
        if row["image_id"] != ids[row["image"]]:
            raise SystemExit(
                f"{csv_path}: строка {i}: image_id != manifest"
            )
        if row["swap_observed"] != "false":
            raise SystemExit(
                f"{csv_path}: строка {i}: swap_observed=true"
            )
        if int(row["n"]) != expected_n:
            raise SystemExit(
                f"{csv_path}: строка {i}: n={row['n']} != {expected_n}"
            )
        key = (row["implementation"], row["operation"],
               int(row["n"]), int(row["rep"]))
        if key in seen:
            raise SystemExit(f"{csv_path}: строка {i}: дубль {key}")
        seen.add(key)

    seen_pairs = {(row["implementation"], row["operation"]) for row in rows}
    if seen_pairs != expected_pairs:
        missing = expected_pairs - seen_pairs
        extra = seen_pairs - expected_pairs
        raise SystemExit(
            f"{csv_path}: не совпадает набор (impl, op): "
            f"missing={sorted(missing)}, extra={sorted(extra)}"
        )

    counts = Counter(row["image"] for row in rows)
    for image in ids:
        if counts[image] != expected_reps:
            raise SystemExit(
                f"{csv_path}: image={image!r} встречается "
                f"{counts[image]} раз, ожидалось {expected_reps}"
            )

    return ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--prefix", default="series9")
    p.add_argument("--dry-run", action="store_true")
    p.add_argument("--sizes", default=None)
    p.add_argument("--repetitions", type=int, default=None)
    p.add_argument("--allow-dirty", action="store_true",
                   help="Разрешить грязное дерево (только для пробного прогона)")
    args = p.parse_args()

    root = repo_root()
    results = root / "results"
    runks = root / "runks"
    results.mkdir(exist_ok=True)

    run_grid = load_run_grid(root)
    expected_pairs = {(impl, op) for _, impl, op in run_grid.COMBOS}

    if args.sizes and args.repetitions is None:
        raise SystemExit("--sizes требует --repetitions")
    if args.repetitions is not None and not args.sizes:
        raise SystemExit("--repetitions требует --sizes")

    if args.sizes:
        sizes = [int(s) for s in args.sizes.split(",")]
        if len(set(sizes)) != len(sizes):
            raise SystemExit(f"--sizes содержит повторы: {sizes}")
        schedule = [(n, args.repetitions) for n in sizes]
    else:
        schedule = DEFAULT_SERIES

    total = sum(12 * reps for _, reps in schedule)

    if args.dry_run:
        print(f"prefix: {args.prefix}")
        print(f"shuffle_seed: {SHUFFLE_SEED}")
        print(f"всего измерений: {total}")
        for n, reps in schedule:
            print(f"  n={n}: reps={reps}, измерений={12 * reps}")
        return 0

    for n, _ in schedule:
        for suffix in (".csv", ".csv.log", ".manifest.json"):
            path = results / f"{args.prefix}-n{n}{suffix}"
            if path.exists():
                raise SystemExit(
                    f"уже существует: {path}; используй новый --prefix"
                )

   
    head = subprocess.run(
        ["git", "-C", str(root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=True,
    ).stdout.strip()
    print(f"HEAD: {head}", file=sys.stderr)


    if not args.allow_dirty:
        dirty1 = subprocess.run(
            ["git", "-C", str(root), "diff", "--quiet"], capture_output=True
        ).returncode != 0
        dirty2 = subprocess.run(
            ["git", "-C", str(root), "diff", "--cached", "--quiet"],
            capture_output=True,
        ).returncode != 0
        if dirty1 or dirty2:
            raise SystemExit(
                "рабочее дерево грязное; закоммить или используй --allow-dirty"
            )

    def fwd_dirty():
        return ["--allow-dirty"] if args.allow_dirty else []

    csv_paths = []
    ref_ids = None
    series_start = time.time()

    for n, reps in schedule:
        block_start = time.time()
        csv_path = results / f"{args.prefix}-n{n}.csv"
        manifest_path = results / f"{args.prefix}-n{n}.manifest.json"

        print(f"\n=== n={n}, reps={reps} ===", file=sys.stderr, flush=True)

        run([sys.executable, str(runks / "manifest.py"),
             "--sizes", str(n), "--repetitions", str(reps),
             "--thread-mode", "default",
             "--shuffle-seed", str(SHUFFLE_SEED),
             "--output", str(csv_path)] + fwd_dirty())

        run([sys.executable, str(runks / "run_grid.py"),
             "--sizes", str(n), "--repetitions", str(reps),
             "--thread-mode", "default",
             "--shuffle-seed", str(SHUFFLE_SEED),
             "--output", str(csv_path)] + fwd_dirty())

        ids = check_block(csv_path, manifest_path, n, reps, head,
                          ref_ids, expected_pairs)
        if ref_ids is None:
            ref_ids = ids
        csv_paths.append(csv_path)

        block_elapsed = time.time() - block_start
        print(f"=== n={n} готово за {block_elapsed:.1f} с "
              f"({block_elapsed/60:.1f} мин) ===",
              file=sys.stderr, flush=True)

   
    print("\n=== анализ ===", file=sys.stderr, flush=True)
    analysis_dir = root / "analysis" / args.prefix
    all_sizes = ",".join(str(n) for n, _ in schedule)
    run([sys.executable, str(runks / "analyze.py"),
         "--csv"] + [str(p) for p in csv_paths] +
        ["--sizes", all_sizes, "--all",
         "--output-dir", str(analysis_dir)])
    with open(analysis_dir / "summary.csv") as f:
        summary_rows = sum(1 for _ in csv.DictReader(f))
    if summary_rows != 12 * len(schedule):
        raise SystemExit(
            f"{analysis_dir}/summary.csv: {summary_rows} строк, "
            f"ожидалось {12 * len(schedule)}"
        )
    with open(analysis_dir / "fits.csv") as f:
        fits_rows = sum(1 for _ in csv.DictReader(f))
    if fits_rows != 12:
        raise SystemExit(
            f"{analysis_dir}/fits.csv: {fits_rows} строк, ожидалось 12"
        )
    pngs = list(analysis_dir.glob("*.png"))
    if len(pngs) != 8:
        raise SystemExit(
            f"{analysis_dir}: {len(pngs)} PNG, ожидалось 8"
        )
    for kind in ("summary", "fits"):
        src = analysis_dir / f"{kind}.csv"
        dst = results / f"{args.prefix}-{kind}.csv"
        dst.write_bytes(src.read_bytes())

    total_elapsed = time.time() - series_start
    print(f"\nГотово за {total_elapsed:.1f} с ({total_elapsed/3600:.2f} ч)",
          file=sys.stderr)
    print(f"CSV: {len(csv_paths)} блоков, {total} измерений",
          file=sys.stderr)
    print(f"summary: {results/(args.prefix+'-summary.csv')}",
          file=sys.stderr)
    print(f"fits:    {results/(args.prefix+'-fits.csv')}",
          file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
