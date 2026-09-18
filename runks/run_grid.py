#!/usr/bin/env python3

import argparse
import csv
import datetime
import hashlib
import os
import pathlib
import subprocess
import sys
import tempfile
from collections import namedtuple


COMBOS = [
    ("mkl_chol", "mkl", "cholesky"),
    ("lapack_chol", "openblas", "cholesky"),
    ("numpy_chol", "numpy", "cholesky"),
    ("mkl_lu", "mkl", "lu"),
    ("lapack_lu", "openblas", "lu"),
    ("numpy_lu", "numpy", "lu"),
    ("mkl_mul", "mkl", "multiplication"),
    ("lapack_mul", "openblas", "multiplication"),
    ("numpy_mul", "numpy", "multiplication"),
    ("mkl_svd", "mkl", "svd"),
    ("lapack_svd", "openblas", "svd"),
    ("numpy_svd", "numpy", "svd"),
]

HEADER = [
    "run_order",
    "started_at_utc",
    "implementation",
    "operation",
    "n",
    "rep",
    "seconds",
    "thread_pools",
    "peak_rss_kb",
    "swap_in_pages",
    "swap_out_pages",
    "swap_observed",
    "thread_mode",
    "image",
    "image_id",
    "commit",
    "matrix_seed_a",
    "matrix_seed_b",
    "shuffle_seed",
    "routine_1",
    "routine_2",
    "checksum_a",
    "checksum_b",
]

ScheduleEntry = namedtuple(
    "ScheduleEntry",
    ["run_order", "n", "rep", "image", "implementation", "operation"],
)


class GridError(Exception):
    """Ошибка сетки запусков."""


def parse_sizes(raw):
    """'256,512' -> [256, 512]. Проверяет повторы, ноль, мусор."""
    if not isinstance(raw, str) or not raw.strip():
        raise ValueError("--sizes пуст")
    parts = raw.split(",")
    sizes = []
    for part in parts:
        s = part.strip()
        if not s:
            raise ValueError(f"--sizes содержит пустой элемент: {raw!r}")
        try:
            n = int(s)
        except ValueError as exc:
            raise ValueError(f"--sizes: не число: {s!r}") from exc
        if n <= 0:
            raise ValueError(f"--sizes: должно быть положительным: {s!r}")
        sizes.append(n)
    if len(set(sizes)) != len(sizes):
        seen = set()
        dups = set()
        for s in sizes:
            if s in seen:
                dups.add(s)
            seen.add(s)
        raise ValueError(f"--sizes: повторяющиеся размеры: {sorted(dups)}")
    return sizes


def shuffled_combos(shuffle_seed, n, rep):
    """Возвращает все 12 сочетаний в порядке, зависящем от (seed, n, rep)."""
    def sort_key(combo):
        material = f"{shuffle_seed}:{n}:{rep}:{combo[0]}"
        return hashlib.sha256(material.encode("utf-8")).hexdigest()
    return sorted(COMBOS, key=sort_key)


def build_schedule(sizes, repetitions, shuffle_seed):
    """Возвращает список ScheduleEntry, run_order сквозной с 1."""
    schedule = []
    run_order = 1
    for n in sizes:
        for rep in range(1, repetitions + 1):
            for image, implementation, operation in shuffled_combos(shuffle_seed, n, rep):
                schedule.append(ScheduleEntry(
                    run_order=run_order,
                    n=n,
                    rep=rep,
                    image=image,
                    implementation=implementation,
                    operation=operation,
                ))
                run_order += 1
    return schedule


def check_row(row, expected):
    """
    Проверяет одну строку данных из run_one.py.
    """
    if len(row) != 23:
        raise GridError(
            f"run_order={expected['run_order']} image={expected['image']} "
            f"n={expected['n']} rep={expected['rep']}: "
            f"строка содержит {len(row)} полей вместо 23"
        )

    checks = [
        (0, "run_order", expected["run_order"]),
        (2, "implementation", expected["implementation"]),
        (3, "operation", expected["operation"]),
        (4, "n", expected["n"]),
        (5, "rep", expected["rep"]),
        (12, "thread_mode", expected["thread_mode"]),
        (13, "image", expected["image"]),
        (15, "commit", expected["commit"]),
        (18, "shuffle_seed", expected["shuffle_seed"]),
    ]
    for idx, name, exp in checks:
        if row[idx] != str(exp):
            raise GridError(
                f"run_order={expected['run_order']} image={expected['image']} "
                f"n={expected['n']} rep={expected['rep']}: "
                f"поле {name}={row[idx]!r}, ожидалось {str(exp)!r}"
            )


def check_block(rows, expected):
    """
    Проверяет весь блок.

    rows — список списков (первая строка — заголовок).
    expected — dict с ключами:
        num_runs, commit, thread_mode, shuffle_seed
    """
    num_runs = expected["num_runs"]
    if len(rows) != num_runs + 1:
        raise GridError(
            f"в блоке {len(rows)} строк (с заголовком), "
            f"ожидалось {num_runs + 1} ({num_runs} запусков)"
        )

    if rows[0] != HEADER:
        raise GridError(f"заголовок не совпадает с ожидаемым: {rows[0]!r}")

    data = rows[1:]

    for i, row in enumerate(data, start=1):
        if len(row) != 23:
            raise GridError(f"строка {i}: содержит {len(row)} полей вместо 23")

    for i, row in enumerate(data, start=1):
        if row[0] != str(i):
            raise GridError(f"строка {i}: run_order={row[0]!r}, ожидалось {i}")

    expected_commit = expected["commit"]
    expected_tm = expected["thread_mode"]
    expected_ss = str(expected["shuffle_seed"])

    for i, row in enumerate(data, start=1):
        if row[12] != expected_tm:
            raise GridError(
                f"строка {i}: thread_mode={row[12]!r}, ожидалось {expected_tm!r}"
            )
        if row[15] != expected_commit:
            raise GridError(
                f"строка {i}: commit={row[15]!r}, ожидалось {expected_commit!r}"
            )
        if row[18] != expected_ss:
            raise GridError(
                f"строка {i}: shuffle_seed={row[18]!r}, ожидалось {expected_ss!r}"
            )

    # Контрольные суммы между повторами одной (implementation, operation, n)
    groups = {}
    for i, row in enumerate(data, start=1):
        key = (row[2], row[3], row[4])  # implementation, operation, n
        groups.setdefault(key, []).append((i, (row[21], row[22])))
    for key, items in groups.items():
        first_i, first_sum = items[0]
        for i, s in items[1:]:
            if s != first_sum:
                raise GridError(
                    f"контрольные суммы разошлись для "
                    f"implementation={key[0]} operation={key[1]} n={key[2]}: "
                    f"строка {first_i} {first_sum} vs строка {i} {s}"
                )

    # Совпадение MKL и OpenBLAS для одной (operation, n)
    groups2 = {}
    for i, row in enumerate(data, start=1):
        if row[2] in ("mkl", "openblas"):
            key = (row[3], row[4])
            groups2.setdefault(key, {})[row[2]] = (i, (row[21], row[22]))
    for key, by_impl in groups2.items():
        if "mkl" in by_impl and "openblas" in by_impl:
            mkl_i, mkl_sum = by_impl["mkl"]
            obl_i, obl_sum = by_impl["openblas"]
            if mkl_sum != obl_sum:
                raise GridError(
                    f"контрольные суммы MKL и OpenBLAS разошлись для "
                    f"operation={key[0]} n={key[1]}: "
                    f"строка {mkl_i} (mkl) {mkl_sum} "
                    f"vs строка {obl_i} (openblas) {obl_sum}"
                )


#Вспомогательные функции для main


def _now_utc():
    return datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")


def _run_one_entry(entry, run_one_path, thread_mode, shuffle_seed,
                   allow_dirty, recorded_head, writer, log_handle):
    """Запускает один контейнер через run_one.py, проверяет строку, пишет в CSV."""
    started = _now_utc()
    print(
        f"[{started}] run_order={entry.run_order} image={entry.image} "
        f"n={entry.n} rep={entry.rep}",
        file=sys.stderr,
    )
    log_handle.write(
        f"=== run_order={entry.run_order} image={entry.image} "
        f"n={entry.n} rep={entry.rep} ===\n"
    )
    log_handle.write(f"started_at={started}\n")
    log_handle.flush()

    command = [
        sys.executable, str(run_one_path),
        "--image", entry.image,
        "--implementation", entry.implementation,
        "--operation", entry.operation,
        "--n", str(entry.n),
        "--rep", str(entry.rep),
        "--thread-mode", thread_mode,
        "--run-order", str(entry.run_order),
        "--shuffle-seed", str(shuffle_seed),
    ]
    if allow_dirty:
        command.append("--allow-dirty")

    result = subprocess.run(command, capture_output=True, text=True, check=False)

    finished = _now_utc()
    log_handle.write(f"finished_at={finished}\n")
    log_handle.write(f"returncode={result.returncode}\n")

    if result.stderr:
        log_handle.write(f"child_stderr:\n{result.stderr}")
        print(result.stderr, file=sys.stderr)

    if result.returncode != 0:
        log_handle.write("status=ERROR\n")
        log_handle.flush()
        raise GridError(
            f"run_order={entry.run_order} image={entry.image} "
            f"n={entry.n} rep={entry.rep}: "
            f"run_one.py завершился с кодом {result.returncode}"
        )

    lines = [line for line in result.stdout.splitlines() if line.strip()]
    if len(lines) != 1:
        log_handle.write(f"stdout:\n{result.stdout}")
        log_handle.write("status=ERROR\n")
        log_handle.flush()
        raise GridError(
            f"run_order={entry.run_order} image={entry.image} "
            f"n={entry.n} rep={entry.rep}: "
            f"в stdout {len(lines)} непустых строк, ожидалась одна"
        )

    parsed = list(csv.reader([lines[0]]))
    if len(parsed) != 1:
        log_handle.write(f"stdout:\n{result.stdout}")
        log_handle.write("status=ERROR\n")
        log_handle.flush()
        raise GridError(
            f"run_order={entry.run_order} image={entry.image} "
            f"n={entry.n} rep={entry.rep}: "
            f"csv.reader вернул {len(parsed)} строк вместо одной"
        )
    row = parsed[0]

    expected = {
        "run_order": entry.run_order,
        "n": entry.n,
        "rep": entry.rep,
        "image": entry.image,
        "implementation": entry.implementation,
        "operation": entry.operation,
        "thread_mode": thread_mode,
        "shuffle_seed": shuffle_seed,
        "commit": recorded_head,
    }

    try:
        check_row(row, expected)
    except GridError:
        log_handle.write(f"stdout:\n{result.stdout}")
        log_handle.write("status=ERROR\n")
        log_handle.flush()
        raise

    log_handle.write(f"data={lines[0]}\n")
    log_handle.write("status=OK\n")
    log_handle.flush()

    writer.writerow(row)


# main


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Сетка запусков над run_one.py"
    )
    parser.add_argument("--sizes", required=True,
                        help="Список размеров через запятую, например 256,512")
    parser.add_argument("--repetitions", type=int, required=True,
                        help="Число повторов каждого размера")
    parser.add_argument("--thread-mode", required=True,
                        choices=["default", "single"])
    parser.add_argument("--shuffle-seed", type=int, required=True,
                        help="Seed для перемешивания порядка")
    parser.add_argument("--output", required=True,
                        help="Путь к целевому CSV")
    parser.add_argument("--dry-run", action="store_true",
                        help="Печать расписания и выход")
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Разрешить грязное рабочее дерево")
    args = parser.parse_args(argv)

    try:
        sizes = parse_sizes(args.sizes)
    except ValueError as exc:
        parser.error(str(exc))

    if args.repetitions <= 0:
        parser.error("--repetitions должно быть положительным")
    if args.shuffle_seed <= 0:
        parser.error("--shuffle-seed должно быть положительным")

    repo_root = pathlib.Path(__file__).resolve().parent.parent

    schedule = build_schedule(sizes, args.repetitions, args.shuffle_seed)

    # --dry-run
    if args.dry_run:
        for entry in schedule:
            print(
                f"{entry.run_order}\t{entry.n}\t{entry.rep}\t"
                f"{entry.image}\t{entry.implementation}\t{entry.operation}"
            )
        return 0

    output_path = pathlib.Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path = output_path.resolve()
    log_path = output_path.with_name(output_path.name + ".log")

    if output_path.exists():
        print(f"Ошибка: целевой CSV уже существует: {output_path}",
              file=sys.stderr)
        return 1
    if log_path.exists():
        print(f"Ошибка: журнал уже существует: {log_path}", file=sys.stderr)
        return 1

    # docker и git
    for tool in ("docker", "git"):
        try:
            res = subprocess.run([tool, "--version"],
                                 capture_output=True, text=True)
        except FileNotFoundError:
            print(f"Ошибка: {tool} недоступен", file=sys.stderr)
            return 1
        if res.returncode != 0:
            print(f"Ошибка: {tool} --version завершился с кодом "
                  f"{res.returncode}: {res.stderr.strip()}", file=sys.stderr)
            return 1

    # образы
    unique_images = sorted({e.image for e in schedule})
    for image in unique_images:
        res = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", image],
            capture_output=True, text=True, check=False,
        )
        if res.returncode != 0:
            print(f"Ошибка: образ недоступен: {image}: {res.stderr.strip()}",
                   file=sys.stderr)
            return 1

    # чистота рабочего дерева
    dirty = False
    for diff_cmd in (
        ["git", "-C", str(repo_root), "diff", "--quiet"],
        ["git", "-C", str(repo_root), "diff", "--cached", "--quiet"],
    ):
        res = subprocess.run(diff_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            dirty = True
    if dirty and not args.allow_dirty:
        print("Ошибка: рабочее дерево содержит незакоммиченные изменения. "
              "Закоммитьте их или используйте --allow-dirty.", file=sys.stderr)
        return 1
    if dirty:
        print("Предупреждение: рабочее дерево грязное, продолжаю "
              "из-за --allow-dirty.", file=sys.stderr)

    # HEAD
    head_res = subprocess.run(
        ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
        capture_output=True, text=True, check=False,
    )
    if head_res.returncode != 0:
        print(f"Ошибка: git rev-parse HEAD завершился с кодом "
              f"{head_res.returncode}: {head_res.stderr.strip()}",
              file=sys.stderr)
        return 1
    recorded_head = head_res.stdout.strip()
    if len(recorded_head) != 40:
        print(f"Ошибка: HEAD не является полным 40-символьным hash: "
              f"{recorded_head!r}", file=sys.stderr)
        return 1

    # каталог для output
    output_path.parent.mkdir(parents=True, exist_ok=True)

    run_one_path = repo_root / "runks" / "run_one.py"
    if not run_one_path.exists():
        print(f"Ошибка: не найден {run_one_path}", file=sys.stderr)
        return 1

    expected_block = {
        "num_runs": len(schedule),
        "commit": recorded_head,
        "thread_mode": args.thread_mode,
        "shuffle_seed": args.shuffle_seed,
    }

    log_handle = open(log_path, "w", encoding="utf-8")
    temp_handle = tempfile.NamedTemporaryFile(
        "w", newline="", dir=str(output_path.parent),
        prefix=output_path.name + ".", suffix=".part", delete=False,
        encoding="utf-8",
    )
    temp_path = pathlib.Path(temp_handle.name)

    try:
        writer = csv.writer(temp_handle, lineterminator="\n")
        writer.writerow(HEADER)

        for entry in schedule:
            _run_one_entry(
                entry=entry,
                run_one_path=run_one_path,
                thread_mode=args.thread_mode,
                shuffle_seed=args.shuffle_seed,
                allow_dirty=args.allow_dirty,
                recorded_head=recorded_head,
                writer=writer,
                log_handle=log_handle,
            )

        temp_handle.close()

        with open(temp_path, newline="", encoding="utf-8") as f:
            rows = list(csv.reader(f))

        try:
            check_block(rows, expected_block)
        except GridError as exc:
            log_handle.write(f"block_check_error={exc}\n")
            log_handle.flush()
            print(f"Ошибка проверки блока: {exc}", file=sys.stderr)
            temp_path.unlink(missing_ok=True)
            return 1

        os.replace(str(temp_path), str(output_path))
        log_handle.write(f"published: {output_path}\n")
        log_handle.flush()
        print(f"Опубликовано: {output_path}", file=sys.stderr)
        return 0
    except GridError as exc:
        try:
            temp_handle.close()
        except Exception:
            pass
        temp_path.unlink(missing_ok=True)
        log_handle.write(f"aborted: GridError: {exc}\n")
        log_handle.flush()
        print(f"Ошибка: {exc}", file=sys.stderr)
        return 1
    except BaseException as exc:
        try:
            temp_handle.close()
        except Exception:
            pass
        temp_path.unlink(missing_ok=True)
        try:
            log_handle.write(f"aborted: {type(exc).__name__}: {exc}\n")
            log_handle.flush()
        except Exception:
            pass
        raise
    finally:
        try:
            log_handle.close()
        except Exception:
            pass


if __name__ == "__main__":
    sys.exit(main())
