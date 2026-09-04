#!/usr/bin/env python3

import argparse
import csv
import datetime
import math
import pathlib
import subprocess
import sys
import os


class OutputError(Exception):"""Ошибка разбора или проверки вывода контейнера"""


def _parse_finite_float(raw, label):
    """Проверяет raw  конечное число, возвращает float"""
    try:
        value = float(raw)
    except (ValueError, TypeError) as exc:
        raise OutputError(f"{label} не число: {raw!r}") from exc
    if not math.isfinite(value):
        raise OutputError(f"{label} не конечное число: {raw!r}")
    return value


def expected_routines(implementation, operation):
    """Возвращает строку DIAG_ROUTINES"""
    if operation == "cholesky":
        if implementation == "numpy":
            return "dpotrf,dpotrs"
        return "dpotrf,dpotri"
    if operation == "lu":
        return "dgetrf,dgetri"
    if operation == "multiplication":
        return "dgemm"
    if operation == "svd":
        return "dgesdd,dgemm"
    raise ValueError(f"Неизвестная операция: {operation}")


def parse_output(stdout_text, implementation, operation, thread_mode):
    """Разбирает stdout контейнера и проверяет корректность
    """
    lines = [line for line in stdout_text.splitlines() if line.strip()]
    if len(lines) != 5:
        raise OutputError(f"Ожидалось 5 непустых строк, получено {len(lines)}")

    allowed = {
        "RESULT_SECONDS",
        "DIAG_THREADS",
        "DIAG_PEAK_RSS_KB",
        "DIAG_ROUTINES",
        "DIAG_CHECKSUM",
    }
    parsed = {}

    for line in lines:
        if "=" not in line:
            raise OutputError(f"Строка без '=': {line!r}")
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if key not in allowed:
            raise OutputError(f"Неизвестный ключ: {key!r}")
        if key in parsed:
            raise OutputError(f"Повторный ключ: {key!r}")
        parsed[key] = value

    for key in allowed:
        if key not in parsed:
            raise OutputError(f"Отсутствует ключ: {key}")

    # RESULT_SECONDS
    seconds_raw = parsed["RESULT_SECONDS"]
    seconds = _parse_finite_float(seconds_raw, "RESULT_SECONDS")
    if seconds <= 0:
        raise OutputError(f"RESULT_SECONDS должно быть положительным: {seconds_raw!r}")

    # DIAG_PEAK_RSS_KB
    rss_raw = parsed["DIAG_PEAK_RSS_KB"]
    try:
        rss = int(rss_raw)
    except ValueError as exc:
        raise OutputError(f"DIAG_PEAK_RSS_KB нецелое: {rss_raw!r}") from exc
    if rss <= 0:
        raise OutputError(f"DIAG_PEAK_RSS_KB должно быть положительным: {rss_raw!r}")

    # DIAG_THREADS
    thread_pools = parsed["DIAG_THREADS"]
    if not thread_pools:
        raise OutputError("DIAG_THREADS пусто")
    for entry in thread_pools.split(";"):
        entry = entry.strip()
        if ":" not in entry:
            raise OutputError(f"Запись потоков без ':': {entry!r}")
        left, right = entry.rsplit(":", 1)
        left = left.strip()
        right = right.strip()
        parts=left.split("/")
        if len(parts) != 2 or not parts[0] or not parts[1]:
            raise OutputError(f"Запись потоков должна иметь вид backend/prefix: {entry!r}")
        try:
            num_threads = int(right)
        except ValueError as exc:
            raise OutputError(f"Число потоков не целое в {entry!r}") from exc
        if num_threads <= 0:
            raise OutputError(f"Число потоков должно быть положительным: {entry!r}")
        if thread_mode == "single" and num_threads != 1:
            raise OutputError(f"В режиме single все пулы должны иметь 1 поток: {entry!r}")

    # DIAG_ROUTINES
    routines = parsed["DIAG_ROUTINES"]
    expected = expected_routines(implementation, operation)
    if routines != expected:
        raise OutputError(
            f"DIAG_ROUTINES={routines!r} не совпадает с ожидаемым {expected!r}"
        )

    # DIAG_CHECKSUM
    raw_checksum = parsed["DIAG_CHECKSUM"]
    if operation == "multiplication":
        parts = [p.strip() for p in raw_checksum.split(",")]
        if len(parts) != 2:
            raise OutputError(
                f"Для multiplication ожидалось две контрольные суммы, получено {len(parts)}"
            )
        for p in parts:
            _parse_finite_float(p, "DIAG_CHECKSUM")
        checksum_values = parts
    else:
        _parse_finite_float(raw_checksum, "DIAG_CHECKSUM")
        checksum_values = [raw_checksum]

    return {
        "seconds": seconds_raw,
        "thread_pools": thread_pools,
        "peak_rss_kb": str(rss),
        "routines": routines,
        "checksums": checksum_values,
    }


def vmstat_counter(name):
    """Читает глобальный счётчик из /proc/vmstat"""
    path = pathlib.Path("/proc/vmstat")
    if not path.exists():
        raise RuntimeError(f"Файл {path} не найден")
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2 and parts[0] == name:
            return int(parts[1])
    raise RuntimeError(f"В /proc/vmstat нет счётчика {name}")


def run_container(image, n, thread_mode):
    """Запускает docker run --rm синхронно и возвращает CompletedProcess"""
    command = ["docker", "run", "--rm"]
    if thread_mode == "single":
        command += [
            "-e", "OMP_NUM_THREADS=1",
            "-e", "MKL_NUM_THREADS=1",
            "-e", "OPENBLAS_NUM_THREADS=1",
        ]
    command += [image, str(n)]
    return subprocess.run(command, capture_output=True, text=True, check=False)


def main():
    parser = argparse.ArgumentParser(description="Запуск одного контейнера и проверка вывода")
    parser.add_argument("--image", required=True)
    parser.add_argument("--implementation", required=True, choices=["mkl", "openblas", "numpy"])
    parser.add_argument("--operation", required=True,
                        choices=["cholesky", "lu", "multiplication", "svd"])
    parser.add_argument("--n", type=int, required=True)
    parser.add_argument("--rep", type=int, required=True)
    parser.add_argument("--thread-mode", required=True, choices=["default", "single"])
    parser.add_argument("--run-order", type=int, required=True)
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--allow-dirty", action="store_true",
                        help="Предупреждать, а не останавливаться при грязном дереве")
    args = parser.parse_args()

    if args.n <= 0 or args.rep <= 0 or args.run_order <= 0 or args.shuffle_seed <= 0:
        parser.error("--n, --rep, --run-order, --shuffle-seed должны быть положительными")

    repo_root = pathlib.Path(__file__).resolve().parent.parent

    # Проверка чистоты рабочего дерева
    dirty = False
    for diff_cmd in (
        ["git", "-C", str(repo_root), "diff", "--quiet"],
        ["git", "-C", str(repo_root), "diff", "--cached", "--quiet"],
    ):
        res = subprocess.run(diff_cmd, capture_output=True, text=True)
        if res.returncode != 0:
            dirty = True

    if dirty and not args.allow_dirty:
        print(
            "Ошибкаь =  рабочее дерево содержит незакоммиченные изменения "
            "Закоммить их или --allow-dirty",
            file=sys.stderr,
        )
        sys.exit(1)
    if dirty:
        print(
            "Предупреждение: рабочее дерево грязное, но продолжаю из-за --allow-dirty.",
            file=sys.stderr,
        )

    # Полный commit
    try:
        commit_proc = subprocess.run(
            ["git", "-C", str(repo_root), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
    )
    except subprocess.CalledProcessError as exc:
        print(f"Ошибка при получении commit: {exc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    commit = commit_proc.stdout.strip()
    if len(commit) != 40:
        print(f"Ошибка: ожидался полный 40-символьный hash commit, получен {commit!r}", file=sys.stderr)
        sys.exit(1)

    # ID образа
    try:
        image_id_proc = subprocess.run(
            ["docker", "image", "inspect", "--format", "{{.Id}}", args.image],
            capture_output=True,
            text=True,
            check=True,
    )

    except subprocess.CalledProcessError as exc:
        print(f"Ошибка при получении образа {args.image}: {exc.stderr.strip()}", file=sys.stderr)
        sys.exit(1)
    image_id = image_id_proc.stdout.strip()

    # Счётчики свопа до запуска
    swap_in_before = vmstat_counter("pswpin")
    swap_out_before = vmstat_counter("pswpout")

    started_at = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ")

    result = run_container(args.image, args.n, args.thread_mode)

    swap_in_after = vmstat_counter("pswpin")
    swap_out_after = vmstat_counter("pswpout")
    swap_in_pages = max(0, swap_in_after - swap_in_before)
    swap_out_pages = max(0, swap_out_after - swap_out_before)
    swap_observed = swap_in_pages > 0 or swap_out_pages > 0

    if result.returncode != 0:
        print(f"Контейнер завершился с ненулевым кодом: {result.returncode}", file=sys.stderr)
        if result.stdout:
            print("stdout контейнера:\n" + result.stdout, file=sys.stderr)
        if result.stderr:
            print("stderr контейнера:\n" + result.stderr, file=sys.stderr)
        sys.exit(result.returncode or 1)

    if result.stderr:
        print("stderr контейнера (непустой):\n" + result.stderr, file=sys.stderr)

    try:
        parsed = parse_output(result.stdout, args.implementation, args.operation, args.thread_mode)
    except OutputError as exc:
        print(f"Ошибка разбора вывода: {exc}", file=sys.stderr)
        if result.stdout:
            print("stdout контейнера:\n" + result.stdout, file=sys.stderr)
        sys.exit(1)

    routines = parsed["routines"].split(",")
    routine_1 = routines[0] if len(routines) > 0 else ""
    routine_2 = routines[1] if len(routines) > 1 else ""

    checksums = parsed["checksums"]
    checksum_a = checksums[0] if len(checksums) > 0 else ""
    checksum_b = checksums[1] if len(checksums) > 1 else ""

    matrix_seed_a = args.n
    matrix_seed_b = args.n + 1 if args.operation == "multiplication" else ""
    
    for field in row:
        if isinstance (field,str) and "," in field: 
            print(f"щшибка поле содержит запятую: {field!r}", file=sys.stderr)
            sys.exit(1)
    row = [
        args.run_order,
        started_at,
        args.implementation,
        args.operation,
        args.n,
        args.rep,
        parsed["seconds"],
        parsed["thread_pools"],
        parsed["peak_rss_kb"],
        swap_in_pages,
        swap_out_pages,
        "true" if swap_observed else "false",
        args.thread_mode,
        args.image,
        image_id,
        commit,
        matrix_seed_a,
        matrix_seed_b,
        args.shuffle_seed,
        routine_1,
        routine_2,
        checksum_a,
        checksum_b,
    ]

    writer = csv.writer(sys.stdout, lineterminator="\n")
    writer.writerow(row)


if __name__ == "__main__":
    main()

#dirty
