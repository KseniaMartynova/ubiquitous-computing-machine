#!/usr/bin/env python3

import argparse
import pathlib
import subprocess
import sys


BUILD_FILES = {
    "mkl_chol":    ("mkl/cholesky",    "Dockerfile.mklcho"),
    "lapack_chol": ("lapack/cholesky", "Dockerfile.lachol"),
    "numpy_chol":  ("numpy/cholesky",  "Dockerfile.numcho"),
    "mkl_lu":      ("mkl/lu",          "Dockerfile.mkllu"),
    "lapack_lu":   ("lapack/lu",       "Dockerfile.lapackLU"),
    "numpy_lu":    ("numpy/lu",        "Dockerfile.numlu"),
    "mkl_mul":     ("mkl/multiplication",         "Dockerfile.mklmul"),
    "lapack_mul":  ("lapack/multiplication",      "Dockerfile.lamul"),
    "numpy_mul":   ("numpy/multiplication",       "Dockerfile.nummul"),
    "mkl_svd":     ("mkl/svd",         "Dockerfile.mklsvd"),
    "lapack_svd":  ("lapack/svd",      "Dockerfile.lasvd"),
    "numpy_svd":   ("numpy/svd",       "Dockerfile.numsvd"),
}


def _repo_root():
    return pathlib.Path(__file__).resolve().parent.parent

def validate_only_name(only, build_files):
    if only is not None and only not in build_files:
        raise SystemExit(f"Ошибка: неизвестный образ: {only}")

def _import_combos():
    runks_dir = pathlib.Path(__file__).resolve().parent
    if str(runks_dir) not in sys.path:
        sys.path.insert(0, str(runks_dir))
    from run_grid import COMBOS
    return COMBOS


def _check_consistency(combos):
    """BUILD_FILES и COMBOS должны описывать один и тот же набор образов"""
    combos_images = {c[0] for c in combos}
    build_images = set(BUILD_FILES.keys())
    if combos_images == build_images:
        return
    parts = []
    missing = combos_images - build_images
    extra = build_images - combos_images
    if missing:
        parts.append(f"нет в BUILD_FILES: {sorted(missing)}")
    if extra:
        parts.append(f"лишние в BUILD_FILES: {sorted(extra)}")
    raise SystemExit("Ошибка: BUILD_FILES не совпадает с COMBOS: " + "; ".join(parts))


def _check_docker():
    try:
        res = subprocess.run(["docker", "--version"],
                             capture_output=True, text=True)
    except FileNotFoundError:
        raise SystemExit("Ошибка: docker недоступен")
    if res.returncode != 0:
        raise SystemExit(f"Ошибка: docker --version: {res.stderr.strip()}")


def _check_dockerfiles(build_dir):
    problems = []
    for image, (rel_dir, dockerfile) in BUILD_FILES.items():
        path = build_dir / rel_dir / dockerfile
        if not path.exists():
            problems.append(f"{image}: {path}")
    if problems:
        raise SystemExit(
            "Ошибка: не найдены Dockerfile:\n  " + "\n  ".join(problems)
        )


def _build_one(image, build_dir):
    rel_dir, dockerfile = BUILD_FILES[image]
    context = build_dir / rel_dir
    dockerfile_path = context / dockerfile

    print(f" {image} <- {dockerfile_path}", file=sys.stderr)
    cmd = ["docker", "build", "-t", image,
           "-f", str(dockerfile_path), str(context)]
    res = subprocess.run(cmd, check=False)
    if res.returncode != 0:
        raise SystemExit(
            f"Ошибка: сборка {image} завершилась с кодом {res.returncode}. "
            f"Dockerfile: {dockerfile_path}"
        )

    inspect = subprocess.run(
        ["docker", "image", "inspect", "--format", "{{.Id}}", image],
        capture_output=True, text=True, check=False,
    )
    if inspect.returncode != 0:
        raise SystemExit(
            f"Ошибка: после сборки {image} тег не найден: "
            f"{inspect.stderr.strip()}"
        )
    return inspect.stdout.strip()


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Собрать двенадцать образов одной командой"
    )
    parser.add_argument("--only",
                        help="Собрать только указанный образ")
    args = parser.parse_args(argv)

    combos = _import_combos()
    _check_consistency(combos)

    validate_only_name(args.only, BUILD_FILES)

    build_dir = _repo_root() / "runks" / "build"
    if not build_dir.exists():
        raise SystemExit(f"Ошибка: каталог сборки не найден: {build_dir}")

    _check_docker()
    _check_dockerfiles(build_dir)

    if args.only:
        images = [args.only]
    else:
        images = [c[0] for c in combos]

    results = []
    for image in images:
        image_id = _build_one(image, build_dir)
        results.append((image, image_id))

    for image, image_id in results:
        print(f"{image}\t{image_id}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
