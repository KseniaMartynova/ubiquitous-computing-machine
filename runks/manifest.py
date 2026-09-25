#!/usr/bin/env python3

import argparse
import datetime
import json
import os
import pathlib
import platform
import re
import subprocess
import sys
import tempfile
import time


class ManifestError(Exception):
    pass

def _repo_root():
    return pathlib.Path(__file__).resolve().parent.parent


def _runks_dir():
    return pathlib.Path(__file__).resolve().parent


def _ensure_runks_on_path():
    runks_dir = _runks_dir()
    if str(runks_dir) not in sys.path:
        sys.path.insert(0, str(runks_dir))


def _import_run_grid():
    _ensure_runks_on_path()
    import run_grid
    return run_grid


def _import_build_all():
    _ensure_runks_on_path()
    import build_all
    return build_all



def _run(cmd):
    return subprocess.run(cmd, capture_output=True, text=True, check=False)


def _check_nonzero(result, what):
    if result.returncode != 0:
        raise ManifestError(
            f"{what}: команда {' '.join(result.args)} завершилась с кодом "
            f"{result.returncode}: {result.stderr.strip()}"
        )




def _logical_lines_text(text):
 
    result = []
    current = ""
    for line in text.splitlines():
        current = (current + " " + line.strip()) if current else line.strip()
        if current.endswith("\\"):
            current = current[:-1].rstrip()
        else:
            if current:
                result.append(current)
            current = ""
    if current:
        result.append(current)
    return result


def from_line(text):
    for line in _logical_lines_text(text):
        m = re.match(r"^FROM\s+(.+)$", line)
        if m:
            value = m.group(1).strip()
            if "@sha256:" not in value:
                raise ManifestError(f"FROM без digest: {value!r}")
            return value
    raise ManifestError("не найдена строка FROM")


def compile_line(text):
    """Возвращает единственную строку RUN с icpx или g++ """
    candidates = []
    for line in _logical_lines_text(text):
        m = re.match(r"^RUN\s+(.+)$", line)
        if m:
            body = m.group(1).strip()
            if "icpx" in body or "g++" in body:
                candidates.append(body)
    if len(candidates) != 1:
        raise ManifestError(
            f"ожидалась ровно одна строка RUN с icpx/g++, найдено "
            f"{len(candidates)}"
        )
    return candidates[0]


def entrypoint_binary(text):
   
    for line in _logical_lines_text(text):
        m = re.match(r"^ENTRYPOINT\s+(.+)$", line)
        if m:
            body = m.group(1).strip()
            try:
                parsed = json.loads(body)
            except json.JSONDecodeError as exc:
                raise ManifestError(
                    f"ENTRYPOINT не JSON-массив: {body!r}"
                ) from exc
            if isinstance(parsed, list) and len(parsed) == 1:
                return parsed[0]
            raise ManifestError(
                f"ENTRYPOINT должен быть массивом из одного элемента: "
                f"{parsed!r}"
            )
    raise ManifestError("не найдена строка ENTRYPOINT")


def parse_lscpu(text):
    fields = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            fields[key.strip()] = value.strip()
    if "Model name" not in fields:
        raise ManifestError("в lscpu нет строки 'Model name'")

    def to_int(x):
        try:
            return int(x)
        except (TypeError, ValueError):
            return None

    return {
        "cpu_model": fields["Model name"],
        "sockets": to_int(fields.get("Socket(s)")),
        "cores_per_socket": to_int(fields.get("Core(s) per socket")),
        "threads_per_core": to_int(fields.get("Thread(s) per core")),
        "logical_cpus": to_int(fields.get("CPU(s)")),
    }


def parse_meminfo(text):
    """Возвращает MemTotal и MemAvailable в килобайтах"""
    values = {}
    for line in text.splitlines():
        if ":" in line:
            key, value = line.split(":", 1)
            values[key.strip()] = value.strip()
    if "MemTotal" not in values:
        raise ManifestError("в /proc/meminfo нет MemTotal")
    if "MemAvailable" not in values:
        raise ManifestError("в /proc/meminfo нет MemAvailable")
    return {
        "mem_total_kb": int(values["MemTotal"].split()[0]),
        "mem_available_kb": int(values["MemAvailable"].split()[0]),
    }


def parse_dpkg(text, packages):
    """Разбирает вывод dpkg-query -W"""
    result = {}
    for line in text.splitlines():
        line = line.strip()
        if not line:
            continue
        parts = line.split(None, 1)
        if len(parts) == 2:
            result[parts[0]] = parts[1]
    for pkg in packages:
        if pkg not in result:
            raise ManifestError(
                f"dpkg-query не вернул версию для {pkg}; "
                f"получено: {sorted(result.keys())}"
            )
    return result


def manifest_path(csv_path):
    """x.csv -> x.manifest.json"""
    p = pathlib.Path(csv_path)
    if p.suffix != ".csv":
        raise ManifestError(f"путь должен оканчиваться на .csv: {csv_path}")
    return p.with_suffix(".manifest.json")


def check_manifest_absent(path):

    path = pathlib.Path(path)
    if path.exists():
        raise ManifestError(f"manifest уже существует: {path}")




def _collect_machine():
    lscpu_res = _run(["lscpu"])
    _check_nonzero(lscpu_res, "machine: lscpu")
    lscpu_fields = parse_lscpu(lscpu_res.stdout)
    lscpu_lines = [line for line in lscpu_res.stdout.splitlines() if line.strip()]

    meminfo_text = pathlib.Path("/proc/meminfo").read_text()
    mem = parse_meminfo(meminfo_text)

    governor = None
    governor_path = pathlib.Path(
        "/sys/devices/system/cpu/cpu0/cpufreq/scaling_governor"
    )
    if governor_path.exists():
        governor = governor_path.read_text().strip()

    os_pretty = None
    os_release = pathlib.Path("/etc/os-release")
    if os_release.exists():
        for line in os_release.read_text().splitlines():
            if line.startswith("PRETTY_NAME="):
                os_pretty = line.split("=", 1)[1].strip().strip('"')
                break

    return {
        "cpu_model": lscpu_fields["cpu_model"],
        "sockets": lscpu_fields["sockets"],
        "cores_per_socket": lscpu_fields["cores_per_socket"],
        "threads_per_core": lscpu_fields["threads_per_core"],
        "logical_cpus": lscpu_fields["logical_cpus"],
        "host_cpu_count": os.cpu_count(),
        "mem_total_kb": mem["mem_total_kb"],
        "mem_available_kb": mem["mem_available_kb"],
        "governor": governor,
        "os": os_pretty,
        "kernel": platform.release(),
        "lscpu": lscpu_lines,
    }


def _collect_tooling():
    docker_res = _run(["docker", "version", "--format", "{{.Server.Version}}"])
    _check_nonzero(docker_res, "tooling: docker version")
    docker_version = docker_res.stdout.strip()
    if not docker_version:
        raise ManifestError("tooling: docker version пустой ответ")
    return {
        "docker": docker_version,
        "python": sys.version,
    }




def _image_id(image):
    res = _run(["docker", "image", "inspect", "--format", "{{.Id}}", image])
    _check_nonzero(res, f"image {image}: inspect")
    iid = res.stdout.strip()
    if not iid:
        raise ManifestError(f"image {image}: пустой image ID")
    return iid


def _collect_ldd(image, binary):
    res = _run(["docker", "run", "--rm", "--entrypoint", "ldd", image, binary])
    _check_nonzero(res, f"image {image}: ldd {binary}")
    lines = [line for line in res.stdout.splitlines() if line.strip()]
    if not lines:
        raise ManifestError(f"image {image}: ldd {binary} пустой вывод")
    return lines


def _collect_apt(image):
    packages = ["libopenblas-dev", "liblapack-dev", "liblapacke-dev"]
    fmt = "-f=${Package} ${Version}\\n"
    cmd = ["docker", "run", "--rm", "--entrypoint", "dpkg-query",
           image, "-W", fmt] + packages
    res = _run(cmd)
    _check_nonzero(res, f"image {image}: dpkg-query")
    return parse_dpkg(res.stdout, packages)


def _collect_icpx_and_mklroot(image):
    ver_res = _run(["docker", "run", "--rm", "--entrypoint", "icpx",
                    image, "--version"])
    _check_nonzero(ver_res, f"image {image}: icpx --version")
    icpx_version = ver_res.stdout.strip()
    if not icpx_version:
        raise ManifestError(f"image {image}: icpx --version пустой вывод")

    mkl_res = _run(["docker", "run", "--rm", "--entrypoint", "sh",
                    image, "-c", 'echo "$MKLROOT"'])
    _check_nonzero(mkl_res, f"image {image}: echo $MKLROOT")
    mklroot = mkl_res.stdout.strip()
    if not mklroot:
        raise ManifestError(f"image {image}: MKLROOT пустой")

    return icpx_version, mklroot


_PYTHON_PACKAGES_SCRIPT = (
    "import json, numpy, scipy, scipy.linalg, threadpoolctl; "
    "print(json.dumps({'numpy': numpy.__version__, "
    "'scipy': scipy.__version__, "
    "'threadpoolctl': threadpoolctl.__version__, "
    "'pools': threadpoolctl.threadpool_info()}))"
)


def _collect_python_packages(image):
    res = _run(["docker", "run", "--rm", "--entrypoint", "python3", image,
                "-c", _PYTHON_PACKAGES_SCRIPT])
    _check_nonzero(res, f"image {image}: python packages")
    try:
        data = json.loads(res.stdout)
    except json.JSONDecodeError as exc:
        raise ManifestError(
            f"image {image}: не удалось разобрать JSON от python: {exc}: "
            f"{res.stdout[:200]!r}"
        ) from exc
    return data


def _container_cpu_count(image):
    res = _run(["docker", "run", "--rm", "--entrypoint", "nproc", image])
    _check_nonzero(res, f"image {image}: nproc")
    try:
        return int(res.stdout.strip())
    except ValueError as exc:
        raise ManifestError(
            f"image {image}: nproc вернул {res.stdout!r}"
        ) from exc


def _collect_image(image, implementation, build_dir, build_files):
    rel_dir, dockerfile = build_files[image]
    dockerfile_path = build_dir / rel_dir / dockerfile
    if not dockerfile_path.exists():
        raise ManifestError(
            f"image {image}: Dockerfile не найден: {dockerfile_path}"
        )

    text = dockerfile_path.read_text(encoding="utf-8")
    info = {
        "id": _image_id(image),
        "from": from_line(text),
    }

    if implementation in ("mkl", "openblas"):
        info["compile"] = compile_line(text)
        binary = entrypoint_binary(text)
        info["ldd"] = _collect_ldd(image, binary)

    if implementation == "openblas":
        info["apt"] = _collect_apt(image)

    if implementation == "mkl":
        icpx_version, mklroot = _collect_icpx_and_mklroot(image)
        info["icpx"] = icpx_version
        info["mklroot"] = mklroot

    if implementation == "numpy":
        info["python"] = _collect_python_packages(image)

    return info

def _check_docker():
    try:
        res = _run(["docker", "--version"])
    except FileNotFoundError as exc:
        raise ManifestError("docker недоступен") from exc
    _check_nonzero(res, "docker --version")


def _check_git():
    try:
        res = _run(["git", "--version"])
    except FileNotFoundError as exc:
        raise ManifestError("git недоступен") from exc
    _check_nonzero(res, "git --version")


def _check_images(images):
    missing = []
    for image in images:
        res = _run(["docker", "image", "inspect", "--format", "{{.Id}}", image])
        if res.returncode != 0:
            missing.append(image)
    if missing:
        raise ManifestError("не найдены образы: " + ", ".join(missing))


def _check_worktree(repo_root, allow_dirty):
    dirty = False
    for cmd in (
        ["git", "-C", str(repo_root), "diff", "--quiet"],
        ["git", "-C", str(repo_root), "diff", "--cached", "--quiet"],
    ):
        res = _run(cmd)
        if res.returncode != 0:
            dirty = True
    if dirty and not allow_dirty:
        raise ManifestError(
            "рабочее дерево содержит незакоммиченные изменения. "
            "Закоммитьте их или используйте --allow-dirty"
        )
    return not dirty


def _git_head(repo_root):
    res = _run(["git", "-C", str(repo_root), "rev-parse", "HEAD"])
    _check_nonzero(res, "git rev-parse HEAD")
    head = res.stdout.strip()
    if len(head) != 40:
        raise ManifestError(f"HEAD не 40-символьный hash: {head!r}")
    return head


def main(argv=None):
    parser = argparse.ArgumentParser(
        description="Собрать manifest.json для блока измерений"
    )
    parser.add_argument("--sizes", required=True)
    parser.add_argument("--repetitions", type=int, required=True)
    parser.add_argument("--thread-mode", required=True,
                        choices=["default", "single"])
    parser.add_argument("--shuffle-seed", type=int, required=True)
    parser.add_argument("--output", required=True,
                        help="Путь к будущему CSV "
                             "(например, results/pilot-default.csv)")
    parser.add_argument("--allow-dirty", action="store_true")
    args = parser.parse_args(argv)

    run_grid = _import_run_grid()
    build_all = _import_build_all()

    try:
        sizes = run_grid.parse_sizes(args.sizes)
    except ValueError as exc:
        parser.error(str(exc))

    if args.repetitions <= 0:
        parser.error("--repetitions должно быть положительным")
    if args.shuffle_seed <= 0:
        parser.error("--shuffle-seed должно быть положительным")

    repo_root = _repo_root()

    output_path = pathlib.Path(args.output)
    if not output_path.is_absolute():
        output_path = repo_root / output_path
    output_path = output_path.resolve()

    try:
        manifest_path_obj = manifest_path(output_path)
    except ManifestError as exc:
        parser.error(str(exc))

    try:
        check_manifest_absent(manifest_path_obj)
    except ManifestError as exc:
        raise SystemExit(f"Ошибка: {exc}")

    build_dir = repo_root / "runks" / "build"
    if not build_dir.exists():
        raise SystemExit(f"Ошибка: каталог сборки не найден: {build_dir}")

    combos = run_grid.COMBOS
    build_files = build_all.BUILD_FILES

    combos_images = {c[0] for c in combos}
    build_images = set(build_files.keys())
    if combos_images != build_images:
        raise SystemExit(
            "Ошибка: BUILD_FILES и COMBOS не совпадают: "
            f"только в COMBOS: {sorted(combos_images - build_images)}, "
            f"только в BUILD_FILES: {sorted(build_images - combos_images)}"
        )

    try:
        _check_docker()
        _check_git()
        _check_images([c[0] for c in combos])
        worktree_clean = _check_worktree(repo_root, args.allow_dirty)
        commit = _git_head(repo_root)

        machine = _collect_machine()
        tooling = _collect_tooling()
        machine["container_cpu_count"] = _container_cpu_count(combos[0][0])

        images_info = {}
        for image, implementation, _operation in combos:
            images_info[image] = _collect_image(
                image, implementation, build_dir, build_files
            )
    except ManifestError as exc:
        raise SystemExit(f"Ошибка: {exc}")

    created_utc = datetime.datetime.now(datetime.timezone.utc).strftime(
        "%Y-%m-%dT%H:%M:%SZ"
    )
    tz = time.strftime("%Z %z")

    try:
        output_rel = str(output_path.relative_to(repo_root))
    except ValueError:
        output_rel = str(output_path)

    manifest_data = {
        "created_utc": created_utc,
        "timezone": tz,
        "commit": commit,
        "worktree_clean": worktree_clean,
        "machine": machine,
        "tooling": tooling,
        "images": images_info,
        "block": {
            "sizes": sizes,
            "repetitions": args.repetitions,
            "thread_mode": args.thread_mode,
            "shuffle_seed": args.shuffle_seed,
            "matrix_seed_rule":
                "seed = n; для multiplication вторая матрица n + 1",
            "output": output_rel,
        },
    }

    manifest_path_obj.parent.mkdir(parents=True, exist_ok=True)
    handle = tempfile.NamedTemporaryFile(
        "w", dir=str(manifest_path_obj.parent),
        prefix=manifest_path_obj.name + ".", suffix=".part",
        delete=False, encoding="utf-8",
    )
    temp_path = pathlib.Path(handle.name)
    try:
        json.dump(manifest_data, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
        handle.close()

        with open(temp_path, encoding="utf-8") as f:
            json.load(f)

        os.replace(str(temp_path), str(manifest_path_obj))
    except BaseException:
        try:
            handle.close()
        except Exception:
            pass
        temp_path.unlink(missing_ok=True)
        raise

    print(f"Manifest записан: {manifest_path_obj}", file=sys.stderr)
    return 0


if __name__ == "__main__":
    sys.exit(main())
