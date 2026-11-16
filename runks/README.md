## Полная серия окончательных измерений

### План без вычислений
./runks/run_series9.py --dry-run

### Пробный прогон n=256
./runks/run_series9.py --prefix check9 --sizes 256 --repetitions 1 --allow-dirty

### Проверка корректности всех 12 образов
python3 - <<'PY' > results/series9-validate256.log 2>&1
import math, subprocess
from runks.run_grid import COMBOS
for image, _, _ in COMBOS:
    print(f"=== {image} ===", flush=True)
    r = subprocess.run(["docker", "run", "--rm", image, "--validate", "256"],
                       capture_output=True, text=True)
    print(r.stdout, end="", flush=True)
    if r.stderr: print(r.stderr, end="", flush=True)
    lines = r.stdout.splitlines()
    assert r.returncode == 0 and len(lines) == 2
    assert lines[0].startswith("VALIDATE_RESIDUAL=")
    residual = float(lines[0].split("=", 1)[1])
    assert math.isfinite(residual) and 0 <= residual <= 1e-10
    assert lines[1] == "VALIDATE_STATUS=ok"
print("Все 12 проверок пройдены", flush=True)
PY

### Зaпуск всей серии на tmux
tmux new -s series9
./runks/run_series9.py --prefix series9 2>&1 | tee results/series9-run.log
