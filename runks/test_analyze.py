#!/usr/bin/env python3

import contextlib
import csv
import io
import json
import math
import pathlib
import shutil
import tempfile
import unittest

from runks import analyze

FIXTURES = pathlib.Path(__file__).resolve().parent / "fixtures" / "analyze"


def _maybe_float(value):
    if value is None or value == "":
        return None
    return float(value)


def read_summary(path):
    result = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result.append({
                "operation": row["operation"],
                "implementation": row["implementation"],
                "thread_mode": row["thread_mode"],
                "n": int(row["n"]),
                "count": int(row["count"]),
                "median_seconds": float(row["median_seconds"]),
                "q1_seconds": float(row["q1_seconds"]),
                "q3_seconds": float(row["q3_seconds"]),
                "gflops": float(row["gflops"]),
                "peak_rss_kb_max": int(row["peak_rss_kb_max"]),
            })
    return result


def read_fits(path):
    result = []
    with open(path, newline="", encoding="utf-8") as f:
        for row in csv.DictReader(f):
            result.append({
                "operation": row["operation"],
                "implementation": row["implementation"],
                "thread_mode": row["thread_mode"],
                "n_points": int(row["n_points"]),
                "a": _maybe_float(row["a"]),
                "c": _maybe_float(row["c"]),
                "c_ci95_low": _maybe_float(row["c_ci95_low"]),
                "c_ci95_high": _maybe_float(row["c_ci95_high"]),
                "status": row["status"],
            })
    return result


def _read_csv_rows(path):
    with open(path, newline="", encoding="utf-8") as f:
        return list(csv.reader(f))


def _write_csv_rows(path, rows):
    with open(path, "w", newline="", encoding="utf-8") as f:
        csv.writer(f).writerows(rows)


@unittest.skipUnless(analyze.HAS_MPL, "matplotlib недоступен")
class FixtureCasesTest(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        with open(FIXTURES / "expected.json", encoding="utf-8") as f:
            cls.expected = json.load(f)["cases"]

    def _run_case(self, case_name, loose_ci=False):
        case = self.expected[case_name]
        csv_paths = [str(FIXTURES / name) for name in case["csv_files"]]
        out_dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)

        args = (
            ["--csv"] + csv_paths
            + ["--sizes", ",".join(map(str, case["sizes"]))]
            + ["--all", "--output-dir", str(out_dir)]
        )
        with contextlib.redirect_stderr(io.StringIO()):
            rc = analyze.main(args)
        self.assertEqual(rc, 0)

        summary = read_summary(out_dir / "summary.csv")
        fits = read_fits(out_dir / "fits.csv")

        pngs = sorted(out_dir.glob("*.png"))
        self.assertEqual(len(pngs), 8,
                         f"ожидалось 8 PNG, получено {len(pngs)}")

        self._compare_summary(case_name, summary, case["summary"])
        self._compare_fits(case_name, fits, case["fits"], loose_ci=loose_ci)

    def _compare_summary(self, case_name, actual, expected):
        actual_by_key = {
            (r["operation"], r["implementation"], r["thread_mode"], r["n"]): r
            for r in actual
        }
        expected_by_key = {
            (r["operation"], r["implementation"], r["thread_mode"], r["n"]): r
            for r in expected
        }
        self.assertEqual(set(actual_by_key), set(expected_by_key),
                         f"{case_name}: не совпадают наборы строк summary")
        for key, exp in expected_by_key.items():
            act = actual_by_key[key]
            for field in ("median_seconds", "q1_seconds", "q3_seconds", "gflops"):
                self.assertTrue(
                    math.isclose(act[field], exp[field],
                                 rel_tol=1e-9, abs_tol=1e-14),
                    f"{case_name} {key} {field}: "
                    f"{act[field]!r} != {exp[field]!r}",
                )
            for field in ("count", "peak_rss_kb_max", "n"):
                self.assertEqual(act[field], exp[field],
                                 f"{case_name} {key} {field}")

    def _compare_fits(self, case_name, actual, expected, loose_ci=False):
        actual_by_key = {
            (r["operation"], r["implementation"], r["thread_mode"]): r
            for r in actual
        }
        expected_by_key = {
            (r["operation"], r["implementation"], r["thread_mode"]): r
            for r in expected
        }
        self.assertEqual(set(actual_by_key), set(expected_by_key),
                         f"{case_name}: не совпадают наборы строк fits")
        for key, exp in expected_by_key.items():
            act = actual_by_key[key]
            self.assertEqual(act["status"], exp["status"],
                             f"{case_name} {key} status")
            self.assertEqual(act["n_points"], exp["n_points"],
                             f"{case_name} {key} n_points")
            for field in ("a", "c", "c_ci95_low", "c_ci95_high"):
                exp_val = exp[field]
                act_val = act[field]
                if exp_val is None:
                    self.assertIsNone(
                        act_val,
                        f"{case_name} {key} {field}: "
                        f"ожидалось пустое, получено {act_val!r}",
                    )
                    continue
                self.assertIsNotNone(
                    act_val,
                    f"{case_name} {key} {field}: "
                    f"ожидалось {exp_val!r}, получено пустое",
                )
                abs_tol = 1e-14
                if loose_ci and field in ("c_ci95_low", "c_ci95_high"):
                    abs_tol = 1e-6
                self.assertTrue(
                    math.isclose(act_val, exp_val,
                                 rel_tol=1e-9, abs_tol=abs_tol),
                    f"{case_name} {key} {field}: "
                    f"{act_val!r} != {exp_val!r}",
                )

    def test_power(self):
        self._run_case("power", loose_ci=True)

    def test_noisy(self):
        self._run_case("noisy", loose_ci=False)

    def test_single(self):
        self._run_case("single", loose_ci=False)

    def test_unequal(self):
        self._run_case("unequal", loose_ci=True)


class RejectionsTest(unittest.TestCase):

    def test_n_outside_block_sizes(self):
        tmp, csv_path, _ = self._make_temp_case()
        rows = _read_csv_rows(csv_path)
        rows[1][4] = "9999"
        _write_csv_rows(csv_path, rows)
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertIn("block.sizes", err)

    def test_rep_outside_range(self):
        tmp, csv_path, _ = self._make_temp_case()
        rows = _read_csv_rows(csv_path)
        rows[1][5] = "999"
        _write_csv_rows(csv_path, rows)
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertIn("вне диапазона", err)

    def test_image_not_matching_combo(self):
        tmp, csv_path, _ = self._make_temp_case()
        rows = _read_csv_rows(csv_path)
        rows[1][13] = "mkl_lu"      
        _write_csv_rows(csv_path, rows)
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertTrue(
            "run_grid.COMBOS" in err or "image_id" in err,
            f"неожиданное сообщение: {err!r}",
        )

    def test_image_not_matching_combo(self):
        tmp, csv_path, man_path = self._make_temp_case()
        rows = _read_csv_rows(csv_path)
        with open(man_path, encoding="utf-8") as f:
            m = json.load(f)
        images = m.get("images", {})
        if "mkl_lu" not in images:
            self.skipTest("в manifest нет mkl_lu для проверки COMBOS")
        rows[1][3] = "lu"                      
        rows[1][13] = "mkl_lu"                 
        rows[1][14] = images["mkl_lu"]["id"]   
        _write_csv_rows(csv_path, rows)
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertIn("run_grid.COMBOS", err)
    def _make_temp_case(self, base="power"):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        for suffix in (".csv", ".manifest.json"):
            shutil.copy(FIXTURES / f"{base}{suffix}", tmp / f"{base}{suffix}")
        return tmp, tmp / f"{base}.csv", tmp / f"{base}.manifest.json"

    def _run(self, csv_paths, sizes):
        out_dir = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, out_dir, ignore_errors=True)
        args = (
            ["--csv"] + [str(p) for p in csv_paths]
            + ["--sizes", ",".join(map(str, sizes))]
            + ["--all", "--output-dir", str(out_dir)]
        )
        with contextlib.redirect_stderr(io.StringIO()) as err:
            rc = analyze.main(args)
        return rc, err.getvalue()

    def test_missing_csv(self):
        rc, err = self._run(["/nonexistent/path.csv"], [16])
        self.assertNotEqual(rc, 0)
        self.assertIn("не найден", err)

    def test_missing_manifest(self):
        tmp, csv_path, man_path = self._make_temp_case()
        man_path.unlink()
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertIn("manifest", err)

    def test_bad_header(self):
        tmp, csv_path, _ = self._make_temp_case()
        rows = _read_csv_rows(csv_path)
        rows[0] = ["wrong"] * 23
        _write_csv_rows(csv_path, rows)
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertIn("заголовок", err)

    def test_bad_row_width(self):
        tmp, csv_path, _ = self._make_temp_case()
        rows = _read_csv_rows(csv_path)
        rows[1] = rows[1][:-1]
        _write_csv_rows(csv_path, rows)
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertIn("полей", err)

    def test_nonpositive_seconds(self):
        for value in ("0", "-1.5"):
            with self.subTest(value=value):
                tmp, csv_path, _ = self._make_temp_case()
                rows = _read_csv_rows(csv_path)
                rows[1][6] = value
                _write_csv_rows(csv_path, rows)
                rc, err = self._run([csv_path], [16, 64, 256, 1024])
                self.assertNotEqual(rc, 0)
                self.assertIn("положительное", err)

    def test_nonfinite_seconds(self):
        for value in ("nan", "inf"):
            with self.subTest(value=value):
                tmp, csv_path, _ = self._make_temp_case()
                rows = _read_csv_rows(csv_path)
                rows[1][6] = value
                _write_csv_rows(csv_path, rows)
                rc, err = self._run([csv_path], [16, 64, 256, 1024])
                self.assertNotEqual(rc, 0)
                self.assertIn("конечное", err)

    def test_duplicate_run(self):
        tmp, csv_path, _ = self._make_temp_case()
        rows = _read_csv_rows(csv_path)
        rows.append(rows[1])
        _write_csv_rows(csv_path, rows)
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertIn("дубль", err)

    def test_missing_run(self):
        tmp, csv_path, _ = self._make_temp_case()
        rows = _read_csv_rows(csv_path)
        del rows[1]
        _write_csv_rows(csv_path, rows)
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertIn("отсутствует", err)

    def test_commit_mismatch(self):
        tmp, csv_path, _ = self._make_temp_case()
        rows = _read_csv_rows(csv_path)
        rows[1][15] = "f" * 40
        _write_csv_rows(csv_path, rows)
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertIn("commit", err)

    def test_image_id_mismatch(self):
        tmp, csv_path, _ = self._make_temp_case()
        rows = _read_csv_rows(csv_path)
        rows[1][14] = "sha256:" + "f" * 64
        _write_csv_rows(csv_path, rows)
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertIn("image_id", err)

    def test_thread_mode_mismatch(self):
        tmp, csv_path, _ = self._make_temp_case()
        rows = _read_csv_rows(csv_path)
        rows[1][12] = "single"
        _write_csv_rows(csv_path, rows)
        rc, err = self._run([csv_path], [16, 64, 256, 1024])
        self.assertNotEqual(rc, 0)
        self.assertIn("thread_mode", err)

    def test_sizes_mismatch(self):
        tmp, csv_path, _ = self._make_temp_case()
        rc, err = self._run([csv_path], [16, 64])
        self.assertNotEqual(rc, 0)
        self.assertIn("размеры", err)


@unittest.skipUnless(analyze.HAS_MPL, "matplotlib недоступен")
class TwoSizesTest(unittest.TestCase):

    def test_two_sizes_summary_without_fits(self):
        tmp = pathlib.Path(tempfile.mkdtemp())
        self.addCleanup(shutil.rmtree, tmp, ignore_errors=True)
        shutil.copy(FIXTURES / "power.csv", tmp / "power.csv")
        shutil.copy(FIXTURES / "power.manifest.json",
                    tmp / "power.manifest.json")

        rows = _read_csv_rows(tmp / "power.csv")
        header = rows[0]
        data = [r for r in rows[1:] if int(r[4]) in (16, 64)]
        _write_csv_rows(tmp / "power.csv", [header] + data)

        with open(tmp / "power.manifest.json", encoding="utf-8") as f:
            m = json.load(f)
        m["block"]["sizes"] = [16, 64]
        with open(tmp / "power.manifest.json", "w", encoding="utf-8") as f:
            json.dump(m, f)

        out_dir = tmp / "out"
        with contextlib.redirect_stderr(io.StringIO()):
            rc = analyze.main([
                "--csv", str(tmp / "power.csv"),
                "--sizes", "16,64",
                "--all",
                "--output-dir", str(out_dir),
            ])
        self.assertEqual(rc, 0)

        summary = read_summary(out_dir / "summary.csv")
        self.assertEqual(len(summary), 24)

        fits = read_fits(out_dir / "fits.csv")
        self.assertEqual(len(fits), 12)
        for row in fits:
            self.assertEqual(row["status"], "insufficient_sizes")
            for field in ("a", "c", "c_ci95_low", "c_ci95_high"):
                self.assertIsNone(row[field])


if __name__ == "__main__":
    unittest.main()
