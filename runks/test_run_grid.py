#!/usr/bin/env python3

import unittest

from runks.run_grid import (
    HEADER,
    COMBOS,
    GridError,
    build_schedule,
    check_block,
    check_row,
    parse_sizes,
)


COMMIT = "a" * 40
SEED = 20260818


def make_row(run_order=1, implementation="mkl", operation="cholesky",
             n=256, rep=1, thread_mode="default", image="mkl_chol",
             commit=COMMIT, shuffle_seed=SEED,
             checksum_a="1.0", checksum_b=""):
    """Собирает валидную строку из 23 полей."""
    row = [""] * 23
    row[0] = str(run_order)
    row[1] = "2026-01-01T00:00:00Z"
    row[2] = implementation
    row[3] = operation
    row[4] = str(n)
    row[5] = str(rep)
    row[6] = "0.1"
    row[7] = "mkl/libmkl_rt:4"
    row[8] = "1000"
    row[9] = "0"
    row[10] = "0"
    row[11] = "false"
    row[12] = thread_mode
    row[13] = image
    row[14] = "sha256:abc"
    row[15] = commit
    row[16] = str(n)
    row[17] = ""
    row[18] = str(shuffle_seed)
    row[19] = "dpotrf"
    row[20] = "dpotri"
    row[21] = checksum_a
    row[22] = checksum_b
    return row


def make_expected_row(**overrides):
    d = {
        "run_order": 1,
        "n": 256,
        "rep": 1,
        "image": "mkl_chol",
        "implementation": "mkl",
        "operation": "cholesky",
        "thread_mode": "default",
        "shuffle_seed": SEED,
        "commit": COMMIT,
    }
    d.update(overrides)
    return d


def make_valid_block(num_runs=1, commit=COMMIT, thread_mode="default",
                     shuffle_seed=SEED):
    rows = [list(HEADER)]
    for i in range(1, num_runs + 1):
        rows.append(make_row(
            run_order=i,
            commit=commit,
            thread_mode=thread_mode,
            shuffle_seed=shuffle_seed,
        ))
    return rows


def make_block_expected(num_runs=1, commit=COMMIT, thread_mode="default",
                        shuffle_seed=SEED):
    return {
        "num_runs": num_runs,
        "commit": commit,
        "thread_mode": thread_mode,
        "shuffle_seed": shuffle_seed,
    }


# ---------- 1–5: расписание ----------

class BuildScheduleTest(unittest.TestCase):

    def test_same_arguments_give_same_schedule(self):
        a = build_schedule([256, 512], 2, SEED)
        b = build_schedule([256, 512], 2, SEED)
        self.assertEqual(a, b)

    def test_each_block_contains_all_combos_exactly_once(self):
        sched = build_schedule([256, 512], 2, SEED)
        blocks = {}
        for e in sched:
            blocks.setdefault((e.n, e.rep), []).append(e.image)
        expected_images = {c[0] for c in COMBOS}
        for (n, rep), images in blocks.items():
            self.assertEqual(len(images), 12,
                             f"block n={n} rep={rep}: {len(images)} images")
            self.assertEqual(set(images), expected_images,
                             f"block n={n} rep={rep}: wrong image set")
            self.assertEqual(len(set(images)), 12,
                             f"block n={n} rep={rep}: duplicates found")

    def test_order_differs_between_repetitions(self):
        sched = build_schedule([256], 2, SEED)
        rep1 = [e.image for e in sched if e.n == 256 and e.rep == 1]
        rep2 = [e.image for e in sched if e.n == 256 and e.rep == 2]
        self.assertNotEqual(rep1, rep2)

    def test_order_differs_with_different_seed(self):
        a = build_schedule([256], 1, 111)
        b = build_schedule([256], 1, 222)
        self.assertNotEqual([e.image for e in a], [e.image for e in b])

    def test_run_order_is_consecutive(self):
        sched = build_schedule([128, 256, 512], 2, SEED)
        self.assertEqual(
            [e.run_order for e in sched],
            list(range(1, len(sched) + 1)),
        )


# ---------- 6–7: разбор --sizes ----------

class ParseSizesTest(unittest.TestCase):

    def test_rejects_duplicates(self):
        with self.assertRaises(ValueError):
            parse_sizes("256,256")
        with self.assertRaises(ValueError):
            parse_sizes("512,256,512")

    def test_rejects_zero_negative_and_garbage(self):
        for bad in ["0", "-1", "abc", "256,,512", "", "   ", "256,", ",256"]:
            with self.assertRaises(ValueError, msg=f"bad={bad!r}"):
                parse_sizes(bad)


# ---------- 8–10: проверка одной строки ----------

class CheckRowTest(unittest.TestCase):

    def test_rejects_wrong_field_count(self):
        row = make_row()
        row.append("extra")            # 24 поля
        with self.assertRaises(GridError):
            check_row(row, make_expected_row())

        row = make_row()[:-1]          # 22 поля
        with self.assertRaises(GridError):
            check_row(row, make_expected_row())

    def test_rejects_wrong_metadata(self):
        # run_one.py вернул n=512, а просили 256
        row = make_row(n=512)
        expected = make_expected_row(n=256)
        with self.assertRaises(GridError):
            check_row(row, expected)

    def test_rejects_wrong_commit(self):
        row = make_row(commit="b" * 40)
        expected = make_expected_row(commit="a" * 40)
        with self.assertRaises(GridError):
            check_row(row, expected)


# ---------- 11–18: проверка блока ----------

class CheckBlockTest(unittest.TestCase):

    def test_rejects_wrong_header(self):
        rows = make_valid_block(2)
        rows[0] = ["wrong"] * 23
        with self.assertRaises(GridError):
            check_block(rows, make_block_expected(2))

    def test_rejects_wrong_row_count(self):
        rows = make_valid_block(2)     # 1 заголовок + 2 строки
        with self.assertRaises(GridError):
            check_block(rows, make_block_expected(3))

    def test_rejects_wrong_run_order(self):
        rows = make_valid_block(2)
        rows[2][0] = "5"               # вместо "2"
        with self.assertRaises(GridError):
            check_block(rows, make_block_expected(2))

    def test_rejects_inconsistent_commit(self):
        rows = make_valid_block(2, commit=COMMIT)
        rows[2][15] = "b" * 40
        with self.assertRaises(GridError):
            check_block(rows, make_block_expected(2))

    def test_rejects_wrong_thread_mode(self):
        rows = make_valid_block(2, thread_mode="default")
        rows[2][12] = "single"
        with self.assertRaises(GridError):
            check_block(rows, make_block_expected(2))

    def test_rejects_wrong_shuffle_seed(self):
        rows = make_valid_block(2, shuffle_seed=SEED)
        rows[2][18] = "99999"
        with self.assertRaises(GridError):
            check_block(rows, make_block_expected(2))

    def test_rejects_diverging_checksums_between_repeats(self):
        rows = make_valid_block(2)
        rows[1][21] = "1.0"
        rows[1][22] = ""
        rows[2][21] = "2.0"            # та же (impl, op, n), другая сумма
        rows[2][22] = ""
        with self.assertRaises(GridError):
            check_block(rows, make_block_expected(2))

    def test_rejects_mkl_openblas_divergence(self):
        rows = [list(HEADER)]
        rows.append(make_row(run_order=1, implementation="mkl",
                             image="mkl_chol", checksum_a="1.0"))
        rows.append(make_row(run_order=2, implementation="openblas",
                             image="lapack_chol", checksum_a="2.0"))
        with self.assertRaises(GridError):
            check_block(rows, make_block_expected(2))


class HappyPathTest(unittest.TestCase):

    def test_valid_block_passes(self):
        rows = make_valid_block(3)
        check_block(rows, make_block_expected(3))


if __name__ == "__main__":
    unittest.main()
