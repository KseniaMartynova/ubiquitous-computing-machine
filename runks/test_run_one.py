
import unittest

from runks.run_one import parse_output, OutputError


VALID_CHOLESKY_OPENBLAS = (
    "RESULT_SECONDS=0.039387590\n"
    "DIAG_THREADS=openblas/libopenblas:4\n"
    "DIAG_PEAK_RSS_KB=26852\n"
    "DIAG_ROUTINES=dpotrf,dpotri\n"
    "DIAG_CHECKSUM=98243.764086\n"
)


class ParseOutputTest(unittest.TestCase):
    def test_valid_output_parses(self):
        parsed = parse_output(
            VALID_CHOLESKY_OPENBLAS,
            implementation="openblas",
            operation="cholesky",
            thread_mode="default",
        )
        self.assertEqual(parsed["seconds"], "0.039387590")
        self.assertEqual(parsed["thread_pools"], "openblas/libopenblas:4")
        self.assertEqual(parsed["routines"], "dpotrf,dpotri")
        self.assertEqual(parsed["checksums"], ["98243.764086"])

    def test_duplicate_result_line_is_rejected(self):
        stdout = (
            "RESULT_SECONDS=0.100000000\n"
            "RESULT_SECONDS=0.200000000\n"
            "DIAG_THREADS=mkl/libmkl_rt:4\n"
            "DIAG_PEAK_RSS_KB=26852\n"
            "DIAG_ROUTINES=dpotrf,dpotri\n"
            "DIAG_CHECKSUM=98243.764086\n"
        )
        with self.assertRaises(OutputError):
            parse_output(stdout, implementation="mkl", operation="cholesky", thread_mode="default")

    def test_missing_line_is_rejected(self):
        stdout = (
            "RESULT_SECONDS=0.100000000\n"
            "DIAG_THREADS=mkl/libmkl_rt:4\n"
            "DIAG_ROUTINES=dpotrf,dpotri\n"
            "DIAG_CHECKSUM=98243.764086\n"
        )
        with self.assertRaises(OutputError):
            parse_output(stdout, implementation="mkl", operation="cholesky", thread_mode="default")

    def test_extra_line_is_rejected(self):
        stdout = (
            "RESULT_SECONDS=0.100000000\n"
            "DIAG_THREADS=mkl/libmkl_rt:4\n"
            "DIAG_PEAK_RSS_KB=26852\n"
            "DIAG_ROUTINES=dpotrf,dpotri\n"
            "DIAG_CHECKSUM=98243.764086\n"
            "DIAG_THREADS_TOTAL=8\n"
        )
        with self.assertRaises(OutputError):
            parse_output(stdout, implementation="mkl", operation="cholesky", thread_mode="default")

    def test_result_nan_rejected(self):
        stdout = VALID_CHOLESKY_OPENBLAS.replace("0.039387590", "nan")
        with self.assertRaises(OutputError):
            parse_output(stdout, implementation="openblas", operation="cholesky", thread_mode="default")

    def test_result_zero_rejected(self):
        stdout = VALID_CHOLESKY_OPENBLAS.replace("0.039387590", "0.000000000")
        with self.assertRaises(OutputError):
            parse_output(stdout, implementation="openblas", operation="cholesky", thread_mode="default")

    def test_wrong_routines_rejected(self):
        stdout = (
            "RESULT_SECONDS=0.100000000\n"
            "DIAG_THREADS=mkl/libmkl_rt:4\n"
            "DIAG_PEAK_RSS_KB=26852\n"
            "DIAG_ROUTINES=dgesvd,dgemm\n"
            "DIAG_CHECKSUM=98243.764086\n"
        )
        with self.assertRaises(OutputError):
            parse_output(stdout, implementation="mkl", operation="svd", thread_mode="default")

    def test_single_mode_with_more_than_one_thread_rejected(self):
        stdout = (
            "RESULT_SECONDS=0.100000000\n"
            "DIAG_THREADS=openblas/libopenblas:4\n"
            "DIAG_PEAK_RSS_KB=26852\n"
            "DIAG_ROUTINES=dpotrf,dpotri\n"
            "DIAG_CHECKSUM=98243.764086\n"
        )
        with self.assertRaises(OutputError):
            parse_output(stdout, implementation="openblas", operation="cholesky", thread_mode="single")

    def test_valid_multiplication_parses_two_checksums(self):
        stdout = (
            "RESULT_SECONDS=0.005704845\n"
            "DIAG_THREADS=openblas/libopenblas:4\n"
            "DIAG_PEAK_RSS_KB=23656\n"
            "DIAG_ROUTINES=dgemm\n"
            "DIAG_CHECKSUM=98243.764086,98325.187950\n"
        )
        parsed = parse_output(stdout, implementation="openblas", operation="multiplication", thread_mode="default")
        self.assertEqual(parsed["checksums"], ["98243.764086", "98325.187950"])


if __name__ == "__main__":
    unittest.main()
