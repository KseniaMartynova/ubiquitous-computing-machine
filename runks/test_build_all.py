#!/usr/bin/env python3

import pathlib
import sys
import unittest


RUNKS = pathlib.Path(__file__).resolve().parent
REPO_ROOT = RUNKS.parent
if str(RUNKS) not in sys.path:
    sys.path.insert(0, str(RUNKS))

import build_all
from run_grid import COMBOS


class BuildFilesTest(unittest.TestCase):

    def test_build_files_keys_match_combos(self):
        combos_images = {c[0] for c in COMBOS}
        build_images = set(build_all.BUILD_FILES.keys())
        self.assertEqual(combos_images, build_images)

    def test_all_dockerfiles_exist(self):
        build_dir = REPO_ROOT / "runks" / "build"
        for image, (rel_dir, dockerfile) in build_all.BUILD_FILES.items():
            path = build_dir / rel_dir / dockerfile
            self.assertTrue(path.exists(),
                            f"нет Dockerfile для {image}: {path}")


class OnlyArgumentTest(unittest.TestCase):

    def test_unknown_only_rejected_without_docker(self):
        with self.assertRaises(SystemExit):
            build_all.validate_only_name("no_such_image",
                                         build_all.BUILD_FILES)


if __name__ == "__main__":
    unittest.main()
