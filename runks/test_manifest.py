#!/usr/bin/env python3

import pathlib
import sys
import tempfile
import unittest


RUNKS = pathlib.Path(__file__).resolve().parent
REPO_ROOT = RUNKS.parent
if str(RUNKS) not in sys.path:
    sys.path.insert(0, str(RUNKS))

import manifest
import build_all
from run_grid import COMBOS


LSCPU_SAMPLE = """\
Architecture:                            x86_64
CPU op-mode(s):                          32-bit, 64-bit
Address sizes:                           46 bits physical, 57 bits virtual
Byte Order:                              Little Endian
CPU(s):                                  16
On-line CPU(s) list:                     0-15
Vendor ID:                               GenuineIntel
Model name:                              Intel(R) Xeon(R) Gold 6334 CPU @ 3.60GHz
CPU family:                              6
Model:                                   106
Thread(s) per core:                      2
Core(s) per socket:                      8
Socket(s):                               1
Stepping:                                6
CPU max MHz:                             3700,0000
CPU min MHz:                             800,0000
BogoMIPS:                                7200.00
Flags:                                   fpu vme de pse tsc msr pae mce cx8 apic sep mtrr pge mca cmov pat pse36 clflush dts acpi mmx fxsr sse sse2 ss ht tm pbe syscall nx pdpe1gb rdtscp lm constant_tsc art arch_perfmon pebs bts rep_good nopl xtopology nonstop_tsc cpuid aperfmperf pni pclmulqdq dtes64 monitor ds_cpl vmx smx est tm2 ssse3 sdbg fma cx16 xtpr pdcm pcid dca sse4_1 sse4_2 x2apic movbe popcnt tsc_deadline_timer aes xsave avx f16c rdrand lahf_lm abm 3dnowprefetch cpuid_fault epb cat_l3 intel_ppin ssbd mba ibrs ibpb stibp ibrs_enhanced tpr_shadow flexpriority ept vpid ept_ad fsgsbase tsc_adjust bmi1 avx2 smep bmi2 erms invpcid cqm rdt_a avx512f avx512dq rdseed adx smap avx512ifma clflushopt clwb intel_pt avx512cd sha_ni avx512bw avx512vl xsaveopt xsavec xgetbv1 xsaves cqm_llc cqm_occup_llc cqm_mbm_total cqm_mbm_local split_lock_detect wbnoinvd dtherm ida arat pln pts hwp hwp_act_window hwp_epp hwp_pkg_req vnmi avx512vbmi umip pku ospke avx512_vbmi2 gfni vaes vpclmulqdq avx512_vnni avx512_bitalg tme avx512_vpopcntdq la57 rdpid fsrm md_clear pconfig flush_l1d arch_capabilities
Virtualization:                          VT-x
L1d cache:                               384 KiB (8 instances)
L1i cache:                               256 KiB (8 instances)
L2 cache:                                10 MiB (8 instances)
L3 cache:                                18 MiB (1 instance)
NUMA node(s):                            2
NUMA node0 CPU(s):                       0-3,8-11
NUMA node1 CPU(s):                       4-7,12-15
Vulnerability Gather data sampling:      Mitigation; Microcode
Vulnerability Indirect target selection: Vulnerable
Vulnerability Itlb multihit:             Not affected
Vulnerability L1tf:                      Not affected
Vulnerability Mds:                       Not affected
Vulnerability Meltdown:                  Not affected
Vulnerability Mmio stale data:           Mitigation; Clear CPU buffers; SMT vulnerable
Vulnerability Reg file data sampling:    Not affected
Vulnerability Retbleed:                  Not affected
Vulnerability Spec rstack overflow:      Not affected
Vulnerability Spec store bypass:         Mitigation; Speculative Store Bypass disabled via prctl
Vulnerability Spectre v1:                Mitigation; usercopy/swapgs barriers and __user pointer sanitization
Vulnerability Spectre v2:                Mitigation; Enhanced / Automatic IBRS; IBPB conditional; PBRSB-eIBRS SW sequence; BHI SW loop, KVM SW loop
Vulnerability Srbds:                     Not affected
Vulnerability Tsa:                       Not affected
Vulnerability Tsx async abort:           Not affected
Vulnerability Vmscape:                   Not affected
"""

MEMINFO_SAMPLE = """\
MemTotal:       263732508 kB
MemFree:        204230684 kB
MemAvailable:   255062412 kB
Buffers:         1116828 kB
Cached:         32400036 kB
SwapCached:            0 kB
Active:          5152160 kB
Inactive:       31763128 kB
Active(anon):    3453588 kB
Inactive(anon):        0 kB
Active(file):    1698572 kB
Inactive(file): 31763128 kB
Unevictable:       33308 kB
Mlocked:           24292 kB
SwapTotal:       2097148 kB
SwapFree:        2097148 kB
Zswap:                 0 kB
Zswapped:              0 kB
Dirty:                84 kB
Writeback:             0 kB
AnonPages:       3431748 kB
Mapped:           580876 kB
Shmem:             51104 kB
KReclaimable:   19798340 kB
Slab:           21521436 kB
SReclaimable:   19798340 kB
SUnreclaim:      1723096 kB
KernelStack:       13296 kB
PageTables:        38168 kB
SecPageTables:         0 kB
NFS_Unstable:          0 kB
Bounce:                0 kB
WritebackTmp:          0 kB
CommitLimit:    133963400 kB
Committed_AS:   26123644 kB
VmallocTotal:   13743895347199 kB
VmallocUsed:      416080 kB
VmallocChunk:          0 kB
Percpu:            50432 kB
HardwareCorrupted:     0 kB
AnonHugePages:         0 kB
ShmemHugePages:        0 kB
ShmemPmdMapped:        0 kB
FileHugePages:         0 kB
FilePmdMapped:         0 kB
Unaccepted:            0 kB
HugePages_Total:       0
HugePages_Free:        0
HugePages_Rsvd:        0
HugePages_Surp:        0
Hugepagesize:       2048 kB
Hugetlb:               0 kB
DirectMap4k:      419428 kB
DirectMap2M:     8626176 kB
DirectMap1G:    261095424 kB
"""

DPKG_SAMPLE = """\
liblapack-dev 3.11.0-2
liblapacke-dev 3.11.0-2
libopenblas-dev 0.3.21+ds-4
"""


CPP_IMAGES = [c[0] for c in COMBOS if c[1] in ("mkl", "openblas")]


class FromLineTest(unittest.TestCase):

    def test_pinned_images_have_digest(self):
        build_dir = REPO_ROOT / "runks" / "build"
        for image, (rel_dir, dockerfile) in build_all.BUILD_FILES.items():
            text = (build_dir / rel_dir / dockerfile).read_text(encoding="utf-8")
            value = manifest.from_line(text)
            self.assertIn("@sha256:", value,
                          f"{image}: FROM без digest: {value!r}")

    def test_floating_tag_rejected(self):
        text = "FROM gcc:12.4\n"
        with self.assertRaises(manifest.ManifestError):
            manifest.from_line(text)


class CompileLineTest(unittest.TestCase):

    def test_one_compile_line_per_cpp_dockerfile(self):
        build_dir = REPO_ROOT / "runks" / "build"
        for image in CPP_IMAGES:
            rel_dir, dockerfile = build_all.BUILD_FILES[image]
            text = (build_dir / rel_dir / dockerfile).read_text(encoding="utf-8")
            line = manifest.compile_line(text)
            self.assertTrue(line.startswith("g++") or line.startswith("icpx"),
                            f"{image}: строка компиляции: {line!r}")

    def test_no_compile_line_rejected(self):
        text = (
            "FROM gcc:12.4@sha256:abc\n"
            "COPY prog.cpp /prog.cpp\n"
            "ENTRYPOINT [\"./prog\"]\n"
        )
        with self.assertRaises(manifest.ManifestError):
            manifest.compile_line(text)

    def test_two_compile_lines_rejected(self):
        text = (
            "FROM gcc:12.4@sha256:abc\n"
            "RUN g++ -o a a.cpp\n"
            "RUN g++ -o b b.cpp\n"
            "ENTRYPOINT [\"./a\"]\n"
        )
        with self.assertRaises(manifest.ManifestError):
            manifest.compile_line(text)


class EntrypointBinaryTest(unittest.TestCase):

    def test_cpp_entrypoints_start_with_dot_slash(self):
        build_dir = REPO_ROOT / "runks" / "build"
        for image in CPP_IMAGES:
            rel_dir, dockerfile = build_all.BUILD_FILES[image]
            text = (build_dir / rel_dir / dockerfile).read_text(encoding="utf-8")
            binary = manifest.entrypoint_binary(text)
            self.assertTrue(binary.startswith("./"),
                            f"{image}: entrypoint={binary!r}")


class ParseLscpuTest(unittest.TestCase):

    def test_correct_types(self):
        info = manifest.parse_lscpu(LSCPU_SAMPLE)
        self.assertEqual(info["cpu_model"],
                         "Intel(R) Xeon(R) Gold 6334 CPU @ 3.60GHz")
        self.assertEqual(info["sockets"], 1)
        self.assertEqual(info["cores_per_socket"], 8)
        self.assertEqual(info["threads_per_core"], 2)
        self.assertEqual(info["logical_cpus"], 16)

    def test_missing_model_name_rejected(self):
        text = "Architecture: x86_64\nSocket(s): 2\n"
        with self.assertRaises(manifest.ManifestError):
            manifest.parse_lscpu(text)


class ParseMeminfoTest(unittest.TestCase):

    def test_both_values_returned(self):
        info = manifest.parse_meminfo(MEMINFO_SAMPLE)
        self.assertEqual(info["mem_total_kb"], 263732508)
        self.assertEqual(info["mem_available_kb"], 255062412)

    def test_missing_memavailable_rejected(self):
        text = "MemTotal: 1024 kB\nMemFree: 512 kB\n"
        with self.assertRaises(manifest.ManifestError):
            manifest.parse_meminfo(text)


class ParseDpkgTest(unittest.TestCase):

    def test_three_versions(self):
        packages = ["libopenblas-dev", "liblapack-dev", "liblapacke-dev"]
        info = manifest.parse_dpkg(DPKG_SAMPLE, packages)
        self.assertEqual(info["libopenblas-dev"], "0.3.21+ds-4")
        self.assertEqual(info["liblapack-dev"], "3.11.0-2")
        self.assertEqual(info["liblapacke-dev"], "3.11.0-2")

    def test_missing_package_rejected(self):
        packages = ["libopenblas-dev", "liblapack-dev", "liblapacke-dev"]
        text = "libopenblas-dev 0.3.21+ds-4\nliblapack-dev 3.11.0-2\n"
        with self.assertRaises(manifest.ManifestError):
            manifest.parse_dpkg(text, packages)


class ManifestPathTest(unittest.TestCase):

    def test_csv_becomes_manifest_json(self):
        p = manifest.manifest_path("results/x.csv")
        self.assertEqual(str(p), "results/x.manifest.json")

    def test_non_csv_rejected(self):
        with self.assertRaises(manifest.ManifestError):
            manifest.manifest_path("results/x.txt")


class CheckManifestAbsentTest(unittest.TestCase):

    def test_existing_manifest_rejected_and_untouched(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "pilot.manifest.json"
            path.write_text('{"original": true}\n')
            before = path.read_bytes()
            with self.assertRaises(manifest.ManifestError):
                manifest.check_manifest_absent(path)
            after = path.read_bytes()
            self.assertEqual(before, after)

    def test_absent_manifest_ok(self):
        with tempfile.TemporaryDirectory() as tmp:
            path = pathlib.Path(tmp) / "pilot.manifest.json"
            manifest.check_manifest_absent(path)


if __name__ == "__main__":
    unittest.main()
