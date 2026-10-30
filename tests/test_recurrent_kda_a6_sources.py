"""Static contract checks for the RecurrentKda A6 source routing."""

from __future__ import annotations

import re
import unittest
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[1]
OP_ROOT = REPO_ROOT / "fla" / "ops" / "ascendc" / "kda" / "recurrent_kda"


class RecurrentKdaA6SourceTest(unittest.TestCase):
    def test_a6_maps_to_arch92(self) -> None:
        cmake = (REPO_ROOT / "CMakeLists.txt").read_text(encoding="utf-8")
        socs = re.search(r"set\(SOC_VERSION_LIST (.*?)\)", cmake).group(1).split()
        archs = re.search(r"set\(ARCH_DIRECTORY_LIST (.*?)\)", cmake).group(1).split()
        self.assertEqual(dict(zip(socs, archs))["ascend960dt"], "arch92")

    def test_kernel_selects_arch92_for_dav_9201(self) -> None:
        source = (OP_ROOT / "op_kernel" / "recurrent_kda.cpp").read_text(encoding="utf-8")
        self.assertIn("__NPU_ARCH__ == 9201", source)
        self.assertIn('#include "arch92/recurrent_kda.h"', source)
        self.assertIn("KERNEL_TYPE_AIV_ONLY", source)
        self.assertTrue((OP_ROOT / "op_kernel" / "arch92" / "recurrent_kda.h").is_file())

    def test_operator_registers_a6_and_uses_regbase_tiling(self) -> None:
        op_def = (OP_ROOT / "op_host" / "recurrent_kda_def.cpp").read_text(encoding="utf-8")
        tiling = (OP_ROOT / "op_host" / "recurrent_kda_tiling.cpp").read_text(encoding="utf-8")
        cmake = (OP_ROOT / "op_host" / "CMakeLists.txt").read_text(encoding="utf-8")
        self.assertIn('AddConfig("ascend960dt"', op_def)
        self.assertIn("static_cast<int32_t>(npuArch) == DAV_9201", tiling)
        self.assertIn('STREQUAL "ascend960dt"', cmake)
        self.assertIn("--cce-no-dcache-flush", cmake)

    def test_a6_k128_hot_paths_expose_three_row_ilp(self) -> None:
        source = (
            OP_ROOT / "op_kernel" / "recurrent_kda_regbase.h"
        ).read_text(encoding="utf-8")
        self.assertIn("A6_TRIPLE_ISSUE_ROWS = 3", source)
        self.assertIn("DecayMatVecReduce128A6", source)
        self.assertIn("ProcessDeltaKQReduce128A6", source)
        self.assertRegex(
            source,
            r"#if defined\(__NPU_ARCH__\) && __NPU_ARCH__ == 9201"
            r"\s+DecayMatVecReduce128A6",
        )
        self.assertRegex(
            source,
            r"#if defined\(__NPU_ARCH__\) && __NPU_ARCH__ == 9201"
            r"\s+ProcessDeltaKQReduce128A6",
        )

    def test_atk_harness_accepts_a6_soc(self) -> None:
        script = (REPO_ROOT / "tests" / "atk" / "run_test_cpu.sh").read_text(
            encoding="utf-8"
        )
        self.assertIn("a6|A6|ascend960dt) SOC=\"ascend960dt\"", script)
        self.assertIn("*960DT*|*960dt*|*Ascend960*) echo \"ascend960dt\"", script)


if __name__ == "__main__":
    unittest.main()
