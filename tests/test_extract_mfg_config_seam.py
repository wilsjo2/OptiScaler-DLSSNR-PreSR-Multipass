"""Formatting tolerance must not weaken the production-region anchors."""
import importlib.util
from pathlib import Path
import unittest

SPEC = importlib.util.spec_from_file_location(
    "extractor", Path(__file__).resolve().parents[1] / "tools" / "extract_mfg_config_seam.py"
)
seam = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(seam)


class RegionTests(unittest.TestCase):
    def extract(self, source):
        return seam.extract_region(source, seam.FIELDS_START, seam.FIELDS_END, "fixture",
                                   seam.FIELDS_REQUIRED, seam.FIELDS_FORBIDDEN)

    def fixture(self, declaration):
        return "\n".join((declaration,
                          "CustomOptional<bool> FGDLSSGAmpereMfgUnlock { false };",
                          "CustomOptional<int> FGDLSSGAmpereMfgMaxFrames { 3 };",
                          seam.FIELDS_END))

    def test_whitespace_preserves_verbatim_region(self):
        for declaration in (seam.FIELDS_START,
                            "CustomOptional < bool > ExternalFrameGeneration\n{\n false\n};"):
            source = self.fixture(declaration)
            self.assertEqual(self.extract(source), source)

    def test_changed_default_missing_duplicate_and_forbidden_fail(self):
        valid = self.fixture(seam.FIELDS_START)
        for source in (valid.replace("ExternalFrameGeneration { false }", "ExternalFrameGeneration { true }"),
                       valid.replace("FGDLSSGAmpereMfgMaxFrames", "AnotherField"),
                       valid + "\n" + seam.FIELDS_START,
                       valid.replace(seam.FIELDS_END, "bool FGDLSSGAda;\n" + seam.FIELDS_END),
                       seam.FIELDS_END + "\n" + seam.FIELDS_START):
            with self.subTest(source=source), self.assertRaises(SystemExit):
                self.extract(source)

    def test_wrapped_save_call(self):
        start = 'ini.SetValue("FrameGen", "External", value);'
        end = 'ini.SetValue("DLSSG", "AmpereMfgKernelImage", kernel);'
        source = 'ini.SetValue(\n "FrameGen", "External", value);\n' + end
        self.assertEqual(seam.extract_region(source, start, end, "fixture", (start,)), source)


if __name__ == "__main__":
    unittest.main()
