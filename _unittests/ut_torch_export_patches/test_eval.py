import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_torch
from onnx_diagnostic.torch_export_patches.eval import discover, evaluation


class TestEval(ExtTestCase):
    @requires_torch("2.7", "scan")
    def test_discover(self):
        res = discover()
        self.assertNotEmpty(res)
        for mod in res.values():
            if mod.__name__ == "ControlFlowCondIdentity_153832":
                continue
            with self.subTest(name=mod.__name__):
                m = mod()
                if isinstance(m._inputs, tuple):
                    m(*m._inputs)
                else:
                    m(*m._inputs[0])

    def test_eval(self):
        d = list(discover().items())[0]  # noqa: RUF015
        ev = evaluation(
            quiet=False,
            cases={d[0]: d[1]},
            exporters=(
                "export-strict",
                "export-nostrict",
                "custom",
                "dynamo",
                "dynamo-ir",
                "export-tracing",
            ),
        )
        self.assertIsInstance(ev, list)
        self.assertIsInstance(ev[0], dict)

    def test_run_exporter(self):
        evaluation(
            cases="SignatureListFixedLength",
            exporters="custom-strict",
            quiet=False,
            dynamic=False,
        )

    def test_run_exporter_regex(self):
        evaluation(cases=".*Aten.*", exporters="custom-strict", quiet=False, dynamic=False)


if __name__ == "__main__":
    unittest.main(verbosity=2)
