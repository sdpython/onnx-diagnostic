import unittest
from onnx_diagnostic.ext_test_case import ExtTestCase, requires_torch, long_test
from onnx_diagnostic.torch_export_patches.eval import discover, evaluation


class TestEval(ExtTestCase):
    @requires_torch("2.7", "scan")
    def test_discover(self):
        res = discover()
        self.assertNotEmpty(res)
        for mod in res.values():
            with self.subTest(name=mod.__name__):
                if mod.__name__ == "ControlFlowCondIdentity_153832":
                    raise unittest.SkipTest(
                        "ControlFlowCondIdentity_153832 needs missing clone."
                    )
                m = mod()
                if isinstance(m._inputs, tuple):
                    m(*m._inputs)
                else:
                    for v in m._inputs:
                        m(*v)
                if hasattr(m, "_valid"):
                    for v in m._valid:
                        m(*v)

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

    def test_run_exporter_custom(self):
        evaluation(
            cases="SignatureListFixedLength",
            exporters="custom",
            quiet=False,
            dynamic=False,
        )

    def test_run_exporter_dynamo(self):
        evaluation(
            cases="SignatureListFixedLength",
            exporters="dynamo",
            quiet=False,
            dynamic=False,
        )

    def test_run_exporter_dynamo_ir(self):
        evaluation(
            cases="SignatureListFixedLength",
            exporters="dynamo-ir",
            quiet=False,
            dynamic=False,
        )

    def test_run_exporter_nostrict(self):
        evaluation(
            cases="SignatureListFixedLength",
            exporters="export-nostrict",
            quiet=False,
            dynamic=False,
        )

    def test_run_exporter_tracing(self):
        evaluation(
            cases="SignatureListFixedLength",
            exporters="export-tracing",
            quiet=False,
            dynamic=False,
        )

    def test_run_exporter_regex(self):
        evaluation(cases=".*Aten.*", exporters="custom-strict", quiet=False, dynamic=False)

    def test_run_exporter_custom_nested_cond(self):
        evaluation(
            cases="ControlFlowNestCond",
            exporters="custom",
            quiet=False,
            dynamic=False,
        )

    def test_run_exporter_dimension0(self):
        evaluation(
            cases="ExportWithDimension0",
            exporters="export-nostrict-oblivious",
            quiet=False,
            dynamic=True,
        )

    def test_run_exporter_dimension1(self):
        evaluation(
            cases="ExportWithDimension1",
            exporters="export-nostrict-oblivious",
            quiet=False,
            dynamic=True,
        )

    @long_test()
    def test_documentation(self):
        import inspect
        import textwrap
        import pandas
        from onnx_diagnostic.helpers import string_type
        from onnx_diagnostic.torch_export_patches.eval import discover, run_exporter
        from onnx_diagnostic.ext_test_case import unit_test_going

        cases = discover()
        print()
        print(":ref:`Summary <led-summary-exported-program>`")
        print()
        sorted_cases = sorted(cases.items())
        if unit_test_going():
            sorted_cases = sorted_cases[:3]
        for name, _cls_model in sorted_cases:
            print(f"* :ref:`{name} <led-model-case-export-{name}>`")
        print()
        print()

        obs = []
        for name, cls_model in sorted(cases.items()):
            print()
            print(f".. _led-model-case-export-{name}:")
            print()
            print(name)
            print("=" * len(name))
            print()
            print("forward")
            print("+++++++")
            print()
            print(".. code-block:: python")
            print()
            src = inspect.getsource(cls_model.forward)
            if src:
                print(textwrap.indent(textwrap.dedent(src), "    "))
            else:
                print("    # code is missing")
            print()
            print()
            for exporter in (
                "export-strict",
                "export-nostrict",
                "export-nostrict-oblivious",
                "export-nostrict-decall",
                "export-tracing",
            ):
                expname = exporter.replace("export-", "")
                print()
                print(expname)
                print("+" * len(expname))
                print()
                res = run_exporter(exporter, cls_model, True, quiet=True)
                case_ref = f":ref:`{name} <led-model-case-export-{name}>`"
                expo = exporter.split("-", maxsplit=1)[-1]
                if "inputs" in res:
                    print(f"* **inputs:** ``{string_type(res['inputs'], with_shape=True)}``")
                if "dynamic_shapes" in res:
                    print(f"* **shapes:** ``{string_type(res['dynamic_shapes'])}``")
                print()
                print()
                if "exported" in res:
                    print(".. code-block:: text")
                    print()
                    print(textwrap.indent(str(res["exported"].graph), "    "))
                    print()
                    print()
                    obs.append(dict(case=case_ref, error="", exporter=expo))
                else:
                    print("**FAILED**")
                    print()
                    print(".. code-block:: text")
                    print()
                    err = str(res["error"])
                    if err:
                        print(textwrap.indent(err, "    "))
                    else:
                        print("    # no error found for the failure")
                    print()
                    print()
                    obs.append(dict(case=case_ref, error="FAIL", exporter=expo))

        print()
        print(".. _led-summary-exported-program:")
        print()
        print("Summary")
        print("+++++++")
        print()
        df = pandas.DataFrame(obs)
        piv = df.pivot(index="case", columns="exporter", values="error")
        print(piv.to_markdown(tablefmt="rst"))
        print()


if __name__ == "__main__":
    unittest.main(verbosity=2)
