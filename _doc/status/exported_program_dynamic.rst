=====================================
Exported Programs with Dynamic Shapes
=====================================

The following script shows the exported program for many short cases
and various l-plot-export-with-dynamic-shape to retrieve an ONNX model equivalent
to the original model.

.. runpython::
    :showcode:
    :rst:
    :toggle: code
    :warningout: UserWarning

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
    for name, cls_model in sorted_cases:
        print(f"* :ref:`{name} <led-model-case-export-{name}>`")
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
        print("::")
        print()
        print(textwrap.indent(textwrap.dedent(inspect.getsource(cls_model.forward)), "    "))
        print()
        for exporter in (
            "export-strict",
            "export-nostrict",
            "export-nostrict-decall",
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
            if "exported" in res:
                print("::")
                print()
                print(textwrap.indent(str(res["exported"].graph), "    "))
                print()
                obs.append(dict(case=case_ref, error="", exporter=expo))
            else:
                print("**FAILED**")
                print()
                print("::")
                print()
                print(textwrap.indent(str(res["error"]), "    "))
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
