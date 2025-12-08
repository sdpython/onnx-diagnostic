
clear&&python -m onnx_diagnostic sbs \
    -i qwen25_vli_visual.inputs.pt \
    -e test_qwen25_vli_visual.cpu.float32.LOOPMHA.custom.graph.ep.pt2 \
    -m test_qwen25_vli_visual.cpu.float32.LOOPMHA.custom.onnx \
    -o test_qwen25_vli_visual.cpu.float32.LOOPMHA.custom.xlsx \
    -v 1 --atol 0.1 --rtol 1000
