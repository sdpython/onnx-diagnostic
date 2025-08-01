Change Logs
===========

0.7.7
+++++

0.7.6
+++++

* :pr:`193`: validates with 4.53.3 
* :pr:`189`: support for task mask-generation
* :pr:`192`: add support for Gemma-3, add serialization for HybridCache,
  changes to support ``transformers>=4.54``

0.7.5
+++++

* :pr:`186`: add parameter --output_names to command line validate to change the output names of the onnx exported model
* :pr:`185`: remove the use of _seen_tokens in DynamicCache (removed in transformers>4.53),
  updates dummpy inputs for feature-extraction
* :pr:`184`: implements side-by-side

0.7.4
+++++

* :pr:`178`: add a patch for eager_mask to handle ``assert len(flat_dynamic_shapes) == num_placeholders - num_lifted_inputs``
* :pr:`177`: changes for the next version of onnx, fixes all_dynamic_shape_from_inputs

0.7.3
+++++

* :pr:`173`: fixes function to_any for BaseModelOutput

0.7.2
+++++

* :pr:`170`: patches LlamaRotaryEmbedding
* :pr:`168`, :pr:`169`: introduces patch_diffusers
* :pr:`166`: improves handling of StaticCache
* :pr:`165`: support for task text-to-image
* :pr:`162`: improves graphs rendering for historical data

0.7.1
+++++

* :pr:`159`: supports for models with custom code in huggingface
* :pr:`158`: fix uses of pretrained version
* :pr:`156`, :pr:`157`: add plots and other options to deal with the unpredictable
* :pr:`155`: better aggregation of historical data
* :pr:`151`, :pr:`153`: adds command line ``agg``, class CubeLogsPerformance to produce timeseries
* :pr:`152`: add a function to compute fully dynamic shapes given any inputs

0.7.0
+++++

* :pr:`149`: supports for StaticCache
* :pr:`147`: simplified log processing
* :pr:`146`: patch for IdeficsAttention, IdeficsEmbedding
* :pr:`145`: patch for _compute_dynamic_ntk_parameters (Phi3RotaryEmbedding)
* :pr:`144`: support for second inputs with different dimension,
  rename test_helper into validate,
  support ``interpolate_pos_encoding`` for ``VitModel``,
  update model builder helpers for this PR
  `Use ONNX IR for model builder
  <https://github.com/microsoft/onnxruntime-genai/pull/1416>`_
* :pr:`143`: compares intermediate results,

0.6.3
+++++

* :pr:`140`: improves command line find

0.6.2
+++++

* :pr:`131`: support for custom kernels in TorchOnnxEvaluator

0.6.1
+++++

* :pr:`128`: patch for Phi3RotaryEmbedding
* :pr:`126`: add repeat and warmup to command line validate
* :pr:`125`: handles sequences in TorchOnnxEvaluator
* :pr:`123`: add subgraphs to TorchOnnxEvaluator
* :pr:`122`: add local functions to TorchOnnxEvaluator
* :pr:`120`: enables TorchOnnxEvaluator in command line ``python -m onnx_diagnostic validate ...``
* :pr:`115`, :pr:`116`, :pr:`117`, :pr:`118`, :pr:`119`, :pr:`127`:
  first steps for TorchOnnxEvaluator
* :pr:`114`: extends the list of known rewritings
* :pr:`113`: fixes a couple of issues with ModelBuilder

0.6.0
+++++

* :pr:`111`: support ModelBuilder with command line validate
* :pr:`108`, :pr:`109`, :pr:`110`: first version of an algorithm rendering
  small onnx graph in ascii, patch for ``torch.vmap``

0.5.0
+++++

* :pr:`105`: more options to tune control flow rewriting
* :pr:`104`: add summarization task, add rewrite to command line validate
* :pr:`101`: first draft to rewrite loops
* :pr:`100`: implements a context to automatically rewrite methods or function with control flows
* :pr:`96`: implements ``is_stealing``, ``steal_append`` to complement ``steal_forward``
* :pr:`95`: fixzq Scan implementation for ``OnnxruntimeEvaluator``
* :pr:`93`: introduces patched expressions to get around annoying export issues
* :pr:`92`: supports errors distribution in max_diff
* :pr:`91`: enables strings in ``guess_dynamic_shapes``
* :pr:`88`, :pr:`89`: extends ``steal_forward`` to dump input, outputs in onnx models
* :pr:`83`, :pr:`85`: improves the automated rewriting of control flow (test)

0.4.4
+++++

* :pr:`82`: exposes ``register_flattening_functions``, add option ``--subfolder``
* :pr:`81`: fixes missing ``intermediate_size`` in configuration
* :pr:`79`: implements task ``object-detection``
* :pr:`78`: uses *onnx-weekly* instead of *onnx* to avoid conflicts with *onnxscript*

0.4.3
+++++

* :pr:`75`: renames bypass_export_some_patches into torch_export_patches, keeps the old name
* :pr:`74`: increases the list of class/architectures

0.4.2
+++++

* :pr:`73`: supports MambaCache in max_diff, torch_deepcopy

0.4.1
+++++

* :pr:`72`: fix change_dynamic_dimension for custom classes
* :pr:`70`: support models options in command lines

0.4.0
+++++

* :pr:`65`: support SlidingWindowCache
* :pr:`63`: support option ``--trained``
* :pr:`61`: improves dynamic shapes for EncoderDecoderCache
* :pr:`58`: add function use_dyn_not_str to replace string by ``torch.export.Dim.DYNAMIC``,
  use string instead of ``torch.export.Dim.DYNAMIC`` when returning the dynamic shapes
  for a specific models, it is a valid definition for ``torch.onnx.export``
  which can reuse the names
* :pr:`55`: add support for text-classification
* :pr:`54`: add support for fill-mask, refactoring
* :pr:`52`: add support for zero-shot-image-classification
* :pr:`50`: add support for onnxruntime fusion
* :pr:`48`: add support for EncoderDecoderCache, test with openai/whisper-tiny
* :pr:`45`: improve change_dynamic_dimension to fix some dimensions

0.3.0
+++++

* :pr:`43`: uses custom patches
* :pr:`38`: uses the registered serialization functions when it is available
* :pr:`30`, :pr:`31`: adds command to test a model id, validate the export
* :pr:`29`: adds helpers to measure the memory peak and run benchmark
  on different processes
* :pr:`28`: adds command line to print out the configuration for a model id,
  support image-text-to-text
* :pr:`26`: creates a folder ``helpers`` to gather all the functions
  used in many places
* :pr:`25`: improve patches for DynamicCache
  (issue with register_pytree_flatten_spec being deprecated)
* :pr:`24`: dummy inputs for ``text2text-generation``, add new function
  ``convert_dynamic_axes_into_dynamic_shapes`` to convert dynamic axes
  into dynamic shapes, add support for ``T5ForConditionalGeneration``
* :pr:`23`: dummy inputs for ``image-classification``
* :pr:`22`, :pr:`27`: api to create untrained model copying the architecture
  of the trained models and dummy inputs for them,
  support for ``text-generation``

0.2.1
+++++

* :pr:`16`: refactors patches, add model Phi2, implements
  a tweak to raise an exception with a dynamic dimension
  becomes static when exporting a model

0.2.0
+++++

* :pr:`11`: adds ``ModelInputs`` to guess dynamic shapes
* :pr:`9`: adds ``OnnxruntimeEvaluator``
* :pr:`8`: adds ``ExtendedReferenceEvaluator``
* :pr:`7`: improves function ``investigate_onnxruntime_issue``

0.1.0
+++++

first version
