Change Logs
===========

0.9.1
+++++

0.9.0
+++++

* :pr:`403`: update the serialization of SlidingWindowCache to include parameter slidinw_window, patch for sdpa_mask
* :pr:`400`, :pr:`401`:, :pr:`402`: improves InputObserver (investigations), add it the documentation
* :pr:`399`: update CI

0.8.11
++++++

* :pr:`396`: fix serialization for DynamicCache with different layer classes
* :pr:`394`: add function make_model_with_local_functions to partition a model into local functions

0.8.10
++++++

* :pr:`384`: add ``weights_only=False`` when using ``torch.load``

0.8.9
+++++

* :pr:`383`: removed bool, int, float, None as input dummies for the exporter in ``method_to_onnx``
* :pr:`382`: make the ordering of the inferred dynamic shapes more robust
* :pr:`381`: add parameter *expand_batch_for* to ``method_to_onnx``
* :pr:`378`: implements the computation of discrepancies in ``method_to_onnx``
* :pr:`379`: update the handling of cache after the removal of HybridCache, SlidingWindowCache in ``transformers>=5``,

0.8.8
+++++

* :pr:`375`: export a method to onnx in order to export using method generate
* :pr:`376`: fix patched lazy_initialization for ``transformers>=5``
* :pr:`372`: fix patch on rotary embedding
* :pr:`371`: fix make_fake_with_dynamic_dimensions

0.8.7
+++++

* :pr:`366`: add command line to optimize a model
* :pr:`363`: patch for DynamicDimConstraintPrinter
* :pr:`360`, :pr:`364`: preliminary work for phi4

0.8.6
+++++

* :pr:`357`: complete simple_loop_for, an easier to rewrite loops
* :pr:`356`: include qwen embedding part
* :pr:`355`: better command line to export models
* :pr:`353`, :pr:`354`: add command line to compare two onnx models

0.8.5
+++++

* :pr:`349`: fixes function max_diff (parameter hist)
* :pr:`348`: add format dot, shape to command line print
* :pr:`346`: fix patch for sdpa_mask_recent_torch even if it was removed in transformers>=5.0

0.8.4
+++++

* :pr:`341`: preliminary support to export submodule
* :pr:`340`: supports devices in onnx plugs
* :pr:`338`: fixes ReplayConfiguration.dump, add function to select of part of a model
* :pr:`337`: fixes extract_subset_of_nodes
* :pr:`336`: implements versioned onnx plugs

0.8.3
+++++

* :pr:`331`: adds a helper to convert an onnx model into dot
* :pr:`330`: fixes access rope_parameters for ``transformers>=5``
* :pr:`329`: supports lists with OnnxruntimeEvaluator
* :pr:`326`: use ConcatFromSequence in LoopMHA with the loop
* :pr:`325`: adds plug for LoopMHA, extends the unit tests to measure the discrepancies
* :pr:`324`: supports FunctionProto with arguments in OnnxruntimeEvaluator
* :pr:`323`: drops torch 2.8 on CI
* :pr:`322`: support rerunning onnx kernels with torch intermediate results in side-by-side
* :pr:`314`: fix modelbuilder download needed after this change https://github.com/microsoft/onnxruntime-genai/pull/1862
* :pr:`311`: use custom and local function to use PackedMultiHeadAttention from onnxruntime
* :pr:`310`: splits patches into multiple files 
* :pr:`308`: add option --save_ep to dump the exported program as well as torch input
* :pr:`304`, :pr:`306`, :pr:`316`, :pr:`317`, :pr:`318`, :pr:`319`: improves side-by-side comparison, creates command line sbs

0.8.2
+++++

* :pr:`303`: fix inputs for summarization, feature extraction tasks
* :pr:`302`: adds helpers to analyse onnxruntime profiling
* :pr:`297`: experiment around a higher ops ``loop_for_onnx``
* :pr:`292`, :pr:`293`, :pr:`294`, :pr:`295`: new patches for Qwen models

0.8.1
+++++

* :pr:`290`: adds one prompt for text2text-generation
* :pr:`289`: adds command line options ``--exppo`` to give the exporter additional options
* :pr:`287`: adds input ``'inputs_prompt'`` to test a LLM, meant to be used during validation
* :pr:`288`: add .contiguous in torch.cond branch (attention patch for sdpa implementation)
* :pr:`286`: adds variable to track random nodes in models

0.8.0
+++++

* :pr:`283`: fix historical aggregation when multiple input sets are used
* :pr:`282`: add tools to understand better which functions were patched
* :pr:`280`: fixes patches for sdpa_attention_forward for different version of transformers
* :pr:`278`: implements ``onnx_generate_with_genai``
* :pr:`277`: changes the serialization for all caches to reorder the model outputs (key_1, value_1, key_2, ...)
* :pr:`276`: implements ``onnx_generate`` which implements method generate for an onnx model,
* :pr:`275`: fixes function ``patched_vmap``

0.7.16
++++++

* :pr:`273`: enables export with FakeTensor
* :pr:`272`: makes patches work with FakeTensor
* :pr:`270`: add export sample code to export a specific model id with the appropriate inputs
* :pr:`269`: adds one unit test to track a patch fixing broadcast output shape
* :pr:`267`: patches ``sdpa_attention_forward`` because of a control flow (``transformers>=5.0``)
* :pr:`266`: makes ``patch_torch`` an integer in ``torch_export_patches`` to enable more patches 

0.7.15
++++++

* :pr:`264`: allows to validate a model with inputs defined from another task
* :pr:`261`: updates to support ``transformers>=5.0``

0.7.14
++++++

* :pr:`257`: patch to disable one exception in pytorch
* :pr:`256`: extract subfolder from modelid//subfolder
* :pr:`252`: adds new sets of inputs for task texgt-generation
* :pr:`250`: add variables to track sequence nodes
* :pr:`249`: patches _maybe_broadcast to support a corner case

0.7.13
++++++

* :pr:`247`: supports more gemma models with ModelBuilder
* :pr:`246`: add a set of inputs checking models works for an empty cache on task text-generation
* :pr:`237`: dummy inputs for google/gemma-3-4b-it
* :pr:`244`: add a patch to bypass the exception raised when the dynamic dimension is in {0,1}

0.7.12
++++++

* :pr:`232`: fixes ``--patch`` argument so that ``--patch=0`` works
* :pr:`231`: better statistics about fusions
* :pr:`227`: better support for ``model_id//pretrained``, adds speed up when running command validate
* :pr:`226`: fix input order for models created with modelbuilder

0.7.11
++++++

* :pr:`224`: support model_id with // to specify a subfolder 
* :pr:`223`: adds task image-to-video
* :pr:`220`: adds option --ort-logs to display onnxruntime logs when creating the session
* :pr:`220`: adds a patch for PR `#40791 <https://github.com/huggingface/transformers/pull/40791>`_ in transformers

0.7.10
++++++

* :pr:`218`: patches used sdpa_mask_recent_torch used from _vmap_for_bhqkv

0.7.9
+++++

* :pr:`214`: fix modelbuilder export
* :pr:`213`: use DYNAMIC on batch size

0.7.8
+++++

* :pr:`210`: add utilities to investigate models
* :pr:`208`: add a patch for Qwen3 (rewrite a loop)

0.7.7
+++++

* :pr:`205`: add in_channels in image_text_to_text
* :pr:`204`: switch default num_hidden_layers to 4
* :pr:`203`: Add option to disable patches for torch in command line validate
* :pr:`202`: add models DeepseekV3ForCausalLM, Gemma3ForCausalLM, Glm4vMoeForConditionalGeneration
* :pr:`201`: switch CI to 4.55.4
* :pr:`200`: fixes patches for 4.55.1+, DynamicCache is no longer registered by default, this code moved to executorch.py in transformers
* :pr:`199`: delete hidden_size and num_attention_heads modification in a config
* :pr:`198`: support gpt-oss
* :pr:`197`: updates CI for torch 2.8
* :pr:`196`: implements a patch to rewrite a loop in modeling_qwen2_vl.VisionAttention 

0.7.6
+++++

* :pr:`193`: validates with 4.53.3 
* :pr:`189`: support for task mask-generation
* :pr:`192`: add support for Gemma-3, add serialization for HybridCache, changes to support ``transformers>=4.54``

0.7.5
+++++

* :pr:`186`: add parameter --output_names to command line validate to change the output names of the onnx exported model
* :pr:`185`: remove the use of _seen_tokens in DynamicCache (removed in ``transformers>4.53``), updates dummpy inputs for feature-extraction
* :pr:`184`: implements side-by-side

0.7.4
+++++

* :pr:`178`: add a patch for eager_mask to handle ``assert len(flat_dynamic_shapes) == num_placeholders - num_lifted_inputs``
* :pr:`177`: changes for the next version of onnx, fixes all_dynamic_shapes_from_inputs

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
* :pr:`144`: support for second inputs with different dimension, rename test_helper into validate, support ``interpolate_pos_encoding`` for ``VitModel``, update model builder helpers for this PR
  `Use ONNX IR for model builder <https://github.com/microsoft/onnxruntime-genai/pull/1416>`_
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
* :pr:`115`, :pr:`116`, :pr:`117`, :pr:`118`, :pr:`119`, :pr:`127`: first steps for TorchOnnxEvaluator
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
* :pr:`58`: add function use_dyn_not_str to replace string by ``torch.export.Dim.DYNAMIC``, use string instead of ``torch.export.Dim.DYNAMIC`` when returning the dynamic shapes for a specific models, it is a valid definition for ``torch.onnx.export`` which can reuse the names
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
* :pr:`29`: adds helpers to measure the memory peak and run benchmark on different processes
* :pr:`28`: adds command line to print out the configuration for a model id, support image-text-to-text
* :pr:`26`: creates a folder ``helpers`` to gather all the functions used in many places
* :pr:`25`: improve patches for DynamicCache (issue with register_pytree_flatten_spec being deprecated)
* :pr:`24`: dummy inputs for ``text2text-generation``, add new function ``convert_dynamic_axes_into_dynamic_shapes`` to convert dynamic axes into dynamic shapes, add support for ``T5ForConditionalGeneration``
* :pr:`23`: dummy inputs for ``image-classification``
* :pr:`22`, :pr:`27`: api to create untrained model copying the architecture of the trained models and dummy inputs for them, support for ``text-generation``

0.2.1
+++++

* :pr:`16`: refactors patches, add model Phi2, implements a tweak to raise an exception with a dynamic dimension becomes static when exporting a model

0.2.0
+++++

* :pr:`11`: adds ``ModelInputs`` to guess dynamic shapes
* :pr:`9`: adds ``OnnxruntimeEvaluator``
* :pr:`8`: adds ``ExtendedReferenceEvaluator``
* :pr:`7`: improves function ``investigate_onnxruntime_issue``

0.1.0
+++++

first version
