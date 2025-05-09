Change Logs
===========

0.5.0
+++++

* :pr:`91`: enable strings in ``guess_dynamic_shapes``
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
