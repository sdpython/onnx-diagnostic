Change Logs
===========

0.3.0
+++++

* :pr:`25`: improve patches for DynamicCache
  (issue with register_pytree_flatten_spec being deprecated)
* :pr:`24`: dummy inputs for ``text2text-generation``, add new function
  ``convert_dynamic_axes_into_dynamic_shapes`` to convert dynamic axes
  into dynamic shapes, add support for ``T5ForConditionalGeneration``
* :pr:`23`: dummy inputs for ``image-classification``
* :pr:`22`: api to create untrained model copying the architecture
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
