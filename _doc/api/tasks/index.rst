onnx_diagnostic.tasks
=====================

All submodules contains the three following functions:

* ``reduce_model_config(config) -> kwargs``:
  updates the configuration to get a smaller model more suitable
  for unit tests
* ``random_input_kwargs(config) -> kwargs, get_inputs``:
  produces values ``get_inputs`` can take to generate dummy inputs
  suitable for a model defined by its configuration
* ``get_inputs(model, config, *args, add_second_input=False, **kwargs) -> dict(inputs=..., dynamic_shapes=...)``:
  generates the dummy inputs and dynamic shapes for a specific model and configuration,
  if ``add_second_input`` is True, the function should return a different set of inputs,
  with different values for the dynamic dimension. This is usually better to
  rely on the function as the dynamic dimensions may be correlated.

For a specific task, you would write:

.. code-block:: python

    kwargs, get_inputs = random_input_kwargs(config)
    dummies = get_inputs(model, config, **kwargs)

Or:

.. code-block:: python

    from onnx_diagnostic.tasks import random_input_kwargs

    kwargs, get_inputs = random_input_kwargs(config, task)  # "text-generation" for example
    dummies = get_inputs(model, config, **kwargs)

.. toctree::
    :maxdepth: 1
    :caption: modules

    automatic_speech_recognition
    fill_mask
    feature_extraction
    image_classification
    image_text_to_text
    mixture_of_expert
    object_detection
    sentence_similarity
    summarization
    text_classification
    text_generation
    text2text_generation
    zero_shot_image_classification
    
.. automodule:: onnx_diagnostic.tasks
    :members:
    :no-undoc-members:
