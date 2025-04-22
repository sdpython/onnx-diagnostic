onnx_diagnostic.tasks
=====================

All submodules contains the three following functions:

* ``reduce_model_config(config) -> kwargs``:
  updates the configuration to get a smaller model more suitable
  for unit tests
* ``random_input_kwargs(config) -> kwargs, get_inputs``:
  produces values ``get_inputs`` can take to generate dummy inputs
  suitable for a model defined by its configuration
* ``get_inputs(model, config, *args, **kwargs) -> dict(inputs=..., dynamic_shapes=...)``:
  generates the dummy inputs and dynamic shapes for a specific model and configuration.

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
    image_classification
    image_text_to_text
    sentence_similarity
    text_classification
    text_generation
    text2text_generation
    zero_shot_image_classification
    
.. automodule:: onnx_diagnostic.tasks
    :members:
    :no-undoc-members:
