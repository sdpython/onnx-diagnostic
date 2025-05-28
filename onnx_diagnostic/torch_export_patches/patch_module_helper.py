import ast
import functools
from typing import Any, Dict, List, Optional


class OrToBitOrTransformer(ast.NodeTransformer):
    def visit_BoolOp(self, node):
        self.generic_visit(node)
        if isinstance(node.op, ast.Or):
            new_node = node.values[0]
            for value in node.values[1:]:
                new_node = ast.BinOp(left=new_node, op=ast.BitOr(), right=value)
            return ast.copy_location(new_node, node)
        return node


def ast_or_into_bitor(node: "ast.Node") -> "ast.Node":
    """Replaces every operator ``or`` into ``|``."""
    new_node = OrToBitOrTransformer().visit(node)
    return new_node


@functools.lru_cache
def _rewrite_forward_clamp_float16() -> Dict[str, List[type]]:

    import transformers

    _known = {
        "AutoformerEncoderLayer": [
            transformers.models.autoformer.modeling_autoformer.AutoformerEncoderLayer
        ],
        "BartEncoderLayer": [
            transformers.models.bart.modeling_bart.BartEncoderLayer,
            transformers.models.plbart.modeling_plbart.PLBartEncoderLayer,
        ],
        "BigBirdPegasusEncoderLayer": [
            transformers.models.bigbird_pegasus.modeling_bigbird_pegasus.BigBirdPegasusEncoderLayer
        ],
        "BlenderbotSmallEncoderLayer": [
            transformers.models.blenderbot_small.modeling_blenderbot_small.BlenderbotSmallEncoderLayer
        ],
        "InformerEncoderLayer": [
            transformers.models.informer.modeling_informer.InformerEncoderLayer
        ],
        "LEDEncoderLayer": [transformers.models.led.modeling_led.LEDEncoderLayer],
        "MarianEncoderLayer": [transformers.models.marian.modeling_marian.MarianEncoderLayer],
        "MvpEncoderLayer": [transformers.models.mvp.modeling_mvp.MvpEncoderLayer],
        "NllbMoeEncoderLayer": [
            transformers.models.nllb_moe.modeling_nllb_moe.NllbMoeEncoderLayer
        ],
        "TimeSeriesTransformerEncoderLayer": [
            transformers.models.time_series_transformer.modeling_time_series_transformer.TimeSeriesTransformerEncoderLayer
        ],
    }
    return _known


@functools.lru_cache
def known_transformers_rewritings_clamp_float16() -> Dict[str, str]:
    """
    This functions returns the list of known classes to be rewritten.
    in :epkg:`transformers`. Each class is mapped to an alias,
    this alias is then given to :func:`rewritings_transformers_clamp_float16`
    to rewrite the encoder layers because of a specific control flow.

    .. runpython::
        :showcode:

        import pprint
        from onnx_diagnostic.torch_export_patches.patch_model_helper import (
            known_transformers_rewritings,
        )

        pprint.pprint(known_transformers_rewritings())
    """
    _alias = {
        "AutoformerEncoder": "AutoformerEncoderLayer",
        "AutoformerEncoderLayer": "AutoformerEncoderLayer",
        "AutoformerForPrediction": "AutoformerEncoderLayer",
        "AutoformerModel": "AutoformerEncoderLayer",
        "BartEncoderLayer": "BartEncoderLayer",
        "BartForConditionalGeneration": "BartEncoderLayer",
        "BigBirdPegasusForConditionalGeneration": "BigBirdPegasusEncoderLayer",
        "BigBirdPegasusForQuestionAnswering": "BigBirdPegasusEncoderLayer",
        "BigBirdPegasusForCausalLM": "BigBirdPegasusEncoderLayer",
        "BlenderbotSmallEncoderLayer": "BlenderbotSmallEncoderLayer",
        "BlenderbotSmallForConditionalGeneration": "BlenderbotSmallEncoderLayer",
        "BlenderbotSmallForCausalLM": "BlenderbotSmallEncoderLayer",
        "InformerEncoderLayer": "InformerEncoderLayer",
        "InformerForPrediction": "InformerEncoderLayer",
        "LEDEncoderLayer": "LEDEncoderLayer",
        "LEDClassificationHead": "LEDEncoderLayer",
        "LEDForConditionalGeneration": "LEDEncoderLayer",
        "MarianEncoderLayer": "MarianEncoderLayer",
        "MarianEncoder": "MarianEncoderLayer",
        "MarianModel": "MarianEncoderLayer",
        "MarianMTModel": "MarianEncoderLayer",
        "MvpEncoderLayer": "MvpEncoderLayer",
        "MvpPrompt": "MvpEncoderLayer",
        "MvpForConditionalGeneration": "MvpEncoderLayer",
        "MvpForSequenceClassification": "MvpEncoderLayer",
        "MvpForQuestionAnswering": "MvpEncoderLayer",
        "MvpForCausalLM": "MvpEncoderLayer",
        "NllbMoeEncoderLayer": "NllbMoeEncoderLayer",
        "NllbMoeForConditionalGeneration": "NllbMoeEncoderLayer",
        "PLBartEncoderLayer": "BartEncoderLayer",
        "PLBartForConditionalGeneration": "BartEncoderLayer",
        "TimeSeriesTransformerEncoderLayer": "TimeSeriesTransformerEncoderLayer",
        "TimeSeriesTransformerForPrediction": "TimeSeriesTransformerEncoderLayer",
    }
    return _alias


def rewritings_transformers_clamp_float16(cls_name) -> List[type]:
    """
    Rewrites known control flows equal to this:

    .. code-block:: python

        if hidden_states.dtype == torch.float16 and (
            torch.isinf(hidden_states).any() or torch.isnan(hidden_states).any()
        ):
            clamp_value = torch.finfo(hidden_states.dtype).max - 1000
            hidden_states = torch.clamp(hidden_states, min=-clamp_value, max=clamp_value)

    *cls_name* is the class name. It is mapped with a list of other class names
    to rename. Here is the known list:

    .. runpython::
        :showcode:

        import pprint
        from onnx_diagnostic.torch_export_patches.patch_model_helper import (
            _rewrite_forward_clamp_float16,
        )

        pprint.pprint(_rewrite_forward_clamp_float16()

    Function :func:`known_transformers_rewritings` collects
    all model classes using those layers.
    """
    _known = _rewrite_forward_clamp_float16()

    assert cls_name in _known, f"cls_name={cls_name!r} unknown in {sorted(_known)}."

    bd = dict(
        filter_node=(
            lambda node: isinstance(node, ast.If) and not isinstance(node.test, ast.Name)
        ),
        pre_rewriter=ast_or_into_bitor,
    )

    def _add(f):
        g = bd.copy()
        g["function"] = f
        return g

    return [_add(cls.forward) for cls in _known[cls_name]]


def code_needing_rewriting(cls_name: str) -> Optional[List[Any]]:
    """
    Returns a known list of classes mapped to a knwon rewritings
    because of control flow. See :func:`registered_transformers_rewritings`.

    :param cls_name: name of the class
    :return: a list of rewriting

    .. runpython::
        :showcode:

        import pprint
        from onnx_diagnostic.torch_export_patches.patch_module_helper import (
            code_needing_rewriting,
        )

        pprint.pprint(code_needing_rewriting("BartForConditionalGeneration"))
    """
    aliases = known_transformers_rewritings_clamp_float16()
    if cls_name in aliases:
        alias = aliases[cls_name]
        return rewritings_transformers_clamp_float16(alias)
    return None
