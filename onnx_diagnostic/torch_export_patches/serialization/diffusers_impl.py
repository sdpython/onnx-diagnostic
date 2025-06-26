from typing import Dict, Optional, Set

try:
    from diffusers.models.unets.unet_2d_condition import UNet2DConditionOutput
except ImportError as e:
    try:
        import diffusers
    except ImportError:
        diffusers = None
        UNet2DConditionOutput = None
    if diffusers:
        raise e

from . import make_serialization_function_for_dataclass


def _make_wrong_registrations() -> Dict[type, Optional[str]]:
    res: Dict[type, Optional[str]] = {}
    for c in [UNet2DConditionOutput]:
        if c is not None:
            res[c] = None
    return res


SUPPORTED_DATACLASSES: Set[type] = set()
WRONG_REGISTRATIONS = _make_wrong_registrations()


if UNet2DConditionOutput is not None:
    (
        flatten_u_net2_d_condition_output,
        flatten_with_keys_u_net2_d_condition_output,
        unflatten_u_net2_d_condition_output,
    ) = make_serialization_function_for_dataclass(UNet2DConditionOutput, SUPPORTED_DATACLASSES)
