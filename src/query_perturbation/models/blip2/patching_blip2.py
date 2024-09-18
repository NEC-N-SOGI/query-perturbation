import types

from lavis.models.blip2_models.blip2_qformer import Blip2Qformer
from torch import nn

from .qpert_attn_forward import attention_forward
from .qpert_proj import QueryProjection


# Monkey patches query, and forward of crossattention in model
def monkey_patch_crossattention(
    model: Blip2Qformer,
    do_normalize: bool = False,
) -> None:
    for layer in model.Qformer.bert.encoder.layer:
        if hasattr(layer, "crossattention"):
            if isinstance(layer.crossattention.self.query, nn.Linear):
                layer.crossattention.self.query = QueryProjection(
                    layer.crossattention.self.query,
                    do_normalize,
                )
            else:
                layer.crossattention.self.query = QueryProjection(
                    layer.crossattention.self.query.query,
                    do_normalize,
                )

            layer.crossattention.self.forward = attention_forward
            layer.crossattention.self.forward = types.MethodType(
                attention_forward, layer.crossattention.self
            )


def check_patching_results(model: Blip2Qformer) -> None:
    # Confirm monkey patch results
    for layer in model.Qformer.bert.encoder.layer:
        if hasattr(layer, "crossattention"):
            assert isinstance(
                layer.crossattention.self.query, QueryProjection
            ), "Query not monkey patched"
            assert (
                layer.crossattention.self.forward.__code__.co_code
                == attention_forward.__code__.co_code
            ), "Forward not monkey patched"
