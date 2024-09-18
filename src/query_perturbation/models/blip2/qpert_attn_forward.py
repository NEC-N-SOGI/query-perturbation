import math
from typing import Optional

import torch
from lavis.models.blip2_models.Qformer import BertSelfAttention
from torch import nn


def attention_forward(
    self: BertSelfAttention,
    hidden_states: torch.Tensor,
    attention_mask: Optional[torch.Tensor] = None,
    head_mask: Optional[torch.Tensor] = None,
    encoder_hidden_states: Optional[torch.Tensor] = None,
    encoder_attention_mask: Optional[torch.Tensor] = None,
    past_key_value: Optional[tuple[torch.Tensor, torch.Tensor]] = None,
    output_attentions: bool = False,
) -> tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor]]:
    """CrossAttention forward function with Q-Pertuabtion.
    Almost same to the original BLIP-2 code:
    https://github.com/salesforce/LAVIS/blob/ac8fc98c93c02e2dfb727e24a361c4c309c8dbbc/lavis/models/blip2_models/Qformer.py#L111

    Args:
        self (BertSelfAttention): _description_
        hidden_states (torch.Tensor): _description_
        attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
        head_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
        encoder_hidden_states (Optional[torch.Tensor], optional): _description_. Defaults to None.
        encoder_attention_mask (Optional[torch.Tensor], optional): _description_. Defaults to None.
        past_key_value (Optional[tuple[torch.Tensor, torch.Tensor]], optional): _description_. Defaults to None.
        output_attentions (bool, optional): _description_. Defaults to False.

    Returns:
        tuple[torch.Tensor, torch.Tensor, tuple[torch.Tensor]]: _description_
    """
    is_cross_attention = encoder_hidden_states is not None

    assert isinstance(self, BertSelfAttention), "Only support for BertSelfAttention"
    assert is_cross_attention, "Currently only support cross attention"
    assert not (
        self.position_embedding_type == "relative_key"
        or self.position_embedding_type == "relative_key_query"
    ), "Not support for positional embeddings yet"

    key_layer: torch.Tensor = self.transpose_for_scores(
        self.key(encoder_hidden_states)
    )
    value_layer: torch.Tensor = self.transpose_for_scores(
        self.value(encoder_hidden_states)
    )
    attention_mask = encoder_attention_mask

    self.keys = key_layer

    mixed_query_layer = self.query(hidden_states)

    query_layer = self.transpose_for_scores(mixed_query_layer)

    # ----- Q-Pert: Query Projection ---------------
    query_layer = self.query.proj(query_layer, key_layer)

    past_key_value = (key_layer, value_layer)

    # Take the dot product between "query" and "key" to get the raw attention scores.
    attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))

    attention_scores = attention_scores / math.sqrt(self.attention_head_size)
    if attention_mask is not None:
        # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
        attention_scores = attention_scores + attention_mask

    # Normalize the attention scores to probabilities.
    attention_probs: torch.Tensor = nn.Softmax(dim=-1)(attention_scores)

    if is_cross_attention and self.save_attention:
        self.save_attention_map(attention_probs)
        attention_probs.register_hook(self.save_attn_gradients)  # type: ignore

    # This is actually dropping out entire tokens to attend to, which might
    # seem a bit unusual, but is taken from the original Transformer paper.
    attention_probs_dropped = self.dropout(attention_probs)

    # Mask heads if we want to
    if head_mask is not None:
        attention_probs_dropped = attention_probs_dropped * head_mask

    context_layer: torch.Tensor = torch.matmul(attention_probs_dropped, value_layer)

    context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
    new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
    context_layer = context_layer.view(*new_context_layer_shape)

    outputs = (
        (context_layer, attention_probs) if output_attentions else (context_layer,)
    )

    outputs = outputs + (past_key_value,)  # type: ignore
    return outputs  # type: ignore
