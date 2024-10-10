import logging
import os
import os.path
from typing import (
    Any,
    Callable,
    Dict,
    Generic,
    Iterator,
    List,
    Mapping,
    Optional,
    Tuple,
    Union,
    get_args,
)
import torch

def rectify_embed_sizes(param_name: str, tensors: List[torch.Tensor]):
    # TODO: use arch_info.embed_weights() instead
    if ("lm_head" in param_name or "embed_tokens" in param_name) and all(
        len(t.shape) == 2 for t in tensors
    ):
        # special case - if lm_head.weight or embed_tokens.weight have a size
        # mismatch, take the largest common submatrix of all of them
        if take_common_submatrix(tensors):
            logging.warning(
                f"Using common submatrix of size {tensors[0].shape} for {param_name}"
            )
            return tensors
    return


def take_common_submatrix(tensors: List[torch.Tensor]) -> bool:
    min_size = [None, None]
    for t in tensors:
        for idx in range(2):
            if min_size[idx] is None or t.shape[idx] < min_size[idx]:
                min_size[idx] = t.shape[idx]

    if not all(t.shape == torch.Size(min_size) for t in tensors):
        for idx in range(len(tensors)):
            tensors[idx] = tensors[idx][: min_size[0], : min_size[1]]
        return True
    return False


def reorder_embeddings(base_tokenizer, tokenizer, base_embed, embed):
    token_id_mapping = {}
    for token, id1 in base_tokenizer.get_vocab().items():
        if id1 >= base_embed.size(0):
            continue
        id2 = tokenizer.convert_tokens_to_ids(token)
        if id2 != tokenizer.unk_token_id:  # Check if the token exists in Model 2's vocabulary
            token_id_mapping[id1] = id2
        else:
            print(f"Token '{token}' not found in Model 2's vocabulary, assigning to '{tokenizer.unk_token_id}'.")
            # Handle the case for unknown tokens, if necessary
            # For example, map to the unknown token ID in Model 2's tokenizer
            token_id_mapping[id1] = tokenizer.unk_token_id

    # Initialize a new embedding matrix for Model 2 that aligns with Model 1's token order
    new_embed = base_embed.clone()

    # Reorder Model 2's embeddings to align with Model 1's token order
    for id1, id2 in token_id_mapping.items():
        new_embed[id1, :] = embed[id2, :]
    return new_embed