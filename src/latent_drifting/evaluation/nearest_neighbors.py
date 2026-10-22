"""Chunked nearest-neighbor search for memorization analysis."""

from __future__ import annotations

import torch
import torch.nn.functional as F


@torch.no_grad()
def nearest_neighbors(
    query_features: torch.Tensor,
    reference_features: torch.Tensor,
    top_k: int = 5,
    query_chunk_size: int = 256,
    reference_chunk_size: int = 4096,
) -> tuple[torch.Tensor, torch.Tensor]:
    if query_features.ndim != 2 or reference_features.ndim != 2:
        raise ValueError("Feature matrices must be rank two")
    if query_features.shape[1] != reference_features.shape[1]:
        raise ValueError("Query/reference feature dimensions differ")
    if top_k < 1 or top_k > reference_features.shape[0]:
        raise ValueError("top_k must be between one and the number of references")
    query = F.normalize(query_features.float(), dim=-1)
    reference = F.normalize(reference_features.float(), dim=-1)
    all_scores, all_indices = [], []
    for query_batch in query.split(query_chunk_size):
        best_scores = torch.full((query_batch.shape[0], top_k), -torch.inf, device=query.device)
        best_indices = torch.full((query_batch.shape[0], top_k), -1, device=query.device, dtype=torch.long)
        reference_offset = 0
        for reference_batch in reference.split(reference_chunk_size):
            scores = query_batch @ reference_batch.T
            local_k = min(top_k, reference_batch.shape[0])
            local_scores, local_indices = scores.topk(local_k, dim=-1)
            local_indices += reference_offset
            merged_scores = torch.cat((best_scores, local_scores), dim=-1)
            merged_indices = torch.cat((best_indices, local_indices), dim=-1)
            best_scores, selected = merged_scores.topk(top_k, dim=-1)
            best_indices = merged_indices.gather(-1, selected)
            reference_offset += reference_batch.shape[0]
        all_scores.append(best_scores)
        all_indices.append(best_indices)
    return torch.cat(all_scores), torch.cat(all_indices)
