import torch
from transformers import EvalPrediction
from models import *
from loss import *
from typing import Optional, Callable, Dict, List
from torchmetrics.retrieval import RetrievalRecall, RetrievalNormalizedDCG
from torchmetrics.classification import (
    MulticlassPrecision, MulticlassRecall, MulticlassF1Score,
)
import os

PROCESSOR_MAPPING = {
        "colqwen": ColQwen2_5Processor,
        "colqwenomni": ColQwenOmniProcessor,
    }

MODEL_MAPPING = {
        "colqwen": ColQwen2_5,
        "colqwenomni": ColQwenOmni,
    }

LOSS_MAPPING = {
    "contrastive": InfoNCELoss,
    "contrastive_hard_positive": HardPositiveInfoNCELoss,
    "contrastive_hard_negative": HardNegativeInfoNCELoss,
    "pairwise": PairwiseSoftplusLoss,
}

# Similarity calculations

def _colbert_sim(
    query: torch.Tensor,          # [B,Q,D]
    doc: torch.Tensor,            # [B,S,D]
    modality_ids: Optional[torch.Tensor] = None,
    query_types: Optional[torch.Tensor] = None,
    query_mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
    return_argmax: bool = False,
) -> torch.Tensor:                # → [B,B]
    """Late-interaction: max over *tokens*, sum over *query*."""
    sim = torch.einsum("bqd,csd->bcqs", query, doc) # [B,B,Q,S]
    if return_argmax:
        sim_argmax = sim.argmax(dim=-1)
        if query_mask is not None:
            sim_argmax[~query_mask] = 0
    sim = sim.amax(dim=-1).sum(dim=-1) # [B,B]
    if normalize and query_mask is not None:
        sim = sim / torch.clamp(query_mask.sum(dim=1,keepdim=True), min=1.0)
    if return_argmax:
        return sim, sim_argmax
    return sim

def _modality_max_sim(
    query: torch.Tensor,          # [B,Q,D]
    doc: torch.Tensor,            # [B,S,D]
    modality_ids: torch.Tensor,   # [B,S]   ints 0 … M‑1
    query_types: Optional[torch.Tensor] = None,
    query_mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
    return_argmax: bool = False,
    M: Optional[int] = None,      # override #modalities if you know it a‑priori
) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
    """
    ColBERT late‑interaction *per modality* in one shot.
    Returns:
        sim         … [B,B]   similarity matrix
        sim_argmax  … [B,B,Q] winning token indices (only if `return_argmax`)
    """
    B, Q, _ = query.shape
    if M is None:
        M = int(modality_ids.max().item()) + 1                           # #modalities
    
    base = torch.einsum("bqd,csd->bcqs", query, doc)                    # [B,B,Q,S]

    mod_mask = torch.nn.functional.one_hot(modality_ids, M).bool()      # [B,S,M]
    mod_mask = mod_mask.permute(2, 0, 1)                                # [M,B,S]

    # Expand dims so the mask aligns with base:
    #    masked[k] ≡ scores where doc‑token belongs to modality k
    masked = base.unsqueeze(0).masked_fill(                             # [M,B,B,Q,S]
        ~mod_mask[:, None, :, None, :],
        float("-inf"),
    )
    
    max_vals, max_idxs = masked.max(dim=-1)                             # both [M,B,B,Q]
    sims = max_vals.sum(dim=-1)                                         # [M,B,B]
    sims = sims.permute(1, 2, 0).contiguous()                           # [B,B,M]

    # Pick best modality for every (query, doc) pair
    sim, best_mod = sims.max(dim=-1)                                    # each [B,B]
    if normalize and query_mask is not None:
        sim = sim / torch.clamp(query_mask.sum(dim=1, keepdim=True), 1.0)

    if not return_argmax:
        return sim
    
    max_idxs = max_idxs.permute(1, 2, 0, 3).contiguous()                # [B,B,M,Q]
    gather_idx = best_mod.unsqueeze(-1).unsqueeze(-1).expand(-1, -1, 1, Q)
    sim_argmax = max_idxs.gather(2, gather_idx).squeeze(2)              # [B,B,Q]

    return sim, sim_argmax

AGGREGATION_MAPPING = {
    "token_max": _colbert_sim,
    "modality_max": _modality_max_sim,
}

def compute_metrics(
    eval_pred: EvalPrediction,
    *,
    agg: Callable,
    device: torch.device,
    modality_types: List[str],
    batch_size: int = 256,
    output_dir = None,
) -> Dict[str, float]:
    """
    Metrics for multimodal retrieval.

    Parameters injected with functools.partial
    -----------------------------------------
    agg              : callable that produces (similarity, argmax_token)
    device           : torch.device where embeddings fit
    modality_types   : list in the SAME order used for training
    batch_size       : chunk size when computing similarities
    """

    metric_key_prefix = "eval"

    q_np, d_np, mod_ids_np, qtype_ids_np = eval_pred.predictions
    query      = torch.from_numpy(q_np)             # [N,Q,D]
    doc        = torch.from_numpy(d_np)             # [M,S,D]
    mod_ids    = torch.from_numpy(mod_ids_np)       # [M,S]
    qtype_ids  = torch.from_numpy(qtype_ids_np)     # [N]

    # received dummy values, just save
    print(query.shape, doc.shape, mod_ids.shape, qtype_ids.shape)
    if query.shape[1] == 1 or doc.shape[1] == 1:
        if query.shape[1] != 1:
            torch.save(query, os.path.join(output_dir, "query_embeddings.pt") if output_dir is not None else "query_embeddings.pt")
        if doc.shape[1] != 1:
            torch.save(doc, os.path.join(output_dir, "doc_embeddings.pt") if output_dir is not None else "doc_embeddings.pt")
        return {}

    N, M = query.size(0), doc.size(0)
    Q = query.size(1)

    doc        = doc.to(device, non_blocking=True)
    mod_ids_gpu = mod_ids.to(device, non_blocking=True)

    sim_rows, pred_mod_rows = [], []

    with torch.no_grad():
        for start in range(0, N, batch_size):
            end     = min(start + batch_size, N)
            q_chunk = query[start:end].to(device, non_blocking=True)

            # argmax_tok will have shape [B_query_chunk, total_docs_M, Q]
            sims, argmax_tok = agg(
                q_chunk,
                doc,
                mod_ids_gpu,
                qtype_ids[start:end].to(device),
                None,
                return_argmax=True,
            )

            sim_rows.append(sims.cpu())

            query_indices_in_chunk = torch.arange(end - start, device=device)
            doc_indices_in_full_M = torch.arange(start, end, device=device)

            # Get the argmax token indices for the relevant document for each query
            tok_idx = argmax_tok[query_indices_in_chunk, doc_indices_in_full_M, :]
            expanded_doc_indices = doc_indices_in_full_M.unsqueeze(1).expand(-1, Q)

            pred_mod_current_chunk = mod_ids_gpu[expanded_doc_indices, tok_idx]
            pred_mod_rows.append(pred_mod_current_chunk.cpu())

    sim        = torch.cat(sim_rows,      dim=0)     # [N,M]
    pred_mods  = torch.cat(pred_mod_rows, dim=0)     # [N,Q]

    # Compute retrieval metrics (torchmetrics)
    preds_flat  = sim.flatten()
    target_flat = torch.zeros_like(preds_flat, dtype=torch.long)
    target_flat[torch.arange(N) * M + torch.arange(N)] = 1
    group_ids = torch.arange(N).repeat_interleave(M)

    from torchmetrics.retrieval import (
        RetrievalRecall, RetrievalNormalizedDCG
    )

    metrics: Dict[str, float] = {}
    for k in (1, 5, 10):
        metrics[f"{metric_key_prefix}_r@{k}"] = (
            RetrievalRecall(top_k=k)(
                preds_flat, target_flat, indexes=group_ids
            ).item() * 100
        )

    metrics[f"{metric_key_prefix}_ndcg@10"] = (
        RetrievalNormalizedDCG(top_k=10)(
            preds_flat, target_flat, indexes=group_ids
        ).item() * 100
    )

    # Per-class retrieval metrics (nDCG@10 and r@10)
    num_classes = len(modality_types) + 1 # +1 for padding 
    id2mod = {i + 1: m for i, m in enumerate(modality_types)}

    for mid, mname in id2mod.items():
        # Mask for queries of this modality
        query_mask = (qtype_ids == mid)
        if query_mask.any():
            idx = torch.where(query_mask)[0]
            # For these queries, get their similarity rows
            sim_sub = sim[idx]  # [num_q, M]
            N_sub = sim_sub.size(0)
            # Build targets: only the diagonal is relevant
            preds_flat_sub = sim_sub.flatten()
            target_flat_sub = torch.zeros_like(preds_flat_sub, dtype=torch.long)
            target_flat_sub[torch.arange(N_sub) * M + idx] = 1
            group_ids_sub = torch.arange(N_sub).repeat_interleave(M)
            metrics[f"{metric_key_prefix}_r@10_{mname}"] = (
                RetrievalRecall(top_k=10)(
                    preds_flat_sub, target_flat_sub, indexes=group_ids_sub
                ).item() * 100
            )
            metrics[f"{metric_key_prefix}_ndcg@10_{mname}"] = (
                RetrievalNormalizedDCG(top_k=10)(
                    preds_flat_sub, target_flat_sub, indexes=group_ids_sub
                ).item() * 100
            )
        else:
            metrics[f"{metric_key_prefix}_r@10_{mname}"] = 0.0
            metrics[f"{metric_key_prefix}_ndcg@10_{mname}"] = 0.0

    # Modality‑prediction accuracy & stats
    num_classes = len(modality_types) + 1 # +1 for padding
    id2mod = {i + 1: m for i, m in enumerate(modality_types)}
    mod_ids_range = list(id2mod.keys())

    # 4‑a: majority vote per query (ignore padding‑id==0)
    print("pred_mods",pred_mods)
    non_pad = pred_mods > 0
    maj = torch.zeros(N, dtype=torch.long)
    for i in range(N):
        mods = pred_mods[i][non_pad[i]]
        if len(mods):
            vals, cnts = torch.unique(mods, return_counts=True)
            # print(mod_ids[i].tolist(), qtype_ids[i], vals, cnts)
            print(qtype_ids[i], vals, cnts)
            maj[i] = vals[cnts.argmax()]
        # else: keep maj[i] = 0 (handle cases where no valid modality tokens were predicted)

    # Filter out predictions where no valid modality was found (maj == 0)
    valid_mask = maj > 0
    maj_filtered = maj[valid_mask]
    qtype_ids_filtered = qtype_ids[valid_mask]
    num_valid_preds = valid_mask.sum().item()

    if num_valid_preds > 0:
        # 4‑b: overall accuracy
        acc_all = (maj_filtered == qtype_ids_filtered).float().mean().item() * 100
        metrics[f"{metric_key_prefix}_modality_accuracy"] = acc_all

        any_match = []
        for i in range(N):
            if valid_mask[i]:
                ref_mod = qtype_ids[i].item()
                pred_set = set(pred_mods[i][pred_mods[i] > 0].tolist())
                any_match.append(ref_mod in pred_set)
        if any_match:
            acc_any = (sum(any_match) / len(any_match)) * 100
        else:
            acc_any = 0.0
        metrics[f"{metric_key_prefix}_modality_accuracy_any"] = acc_any

        for mid, mname in id2mod.items():
            # Find indices of queries with this reference modality
            mask = (qtype_ids == mid) & valid_mask
            idxs = torch.where(mask)[0]
            if len(idxs) > 0:
                correct = 0
                for i in idxs:
                    ref_mod = qtype_ids[i].item()
                    pred_set = set(pred_mods[i][pred_mods[i] > 0].tolist())
                    if ref_mod in pred_set:
                        correct += 1
                acc_any_mod = (correct / len(idxs)) * 100
            else:
                acc_any_mod = 0.0
            metrics[f"{metric_key_prefix}_modality_acc_any_{mname}"] = acc_any_mod

        # Precision, Recall, F1 (Macro and Per-Class)
        m_prec = MulticlassPrecision(num_classes=num_classes, average='macro', ignore_index=0)(maj_filtered, qtype_ids_filtered).item()
        m_rec  = MulticlassRecall(num_classes=num_classes, average='macro', ignore_index=0)(maj_filtered, qtype_ids_filtered).item()
        m_f1   = MulticlassF1Score(num_classes=num_classes, average='macro', ignore_index=0)(maj_filtered, qtype_ids_filtered).item()
        metrics[f"{metric_key_prefix}_modality_precision_macro"] = m_prec * 100
        metrics[f"{metric_key_prefix}_modality_recall_macro"] = m_rec * 100
        metrics[f"{metric_key_prefix}_modality_f1_macro"] = m_f1 * 100

        prec_pc = MulticlassPrecision(num_classes=num_classes, average='none', ignore_index=0)(maj_filtered, qtype_ids_filtered)
        rec_pc = MulticlassRecall(num_classes=num_classes, average='none', ignore_index=0)(maj_filtered, qtype_ids_filtered)
        f1_pc = MulticlassF1Score(num_classes=num_classes, average='none', ignore_index=0)(maj_filtered, qtype_ids_filtered)

        # Accuracy and PRF per modality
        for mid, mname in id2mod.items():
            class_idx = mid # Assuming class indices match modality IDs (1-based)
            mask = qtype_ids_filtered == mid
            if mask.any():
                 metrics[f"{metric_key_prefix}_modality_acc_{mname}"] = (
                    (maj_filtered[mask] == mid).float().mean().item() * 100
                 )
                 if class_idx < len(prec_pc): # Check if index is valid
                    metrics[f"{metric_key_prefix}_modality_precision_{mname}"] = prec_pc[class_idx].item() * 100
                    metrics[f"{metric_key_prefix}_modality_recall_{mname}"] = rec_pc[class_idx].item() * 100
                    metrics[f"{metric_key_prefix}_modality_f1_{mname}"] = f1_pc[class_idx].item() * 100

        # Predicted Modality Distribution
        pred_counts = torch.bincount(maj_filtered, minlength=num_classes)
        total_preds = maj_filtered.numel()
        if total_preds > 0:
             for mid, mname in id2mod.items():
                 class_idx = mid
                 if class_idx < len(pred_counts): # Check if index is valid
                     metrics[f"{metric_key_prefix}_modality_pred_dist_{mname}"] = (pred_counts[class_idx].item() / total_preds) * 100
    else:
        # Handle case with no valid predictions
        metrics[f"{metric_key_prefix}_modality_accuracy"] = 0.0
        metrics[f"{metric_key_prefix}_modality_precision_macro"] = 0.0
        metrics[f"{metric_key_prefix}_modality_recall_macro"] = 0.0
        metrics[f"{metric_key_prefix}_modality_f1_macro"] = 0.0
        for mid, mname in id2mod.items():
            metrics[f"{metric_key_prefix}_modality_acc_{mname}"] = 0.0
            metrics[f"{metric_key_prefix}_modality_precision_{mname}"] = 0.0
            metrics[f"{metric_key_prefix}_modality_recall_{mname}"] = 0.0
            metrics[f"{metric_key_prefix}_modality_f1_{mname}"] = 0.0
            metrics[f"{metric_key_prefix}_modality_pred_dist_{mname}"] = 0.0


    return metrics