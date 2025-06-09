import torch, torch.nn.functional as F
from torch import nn
from typing import Optional, Callable

class InfoNCELoss(nn.Module):
    """
    In-batch InfoNCE with three aggregation modes:
      - token_max      : classic ColBERT
      - modality_max   : choose the single best modality
      - masked         : positives forced to correct modality
    """
    def __init__(self, temperature: float = 0.02, aggregation_method: Callable= None , normalize_scores: bool = True, modality_types: list[str] = None):
        super().__init__()
        self.tau = temperature
        self.agg = aggregation_method
        self.normalize = normalize_scores
        self.modality_types = modality_types

    def forward(
        self,
        query_emb: torch.Tensor,              # [B,Q,D]
        doc_emb: torch.Tensor,                # [B,S,D]
        modality_ids: Optional[torch.Tensor] = None,  # [B,S]
        query_types: Optional[torch.Tensor] = None,    # [B]
        query_mask: Optional[torch.Tensor] = None,     # [B,Q]
    ) -> torch.Tensor:
        B, device = query_emb.size(0), query_emb.device

        sim = self.agg(
            query_emb,
            doc_emb,
            modality_ids=modality_ids,
            query_types=query_types,
            query_mask=query_mask,
            normalize=self.normalize
        )

        labels = torch.arange(B, device=device)
        return F.cross_entropy(sim / self.tau, labels)

class PairwiseSoftplusLoss(nn.Module):
    """
    Pair‑wise ranking loss for multimodal ColBERT‑style retrieval.

    • Uses the same `aggregation_method` callable as InfoNCELoss to turn
      (query_emb, doc_emb, …) → similarity matrix  S  of shape [B, B].

    • For every query i (row i of S):
        – `pos_score`  = S[i, i]                 (the in‑batch positive)
        – `neg_score`  = max_j≠i  S[i, j]        (hardest in‑batch negative)

      The loss for that example is  softplus(neg_score − pos_score).

    • Final loss = mean over the batch.
    """
    def __init__(self, aggregation_method: Callable, normalize_scores: bool = True, modality_types: Optional[list[str]] = None):
        super().__init__()
        self.agg = aggregation_method
        self.normalize = normalize_scores
        self.modality_types = modality_types

    def forward(
        self,
        query_emb: torch.Tensor,                   # [B, Q, D]
        doc_emb: torch.Tensor,                     # [B, S, D]
        modality_ids: Optional[torch.Tensor] = None,   # [B, S]
        query_types: Optional[torch.Tensor] = None,     # [B]
        query_mask: Optional[torch.Tensor] = None,      # [B, Q]
    ) -> torch.Tensor:

        B, device = query_emb.size(0), query_emb.device

        # 1) Aggregate to a full [B, B] similarity matrix
        sim = self.agg(
            query_emb,
            doc_emb,
            modality_ids=modality_ids,
            query_types=query_types,
            query_mask=query_mask,
            normalize=self.normalize,
        )                                           # shape [B, B]

        # 2) Positive scores: diagonal
        pos_scores = sim.diagonal()                # shape [B]

        # 3) Hard‑negative scores: row‑wise max over j ≠ i
        #    (mask out the diagonal with a large negative value)
        neg_masked = sim - torch.eye(B, device=device) * 1e6
        neg_scores = neg_masked.max(dim=1).values   # shape [B]

        # 4) Pair‑wise softplus loss  =  log(1 + exp(neg - pos))
        loss = F.softplus(neg_scores - pos_scores).mean()

        return loss

def _all_modality_sims(
    query: torch.Tensor,            # [B,Q,D]
    doc: torch.Tensor,              # [B,S,D]
    modality_ids: torch.Tensor,     # [B,S]  (ints 0 … M-1)
    query_mask: Optional[torch.Tensor] = None,
    normalize: bool = True,
    M: int = 5,
) -> torch.Tensor:                  # → [B,B,M]
    """
    Computes ColBERT late‑interaction (token‑max → query‑sum) for *every*
    modality **in one kernel**.

        sims[b, c, m]  =  Σ_q  max_{s | modality_ids[c,s]==m}  <q_b, d_c>
    """
    B, _, _ = query.shape

    # Base dot‑product: [B(query), B(doc), Q, S]
    base = torch.einsum("bqd,csd->bcqs", query, doc)

    # Build a [B, S, M] boolean mask for each *doc* row)
    one_hot = torch.nn.functional.one_hot(modality_ids, M).bool()  # [B,S,M]
    # Move modality to front => [M,B,S] to broadcast nicely
    mod_mask = one_hot.permute(2, 0, 1)                          # [M,B,S]
    # Apply mask in a vectorised way
    base = base.unsqueeze(0)                             # [1,B,B,Q,S]
    masked = base.masked_fill(~mod_mask[:, None, :, None, :], float("-inf"))
    sims = masked.amax(dim=-1).sum(dim=-1)                       # [M,B,B]
    sims = sims.permute(1, 2, 0).contiguous()                    # [B,B,M]

    if normalize and query_mask is not None:
        sims = sims / torch.clamp(query_mask.sum(dim=1), min=1)[:, None, None]

    return sims

class HardPositiveInfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.02, aggregation_method = None, normalize_scores: bool = True, modality_types: list[str] = None):
        super().__init__()
        self.tau = temperature
        self.agg = aggregation_method
        self.normalize = normalize_scores
        self.modality_types = modality_types
    
    def forward(
        self,
        query_emb: torch.Tensor,            # [B,Q,D]
        doc_emb: torch.Tensor,              # [B,S,D]
        modality_ids: torch.Tensor,         # [B,S]
        query_types: torch.Tensor,          # [B]   (int correct modality)
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, device = query_emb.size(0), query_emb.device
        # base still needs to be computed with an einsum as the maxsim can be computed not across modality
        sim = self.agg(query_emb, doc_emb, modality_ids, query_types, query_mask=query_mask, normalize=self.normalize)

        # all modality sims so that we can overwrite the diagonal with the correct modality
        sims = _all_modality_sims(query_emb, doc_emb, modality_ids, query_mask=query_mask, normalize=self.normalize, M=len(self.modality_types)+1) # [B,B,M], modality_ids==0 is padding

        idx = torch.arange(B, device=device, dtype=query_types.dtype)
        sim_fixed = sim.clone()
        sim_fixed[idx, idx] = sims[idx, idx, query_types].to(sim_fixed.dtype)

        logits = sim_fixed / self.tau
        labels = idx

        return F.cross_entropy(logits, labels)

class HardNegativeInfoNCELoss(nn.Module):
    """
    Final logits matrix: [B,  B + (M-1)]
        · first  B        columns → positives + inter‑doc modality‑max negatives
        · next   B*(M-1)  columns → other modalities of every doc (hard negatives)
    """
    def __init__(self, temperature: float = 0.02, aggregation_method = None, normalize_scores: bool = True, modality_types: list[str] = None):
        super().__init__()
        self.tau = temperature
        self.agg = aggregation_method
        self.normalize = normalize_scores
        self.modality_types = modality_types

    # ................................................................. #
    def forward(
        self,
        query_emb: torch.Tensor,            # [B,Q,D]
        doc_emb: torch.Tensor,              # [B,S,D]
        modality_ids: torch.Tensor,         # [B,S]
        query_types: torch.Tensor,          # [B]
        query_mask: Optional[torch.Tensor] = None,
    ) -> torch.Tensor:

        B, device = query_emb.size(0), query_emb.device
        M = len(self.modality_types)+1 # need to add 1 since modality_ids==0 is padding

        # base still needs to be computed with an einsum as the maxsim can be computed not across modality
        sim = self.agg(query_emb, doc_emb, modality_ids, query_types, query_mask, normalize=self.normalize)

        # all modality sims
        sims = _all_modality_sims(query_emb, doc_emb, modality_ids, query_mask=query_mask, normalize=self.normalize, M=M) # [B,B,M] 

        # overwrite diagonal with correct modality
        idx = torch.arange(B, device=device, dtype=query_types.dtype)
        sim_fixed = sim.clone()
        sim_fixed[idx, idx] = sims[idx, idx, query_types].to(sim_fixed.dtype)

        # (b) build *all* (doc, modality) logits except the positive
        idx = torch.arange(B, device=device)
        all_modal = sims.view(B, -1)
        mask_no_pad = torch.ones_like(all_modal, dtype=torch.bool)    # shape [B, B*M]
        mask_no_pad[:, 0::M] = False                                  # mask out pad columns
        
        mask_no_pad[idx, idx * M + query_types] = False                            # mask out positive
        extras = all_modal[mask_no_pad].view(B, -1)                   # [B, B*(M-1)-1]
        logits = torch.cat([sim, extras], dim=1) / self.tau     # [B, B*(M-1)]
        labels = idx
        return F.cross_entropy(logits, labels)