import math
from collections import defaultdict
from typing import Dict, List, Tuple, Optional

import torch
import torch.nn.functional as F
from transformers import Trainer

import datasets
from transformers.utils import is_datasets_available
from transformers.trainer_utils import seed_worker

def _global_max(value: int, device: torch.device) -> int:
    if not torch.distributed.is_initialized():
        return value
    t = torch.tensor([value], device=device, dtype=torch.long)
    torch.distributed.all_reduce(t, op=torch.distributed.ReduceOp.MAX)
    return int(t.item())

def _all_gather(t: torch.Tensor) -> torch.Tensor:
    if not torch.distributed.is_initialized():
        return t
    gathered = torch.distributed.nn.all_gather(t)
    return torch.cat(gathered, dim=0)

class LateInteractionTrainer(Trainer):
    """
    Trainer that handles late‑interaction retrievers with multi‑modal documents.
    """

    META_KEYS = {"labels", "idx", "query_types", "combined_modality_ids"}

    def __init__(self, loss_func, *args, **kwargs):
        self.modality_types: List[str] = kwargs.pop(
            "modality_types", ["video", "ocr", "asr", "description"]
        )
        self.agg = kwargs.pop("agg")
        super().__init__(*args, **kwargs)
        self.loss_func = loss_func
        # map modality → 1‑based id (0 is reserved for padding)
        self._modid = {m: i + 1 for i, m in enumerate(self.modality_types)}

    def compute_loss(self, model, inputs, num_items_in_batch=None,return_outputs=False):
        (
            query_embs,
            doc_embs,
            modality_ids,
            qry_mask,
            qry_types,
            idx,
        ) = self._prepare_batch(model, inputs, gather=True, drop_invalid=True)

        loss = self.loss_func(
            query_embs,
            doc_embs,
            modality_ids,
            qry_types,
            qry_mask,
        )

        return (loss, (query_embs, doc_embs)) if return_outputs else loss

    
    def prediction_step(self, model, inputs, prediction_loss_only=False, ignore_keys=True):
        with torch.no_grad():
            (
                query_embs,
                doc_embs,
                modality_ids,
                qry_mask,
                qry_types,
            idx,
            ) = self._prepare_batch(model, inputs, gather=False, drop_invalid=False)
        # print(query_embs.shape, doc_embs.shape, modality_ids.shape, qry_mask.shape, qry_types.shape, idx.shape)

        return None, (query_embs, doc_embs, modality_ids, qry_types), torch.zeros(modality_ids.shape[0], device=modality_ids.device, dtype=torch.long)

    def _prepare_batch(
        self,
        model,
        raw_inputs: Dict[str, torch.Tensor],
        *,
        gather: bool,
        drop_invalid: bool,
    ):
        """
        → encodes all inputs, detects bad rows, gathers (optional), and returns
          query & document embeddings plus modality id matrix.
        """

        device = self.model.device
        inputs = {k: v.to(device) for k, v in raw_inputs.items()}

        # Infer batch size from any present tensor
        batch_size = next(iter(inputs.values())).size(0) if inputs else 1
        # Try to infer embedding dim from model config, fallback to 768
        D = getattr(self.model, 'dim', 128)

        # ------------------------------------------------------------ queries
        query_inputs, doc_inputs = self._split_by_modality(inputs)
        if query_inputs:
            query_embs = model(**query_inputs)                        # [B,Q,D]
            qry_mask  = query_inputs["attention_mask"]
            qry_types = inputs.get("query_types", None)                     # may be None
        else:
            # No query input: output dummy
            query_embs = torch.zeros((batch_size, 1, D), device=device)
            qry_mask = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
            qry_types = torch.zeros((batch_size, 1), device=device, dtype=torch.long)
        
        idx = inputs.get("idx", None)

        if doc_inputs:
            doc_embs, modality_ids = self._encode_documents(model, doc_inputs, inputs)
        else:
            # No document input: output dummy
            doc_embs = torch.zeros((batch_size, 1, D), device=device)
            modality_ids = torch.zeros((batch_size, 1), device=device, dtype=torch.long)

        if gather and doc_inputs and "combined" not in doc_inputs:
            max_len = _global_max(doc_embs.size(1), device)
            pad_len = max_len - doc_embs.size(1)
            if pad_len > 0:
                # left‑pad with zeros
                pad_emb  = doc_embs.new_zeros(doc_embs.size(0), pad_len, doc_embs.size(2))
                pad_ids  = modality_ids.new_zeros(modality_ids.size(0), pad_len)
                doc_embs     = torch.cat([pad_emb,  doc_embs],  dim=1)  # [B,max_len,D]
                modality_ids = torch.cat([pad_ids, modality_ids], dim=1)  # [B,max_len]
        
        if gather:
            query_embs   = _all_gather(query_embs)
            doc_embs     = _all_gather(doc_embs)
            modality_ids = _all_gather(modality_ids)
            qry_mask     = _all_gather(qry_mask)
            if qry_types is not None:
                qry_types = _all_gather(qry_types)
            if idx is not None:
                idx = _all_gather(idx)
        
        if drop_invalid:
            (
                query_embs,
                doc_embs,
                modality_ids,
                qry_mask,
                qry_types,
                idx,
            ) = self._remove_invalid(
                query_embs, doc_embs, modality_ids, qry_mask, qry_types, idx
            )
        
        return query_embs, doc_embs, modality_ids, qry_mask, qry_types, idx
    
    def _split_by_modality(self, inputs):
        """Separate query_* tensors from document tensors."""
        query_inputs: Dict[str, torch.Tensor] = {}
        doc_inputs: Dict[str, Dict[str, torch.Tensor]] = defaultdict(dict)

        for k, v in inputs.items():
            if k in self.META_KEYS:
                continue
            if k.startswith("query_"):
                query_inputs[k.replace("query_", "")] = v
            else:
                # either "combined_*" or "<mod>_*"
                prefix, field = k.split("_", 1)
                doc_inputs[prefix][field] = v

        return query_inputs, doc_inputs

    
    def _encode_documents(self, model, doc_inputs, full_inputs):
        """
        Returns
        -------
        doc_embs       : torch.Tensor [B,S,D]
        modality_ids   : torch.LongTensor [B,S]
        """
        if "combined" in doc_inputs:
            # -------- combined batch: simply run the encoder once
            emb = model(**doc_inputs["combined"])                # [B,S,D]
            mod_ids = full_inputs["combined_modality_ids"]       # already 1‑based
            return emb, mod_ids

        # -------- separate batch: encode each modality individually
        embs_per_mod = []
        ids_per_mod  = []

        for m in self.modality_types:
            if m not in doc_inputs:
                continue
            out = model(**doc_inputs[m])                         # [B,S_m,D]
            embs_per_mod.append(out)

            # build modality id tensor via attention‑mask
            attn = doc_inputs[m]["attention_mask"]               # [B,S_m]
            ids_per_mod.append(attn * self._modid[m])
        doc_embs     = torch.cat(embs_per_mod,  dim=1)           # [B,∑S_m,D]
        modality_ids = torch.cat(ids_per_mod,   dim=1).long()    # [B,∑S_m]

        return doc_embs, modality_ids

    def _remove_invalid(
        self,
        q_embs: torch.Tensor,          # [B,Q,D]
        d_embs: torch.Tensor,          # [B,S,D]
        mod_ids: torch.Tensor,         # [B,S]
        q_mask: torch.Tensor,          # [B,Q]
        q_types: Optional[torch.Tensor],
        idx: Optional[torch.Tensor],   # [B]  dataset row‑ids
    ):
        """
        Drop rows whose *any* embedding contains NaN / Inf.
        If `self.args.fail_on_invalid` is True, raises a ValueError instead.
        """
        bad = ~torch.isfinite(q_embs.flatten(1)).all(dim=1)
        bad |= ~torch.isfinite(d_embs.flatten(1)).all(dim=1)

        if not bad.any():
            return q_embs, d_embs, mod_ids, q_mask, q_types, idx

        if idx is not None:
            bad_ids = idx[bad].tolist()
        else:
            bad_ids = torch.arange(bad.size(0), device=bad.device)[bad].tolist()

        n_bad = len(bad_ids)
        print(f"[Trainer]  found {n_bad} invalid example(s): {bad_ids}")

        keep = ~bad
        q_embs  = q_embs[keep]
        d_embs  = d_embs[keep]
        mod_ids = mod_ids[keep]
        q_mask  = q_mask[keep]
        q_types = q_types[keep] if q_types is not None else None
        idx     = idx[keep]    if idx is not None    else None

        return q_embs, d_embs, mod_ids, q_mask, q_types, idx