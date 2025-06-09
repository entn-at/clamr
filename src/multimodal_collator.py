from __future__ import annotations

import os
import random
from collections import defaultdict
from typing import Any, Dict, List, Optional

import torch

from models.processing_utils import BaseVisualRetrieverProcessor

def _prefix_keys(d: Dict[str, Any], prefix: str) -> Dict[str, Any]:
    """Return a *new* dict whose keys are prefixed (no in‑place mutation)."""
    return {f"{prefix}{k}": v for k, v in d.items()}

class MultimodalCollator:
    def __init__(
        self,
        processor: BaseVisualRetrieverProcessor,
        max_lengths: Dict[str, int],
        modality_types: Optional[List[str]] = None,
        frames_dir: Optional[str] = None,
        video_dir: Optional[str] = None,
        audio_dir: Optional[str] = None,
        combine_modalities: bool = False,
    ) -> None:
        self.processor = processor
        self.max_lengths = max_lengths
        self.frames_dir = frames_dir
        self.video_dir = video_dir
        self.combine_modalities = combine_modalities
        self.modality_types = modality_types or ["video", "ocr", "asr", "description"]
        self._modality2id = {m: i + 1 for i, m in enumerate(self.modality_types)}
    
    def __call__(self, batch: List[Dict[str, Any]]) -> Dict[str, Any]:
        """
        `batch` is a list of dicts produced by the dataset.__getitem__.
        """

        # Queries
        queries: List[str] = [
            random.choice(example["query"]) if isinstance(example.get("query"), list) and example.get("query") else example.get("query")
            for example in batch
        ]
        # Check if all queries are missing or empty
        queries_missing = all(q is None or (isinstance(q, str) and not q.strip()) for q in queries)
        if not queries_missing:
            query_inputs = self.processor.process_queries(
                queries, max_length=self.max_lengths["query"]
            )

            # each sample already stores the *gold* modality for its query
            query_types: List[str] = [
                "asr" if ex.get("query_type") == "speech" else ex.get("query_type")
                for ex in batch
            ]
            # guard – user error if any modality is unknown
            unknown = {qt for qt in query_types if qt not in self._modality2id}
            # if unknown:
            #     raise ValueError(f"Unknown query_type(s) {unknown} (allowed: {self.modality_types}).")

            query_type_ids = torch.tensor([self._modality2id[qt] if qt in self._modality2id else 0 for qt in query_types], dtype=torch.long)
        else:
            query_inputs = None
            query_type_ids = None

        # Document‑side modalities

        modality_inputs: dict[str, List[Any]] = {m: [] for m in self.modality_types}

        for ex in batch:
            for m in self.modality_types:
                key = "images" if m == "video" else m
                value = ex.get(key)       # may be None

                if value is None or (isinstance(value, list) and len(value) == 0):
                    modality_inputs[m].append(None)   # keep list length == batch
                    continue

                if m == "video":
                    if self.video_dir is not None:
                        video_id = ex.get("video_id")
                        if video_id is not None:
                            value = os.path.join(self.video_dir, video_id + ".mp4")
                        else:
                            value = None
                    else:
                        value = [os.path.join(self.frames_dir or "", f) for f in value]

                if m == "audio" and getattr(self, "audio_dir", None) is not None:
                    # Use audio file path inferred from audio_id or video_id
                    audio_id = ex.get("audio_id") or ex.get("video_id")
                    if audio_id is not None:
                        value = os.path.join(self.audio_dir, audio_id + ".wav")
                    else:
                        value = None

                modality_inputs[m].append(value)
        
        # Encode

        collated: Dict[str, Any] = {}

        if self.combine_modalities:
            # Only include modalities with at least one non-missing input
            present_modalities = [m for m, inputs in modality_inputs.items() if any(x is not None for x in inputs)]
            if present_modalities:
                mod2prefix = {
                    m: (f"{m.upper()}: " if m in {"ocr", "asr"} else f"{m.capitalize()}: ")
                    for m in present_modalities
                }
                mod2maxlen = {m: self.max_lengths.get(m, self.max_lengths["text"]) for m in present_modalities}
                filtered_modality_inputs = {m: modality_inputs[m] for m in present_modalities}
                combined_out = self.processor.process_combined(
                    filtered_modality_inputs, mod2prefix, mod2maxlen, max_length=self.max_lengths["combined"], modality_types=present_modalities
                )
                collated.update(_prefix_keys(combined_out, "combined_"))
        else:
            # keep each modality separate
            for m, inputs in modality_inputs.items():
                if all(x is None for x in inputs):        # modality absent in the *entire* batch
                    continue
                collated.update(_prefix_keys(self._process_single_modality(m, inputs), f"{m}_"))

        if query_inputs is not None:
            collated.update(_prefix_keys(query_inputs, "query_"))
        if query_type_ids is not None:
            collated["query_types"] = query_type_ids
            collated["labels"] = query_type_ids          # dummy target for Trainer

        idx = [ex["idx"] for ex in batch if "idx" in ex]
        if idx:
            collated["idx"] = torch.tensor(idx, dtype=torch.long)
        return collated

    def _process_single_modality(self, modality: str, data: List[Any]) -> Dict[str, torch.Tensor]:
        """Encode one modality (list length == batch size)."""
        if modality == "video":
            return self.processor.process_frames(data)
        if modality == "image":
            return self.processor.process_images(data)
        if modality == "audio":
            return self.processor.process_audio(data)

        # text‑like modalities
        assert modality in {"ocr", "asr", "description"}, modality

        prefix = f"{modality.upper()}: " if modality in {"ocr", "asr"} else f"{modality.capitalize()}: "
        max_len = self.max_lengths.get(modality, self.max_lengths["text"])

        return self.processor.process_text(data, prefix=prefix, max_length=max_len)
