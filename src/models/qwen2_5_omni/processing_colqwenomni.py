from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.qwen2_5_omni import Qwen2_5OmniProcessor

from ..processing_utils import BaseVisualRetrieverProcessor
from qwen_omni_utils import process_mm_info

import json


class ColQwenOmniProcessor(BaseVisualRetrieverProcessor, Qwen2_5OmniProcessor):
    """
    Multi‑modal processor tailored for ColQwenOmni + late‑interaction retrievers,
    with support for vision, text, OCR, ASR and raw audio inputs.
    """
    # ---- static prompt pieces ------------------------------------------------
    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|>"
        "<|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"

    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"

        # cache special‑token ids
        self._t_id = {
            "vision_start": self.tokenizer.convert_tokens_to_ids(self.vision_bos_token),
            "vision_end":   self.tokenizer.convert_tokens_to_ids(self.vision_eos_token),
            "image_pad":    self.tokenizer.convert_tokens_to_ids(self.image_token),
            "audio_pad":    self.tokenizer.convert_tokens_to_ids(self.audio_token),
            "video_pad":    self.tokenizer.convert_tokens_to_ids(self.video_token),
            "user":         self.tokenizer.convert_tokens_to_ids("user"),
        }
        self._image_token_id = self._t_id["image_pad"]
        self._audio_token_id = self._t_id["audio_pad"]
        self._video_token_id = self._t_id["video_pad"]
    def process_combined(
        self,
        inputs: Dict[str, List[Any]],
        modality2prefix: Optional[Dict[str, str]] = None,
        modality2length: Optional[Dict[str, int]] = None,
        max_length: int = 2048,
        modality_types: Tuple[str, ...] = (
            "video", "image", "ocr", "asr", "description", "audio",
        ),
    ) -> BatchFeature:
        """
        Concatenate all modalities (including raw audio) of each sample into ONE chat turn.
        Returns a BatchFeature with fields:
          - input_ids / attention_mask / pixel_values(+videos) / input_features
          - modality_ids  (LongTensor [B,seq])
        """
        modality2prefix = modality2prefix or {}
        modality2length = modality2length or {}
        inputs = {k: list(v) for k, v in inputs.items()}
        B = len(next(iter(inputs.values())))
        # 1. build chat messages
        messages, modality_lens = [], [{} for _ in range(B)]
        for i in range(B):
            content: List[Dict[str, Any]] = []
            # vision first
            if "video" in inputs and "video" in modality_types:
                content.append(
                    {"type": "text", "text": "Video: "}
                )
                content.append(
                    {
                        "type": "video",
                        "video": inputs["video"][i],
                        "nframes": 10,
                        #"fps": 1,
                        "max_pixels": 224*224
                    }
                )
            if "image" in inputs and "image" in modality_types:
                content.append({"type": "image", "image": inputs["image"][i]})
            # raw audio
            if "audio" in inputs and "audio" in modality_types and inputs["audio"][i] is not None:
                content.append({"type": "text", "text": "Audio: "})
                content.append({"type": "audio", "audio": inputs["audio"][i]})
            # text‑like modalities
            for mod in modality_types:
                if mod in {"video", "image", "audio"}:
                    continue
                if mod not in inputs:
                    continue
                lst = inputs[mod]
                prefix = modality2prefix.get(mod, f"{mod.upper()}: " if mod in {"ocr", "asr"} else f"{mod.capitalize()}: ")
                txt = prefix + lst[i]
                enc = self.tokenizer(txt, truncation=True, max_length=modality2length.get(mod, max_length))
                txt = self.tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
                content.append({"type": "text", "text": " "+txt})
                modality_lens[i][mod] = len(enc["input_ids"])
            messages.append(
                [
                    {
                        "role": "system",
                        "content": [
                            {"type": "text", "text": "You are Qwen, a virtual human developed by the Qwen Team, Alibaba Group, capable of perceiving auditory and visual inputs, as well as generating text and speech."}
                        ],
                    },
                    {
                        "   role": "user",
                        "content": content
                    }
                ]
            )
        # print(messages)
        # 2. run upstream processor
        text_prompt = self.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        audio_inputs, img_inputs, vid_inputs = process_mm_info(messages, use_audio_in_video=False)
        # print(audio_inputs[0].shape)
        batch = self(
            text=text_prompt,
            images=img_inputs,
            videos=vid_inputs,
            audio=audio_inputs,
            max_length=modality2length.get("combined", max_length),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            use_audio_in_video=False,
            # videos_kwargs={"fps": 1},
            # padding=True,
        )

        # ! Currently facing bugs where feature_attention_mask is larger than sequence length and so need to truncate
        # print(batch["feature_attention_mask"].shape, batch["input_features"].shape)
        # print(batch["feature_attention_mask"])
        if "feature_attention_mask" in batch and  batch["feature_attention_mask"].shape[1] > batch["input_features"].shape[2]:
            batch["feature_attention_mask"] = batch["feature_attention_mask"][:,:batch["input_features"].shape[2]]
        
        # 3. pad vision tensors to equal length
        self._pad_vision(batch, "video")
        self._pad_vision(batch, "image")
    
        # 4. build modality_ids
        batch["modality_ids"] = self._build_modality_ids(
            batch, modality_lens, vid_inputs, img_inputs, modality_types
        )

        return batch

    def _pad_vision(self, batch: BatchFeature, vision_kind: str) -> None:
        """Pads pixel_values_(videos) or pixel_values to same S per batch row."""
        key_grid = f"{vision_kind}_grid_thw"
        key_pix  = "pixel_values_videos" if vision_kind == "video" else "pixel_values"
        if key_grid not in batch:
            return

        offsets = batch[key_grid].prod(dim=1)          # (B,)
        pieces  = torch.split(batch[key_pix], offsets.tolist())
        batch[key_pix] = torch.nn.utils.rnn.pad_sequence(
            list(pieces), batch_first=True
        )
    
    def _build_modality_ids(
        self,
        batch: BatchFeature,
        modality_lens: List[Dict[str, int]],
        vid_inputs: Any,
        img_inputs: Any,
        modality_types: Tuple[str, ...],
    ) -> torch.LongTensor:
        B, S = batch["input_ids"].shape
        device = batch["input_ids"].device
        M = torch.zeros((B, S), dtype=torch.long, device=device)
        _mod2id = {m: i+1 for i, m in enumerate(modality_types)}
        tid = self._t_id
        # video and image same as ColQwen
        if vid_inputs is not None:
            vs, ve = tid["vision_start"], tid["vision_end"]
            in_vid = (batch["input_ids"]==vs).cumsum(1) - (batch["input_ids"]==ve).cumsum(1)
            M[in_vid>0] = _mod2id["video"]
        if img_inputs is not None:
            is_, ie = tid["vision_start"], tid["vision_end"]  # reuse vision markers
            in_img = (batch["input_ids"]==is_).cumsum(1) - (batch["input_ids"]==ie).cumsum(1)
            M[in_img>0] = _mod2id["image"]
        # raw audio: label each audio placeholder token
        audio_mask = batch["input_ids"] == self._audio_token_id
        M[audio_mask] = _mod2id.get("audio", 0)
        # text modalities
        for i in range(B):
            ids_i = batch["input_ids"][i]
            # determine ptr after vision/audio blocks
            ptr = 0
            for tok in [tid.get("vision_end"), tid.get("audio_pad")]:
                if tok is not None and (ids_i==tok).any():
                    ptr = max(ptr, (ids_i==tok).nonzero()[0,0]+1)
            if ptr==0:
                ptr = (ids_i==tid["user"]).nonzero()[0,0]+1
            for mod in modality_types:
                if mod in {"video","image","audio"}:
                    continue
                ln = modality_lens[i].get(mod)
                if ln is None:
                    continue
                M[i, ptr:ptr+ln] = _mod2id[mod]
                ptr += ln
        return M

    def process_frames(self, frames: List[List[Image.Image]], context_prompts: Optional[List[str]] = None) -> BatchFeature:
        messages = []
        for frame in frames:
            content = [
                {
                        "type": "video",
                        "video": frame,
                        "fps": 1,
                        "max_pixels": 224 * 224,
                }
            ]
            messages.append([{"role": "user", "content": content}])
        text = self.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        audio_inputs,image_inputs, video_inputs = process_mm_info(messages)
        inputs = self(
            text=text,
            audio=audio_inputs,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            use_audio_in_video=False,
        )

        # NOTE: The following adjustment ensures correct behavior with DDP on multiple GPUs.
        offsets = inputs["video_grid_thw"].prod(dim=1) # (batch_size,)

        # Split the pixel_values tensor into a list of tensors, one per image
        pixel_values = list(
            torch.split(inputs["pixel_values_videos"], offsets.tolist())
        )

        # # Pad the list of pixel_value tensors to the same length along the sequence dimension
        inputs["pixel_values_videos"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )

        return inputs
    
    def process_images(self, images: List[Image.Image], context_prompts: Optional[List[str]] = None) -> BatchFeature:
        """
        Process images for ColQwen2.5.

        Args:
            images: List of PIL images.
            context_prompts: List of optional context prompts, i.e. some text description of the context of the image.
        """
        messages = [
            [
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "image",
                            "image": image,
                        },
                        {
                            "type": "text",
                            "text": "Describe this image."
                        },
                    ],
                }
            ] for image in images
        ]

        text = self.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_mm_info(messages)
        inputs = self(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            truncation=True,
        )

        # NOTE: The following adjustment ensures correct behavior with DDP on multiple GPUs.
        offsets = inputs["image_grid_thw"].prod(dim=1) # (batch_size,)

        # Split the pixel_values tensor into a list of tensors, one per image
        pixel_values = list(
            torch.split(inputs["pixel_values"], offsets.tolist())
        )  # [(num_patches_image_0, pixel_values), ..., (num_patches_image_n, pixel_values)]

        # Pad the list of pixel_value tensors to the same length along the sequence dimension
        inputs["pixel_values"] = torch.nn.utils.rnn.pad_sequence(
            pixel_values, batch_first=True
        )  # (batch_size, max_num_patches, pixel_values)

        return inputs
    
    def process_queries(
        self,
        queries: List[str],
        max_length: int = 50,
        suffix: Optional[str] = None,
    ) -> BatchFeature:
        if suffix is None:
            suffix = self.query_augmentation_token * 10
        texts_query: List[str] = []

        for query in queries:
            query = self.query_prefix + query + suffix
            texts_query.append(query)

        # needs to be padded to the same length as we need to gather all the queries and attention masks from all the GPUs
        batch_query = self(
            text=texts_query,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        return batch_query
    
    def process_text(
        self,
        texts: List[str],
        max_length: int = 128,
        prefix: Optional[str] = None,
    ) -> BatchFeature:
        texts_query: List[str] = []

        for text in texts:
            if prefix is not None:
                text = prefix + text
            texts_query.append(text)
        # needs to be padded to the same length as we need to gather all the queries and attention masks from all the GPUs
        batch_query = self(
            text=texts_query,
            return_tensors="pt",
            padding="max_length",
            max_length=max_length,
            truncation=True,
        )

        return batch_query
    
    def score(
        self,
        qs: List[torch.Tensor],
        ps: List[torch.Tensor],
        device: Optional[Union[str, torch.device]] = None,
        **kwargs,
    ) -> torch.Tensor:
        """
        Compute the MaxSim score (ColBERT-like) for the given multi-vector query and passage embeddings.
        """
        return self.score_multi_vector(qs, ps, device=device, **kwargs)

    def get_n_patches(
        self,
        image_size: Tuple[int, int],
        patch_size: int,
        spatial_merge_size: int,
    ) -> Tuple[int, int]:
        """
        Get the number of patches (n_patches_x, n_patches_y) that will be used to process an image of
        size (height, width) with the given patch size.

        The `spatial_merge_size` is the number of patches that will be merged spatially. It is stored in
        as a `Qwen2VLForConditionalGeneration` attribute under `model.spatial_merge_size`.
        """
        height_new, width_new = smart_resize(
            width=image_size[0],
            height=image_size[1],
            factor=self.factor,
            min_pixels=self.min_pixels,
            max_pixels=self.max_pixels,
        )

        n_patches_x = width_new // patch_size // spatial_merge_size
        n_patches_y = height_new // patch_size // spatial_merge_size

        return n_patches_x, n_patches_y

    def get_image_mask(self, batch_images: BatchFeature) -> torch.Tensor:
        return batch_images.input_ids == self.image_token_id
