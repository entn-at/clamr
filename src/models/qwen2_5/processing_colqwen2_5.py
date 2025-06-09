from __future__ import annotations

from typing import Any, ClassVar, Dict, List, Optional, Tuple, Union

import torch
from PIL import Image
from transformers import BatchFeature
from transformers.models.qwen2_vl import Qwen2VLProcessor
from transformers.models.qwen2_vl.image_processing_qwen2_vl import smart_resize

from ..processing_utils import BaseVisualRetrieverProcessor
from qwen_vl_utils import process_vision_info


class ColQwen2_5Processor(BaseVisualRetrieverProcessor, Qwen2VLProcessor):
    """
    Multi‑modal processor tailored for ColQwen2.5 + late‑interaction retrievers.
    """

    # ---- static prompt pieces ------------------------------------------------
    visual_prompt_prefix: ClassVar[str] = (
        "<|im_start|>user\n<|vision_start|><|image_pad|>"
        "<|vision_end|>Describe the image.<|im_end|><|endoftext|>"
    )
    query_prefix: ClassVar[str] = "Query: "
    query_augmentation_token: ClassVar[str] = "<|endoftext|>"
    image_token: ClassVar[str] = "<|image_pad|>"

    # ---- init ----------------------------------------------------------------
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.tokenizer.padding_side = "left"

        # cache special‑token ids
        self._t_id = {
            "vision_start": self.tokenizer.convert_tokens_to_ids("<|vision_start|>"),
            "vision_end":   self.tokenizer.convert_tokens_to_ids("<|vision_end|>"),
            "video_pad":    self.tokenizer.convert_tokens_to_ids("<|video_pad|>"),
            "image_pad":    self.tokenizer.convert_tokens_to_ids("<|image_pad|>"),
            "im_start":     self.tokenizer.convert_tokens_to_ids("<|im_start|>"),
            "im_end":       self.tokenizer.convert_tokens_to_ids("<|im_end|>"),
            "user":         self.tokenizer.convert_tokens_to_ids("user"),
        }
        self._image_token_id = self.tokenizer.convert_tokens_to_ids(self.image_token)

    # property kept for backward‑compat
    @property
    def image_token_id(self) -> int:  # noqa: D401
        return self._image_token_id
    
    def process_combined(
        self,
        inputs: Dict[str, List[Any]],
        modality2prefix: Optional[Dict[str, str]] = None,
        modality2length: Optional[Dict[str, int]] = None,
        max_length: int = 2048,
        modality_types: Tuple[str, ...] = ("video", "image", "ocr", "asr", "description"),
    ) -> BatchFeature:
        """
        Concatenate all modalities of each sample into ONE chat turn.
        Returns
        -------
        BatchFeature with fields:
            - input_ids / attention_mask / pixel_values(+videos)  (as usual)
            - modality_ids  (LongTensor [B,seq])
        """
        modality2prefix = modality2prefix or {}
        modality2length = modality2length or {}

        inputs = {k: list(v) for k, v in inputs.items()}  # shallow copy
        B = len(next(iter(inputs.values())))

        messages, modality_lens = [], [{} for _ in range(B)]
        for i in range(B):
            content: List[Dict[str, Any]] = []

            # vision first
            if "video" in inputs and "video" in modality_types:
                content.append(
                    {
                        "type": "video",
                        "video": inputs["video"][i],
                        "fps": 1,
                        "max_pixels": 224 * 224,
                    }
                )

            if "image" in inputs and "image" in modality_types:
                content.append({"type": "image", "image": inputs["image"][i]})

            # text‑like modalities
            for mod in modality_types:
                if mod in {"video", "image"}:
                    continue
                if mod not in inputs:
                    continue
                lst = inputs[mod]
                prefix = modality2prefix.get(mod, f"{mod.upper()}: " if mod in {"ocr", "asr"} else f"{mod.capitalize()}: ")
                txt = prefix + lst[i]
                enc = self.tokenizer(txt, truncation=True, max_length=modality2length.get(mod, max_length))
                txt = self.tokenizer.decode(enc["input_ids"], skip_special_tokens=True)
                content.append({"type": "text", "text": " " + txt})
                modality_lens[i][mod] = len(enc["input_ids"])

            messages.append([{"role": "user", "content": content}])
        
        text_prompt = self.apply_chat_template(messages, tokenize=False, add_generation_prompt=False)
        img_inputs, vid_inputs = process_vision_info(messages)
        batch = self(
            text=text_prompt,
            images=img_inputs,
            videos=vid_inputs,
            max_length=modality2length.get("combined", max_length),
            truncation=True,
            padding="max_length",
            return_tensors="pt",
            videos_kwargs={"fps": 1},
        )

        self._pad_vision(batch, "video")
        self._pad_vision(batch, "image")

        batch["modality_ids"] = self._build_modality_ids(batch, modality_lens, vid_inputs, img_inputs, modality_types)
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
        """
        Return a [B, S] LongTensor where each token is labelled with its modality id
        (1‑based, order taken from `modality_types`; 0 = padding / system).
        """

        B, S = batch["input_ids"].shape
        device = batch["input_ids"].device
        M = torch.zeros((B, S), dtype=torch.long, device=device)  # default 0

        _mod2id = {m: i + 1 for i, m in enumerate(modality_types)}
        tid = self._t_id

        if vid_inputs is not None:
            vs, ve = tid["vision_start"], tid["vision_end"]
            in_vid = (batch["input_ids"] == vs).float().cumsum(dim=1) - (
                batch["input_ids"] == ve
            ).float().cumsum(dim=1)
            M[in_vid > 0] = _mod2id["video"]

        if img_inputs is not None:
            is_, ie = tid["im_start"], tid["im_end"]
            in_img = (batch["input_ids"] == is_).float().cumsum(dim=1) - (
                batch["input_ids"] == ie
            ).float().cumsum(dim=1)
            M[in_img > 0] = _mod2id["image"]

        for i in range(B):
            ids_i = batch["input_ids"][i]

            # a) start pointer = first position AFTER video/image block(s)
            ptr = 0
            if vid_inputs is not None:
                ptr = max(ptr, (ids_i == tid["vision_end"]).nonzero(as_tuple=False)[0, 0] + 1)
            if img_inputs is not None:
                ptr = max(ptr, (ids_i == tid["im_end"]).nonzero(as_tuple=False)[0, 0] + 1)
            if ptr == 0:  # no vision → start just after "user"
                ptr = (ids_i == tid["user"]).nonzero(as_tuple=False)[0, 0] + 1

            # b) walk through modality_types order and assign ids
            for mod in modality_types:
                if mod in {"video", "image"}:
                    continue
                ln = modality_lens[i].get(mod)
                if ln is None:
                    continue
                M[i, ptr : ptr + ln] = _mod2id[mod]
                ptr += ln  # advance

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
                # {"type": "text", "text": "Describe this video."}
            ]
            messages.append([{"role": "user", "content": content}])
        text = self.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        image_inputs, video_inputs = process_vision_info(messages)
        inputs = self(
            text=text,
            images=image_inputs,
            videos=video_inputs,
            padding=True,
            return_tensors="pt",
            videos_kwargs= {"fps": 1},
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
        image_inputs, video_inputs = process_vision_info(messages)
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
