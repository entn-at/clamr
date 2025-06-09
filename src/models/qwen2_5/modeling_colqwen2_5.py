from typing import ClassVar, List, Optional

import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen2_5_vl import Qwen2_5_VLConfig, Qwen2_5_VLForConditionalGeneration


class ColQwen2_5(Qwen2_5_VLForConditionalGeneration):  # noqa: N801
    """
    ColQwen2.5 model implementation, following the achitecture from the article "ColPali: Efficient Document Retrieval
    with Vision Language Models" paper. Based on the Qwen2.5-VL backbone.

    Args:
        config (Qwen2.5VLConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore all tokens embeddings
            except those of the image at inference.
            Defaults to False --> Do not mask any embeddings during forward pass.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"  # transformers-related

    def __init__(self, config: Qwen2_5_VLConfig, mask_non_image_embeddings: bool = False):
        super().__init__(config=config)
        self.dim = 128
        self.custom_text_proj = nn.Linear(self.model.config.hidden_size, self.dim)
        self.padding_side = "left"
        self.mask_non_image_embeddings = mask_non_image_embeddings
        
        self.post_init()
    
    def forward(self, *args, **kwargs) -> torch.Tensor:
        kwargs.pop("output_hidden_states", None)

        compute_modality_labels = kwargs.pop("compute_modality_labels", False)

        # Handle the custom "pixel_values" input obtained with `ColQwen2Processor` through unpadding
        if "pixel_values" in kwargs:
            offsets = kwargs["image_grid_thw"].prod(dim=1)  # (batch_size,)
            kwargs["pixel_values"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values"], offsets)],
                dim=0,
            )

        if "pixel_values_videos" in kwargs:
            offsets = kwargs["video_grid_thw"].prod(dim=1)  # (batch_size,)
            kwargs["pixel_values_videos"] = torch.cat(
                [pixel_sequence[:offset] for pixel_sequence, offset in zip(kwargs["pixel_values_videos"], offsets)],
                dim=0,
            )
        
        outputs = super().forward(*args, output_hidden_states=True, **kwargs)  # (batch_size, sequence_length, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]  # (batch_size, sequence_length, hidden_size)

        proj = self.custom_text_proj(last_hidden_states)  # (batch_size, sequence_length, dim)

        # L2 normalization
        proj = F.normalize(proj, p=2, dim=-1)  # (batch_size, sequence_length, dim)
        proj = proj * kwargs["attention_mask"].unsqueeze(-1)  # (batch_size, sequence_length, dim)

        if "pixel_values" in kwargs and self.mask_non_image_embeddings:
            # Pools only the image embeddings
            image_mask = (kwargs["input_ids"] == self.config.image_token_id).unsqueeze(-1)
            proj = proj * image_mask
        
        if "pixel_values_videos" in kwargs and self.mask_non_image_embeddings:
            # Pools only the video embeddings
            video_mask = (kwargs["input_ids"] == self.config.video_token_id).unsqueeze(-1)
            proj = proj * video_mask
        
        return proj
