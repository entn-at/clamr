from typing import ClassVar, Optional, List
import torch
from torch import nn
import torch.nn.functional as F
from transformers.models.qwen2_5_omni import Qwen2_5OmniConfig, Qwen2_5OmniForConditionalGeneration


class ColQwenOmni(Qwen2_5OmniForConditionalGeneration):
    """
    ColQwenOmni model implementation for retrieval, adapting the ColQwen2.5 structure to Qwen2.5-Omni.

    Args:
        config (Qwen2_5OmniConfig): The model configuration.
        mask_non_image_embeddings (Optional[bool]): Whether to ignore tokens unrelated to the active modality.
            Defaults to False.
    """

    main_input_name: ClassVar[str] = "doc_input_ids"

    def __init__(
        self,
        config: Qwen2_5OmniConfig,
        mask_non_image_embeddings: bool = False,
        dim: int = 128,
    ):
        super().__init__(config)
        # Projection dimension
        self.dim = dim
        # Linear projection from hidden_size to dim
        hidden_size = self.thinker.config.text_config.hidden_size
        self.custom_text_proj = nn.Linear(hidden_size, self.dim)
        # Follow left padding convention
        self.padding_side = "left"
        # Control masking of non-active modalities
        self.mask_non_image_embeddings = mask_non_image_embeddings
        # Initialize weights and biases
        self.post_init()

    def forward(self, *args, **kwargs) -> torch.Tensor:
        # Ensure hidden states are returned
        kwargs.pop("output_hidden_states", None)
        # Handle modality-specific labels if used downstream
        kwargs.pop("compute_modality_labels", None)

        # Process inputs through the thinker backbone
        outputs = self.thinker.forward(
            *args,
            output_hidden_states=True,
            **kwargs,
        )
        # Extract last hidden layer: (batch_size, seq_len, hidden_size)
        last_hidden_states = outputs.hidden_states[-1]

        # Project and normalize embeddings
        proj = self.custom_text_proj(last_hidden_states)  # → (B, L, dim)
        proj = F.normalize(proj, p=2, dim=-1)
        # Apply attention mask to zero-out padding embeddings
        attention_mask = kwargs.get("attention_mask")
        if attention_mask is not None:
            proj = proj * attention_mask.unsqueeze(-1)

        # Optionally mask out tokens not part of the requested modality
        if self.mask_non_image_embeddings and "input_ids" in kwargs:
            input_ids = kwargs["input_ids"]
            cfg = self.thinker.config
            # Image tokens
            if kwargs.get("pixel_values") is not None:
                mask = (input_ids == cfg.image_token_id).unsqueeze(-1)
                proj = proj * mask
            # Video tokens
            if kwargs.get("pixel_values_videos") is not None:
                mask = (input_ids == cfg.video_token_id).unsqueeze(-1)
                proj = proj * mask
            # Audio tokens
            if kwargs.get("input_features") is not None:
                mask = (input_ids == cfg.audio_token_id).unsqueeze(-1)
                proj = proj * mask

        return proj