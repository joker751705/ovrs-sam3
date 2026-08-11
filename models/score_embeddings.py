from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


_L2_NORM_EPS = 1e-6
_SCORE_SCALE = 20.0
_NUM_PROMPT_TEMPLATES = 64


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_channels = int(num_channels)
    num_groups = min(8, num_channels)
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, num_channels)


class ClipScoreEmbedding(nn.Module):
    """Build a 36×36 score embedding from RemoteCLIP text scores and features.

    Inputs:
        remoteclip_feat_map: [B, D_clip, 36, 36]
        template text:       [C, 64, D_clip]

    Process:
        1. Compute 64 normalized template score maps.
        2. Project score maps from 64 to 256 channels.
        3. L2-normalize score features and dense CLIP features separately.
        4. Concatenate and fuse them with a 1×1 convolution.
        5. L2-normalize both 256-channel fusion paths separately.
        6. Concatenate and apply two ordinary 3×3 convolutions.

    Outputs:
        clip_score_embed_36: [B, C, 256, 36, 36]
        score_maps_36:       [B, C, 64, 36, 36]
        template_clip_text:  [C, 64, D_clip]
    """

    def __init__(
        self,
        clip_text_encoder,
        prompt_templates: list[str],
        normalize_label: bool = True,
        clip_output_dim: int = 768,
        score_embed_dim: int = 256,
        text_prompt_batch_size: int = 64,
        text_prompt_use_checkpoint: bool = True,
    ):
        super().__init__()

        object.__setattr__(self, "clip_text_encoder", clip_text_encoder)

        self.prompt_templates = list(prompt_templates)
        self.normalize_label = bool(normalize_label)
        self.clip_output_dim = int(clip_output_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_prompt_templates = len(self.prompt_templates)
        self.text_prompt_batch_size = int(text_prompt_batch_size)
        self.text_prompt_use_checkpoint = bool(
            text_prompt_use_checkpoint
        )

        if self.num_prompt_templates != _NUM_PROMPT_TEMPLATES:
            raise ValueError(
                f"Expected {_NUM_PROMPT_TEMPLATES} prompt templates, "
                f"got {self.num_prompt_templates}."
            )

        if self.clip_output_dim <= 0:
            raise ValueError(
                "clip_output_dim must be positive, "
                f"got {self.clip_output_dim}."
            )

        if self.score_embed_dim <= 0:
            raise ValueError(
                "score_embed_dim must be positive, "
                f"got {self.score_embed_dim}."
            )

        # 64-channel template score maps → 256-channel intermediate feature 1.
        self.score_stem = nn.Sequential(
            nn.Conv2d(
                self.num_prompt_templates,
                self.score_embed_dim,
                kernel_size=1,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        # Normalized intermediate feature 1 + normalized dense CLIP feature.
        self.score_clip_fusion = nn.Sequential(
            nn.Conv2d(
                self.score_embed_dim + self.clip_output_dim,
                self.score_embed_dim,
                kernel_size=1,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        # Normalized intermediate feature 1 + normalized intermediate feature 2.
        self.spatial_fusion = nn.Sequential(
            nn.Conv2d(
                self.score_embed_dim * 2,
                self.score_embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
            nn.Conv2d(
                self.score_embed_dim,
                self.score_embed_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.score_embed_dim),
            nn.GELU(),
        )

        self._text_feature_cache: dict[tuple, torch.Tensor] = {}

    def _has_trainable_clip_text_params(self) -> bool:
        return any(
            parameter.requires_grad
            for parameter in self.clip_text_encoder.parameters()
        )

    def _make_text_cache_key(
        self,
        class_names: list[str],
        device: torch.device,
    ) -> tuple:
        return tuple(class_names), str(device)

    def clear_text_cache(self) -> None:
        self._text_feature_cache.clear()

    def _encode_template_text(
        self,
        class_names: list[str],
        device: torch.device,
    ) -> torch.Tensor:
        trainable = self._has_trainable_clip_text_params()
        grad_enabled = torch.is_grad_enabled()
        cache_allowed = (not trainable) or (not grad_enabled)

        if cache_allowed:
            cache_key = self._make_text_cache_key(
                class_names=class_names,
                device=device,
            )
            cached = self._text_feature_cache.get(cache_key)
            if cached is not None:
                return cached.to(device=device)

        result = self.clip_text_encoder.encode_prompt_templates(
            class_names=class_names,
            templates=self.prompt_templates,
            device=device,
            normalize_label=self.normalize_label,
            normalize=False,
            prompt_batch_size=self.text_prompt_batch_size,
            use_checkpoint=self.text_prompt_use_checkpoint,
        )

        if cache_allowed:
            cached = result.detach().contiguous()
            self._text_feature_cache[cache_key] = cached
            return cached.to(device=device)

        return result

    def forward(
        self,
        class_names: list[str],
        remoteclip_feat_map: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        if remoteclip_feat_map.ndim != 4:
            raise ValueError(
                "remoteclip_feat_map must be [B, D_clip, H, W], "
                f"got {tuple(remoteclip_feat_map.shape)}."
            )

        batch_size, image_clip_dim, height, width = (
            remoteclip_feat_map.shape
        )

        if image_clip_dim != self.clip_output_dim:
            raise ValueError(
                "CLIP dimension mismatch: expected "
                f"{self.clip_output_dim}, got {image_clip_dim}."
            )

        if (height, width) != (36, 36):
            raise ValueError(
                "Expected a 36×36 RemoteCLIP feature map, "
                f"got {(height, width)}."
            )

        num_classes = len(class_names)
        if num_classes == 0:
            raise ValueError("class_names is empty.")

        template_clip_text = self._encode_template_text(
            class_names=class_names,
            device=remoteclip_feat_map.device,
        )

        expected_text_shape = (
            num_classes,
            self.num_prompt_templates,
            self.clip_output_dim,
        )
        if tuple(template_clip_text.shape) != expected_text_shape:
            raise ValueError(
                "template_clip_text shape mismatch: expected "
                f"{expected_text_shape}, "
                f"got {tuple(template_clip_text.shape)}."
            )

        template_clip_text = template_clip_text.to(
            device=remoteclip_feat_map.device,
            dtype=remoteclip_feat_map.dtype,
        )

        # Normalize every template text vector along D_clip.
        text_norm = F.normalize(
            template_clip_text,
            p=2,
            dim=-1,
            eps=_L2_NORM_EPS,
        )

        # Normalize every spatial CLIP vector along D_clip.
        # This normalized feature is used both for similarity calculation
        # and for the direct dense CLIP fusion branch.
        image_norm = F.normalize(
            remoteclip_feat_map,
            p=2,
            dim=1,
            eps=_L2_NORM_EPS,
        )

        score_maps_36 = (
            torch.einsum(
                "ckd,bdhw->bckhw",
                text_norm,
                image_norm,
            )
            * _SCORE_SCALE
        )

        score_flat = score_maps_36.reshape(
            batch_size * num_classes,
            self.num_prompt_templates,
            height,
            width,
        )

        # 64 template channels → 256-channel intermediate feature 1.
        score_mid_1 = self.score_stem(score_flat)

        # Perform per-pixel channel-wise L2 normalization before fusion.
        score_mid_1_norm = F.normalize(
            score_mid_1,
            p=2,
            dim=1,
            eps=_L2_NORM_EPS,
        )

        # Broadcast the normalized image-level CLIP feature over classes.
        clip_feature_flat = (
            image_norm[:, None]
            .expand(
                batch_size,
                num_classes,
                self.clip_output_dim,
                height,
                width,
            )
            .reshape(
                batch_size * num_classes,
                self.clip_output_dim,
                height,
                width,
            )
            .contiguous()
        )

        score_clip_input = torch.cat(
            [score_mid_1_norm, clip_feature_flat],
            dim=1,
        )
        score_mid_2 = self.score_clip_fusion(score_clip_input)

        # Normalize both 256-channel paths before the second concatenation.
        score_mid_2_norm = F.normalize(
            score_mid_2,
            p=2,
            dim=1,
            eps=_L2_NORM_EPS,
        )

        spatial_fusion_input = torch.cat(
            [score_mid_1_norm, score_mid_2_norm],
            dim=1,
        )
        clip_score_flat = self.spatial_fusion(
            spatial_fusion_input
        )

        clip_score_embed_36 = clip_score_flat.reshape(
            batch_size,
            num_classes,
            self.score_embed_dim,
            height,
            width,
        ).contiguous()

        return (
            clip_score_embed_36,
            score_maps_36.contiguous(),
            template_clip_text.contiguous(),
        )
