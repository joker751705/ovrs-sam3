from __future__ import annotations

from typing import List

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

from .score_embeddings import ClipScoreEmbedding
from .encoder_refiner_attention import (
    EncoderRefinerLayer,
    apply_layer_norm_bcdhw,
)
from .refiner_pyramid_decoder import RefinerPyramidDecoder


class ClassConditionedEncoderRefiner(nn.Module):
    """Encoder feature refiner operating at 36×36.

    Takes the full 6-layer SAM3 encoder output after prompt cross-attention
    as input. The feature stream is the bilinear-downsampled encoder feature
    without any FPN injection. The score stream comes directly from
    ClipScoreEmbedding (score_embeddings.py) without any SAM3 FPN fusion.

    The Refiner outputs 36×36 features. High-resolution decoding is handled
    by RefinerPyramidDecoder: three-stage semantic–detail dual-branch
    fusion (72→144→288), where the semantic branch fuses upsampled Refiner
    with frozen Pixel Decoder features and the detail branch fuses Refiner
    with original SAM3 backbone FPN. Both branches use independent Refiner
    projections. stage_288 output goes directly into the frozen
    semantic_seg_head.

    After all Refiner layers, the accumulated 36×36 feature stream is
    normalized once with a channel-wise LayerNorm before being returned
    and passed to RefinerPyramidDecoder. The final score stream is not
    post-normalized.

    Forward inputs:
        encoder_features_72:  [B, C, 256, 72, 72]  (full encoder + cross-attention)
        clip_image_feat_map:  [B, D_clip, 36, 36]
        sam_text_mean:        [B, C, 256]
        class_names:          list of C class names

    Forward outputs:
        refiner_features_36:  [B, C, 256, 36, 36]
        score_embed_36:       [B, C, 256, 36, 36]
        clip_score_embed_36:  [B, C, 256, 36, 36]
        clip_score_maps_36:   [B, C,  64, 36, 36]
        template_clip_text:   [C, 64, D_clip]
    """

    def __init__(
        self,
        clip_text_encoder,
        hidden_dim: int = 256,
        clip_dim: int = 768,
        score_embed_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 12,
        shift_size: int = 6,
        fusion_layers: int = 4,
        dropout: float = 0.1,
        prompt_templates: list[str] | None = None,
        normalize_label_for_clip: bool = True,
        use_checkpoint: bool = True,
        text_prompt_batch_size: int = 64,
        text_prompt_use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.use_checkpoint = bool(use_checkpoint)
        self.num_fusion_layers = int(fusion_layers)

        if prompt_templates is None:
            raise ValueError(
                "prompt_templates must be a list of 64 prompt templates."
            )
        if len(prompt_templates) != 64:
            raise ValueError(
                f"Expected 64 prompt templates, got {len(prompt_templates)}."
            )

        self.clip_score_embed = ClipScoreEmbedding(
            clip_text_encoder=clip_text_encoder,
            prompt_templates=list(prompt_templates),
            normalize_label=bool(normalize_label_for_clip),
            clip_output_dim=int(clip_dim),
            score_embed_dim=int(score_embed_dim),
            text_prompt_batch_size=int(text_prompt_batch_size),
            text_prompt_use_checkpoint=bool(text_prompt_use_checkpoint),
        )

        self.layers = nn.ModuleList([
            EncoderRefinerLayer(
                hidden_dim=self.hidden_dim,
                score_embed_dim=self.score_embed_dim,
                num_heads=int(num_heads),
                window_size=int(window_size),
                shift_size=int(shift_size),
                dropout=float(dropout),
            )
            for _ in range(self.num_fusion_layers)
        ])

        # Final normalization for the accumulated feature residual stream.
        # This is applied once after all Refiner layers and before the
        # 36×36 feature enters the Pyramid Decoder.
        self.feature_output_norm = nn.LayerNorm(self.hidden_dim)

        self.pyramid_decoder = RefinerPyramidDecoder(
            hidden_dim=self.hidden_dim,
            branch_dim=128,
            use_checkpoint=self.use_checkpoint,
        )

    def decode_feature_pyramid_chunk(
        self,
        refiner_feature_36: torch.Tensor,
        original_pixel_feature_72: torch.Tensor,
        original_pixel_feature_144: torch.Tensor,
        original_pixel_feature_288: torch.Tensor,
        sam_fpn_72: torch.Tensor,
        sam_fpn_144: torch.Tensor,
        sam_fpn_288: torch.Tensor,
    ) -> torch.Tensor:
        """Three-stage semantic–detail dual-branch upsampling with
        independent Refiner projections per branch.

        Args:
            refiner_feature_36:          [B×C_chunk, 256, 36, 36]
            original_pixel_feature_72:   [B×C_chunk, 256, 72, 72]
            original_pixel_feature_144:  [B×C_chunk, 256, 144, 144]
            original_pixel_feature_288:  [B×C_chunk, 256, 288, 288]
            sam_fpn_72:                  [B, 256, 72, 72]
            sam_fpn_144:                 [B, 256, 144, 144]
            sam_fpn_288:                 [B, 256, 288, 288]

        Returns:
            final_feature_288: [B×C_chunk, 256, 288, 288]
        """
        return self.pyramid_decoder(
            refiner_feature_36=refiner_feature_36,
            original_pixel_feature_72=original_pixel_feature_72,
            original_pixel_feature_144=original_pixel_feature_144,
            original_pixel_feature_288=original_pixel_feature_288,
            sam_fpn_72=sam_fpn_72,
            sam_fpn_144=sam_fpn_144,
            sam_fpn_288=sam_fpn_288,
        )

    def forward(
        self,
        encoder_features_72: torch.Tensor,
        clip_image_feat_map: torch.Tensor,
        sam_text_mean: torch.Tensor,
        class_names: List[str],
    ) -> dict[str, torch.Tensor]:
        """Run the full 36×36 Refiner on all classes simultaneously.

        Args:
            encoder_features_72:
                [B, C, 256, 72, 72] full encoder output after
                prompt cross-attention.
            clip_image_feat_map:
                [B, D_clip, 36, 36] dense RemoteCLIP feature map.
            sam_text_mean:
                [B, C, 256] mean SAM3 text features.
            class_names:
                List containing C prompt names.

        Returns:
            refiner_features_36:
                [B, C, 256, 36, 36] feature stream after all Refiner
                layers and the final channel-wise LayerNorm.
            score_embed_36:
                [B, C, 256, 36, 36] score stream after all Refiner layers.
            clip_score_embed_36:
                [B, C, 256, 36, 36] initial pure RemoteCLIP score embedding.
            clip_score_maps_36:
                [B, C, 64, 36, 36]
            template_clip_text:
                [C, 64, D_clip]
        """
        if encoder_features_72.ndim != 5:
            raise ValueError(
                "encoder_features_72 must be [B, C, D, 72, 72], "
                f"got {tuple(encoder_features_72.shape)}."
            )

        batch_size, num_classes, hidden_dim, height, width = (
            encoder_features_72.shape
        )

        if hidden_dim != self.hidden_dim:
            raise ValueError(
                "encoder_features_72 hidden_dim mismatch: expected "
                f"{self.hidden_dim}, got {hidden_dim}."
            )

        if (height, width) != (72, 72):
            raise ValueError(
                "ClassConditionedEncoderRefiner expects 72×72 "
                f"encoder features, got {(height, width)}."
            )

        if clip_image_feat_map.ndim != 4:
            raise ValueError(
                "clip_image_feat_map must be [B, D_clip, 36, 36], "
                f"got {tuple(clip_image_feat_map.shape)}."
            )

        if clip_image_feat_map.shape[0] != batch_size:
            raise ValueError(
                "clip_image_feat_map batch mismatch: expected "
                f"{batch_size}, got {clip_image_feat_map.shape[0]}."
            )

        if tuple(clip_image_feat_map.shape[-2:]) != (36, 36):
            raise ValueError(
                "clip_image_feat_map must be 36×36, "
                f"got {tuple(clip_image_feat_map.shape[-2:])}."
            )

        expected_text_shape = (
            batch_size,
            num_classes,
            hidden_dim,
        )
        if tuple(sam_text_mean.shape) != expected_text_shape:
            raise ValueError(
                "sam_text_mean shape mismatch: expected "
                f"{expected_text_shape}, "
                f"got {tuple(sam_text_mean.shape)}."
            )

        (
            clip_score_embed_36,
            clip_score_maps_36,
            template_clip_text,
        ) = self.clip_score_embed(
            class_names=class_names,
            remoteclip_feat_map=clip_image_feat_map,
        )

        base_feature_36 = F.interpolate(
            encoder_features_72.reshape(
                batch_size * num_classes,
                hidden_dim,
                72,
                72,
            ),
            size=(36, 36),
            mode="bilinear",
            align_corners=False,
        ).reshape(
            batch_size,
            num_classes,
            hidden_dim,
            36,
            36,
        )

        feature_36 = base_feature_36

        # The score embedding produced by score_embeddings.py is now used
        # directly. No SAM3 FPN injection is performed before attention.
        score_embed_36 = clip_score_embed_36

        for layer in self.layers:
            if self.use_checkpoint and self.training:
                feature_36, score_embed_36 = checkpoint(
                    layer,
                    feature_36,
                    score_embed_36,
                    sam_text_mean,
                    use_reentrant=False,
                )
            else:
                feature_36, score_embed_36 = layer(
                    feature_36=feature_36,
                    score_embed_36=score_embed_36,
                    sam_text_mean=sam_text_mean,
                )

        # Normalize the accumulated feature stream once after all Refiner layers.
        feature_36 = apply_layer_norm_bcdhw(
            feature_36,
            self.feature_output_norm,
        )

        return {
            "refiner_features_36": feature_36,
            "score_embed_36": score_embed_36,
            "clip_score_embed_36": clip_score_embed_36,
            "clip_score_maps_36": clip_score_maps_36,
            "template_clip_text": template_clip_text,
        }
