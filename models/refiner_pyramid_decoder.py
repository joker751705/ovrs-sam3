from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, int(num_channels))
    if int(num_channels) % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, int(num_channels))


class SemanticDetailFusionStage(nn.Module):
    """Single-scale dual-branch fusion: semantic (Refiner + Pixel Decoder)
    and detail (Refiner + original SAM3 FPN).

    Both branches operate in 128-channel compact space with independent
    standard 3×3 conv blocks. Each branch uses its own dedicated Refiner
    projection. After projecting back to 256 channels, the two branches
    are summed with equal weight and passed through a final 1×1 Conv.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        branch_dim: int = 128,
    ):
        super().__init__()
        if hidden_dim <= 0:
            raise ValueError(f"hidden_dim must be > 0, got {hidden_dim}")
        if branch_dim <= 0:
            raise ValueError(f"branch_dim must be > 0, got {branch_dim}")

        self.hidden_dim = int(hidden_dim)
        self.branch_dim = int(branch_dim)

        # Four independent 256→128 projections (no activation).
        self.semantic_refiner_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.branch_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.branch_dim),
        )

        self.detail_refiner_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.branch_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.branch_dim),
        )

        self.pixel_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.branch_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.branch_dim),
        )

        self.fpn_proj = nn.Sequential(
            nn.Conv2d(self.hidden_dim, self.branch_dim, kernel_size=1, bias=False),
            _safe_group_norm(self.branch_dim),
        )

        self.semantic_block = self._make_branch_block()
        self.detail_block = self._make_branch_block()

        self.semantic_out_proj = nn.Conv2d(
            self.branch_dim, self.hidden_dim, kernel_size=1, bias=False,
        )
        self.detail_out_proj = nn.Conv2d(
            self.branch_dim, self.hidden_dim, kernel_size=1, bias=False,
        )

        self.fusion_out_proj = nn.Conv2d(
            self.hidden_dim, self.hidden_dim, kernel_size=1, bias=False,
        )

    def _make_branch_block(self) -> nn.Sequential:
        return nn.Sequential(
            nn.Conv2d(
                self.branch_dim,
                self.branch_dim,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            _safe_group_norm(self.branch_dim),
            nn.GELU(),
            nn.Conv2d(
                self.branch_dim,
                self.branch_dim,
                kernel_size=1,
                bias=False,
            ),
            _safe_group_norm(self.branch_dim),
        )

    def forward(
        self,
        refiner_feature: torch.Tensor,
        original_pixel_feature: torch.Tensor,
        sam_fpn: torch.Tensor,
    ) -> torch.Tensor:
        """Forward one dual-branch fusion stage.

        Args:
            refiner_feature:        [N, 256, H_prev, W_prev]
            original_pixel_feature: [N, 256, H, W]
            sam_fpn:                [B, 256, H, W]  (image-level, no class dim)

        Returns:
            output: [N, 256, H, W]
        """
        N, _, H_prev, W_prev = refiner_feature.shape
        target_hw = original_pixel_feature.shape[-2:]

        # 1. Bilinear upsample refiner to current resolution.
        upsampled_refiner = F.interpolate(
            refiner_feature,
            size=target_hw,
            mode="bilinear",
            align_corners=False,
        )  # [N, 256, H, W]

        # 2. Project all inputs to 128 channels with dedicated projections.
        semantic_refiner_compact = self.semantic_refiner_proj(
            upsampled_refiner
        )  # [N, 128, H, W]
        detail_refiner_compact = self.detail_refiner_proj(
            upsampled_refiner
        )  # [N, 128, H, W]
        pixel_compact = self.pixel_proj(original_pixel_feature)  # [N, 128, H, W]
        fpn_compact = self.fpn_proj(sam_fpn)                     # [B, 128, H, W]

        # 3. Recover batch/image layout for broadcast-based fusion.
        B = sam_fpn.shape[0]
        if N % B != 0:
            raise ValueError(
                f"refiner N={N} must be divisible by batch B={B}. "
                f"refiner: {tuple(refiner_feature.shape)}, "
                f"sam_fpn: {tuple(sam_fpn.shape)}"
            )
        C_chunk = N // B
        _, branch_dim, H, W = semantic_refiner_compact.shape

        semantic_refiner_compact_5d = semantic_refiner_compact.reshape(
            B, C_chunk, branch_dim, H, W
        )
        detail_refiner_compact_5d = detail_refiner_compact.reshape(
            B, C_chunk, branch_dim, H, W
        )
        pixel_compact_5d = pixel_compact.reshape(B, C_chunk, branch_dim, H, W)
        # fpn_compact stays [B, branch_dim, H, W] and broadcasts over classes.

        # 4. Build semantic and detail inputs via broadcasting.
        semantic_input = (
            semantic_refiner_compact_5d + pixel_compact_5d
        ).reshape(N, branch_dim, H, W)

        detail_input = (
            detail_refiner_compact_5d + fpn_compact[:, None]
        ).reshape(N, branch_dim, H, W)

        # 5. Branch blocks with internal residual.
        semantic_feature = semantic_input + self.semantic_block(semantic_input)
        detail_feature = detail_input + self.detail_block(detail_input)

        # 6. Project back to 256, sum with equal weight, final projection.
        semantic_out = self.semantic_out_proj(semantic_feature)
        detail_out = self.detail_out_proj(detail_feature)

        fused_out = semantic_out + detail_out
        return self.fusion_out_proj(fused_out)


class RefinerPyramidDecoder(nn.Module):
    """Three-stage semantic–detail dual-branch upsampling pyramid.

    Each stage fuses:
      - upsampled Refiner feature (semantic main path + detail reference)
      - frozen Pixel Decoder feature (semantic branch)
      - original SAM3 backbone FPN (detail branch)

    Both branches operate in 128-channel compact space with independent
    standard 3×3 conv blocks. Each branch uses its own dedicated Refiner
    projection. The two branches are summed with equal weight and passed
    through a final 1×1 Conv. No learnable residual scale.

    Stages:
        stage_72:  36→72   (refiner_36 + O72 + FPN72)
        stage_144: 72→144  (refined_72 + O144 + FPN144)
        stage_288: 144→288 (refined_144 + O288 + FPN288)

    stage_288 output goes directly into the frozen SAM3 semantic_seg_head.
    There is no final fusion.

    The entire pyramid is wrapped in a single non-reentrant checkpoint
    during training when ``use_checkpoint=True``.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        branch_dim: int = 128,
        use_checkpoint: bool = True,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.use_checkpoint = bool(use_checkpoint)

        stage_kwargs = dict(
            hidden_dim=self.hidden_dim,
            branch_dim=int(branch_dim),
        )

        self.stage_72 = SemanticDetailFusionStage(**stage_kwargs)
        self.stage_144 = SemanticDetailFusionStage(**stage_kwargs)
        self.stage_288 = SemanticDetailFusionStage(**stage_kwargs)

    def _forward_impl(
        self,
        refiner_feature_36: torch.Tensor,
        original_pixel_feature_72: torch.Tensor,
        original_pixel_feature_144: torch.Tensor,
        original_pixel_feature_288: torch.Tensor,
        sam_fpn_72: torch.Tensor,
        sam_fpn_144: torch.Tensor,
        sam_fpn_288: torch.Tensor,
    ) -> torch.Tensor:
        refined_72 = self.stage_72(
            refiner_feature_36,
            original_pixel_feature_72,
            sam_fpn_72,
        )

        refined_144 = self.stage_144(
            refined_72,
            original_pixel_feature_144,
            sam_fpn_144,
        )

        refined_288 = self.stage_288(
            refined_144,
            original_pixel_feature_288,
            sam_fpn_288,
        )

        return refined_288

    def _validate_inputs(
        self,
        refiner_feature_36: torch.Tensor,
        original_pixel_feature_72: torch.Tensor,
        original_pixel_feature_144: torch.Tensor,
        original_pixel_feature_288: torch.Tensor,
        sam_fpn_72: torch.Tensor,
        sam_fpn_144: torch.Tensor,
        sam_fpn_288: torch.Tensor,
    ) -> None:
        """Check shapes before entering checkpoint so errors are clear."""
        # Class-conditioned tensors: [N, 256, H, W].
        cond_expected = [
            (refiner_feature_36, 36, "refiner_feature_36"),
            (original_pixel_feature_72, 72, "original_pixel_feature_72"),
            (original_pixel_feature_144, 144, "original_pixel_feature_144"),
            (original_pixel_feature_288, 288, "original_pixel_feature_288"),
        ]

        N_ref = None
        device_ref = None

        for tensor, hw, name in cond_expected:
            if tensor.ndim != 4:
                raise ValueError(
                    f"{name} must be [N, 256, {hw}, {hw}], "
                    f"got {tuple(tensor.shape)}."
                )
            N, C, H, W_val = tensor.shape
            if C != self.hidden_dim:
                raise ValueError(
                    f"{name} channel must be {self.hidden_dim}, got {C}."
                )
            if (H, W_val) != (hw, hw):
                raise ValueError(
                    f"{name} spatial size must be {hw}×{hw}, got {(H, W_val)}."
                )
            if N_ref is None:
                N_ref = N
                device_ref = tensor.device
            else:
                if N != N_ref:
                    raise ValueError(
                        f"All pyramid class-conditioned inputs must share N, but "
                        f"{name} has N={N} (expected {N_ref})."
                    )
                if tensor.device != device_ref:
                    raise ValueError(
                        f"All pyramid inputs must be on the same device, "
                        f"but {name} is on {tensor.device} "
                        f"(expected {device_ref})."
                    )

        # Image-level FPN tensors: [B, 256, H, W].
        fpn_expected = [
            (sam_fpn_72, 72, "sam_fpn_72"),
            (sam_fpn_144, 144, "sam_fpn_144"),
            (sam_fpn_288, 288, "sam_fpn_288"),
        ]

        B_ref = None

        for tensor, hw, name in fpn_expected:
            if tensor.ndim != 4:
                raise ValueError(
                    f"{name} must be [B, 256, {hw}, {hw}], "
                    f"got {tuple(tensor.shape)}."
                )
            B, C, H, W_val = tensor.shape
            if C != self.hidden_dim:
                raise ValueError(
                    f"{name} channel must be {self.hidden_dim}, got {C}."
                )
            if (H, W_val) != (hw, hw):
                raise ValueError(
                    f"{name} spatial size must be {hw}×{hw}, got {(H, W_val)}."
                )
            if B_ref is None:
                B_ref = B
            else:
                if B != B_ref:
                    raise ValueError(
                        f"All FPN inputs must share B, but "
                        f"{name} has B={B} (expected {B_ref})."
                    )
                if tensor.device != device_ref:
                    raise ValueError(
                        f"All pyramid inputs must be on the same device, "
                        f"but {name} is on {tensor.device} "
                        f"(expected {device_ref})."
                    )

        if N_ref is None or B_ref is None:
            raise ValueError("No inputs provided for validation.")
        if B_ref <= 0:
            raise ValueError(f"Batch size B must be > 0, got {B_ref}.")
        if N_ref <= 0:
            raise ValueError(f"Class-conditioned N must be > 0, got {N_ref}.")
        if N_ref % B_ref != 0:
            raise ValueError(
                f"N ({N_ref}) must be divisible by B ({B_ref}). "
                f"Got N={N_ref}, B={B_ref}."
            )

    def forward(
        self,
        refiner_feature_36: torch.Tensor,
        original_pixel_feature_72: torch.Tensor,
        original_pixel_feature_144: torch.Tensor,
        original_pixel_feature_288: torch.Tensor,
        sam_fpn_72: torch.Tensor,
        sam_fpn_144: torch.Tensor,
        sam_fpn_288: torch.Tensor,
    ) -> torch.Tensor:
        self._validate_inputs(
            refiner_feature_36=refiner_feature_36,
            original_pixel_feature_72=original_pixel_feature_72,
            original_pixel_feature_144=original_pixel_feature_144,
            original_pixel_feature_288=original_pixel_feature_288,
            sam_fpn_72=sam_fpn_72,
            sam_fpn_144=sam_fpn_144,
            sam_fpn_288=sam_fpn_288,
        )

        if self.use_checkpoint and self.training:
            return checkpoint(
                self._forward_impl,
                refiner_feature_36,
                original_pixel_feature_72,
                original_pixel_feature_144,
                original_pixel_feature_288,
                sam_fpn_72,
                sam_fpn_144,
                sam_fpn_288,
                use_reentrant=False,
            )
        return self._forward_impl(
            refiner_feature_36=refiner_feature_36,
            original_pixel_feature_72=original_pixel_feature_72,
            original_pixel_feature_144=original_pixel_feature_144,
            original_pixel_feature_288=original_pixel_feature_288,
            sam_fpn_72=sam_fpn_72,
            sam_fpn_144=sam_fpn_144,
            sam_fpn_288=sam_fpn_288,
        )
