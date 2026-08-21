from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def flatten_batch_class(
    features: torch.Tensor,
) -> tuple[torch.Tensor, int, int]:
    """[B, C, D, H, W] → [B*C, D, H, W]"""
    batch_size, num_classes, channels, height, width = features.shape
    return (
        features.reshape(batch_size * num_classes, channels, height, width),
        batch_size,
        num_classes,
    )


def unflatten_batch_class(
    features: torch.Tensor,
    batch_size: int,
    num_classes: int,
) -> torch.Tensor:
    """[B*C, D, H, W] → [B, C, D, H, W]"""
    _, channels, height, width = features.shape
    return features.reshape(
        batch_size, num_classes, channels, height, width
    ).contiguous()


def apply_layer_norm_bcdhw(
    x: torch.Tensor,
    norm: nn.LayerNorm,
) -> torch.Tensor:
    """Apply LayerNorm on the channel dim of [B, C, D, H, W]."""
    return norm(
        x.permute(0, 1, 3, 4, 2)
    ).permute(0, 1, 4, 2, 3).contiguous()


def _safe_group_norm(num_channels: int) -> nn.GroupNorm:
    num_groups = min(8, num_channels)
    if num_channels % num_groups != 0:
        num_groups = 1
    return nn.GroupNorm(num_groups, num_channels)


# ---------------------------------------------------------------------------
# ClassScoreAttention
# ---------------------------------------------------------------------------


class ClassScoreAttention(nn.Module):
    """
    Inter-class attention at each spatial position with dual value updates.

    feature, score_embed and sam_text_mean are pre-normalized by the outer layer.
    q, k and both value paths are produced from the normalized inputs.

    q/k = concat(feature, sam_text_mean, score_embed)  → 768 dims
    v_feature = feature
    v_score   = score_embed

    Attention happens across C classes at every spatial position.
    Returns feature_update and score_update (no residual, no LayerNorm).
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 256,
        num_heads: int = 8,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_heads = int(num_heads)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}"
            )

        qk_in_dim = self.hidden_dim * 2 + self.score_embed_dim  # 256+256+256=768

        self.q_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.k_proj = nn.Linear(qk_in_dim, self.hidden_dim)

        self.v_feature_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_score_proj = nn.Linear(self.score_embed_dim, self.hidden_dim)

        self.out_feature_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_score_proj = nn.Linear(self.hidden_dim, self.score_embed_dim)

        self.dropout = nn.Dropout(float(dropout))

    def forward(
        self,
        feature: torch.Tensor,
        score_embed: torch.Tensor,
        sam_text_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature:       [B, C, D, H, W]  pre-normalized
            score_embed:   [B, C, D_score, H, W]  pre-normalized
            sam_text_mean: [B, C, D]  pre-normalized

        Returns:
            feature_update: [B, C, D, H, W]
            score_update:   [B, C, D_score, H, W]
        """
        B, C, D, H, W = feature.shape
        D_score = self.score_embed_dim

        if tuple(score_embed.shape) != (B, C, D_score, H, W):
            raise ValueError(
                f"score_embed must be [{B}, {C}, {D_score}, {H}, {W}], "
                f"got {tuple(score_embed.shape)}"
            )
        if tuple(sam_text_mean.shape) != (B, C, D):
            raise ValueError(
                f"sam_text_mean must be [{B}, {C}, {D}], "
                f"got {tuple(sam_text_mean.shape)}"
            )

        N = H * W

        # Flatten spatial dims into batch for per-position attention.
        # feature: [B, C, D, H, W] → [B*N, C, D]
        f_flat = feature.permute(0, 3, 4, 1, 2).reshape(B * N, C, D)

        # score_embed: [B, C, D_score, H, W] → [B*N, C, D_score]
        s_flat = score_embed.permute(0, 3, 4, 1, 2).reshape(B * N, C, D_score)

        # Broadcast sam_text_mean to each spatial position.
        text_broadcast = (
            sam_text_mean.to(device=f_flat.device, dtype=f_flat.dtype)[:, None]
            .expand(B, N, C, D)
            .reshape(B * N, C, D)
        )

        # q/k from concat of pre-normalized inputs.
        qk_input = torch.cat([f_flat, text_broadcast, s_flat], dim=-1)  # [B*N, C, 768]

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        v_feat = self.v_feature_proj(f_flat)
        v_score = self.v_score_proj(s_flat)

        head_dim = D // self.num_heads
        q = q.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v_feat = v_feat.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v_score = v_score.reshape(B * N, C, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)
        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out_feat = torch.matmul(attn, v_feat)
        out_feat = out_feat.permute(0, 2, 1, 3).reshape(B * N, C, D)
        out_feat = self.out_feature_proj(out_feat)
        out_feat = self.dropout(out_feat)

        out_score = torch.matmul(attn, v_score)
        out_score = out_score.permute(0, 2, 1, 3).reshape(B * N, C, D)
        out_score = self.out_score_proj(out_score)
        out_score = self.dropout(out_score)

        feature_update = out_feat.reshape(B, H, W, C, D).permute(0, 3, 4, 1, 2).contiguous()
        score_update = out_score.reshape(B, H, W, C, D_score).permute(0, 3, 4, 1, 2).contiguous()

        return feature_update, score_update


# ---------------------------------------------------------------------------
# WindowScoreAttention
# ---------------------------------------------------------------------------


class WindowScoreAttention(nn.Module):
    """
    Intra-class window attention with relative position bias and dual value updates.

    feature and score_embed are pre-normalized by the outer layer.
    q, k and both value paths are produced from the normalized inputs.

    q/k = concat(feature, score_embed) → 512 dims
    v_feature = feature
    v_score   = score_embed

    Returns feature_update and score_update (no residual, no LayerNorm).
    Window size = 12, shift_size = 0 for regular, 6 for shifted.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 12,
        shift_size: int = 0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_dim = int(hidden_dim)
        self.score_embed_dim = int(score_embed_dim)
        self.num_heads = int(num_heads)
        self.window_size = int(window_size)
        self.shift_size = int(shift_size)

        if self.hidden_dim % self.num_heads != 0:
            raise ValueError(
                f"hidden_dim={hidden_dim} not divisible by num_heads={num_heads}"
            )
        if not 0 <= self.shift_size < self.window_size:
            raise ValueError(
                f"shift_size={shift_size} must be in [0, window_size={window_size})"
            )

        qk_in_dim = self.hidden_dim + self.score_embed_dim  # 256+256=512

        self.q_proj = nn.Linear(qk_in_dim, self.hidden_dim)
        self.k_proj = nn.Linear(qk_in_dim, self.hidden_dim)

        self.v_feature_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.v_score_proj = nn.Linear(self.score_embed_dim, self.hidden_dim)

        self.out_feature_proj = nn.Linear(self.hidden_dim, self.hidden_dim)
        self.out_score_proj = nn.Linear(self.hidden_dim, self.score_embed_dim)

        self.dropout = nn.Dropout(float(dropout))

        # Relative position bias (GSNet / Swin style).
        ws = self.window_size
        self.relative_position_bias_table = nn.Parameter(
            torch.zeros((2 * ws - 1) * (2 * ws - 1), num_heads)
        )
        nn.init.trunc_normal_(self.relative_position_bias_table, std=0.02)

        coords_h = torch.arange(ws)
        coords_w = torch.arange(ws)
        coords = torch.stack(
            torch.meshgrid(coords_h, coords_w, indexing="ij")
        )  # [2, ws, ws]
        coords_flatten = torch.flatten(coords, 1)          # [2, ws*ws]

        relative_coords = (
            coords_flatten[:, :, None] - coords_flatten[:, None, :]
        )  # [2, ws*ws, ws*ws]
        relative_coords = relative_coords.permute(1, 2, 0).contiguous()  # [N, N, 2]

        relative_coords[:, :, 0] += ws - 1
        relative_coords[:, :, 1] += ws - 1
        relative_coords[:, :, 0] *= 2 * ws - 1

        relative_position_index = relative_coords.sum(-1)  # [N, N]
        self.register_buffer("relative_position_index", relative_position_index)

    @staticmethod
    def _pad_to_window(x: torch.Tensor, window_size: int):
        H, W = x.shape[-2], x.shape[-1]
        pad_h = (window_size - H % window_size) % window_size
        pad_w = (window_size - W % window_size) % window_size
        if pad_h == 0 and pad_w == 0:
            return x, H, W
        return F.pad(x, (0, pad_w, 0, pad_h)), H, W

    def _window_partition(self, x: torch.Tensor):
        B, D, H, W = x.shape
        ws = self.window_size
        x = x.reshape(B, D, H // ws, ws, W // ws, ws)
        x = x.permute(0, 2, 4, 3, 5, 1).reshape(-1, ws * ws, D)
        return x

    def _window_reverse(self, x: torch.Tensor, B: int, H: int, W: int):
        ws = self.window_size
        D = x.shape[-1]
        x = x.reshape(B, H // ws, W // ws, ws, ws, D)
        x = x.permute(0, 5, 1, 3, 2, 4).reshape(B, D, H, W)
        return x

    def _get_relative_position_bias(self) -> torch.Tensor:
        """
        Returns:
            relative_position_bias: [num_heads, N, N] where N = window_size * window_size
        """
        ws = self.window_size
        N = ws * ws

        relative_position_bias = self.relative_position_bias_table[
            self.relative_position_index.reshape(-1)
        ]
        relative_position_bias = relative_position_bias.view(N, N, self.num_heads)
        relative_position_bias = relative_position_bias.permute(2, 0, 1).contiguous()
        return relative_position_bias  # [num_heads, N, N]

    def _build_shift_attn_mask(
        self,
        padded_h: int,
        padded_w: int,
        bc: int,
        device: torch.device,
        dtype: torch.dtype,
    ) -> torch.Tensor | None:
        if self.shift_size == 0:
            return None

        ws = self.window_size
        shift = self.shift_size

        img_mask = torch.zeros(
            (1, padded_h, padded_w), device=device, dtype=torch.float32
        )

        h_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))
        w_slices = (slice(0, -ws), slice(-ws, -shift), slice(-shift, None))

        cnt = 0
        for h in h_slices:
            for w in w_slices:
                img_mask[:, h, w] = cnt
                cnt += 1

        mask_windows = self._window_partition(img_mask.unsqueeze(0))
        mask_windows = mask_windows.squeeze(-1)

        attn_mask = mask_windows.unsqueeze(1) - mask_windows.unsqueeze(2)
        attn_mask = attn_mask.masked_fill(attn_mask != 0, -100.0)
        attn_mask = attn_mask.masked_fill(attn_mask == 0, 0.0)

        win_per_img = attn_mask.shape[0]
        attn_mask = attn_mask.unsqueeze(0).expand(
            bc, win_per_img, ws * ws, ws * ws
        )
        attn_mask = attn_mask.reshape(bc * win_per_img, ws * ws, ws * ws)

        return attn_mask.to(dtype=dtype)

    def forward(
        self,
        feature: torch.Tensor,
        score_embed: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature:     [B, C, D, H, W]  pre-normalized
            score_embed: [B, C, D_score, H, W]  pre-normalized

        Returns:
            feature_update: [B, C, D, H, W]
            score_update:   [B, C, D_score, H, W]
        """
        B, C, D, H, W = feature.shape
        D_score = self.score_embed_dim

        if H % self.window_size != 0 or W % self.window_size != 0:
            raise ValueError(
                f"WindowScoreAttention expects H/W divisible by window_size={self.window_size}, "
                f"got H={H}, W={W}."
            )
        if tuple(score_embed.shape) != (B, C, D_score, H, W):
            raise ValueError(
                f"score_embed must be [{B}, {C}, {D_score}, {H}, {W}], "
                f"got {tuple(score_embed.shape)}"
            )

        bc = B * C
        ws = self.window_size

        f_flat = feature.reshape(bc, D, H, W)
        s_flat = score_embed.reshape(bc, D_score, H, W)

        f_flat, orig_h, orig_w = self._pad_to_window(f_flat, ws)
        s_flat, _, _ = self._pad_to_window(s_flat, ws)

        pad_h, pad_w = f_flat.shape[-2], f_flat.shape[-1]

        shift = self.shift_size
        if shift > 0:
            f_flat = torch.roll(f_flat, shifts=(-shift, -shift), dims=(-2, -1))
            s_flat = torch.roll(s_flat, shifts=(-shift, -shift), dims=(-2, -1))

        f_windows = self._window_partition(f_flat)   # [num_win, ws*ws, D]
        s_windows = self._window_partition(s_flat)   # [num_win, ws*ws, D_score]

        attn_mask = self._build_shift_attn_mask(
            padded_h=pad_h,
            padded_w=pad_w,
            bc=bc,
            device=feature.device,
            dtype=feature.dtype,
        )

        # q/k from concat of pre-normalized inputs.
        qk_input = torch.cat([f_windows, s_windows], dim=-1)  # [num_win, N, 512]

        q = self.q_proj(qk_input)
        k = self.k_proj(qk_input)
        v_feat = self.v_feature_proj(f_windows)
        v_score = self.v_score_proj(s_windows)

        head_dim = D // self.num_heads
        num_win, N = q.shape[0], q.shape[1]

        q = q.reshape(num_win, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        k = k.reshape(num_win, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v_feat = v_feat.reshape(num_win, N, self.num_heads, head_dim).permute(0, 2, 1, 3)
        v_score = v_score.reshape(num_win, N, self.num_heads, head_dim).permute(0, 2, 1, 3)

        attn = torch.matmul(q, k.transpose(-2, -1)) * (head_dim ** -0.5)

        # Add relative position bias.
        rel_pos_bias = self._get_relative_position_bias().to(
            device=attn.device, dtype=attn.dtype
        )
        attn = attn + rel_pos_bias.unsqueeze(0)

        if attn_mask is not None:
            attn = attn + attn_mask.unsqueeze(1)

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        out_feat = torch.matmul(attn, v_feat)
        out_feat = out_feat.permute(0, 2, 1, 3).reshape(num_win, N, D)
        out_feat = self.out_feature_proj(out_feat)
        out_feat = self.dropout(out_feat)

        out_score = torch.matmul(attn, v_score)
        out_score = out_score.permute(0, 2, 1, 3).reshape(num_win, N, D)
        out_score = self.out_score_proj(out_score)
        out_score = self.dropout(out_score)

        out_feat = self._window_reverse(out_feat, bc, pad_h, pad_w)
        out_score = self._window_reverse(out_score, bc, pad_h, pad_w)

        if shift > 0:
            out_feat = torch.roll(out_feat, shifts=(shift, shift), dims=(-2, -1))
            out_score = torch.roll(out_score, shifts=(shift, shift), dims=(-2, -1))

        out_feat = out_feat[:, :, :orig_h, :orig_w]
        out_score = out_score[:, :, :orig_h, :orig_w]

        feature_update = out_feat.reshape(B, C, D, H, W).contiguous()
        score_update = out_score.reshape(B, C, D_score, H, W).contiguous()

        return feature_update, score_update


# ---------------------------------------------------------------------------
# EncoderRefinerLayer
# ---------------------------------------------------------------------------


class EncoderRefinerLayer(nn.Module):
    """
    One refiner layer operating at 36×36 with pre-norm and direct
    residual updates.

    Sequence:
        1. ClassScoreAttention
           (pre-norm → attention → output projection → residual)
        2. WindowScoreAttention regular
           (pre-norm → attention → output projection → residual)
        3. WindowScoreAttention shifted
           (pre-norm → attention → output projection → residual)
        4. Feature FFN
           (pre-norm → FFN output projection → residual)
        5. Score FFN
           (pre-norm → FFN output projection → residual)

    No post-norm is applied inside the layer. Attention update magnitude
    is learned by the attention output projections, while FFN update
    magnitude is learned by the second FFN linear projections.
    """

    def __init__(
        self,
        hidden_dim: int = 256,
        score_embed_dim: int = 256,
        num_heads: int = 8,
        window_size: int = 12,
        shift_size: int = 6,
        dropout: float = 0.1,
    ):
        super().__init__()

        self.class_attn = ClassScoreAttention(
            hidden_dim=hidden_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            dropout=dropout,
        )

        self.window_attn_regular = WindowScoreAttention(
            hidden_dim=hidden_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=0,
            dropout=dropout,
        )

        self.window_attn_shifted = WindowScoreAttention(
            hidden_dim=hidden_dim,
            score_embed_dim=score_embed_dim,
            num_heads=num_heads,
            window_size=window_size,
            shift_size=shift_size,
            dropout=dropout,
        )

        # Pre-norm for class attention.
        self.class_norm_feature = nn.LayerNorm(hidden_dim)
        self.class_norm_score = nn.LayerNorm(score_embed_dim)
        self.class_norm_text = nn.LayerNorm(hidden_dim)

        # Pre-norm for regular window attention.
        self.regular_norm_feature = nn.LayerNorm(hidden_dim)
        self.regular_norm_score = nn.LayerNorm(score_embed_dim)

        # Pre-norm for shifted window attention.
        self.shifted_norm_feature = nn.LayerNorm(hidden_dim)
        self.shifted_norm_score = nn.LayerNorm(score_embed_dim)

        # Pre-norm for FFN.
        self.ffn_norm_feature = nn.LayerNorm(hidden_dim)
        self.ffn_norm_score = nn.LayerNorm(score_embed_dim)

        # Per-token FFN for feature.
        self.ffn_fc1_feature = nn.Linear(hidden_dim, hidden_dim * 4)
        self.ffn_fc2_feature = nn.Linear(hidden_dim * 4, hidden_dim)
        self.ffn_dropout_feature = nn.Dropout(float(dropout))

        # Per-token FFN for score.
        self.ffn_fc1_score = nn.Linear(score_embed_dim, score_embed_dim * 4)
        self.ffn_fc2_score = nn.Linear(score_embed_dim * 4, score_embed_dim)
        self.ffn_dropout_score = nn.Dropout(float(dropout))

    def _ffn_feature_update(
        self,
        feature: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token FFN for pre-normalized feature. Returns update only."""
        B, C, D, H, W = feature.shape
        x = feature.permute(0, 1, 3, 4, 2)
        x = self.ffn_fc2_feature(
            self.ffn_dropout_feature(F.gelu(self.ffn_fc1_feature(x)))
        )
        x = self.ffn_dropout_feature(x)
        return x.permute(0, 1, 4, 2, 3).contiguous()

    def _ffn_score_update(
        self,
        score: torch.Tensor,
    ) -> torch.Tensor:
        """Per-token FFN for pre-normalized score. Returns update only."""
        B, C, Ds, H, W = score.shape
        x = score.permute(0, 1, 3, 4, 2)
        x = self.ffn_fc2_score(
            self.ffn_dropout_score(F.gelu(self.ffn_fc1_score(x)))
        )
        x = self.ffn_dropout_score(x)
        return x.permute(0, 1, 4, 2, 3).contiguous()

    def forward(
        self,
        feature_36: torch.Tensor,
        score_embed_36: torch.Tensor,
        sam_text_mean: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """
        Args:
            feature_36:      [B, C, 256, 36, 36]
            score_embed_36:  [B, C, 256, 36, 36]
            sam_text_mean:   [B, C, 256]

        Returns:
            feature_36:      [B, C, 256, 36, 36]
            score_embed_36:  [B, C, 256, 36, 36]
        """
        # Class attention: pre-norm → attention → direct residual.
        class_feature = apply_layer_norm_bcdhw(
            feature_36,
            self.class_norm_feature,
        )
        class_score = apply_layer_norm_bcdhw(
            score_embed_36,
            self.class_norm_score,
        )
        class_text = self.class_norm_text(sam_text_mean)

        feature_update, score_update = self.class_attn(
            feature=class_feature,
            score_embed=class_score,
            sam_text_mean=class_text,
        )

        feature_36 = feature_36 + feature_update
        score_embed_36 = score_embed_36 + score_update

        # Regular window attention.
        regular_feature = apply_layer_norm_bcdhw(
            feature_36,
            self.regular_norm_feature,
        )
        regular_score = apply_layer_norm_bcdhw(
            score_embed_36,
            self.regular_norm_score,
        )

        feature_update, score_update = self.window_attn_regular(
            feature=regular_feature,
            score_embed=regular_score,
        )

        feature_36 = feature_36 + feature_update
        score_embed_36 = score_embed_36 + score_update

        # Shifted window attention.
        shifted_feature = apply_layer_norm_bcdhw(
            feature_36,
            self.shifted_norm_feature,
        )
        shifted_score = apply_layer_norm_bcdhw(
            score_embed_36,
            self.shifted_norm_score,
        )

        feature_update, score_update = self.window_attn_shifted(
            feature=shifted_feature,
            score_embed=shifted_score,
        )

        feature_36 = feature_36 + feature_update
        score_embed_36 = score_embed_36 + score_update

        # Feature FFN.
        ffn_feature = apply_layer_norm_bcdhw(
            feature_36,
            self.ffn_norm_feature,
        )
        feature_update = self._ffn_feature_update(ffn_feature)
        feature_36 = feature_36 + feature_update

        # Score FFN.
        ffn_score = apply_layer_norm_bcdhw(
            score_embed_36,
            self.ffn_norm_score,
        )
        score_update = self._ffn_score_update(ffn_score)
        score_embed_36 = score_embed_36 + score_update

        return feature_36, score_embed_36
