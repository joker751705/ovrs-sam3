# Copyright (c) Meta Platforms, Inc. and affiliates. All Rights Reserved

# pyre-unsafe

from __future__ import annotations

import math
from pathlib import Path
from typing import Optional, TypeVar

import torch
import torch.nn as nn
from huggingface_hub import hf_hub_download
from iopath.common.file_io import g_pathmgr

from .config_dataclasses import (
    AdapterConfig,
    CheckpointManagerConfig,
    EncoderRefinerConfig,
    FreezeConfig,
    LoggerHookConfig,
    MetricsJsonlHookConfig,
    OpenCLIPConfig,
    SegmentorBuildConfig,
    SemanticCriterionConfig,
    TrainerConfig,
    VisualizerConfig,
    WandbHookConfig,
)
from .losses.semantic_criterion import (
    HybridCriterion,
    SemanticCriterion,
)
from .engine.checkpoint import CheckpointManager
from .engine.experiment_hooks import MetricsJsonlHook, WandbHook
from .engine.hooks import LoggerHook
from .engine.visualization import VisualizationManager
from .models.adapters.semantic_adapter import (
    HybridSegAdapter,
    SemanticSegAdapter,
)
from .models.encoder import TransformerEncoderFusion, TransformerEncoderLayer
from .models.geometry_encoders import SequenceGeometryEncoder
from .models.maskformer_segmentation import PixelDecoder, UniversalSegmentationHead
from .models.model_misc import (
    MultiheadAttentionWrapper as MultiheadAttention,
    TransformerWrapper,
)
from .models.necks import Sam3DualViTDetNeck
from .models.openclip_image_encoder import OpenCLIPImageEncoder
from .models.openclip_text_encoder import OpenCLIPTextEncoder
from .models.position_encoding import PositionEmbeddingSine
from .models.sam3_image import Sam3Image
from .models.segmentor import SAM3Segmentor
from .models.task_modes import (
    TASK_MODE_HYBRID,
    TASK_MODE_SEMANTIC,
    normalize_task_mode,
)
from .models.text_encoder_ve import VETextEncoder
from .models.tokenizer_ve import SimpleTokenizer
from .models.vitdet import ViT
from .models.vl_combiner import SAM3VLBackbone

ConfigT = TypeVar("ConfigT")

PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_bpe_path(explicit_bpe_path=None):
    if explicit_bpe_path is not None:
        p = Path(explicit_bpe_path).expanduser().resolve()
        if not p.exists():
            raise FileNotFoundError(f"BPE vocab file not found: {p}")
        return str(p)

    candidate_paths = [
        PROJECT_ROOT / "assets" / "bpe_simple_vocab_16e6.txt.gz",
        PROJECT_ROOT / "assets" / "clip" / "bpe_simple_vocab_16e6.txt.gz",
        PROJECT_ROOT / "configs" / "bpe_simple_vocab_16e6.txt.gz",
        PROJECT_ROOT / "configs" / "clip" / "bpe_simple_vocab_16e6.txt.gz",
    ]

    for p in candidate_paths:
        if p.exists():
            return str(p)

    tried = "\n".join(str(p) for p in candidate_paths)
    raise FileNotFoundError(
        "Cannot find bpe_simple_vocab_16e6.txt.gz. Tried:\n"
        f"{tried}\n"
        "Please pass `bpe_path` explicitly in config."
    )


def _setup_tf32() -> None:
    if torch.cuda.is_available():
        device_props = torch.cuda.get_device_properties(0)
        if device_props.major >= 8:
            torch.backends.cuda.matmul.allow_tf32 = True
            torch.backends.cudnn.allow_tf32 = True


_setup_tf32()


class FrozenModuleMixin:
    @staticmethod
    def set_requires_grad(module: Optional[nn.Module], requires_grad: bool) -> None:
        if module is None:
            return
        for p in module.parameters():
            p.requires_grad = requires_grad

    @staticmethod
    def set_model_requires_grad(model: nn.Module, requires_grad: bool) -> None:
        for p in model.parameters():
            p.requires_grad = requires_grad

    @staticmethod
    def get_named_modules(model: nn.Module) -> dict[str, nn.Module]:
        return dict(model.named_modules())

    @staticmethod
    def get_named_parameters(model: nn.Module) -> dict[str, nn.Parameter]:
        return dict(model.named_parameters())

    @classmethod
    def set_modules_requires_grad(
        cls,
        model: nn.Module,
        module_names: list[str],
        requires_grad: bool,
        strict: bool = True,
    ) -> None:
        if not module_names:
            return

        named_modules = cls.get_named_modules(model)
        named_parameters = cls.get_named_parameters(model)

        for name in module_names:
            if name in named_modules:
                cls.set_requires_grad(named_modules[name], requires_grad)
                continue

            if name in named_parameters:
                named_parameters[name].requires_grad = requires_grad
                continue

            if strict:
                available_modules = "\n".join(sorted(named_modules.keys()))
                available_parameters = "\n".join(sorted(named_parameters.keys()))
                raise KeyError(
                    f"Unknown module/parameter name: {name}\n"
                    f"Available module names are:\n{available_modules}\n\n"
                    f"Available parameter names are:\n{available_parameters}"
                )


class SAM3ModelBuilder(FrozenModuleMixin):
    @staticmethod
    def _coerce_config(obj, config_cls: type[ConfigT], name: str) -> ConfigT:
        if isinstance(obj, config_cls):
            return obj
        if obj is None:
            return config_cls()
        if isinstance(obj, dict):
            return config_cls(**dict(obj))
        raise TypeError(f"Unsupported {name} type: {type(obj)}")

    @classmethod
    def _coerce_freeze_cfg(cls, obj) -> FreezeConfig:
        return cls._coerce_config(obj, FreezeConfig, "freeze_cfg")

    @classmethod
    def _coerce_openclip_cfg(cls, obj) -> OpenCLIPConfig:
        return cls._coerce_config(obj, OpenCLIPConfig, "openclip_cfg")

    @classmethod
    def _coerce_encoder_refiner_cfg(cls, obj) -> EncoderRefinerConfig:
        return cls._coerce_config(obj, EncoderRefinerConfig, "encoder_refiner_cfg")

    @classmethod
    def _coerce_criterion_cfg(cls, obj) -> SemanticCriterionConfig:
        return cls._coerce_config(obj, SemanticCriterionConfig, "criterion_cfg")

    @classmethod
    def _coerce_adapter_cfg(cls, obj) -> AdapterConfig:
        return cls._coerce_config(obj, AdapterConfig, "adapter_cfg")

    @classmethod
    def _normalize_build_cfg(cls, cfg: SegmentorBuildConfig) -> SegmentorBuildConfig:
        cfg.task_mode = normalize_task_mode(cfg.task_mode)
        cfg.freeze_cfg = cls._coerce_freeze_cfg(cfg.freeze_cfg)
        cfg.openclip_cfg = cls._coerce_openclip_cfg(cfg.openclip_cfg)
        cfg.encoder_refiner_cfg = cls._coerce_encoder_refiner_cfg(cfg.encoder_refiner_cfg)
        cfg.criterion_cfg = cls._coerce_criterion_cfg(cfg.criterion_cfg)
        cfg.adapter_cfg = cls._coerce_adapter_cfg(cfg.adapter_cfg)
        return cfg

    @classmethod
    def build_config(cls, **kwargs) -> SegmentorBuildConfig:
        cfg = SegmentorBuildConfig(**kwargs)
        cfg = cls._normalize_build_cfg(cfg)
        cfg.openclip_cfg = cls.validate_openclip_cfg(cfg.openclip_cfg)
        cfg.encoder_refiner_cfg = cls.validate_encoder_refiner_cfg(cfg.encoder_refiner_cfg)
        return cfg

    @staticmethod
    def _require_dict(obj, name: str) -> dict:
        if not isinstance(obj, dict):
            raise TypeError(f"{name} must be a dict, got {type(obj)}.")
        return dict(obj)

    @staticmethod
    def resolve_work_dir(cfg, work_dir_override: Optional[str] = None) -> str:
        if work_dir_override is not None:
            return str(work_dir_override)
        return str(cfg.get("work_dir", "./work_dirs/default"))

    @staticmethod
    def _resolve_openclip_pretrained(pretrained: Optional[str]) -> Optional[str]:
        if pretrained is None:
            return None

        p = Path(str(pretrained)).expanduser()
        if not p.is_file():
            raise FileNotFoundError(
                f"openclip_cfg.pretrained: expected a local checkpoint file, but got {pretrained!r}."
            )

        return str(p.resolve())

    @classmethod
    def validate_openclip_cfg(cls, openclip_cfg: OpenCLIPConfig) -> OpenCLIPConfig:
        openclip_cfg = cls._coerce_openclip_cfg(openclip_cfg)

        if not openclip_cfg.enabled:
            return openclip_cfg

        _ = cls._resolve_openclip_pretrained(openclip_cfg.pretrained)

        if not isinstance(openclip_cfg.prompt_templates, list) or len(openclip_cfg.prompt_templates) != 64:
            raise ValueError("openclip_cfg.prompt_templates must be a list of 64 templates.")

        for idx, template in enumerate(openclip_cfg.prompt_templates):
            if "{}" not in str(template):
                raise ValueError(
                    f"openclip_cfg.prompt_templates[{idx}] must contain '{{}}'."
                )

        if not isinstance(openclip_cfg.text_prompt_batch_size, int) or openclip_cfg.text_prompt_batch_size <= 0:
            raise ValueError(
                "openclip_cfg.text_prompt_batch_size must be a positive integer, "
                f"got {openclip_cfg.text_prompt_batch_size!r}."
            )

        try:
            _ = bool(openclip_cfg.text_prompt_use_checkpoint)
        except Exception:
            raise ValueError(
                "openclip_cfg.text_prompt_use_checkpoint must be convertible to bool, "
                f"got {openclip_cfg.text_prompt_use_checkpoint!r}."
            )

        return openclip_cfg

    @classmethod
    def validate_encoder_refiner_cfg(cls, cfg: EncoderRefinerConfig) -> EncoderRefinerConfig:
        cfg = cls._coerce_encoder_refiner_cfg(cfg)

        if not cfg.enabled:
            raise ValueError(
                "encoder_refiner_cfg.enabled=False is not supported by the current "
                "semantic training path."
            )

        if cfg.fusion_layers <= 0:
            raise ValueError(
                f"encoder_refiner_cfg.fusion_layers must be positive, got {cfg.fusion_layers}."
            )

        if cfg.num_heads <= 0:
            raise ValueError(
                f"encoder_refiner_cfg.num_heads must be positive, got {cfg.num_heads}."
            )

        if cfg.window_size <= 0:
            raise ValueError(
                "encoder_refiner_cfg.window_size must be positive."
            )
        if not 0 <= cfg.shift_size < cfg.window_size:
            raise ValueError(
                "encoder_refiner_cfg.shift_size must satisfy 0 <= shift_size < window_size."
            )

        if cfg.encoder_hw != 72:
            raise ValueError(
                "encoder_hw=72 表示原始 Pixel Decoder pyramid 的最低分辨率，"
                f"got {cfg.encoder_hw}."
            )

        if cfg.refiner_hw != 36:
            raise ValueError(
                "refiner_hw=36 表示新 pyramid decoder 的起始尺度，"
                f"got {cfg.refiner_hw}."
            )

        if cfg.refiner_hw * 2 != cfg.encoder_hw:
            raise ValueError(
                "当前设计需要 refiner_hw * 2 == encoder_hw（36→72 双线性插值），"
                f"got refiner_hw={cfg.refiner_hw}, encoder_hw={cfg.encoder_hw}."
            )

        if cfg.hidden_dim != 256:
            raise ValueError(
                "Refiner、多尺度注入和冻结 SAM3 head 均固定使用 256 通道，"
                f"got {cfg.hidden_dim}."
            )

        if (
            not math.isfinite(float(cfg.residual_scale_init))
            or float(cfg.residual_scale_init) < 0.0
        ):
            raise ValueError(
                "encoder_refiner_cfg.residual_scale_init must be "
                "finite and non-negative, got "
                f"{cfg.residual_scale_init!r}."
            )

        if cfg.score_embed_dim <= 0:
            raise ValueError(
                "encoder_refiner_cfg.score_embed_dim must be positive, "
                f"got {cfg.score_embed_dim}."
            )

        return cfg

    @staticmethod
    def _create_position_encoding(precompute_resolution=None):
        return PositionEmbeddingSine(
            num_pos_feats=256,
            normalize=True,
            scale=None,
            temperature=10000,
            precompute_resolution=precompute_resolution,
        )

    @staticmethod
    def _create_vit_backbone(compile_mode=None):
        return ViT(
            img_size=1008,
            pretrain_img_size=336,
            patch_size=14,
            embed_dim=1024,
            depth=32,
            num_heads=16,
            mlp_ratio=4.625,
            norm_layer="LayerNorm",
            drop_path_rate=0.1,
            qkv_bias=True,
            use_abs_pos=True,
            tile_abs_pos=True,
            global_att_blocks=(7, 15, 23, 31),
            rel_pos_blocks=(),
            use_rope=True,
            use_interp_rope=True,
            window_size=24,
            pretrain_use_cls_token=True,
            retain_cls_token=False,
            ln_pre=True,
            ln_post=False,
            return_interm_layers=False,
            bias_patch_embed=False,
            compile_mode=compile_mode,
        )

    @classmethod
    def _create_vit_neck(cls, position_encoding, vit_backbone):
        return Sam3DualViTDetNeck(
            position_encoding=position_encoding,
            d_model=256,
            scale_factors=[4.0, 2.0, 1.0, 0.5],
            trunk=vit_backbone,
            add_sam2_neck=False,
        )

    @staticmethod
    def _create_text_encoder(bpe_path: str) -> VETextEncoder:
        tokenizer = SimpleTokenizer(bpe_path=bpe_path)
        return VETextEncoder(
            tokenizer=tokenizer,
            d_model=256,
            width=1024,
            heads=16,
            layers=24,
        )

    @staticmethod
    def _create_vl_backbone(vit_neck, text_encoder):
        return SAM3VLBackbone(visual=vit_neck, text=text_encoder, scalp=1)

    @staticmethod
    def _create_transformer_encoder() -> TransformerEncoderFusion:
        encoder_layer = TransformerEncoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=True,
            pos_enc_at_cross_attn_keys=False,
            pos_enc_at_cross_attn_queries=False,
            pre_norm=True,
            self_attention=MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
                batch_first=True,
            ),
            cross_attention=MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
                batch_first=True,
            ),
        )
        return TransformerEncoderFusion(
            layer=encoder_layer,
            num_layers=6,
            d_model=256,
            num_feature_levels=1,
            frozen=False,
            use_act_checkpoint=True,
            add_pooled_text_to_img_feat=False,
            pool_text_with_mask=True,
        )

    @staticmethod
    def _create_encoder_only_transformer() -> TransformerWrapper:
        encoder = SAM3ModelBuilder._create_transformer_encoder()
        return TransformerWrapper(
            encoder=encoder,
            decoder=None,
            d_model=256,
        )

    @staticmethod
    def _create_segmentation_head(compile_mode=None):
        pixel_decoder = PixelDecoder(
            num_upsampling_stages=3,
            interpolation_mode="nearest",
            hidden_dim=256,
            compile_mode=compile_mode,
        )
        cross_attend_prompt = MultiheadAttention(
            num_heads=8,
            dropout=0,
            embed_dim=256,
        )
        return UniversalSegmentationHead(
            hidden_dim=256,
            upsampling_stages=3,
            aux_masks=False,
            no_dec=True,
            presence_head=False,
            dot_product_scorer=None,
            act_ckpt=True,
            cross_attend_prompt=cross_attend_prompt,
            pixel_decoder=pixel_decoder,
        )

    @classmethod
    def _create_geometry_encoder(cls):
        geo_pos_enc = cls._create_position_encoding()
        geo_layer = TransformerEncoderLayer(
            activation="relu",
            d_model=256,
            dim_feedforward=2048,
            dropout=0.1,
            pos_enc_at_attn=False,
            pre_norm=True,
            self_attention=MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
                batch_first=False,
            ),
            pos_enc_at_cross_attn_queries=False,
            pos_enc_at_cross_attn_keys=True,
            cross_attention=MultiheadAttention(
                num_heads=8,
                dropout=0.1,
                embed_dim=256,
                batch_first=False,
            ),
        )
        return SequenceGeometryEncoder(
            pos_enc=geo_pos_enc,
            encode_boxes_as_points=False,
            points_direct_project=True,
            points_pool=True,
            points_pos_enc=True,
            boxes_direct_project=True,
            boxes_pool=True,
            boxes_pos_enc=True,
            d_model=256,
            num_layers=3,
            layer=geo_layer,
            use_act_ckpt=True,
            add_cls=True,
            add_post_encode_proj=True,
        )

    @classmethod
    def _create_openclip_encoders(
        cls,
        openclip_cfg: OpenCLIPConfig,
    ) -> tuple[OpenCLIPTextEncoder, OpenCLIPImageEncoder]:
        import open_clip

        pretrained = cls._resolve_openclip_pretrained(openclip_cfg.pretrained)
        if pretrained is None:
            raise ValueError(
                "openclip_cfg.enabled=True, but openclip_cfg.pretrained is None."
            )

        clip_model = open_clip.create_model(
            model_name=openclip_cfg.model_name,
            pretrained=pretrained,
            precision="fp32",
            device="cpu",
        )
        clip_model.eval()

        tokenizer = open_clip.get_tokenizer(openclip_cfg.model_name)

        text_width = getattr(getattr(clip_model, "transformer", None), "width", None)
        if text_width is None:
            raise AttributeError(
                "Cannot infer OpenCLIP text width from clip_model.transformer.width."
            )

        text_encoder = OpenCLIPTextEncoder(
            tokenizer=tokenizer,
            token_embedding=clip_model.token_embedding,
            positional_embedding=clip_model.positional_embedding,
            transformer=clip_model.transformer,
            ln_final=clip_model.ln_final,
            text_projection=clip_model.text_projection,
            attn_mask=getattr(clip_model, "attn_mask", None),
            context_length=getattr(clip_model, "context_length", 77),
            width=text_width,
        )

        image_encoder = OpenCLIPImageEncoder(
            visual=clip_model.visual,
            default_output=openclip_cfg.default_output,
            intermediate_layers=list(openclip_cfg.image_intermediate_layers),
            image_size=int(openclip_cfg.image_size),
        )

        return text_encoder, image_encoder

    @staticmethod
    def _load_checkpoint(model, checkpoint_path: str):
        with g_pathmgr.open(checkpoint_path, "rb") as f:
            ckpt = torch.load(f, map_location="cpu", weights_only=True)

        if "model" in ckpt and isinstance(ckpt["model"], dict):
            ckpt = ckpt["model"]

        if any(k.startswith("detector.") for k in ckpt.keys()):
            ckpt = {
                k.replace("detector.", ""): v
                for k, v in ckpt.items()
                if k.startswith("detector.")
            }

        missing_keys, unexpected_keys = model.load_state_dict(ckpt, strict=False)
        if len(missing_keys) > 0 or len(unexpected_keys) > 0:
            print(
                f"Loaded {checkpoint_path} with missing keys={missing_keys} "
                f"and unexpected keys={unexpected_keys}"
            )

    @staticmethod
    def download_ckpt_from_hf():
        model_id = "facebook/sam3"
        _ = hf_hub_download(repo_id=model_id, filename="config.json")
        return hf_hub_download(repo_id=model_id, filename="sam3.pt")

    @classmethod
    def apply_freeze_cfg(cls, model: nn.Module, freeze_cfg: FreezeConfig) -> None:
        if freeze_cfg.train_adapters_only:
            cls.set_model_requires_grad(model, False)
            cls.set_modules_requires_grad(
                model,
                freeze_cfg.trainable_modules,
                True,
                strict=True,
            )
        else:
            cls.set_model_requires_grad(model, True)
            cls.set_modules_requires_grad(
                model,
                freeze_cfg.frozen_modules,
                False,
                strict=True,
            )

    @staticmethod
    def _register_qv_only_grad_mask(param: nn.Parameter) -> None:
        """Register a backward hook that zeros gradients for the k projection
        inside a fused qkv weight/bias (OpenCLIP in_proj_weight / in_proj_bias).

        Fused qkv layout: [q, k, v] along dim 0.
        Only q and v receive gradients; k gradients are zeroed.
        """
        if getattr(param, "_qv_only_grad_mask_registered", False):
            return

        with torch.no_grad():
            mask = torch.zeros_like(param)

            if param.ndim == 2:
                if param.shape[0] % 3 != 0:
                    return
                d = param.shape[0] // 3
                mask[:d, :] = 1.0          # q
                mask[2 * d:, :] = 1.0      # v

            elif param.ndim == 1:
                if param.shape[0] % 3 != 0:
                    return
                d = param.shape[0] // 3
                mask[:d] = 1.0             # q bias
                mask[2 * d:] = 1.0         # v bias

            else:
                return

        def hook(grad):
            return grad * mask.to(device=grad.device, dtype=grad.dtype)

        param.register_hook(hook)
        param._qv_only_grad_mask_registered = True
        param._ovrs_disable_weight_decay = True

    @staticmethod
    def _normalize_finetune_mode(mode: str, name: str) -> str:
        mode = str(mode or "frozen").lower()
        valid = {"frozen", "attention", "transformer", "full"}
        if mode not in valid:
            raise ValueError(
                f"{name} must be one of {sorted(valid)}, got {mode!r}."
            )
        return mode

    @classmethod
    def _apply_attention_finetune(
        cls,
        module: nn.Module,
        attn_scope: str,
    ) -> None:
        """Unfreeze attention q/v + positional embeddings within *module*.

        For fused qkv projections (in_proj_weight / in_proj_bias), a gradient
        mask hook zeros out k gradients while keeping q/v trainable.

        *attn_scope* is the parent attribute name that contains attention layers
        (e.g. "transformer").
        """
        for name, param in module.named_parameters():
            lname = name.lower()
            train = False

            if "positional_embedding" in lname or "position" in lname:
                train = True

            elif attn_scope in lname and "attn" in lname:
                if "q_proj" in lname or "v_proj" in lname:
                    train = True
                elif "in_proj_weight" in lname or "in_proj_bias" in lname:
                    train = True

            if train:
                param.requires_grad_(True)

            if train and ("in_proj_weight" in lname or "in_proj_bias" in lname):
                cls._register_qv_only_grad_mask(param)

    @classmethod
    def set_openclip_text_finetune(
        cls,
        clip_text_encoder: nn.Module,
        mode: str,
    ) -> None:
        mode = cls._normalize_finetune_mode(mode, "freeze_cfg.openclip_text_finetune")

        cls.set_requires_grad(clip_text_encoder, False)

        if mode == "frozen":
            return

        if mode == "full":
            cls.set_requires_grad(clip_text_encoder, True)
            return

        if mode == "transformer":
            transformer = getattr(clip_text_encoder, "transformer", None)
            if transformer is not None:
                cls.set_requires_grad(transformer, True)
            # Also unfreeze positional embedding (lives outside transformer in
            # OpenCLIP's top-level module).
            for name, param in clip_text_encoder.named_parameters():
                if "positional_embedding" in name.lower() or "position" in name.lower():
                    param.requires_grad_(True)
            return

        # mode == "attention"
        cls._apply_attention_finetune(clip_text_encoder, attn_scope="transformer")

    @classmethod
    def set_openclip_image_finetune(
        cls,
        clip_image_encoder: nn.Module,
        mode: str,
    ) -> None:
        mode = cls._normalize_finetune_mode(mode, "freeze_cfg.openclip_image_finetune")

        cls.set_requires_grad(clip_image_encoder, False)

        # Tell OpenCLIPImageEncoder whether it should build an autograd graph.
        if hasattr(clip_image_encoder, "set_enable_grad"):
            clip_image_encoder.set_enable_grad(mode != "frozen")

        if mode == "frozen":
            return

        visual = getattr(clip_image_encoder, "visual", clip_image_encoder)

        if mode == "full":
            cls.set_requires_grad(visual, True)
            return

        if mode == "transformer":
            transformer = getattr(visual, "transformer", None)
            if transformer is not None:
                cls.set_requires_grad(transformer, True)
            # Also unfreeze visual positional embedding.
            for name, param in visual.named_parameters():
                if "positional_embedding" in name.lower() or "position" in name.lower():
                    param.requires_grad_(True)
            return

        # mode == "attention"
        cls._apply_attention_finetune(visual, attn_scope="transformer")

    @classmethod
    def build_semantic_core_model(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        bpe_path = resolve_bpe_path(cfg.bpe_path)
        compile_mode = "default" if cfg.compile else None

        position_encoding = cls._create_position_encoding(precompute_resolution=1008)
        vit_backbone = cls._create_vit_backbone(compile_mode=compile_mode)
        vit_neck = cls._create_vit_neck(position_encoding, vit_backbone)
        text_encoder = cls._create_text_encoder(bpe_path)
        backbone = cls._create_vl_backbone(vit_neck, text_encoder)

        clip_text_encoder = None
        clip_image_encoder = None

        if cfg.openclip_cfg.enabled:
            clip_text_encoder, clip_image_encoder = cls._create_openclip_encoders(
                cfg.openclip_cfg
            )

        transformer = cls._create_encoder_only_transformer()
        segmentation_head = cls._create_segmentation_head(compile_mode=compile_mode)
        input_geometry_encoder = cls._create_geometry_encoder()

        refiner_cfg = cfg.encoder_refiner_cfg

        model = Sam3Image(
            backbone=backbone,
            transformer=transformer,
            input_geometry_encoder=input_geometry_encoder,
            segmentation_head=segmentation_head,
            num_feature_levels=1,
            o2m_mask_predict=True,
            dot_prod_scoring=None,
            use_instance_query=True,
            multimask_output=True,
            matcher=None,
            clip_image_encoder=clip_image_encoder,
            clip_text_encoder=clip_text_encoder,
            openclip_prompt_templates=list(cfg.openclip_cfg.prompt_templates),
            normalize_label_for_clip=bool(cfg.openclip_cfg.normalize_label_for_clip),
            text_prompt_batch_size=int(cfg.openclip_cfg.text_prompt_batch_size),
            text_prompt_use_checkpoint=bool(cfg.openclip_cfg.text_prompt_use_checkpoint),
            encoder_refiner_fusion_layers=int(refiner_cfg.fusion_layers),
            encoder_refiner_num_heads=int(refiner_cfg.num_heads),
            encoder_refiner_dropout=float(refiner_cfg.dropout),
            encoder_refiner_hidden_dim=int(refiner_cfg.hidden_dim),
            encoder_refiner_score_embed_dim=int(refiner_cfg.score_embed_dim),
            encoder_refiner_residual_scale_init=float(
                refiner_cfg.residual_scale_init
            ),
            encoder_refiner_window_size=int(refiner_cfg.window_size),
            encoder_refiner_shift_size=int(refiner_cfg.shift_size),
            encoder_refiner_use_checkpoint=bool(refiner_cfg.use_checkpoint),
            task_mode=TASK_MODE_SEMANTIC,
        )

        checkpoint_path = cfg.checkpoint_path
        if cfg.load_from_hf and checkpoint_path is None:
            checkpoint_path = cls.download_ckpt_from_hf()

        if checkpoint_path is not None:
            cls._load_checkpoint(model, checkpoint_path)

        return model

    @classmethod
    def build_adapter(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        if cfg.task_mode == TASK_MODE_SEMANTIC:
            return SemanticSegAdapter(
                class_relative_prob_thd=cfg.adapter_cfg.class_relative_prob_thd,
                class_relative_eps=cfg.adapter_cfg.class_relative_eps,
            )

        if cfg.task_mode == TASK_MODE_HYBRID:
            return HybridSegAdapter()

        raise ValueError(f"Unsupported task_mode: {cfg.task_mode}")

    @classmethod
    def build_criterion(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        if cfg.task_mode == TASK_MODE_SEMANTIC:
            return SemanticCriterion(cfg=cfg.criterion_cfg)

        if cfg.task_mode == TASK_MODE_HYBRID:
            return HybridCriterion()

        raise ValueError(f"Unsupported task_mode: {cfg.task_mode}")

    @classmethod
    def build_semantic_segmentor(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        core_model = cls.build_semantic_core_model(cfg)
        adapter = cls.build_adapter(cfg)

        model = SAM3Segmentor(
            core=core_model,
            adapter=adapter,
            task_mode=TASK_MODE_SEMANTIC,
        )

        model = model.to(cfg.device)
        cls.apply_freeze_cfg(model, cfg.freeze_cfg)

        core = getattr(model, "core", None)

        # --- Set overall train/eval mode first ---
        if cfg.eval_mode:
            model.eval()
        else:
            model.train()

        # --- Then force frozen modules back to eval ---
        if core is not None:
            # SAM3 frozen modules.
            core.backbone.eval()
            core.transformer.eval()
            core.geometry_encoder.eval()
            core.segmentation_head.eval()

            # OpenCLIP image encoder.
            clip_image_encoder = getattr(core, "clip_image_encoder", None)
            if clip_image_encoder is not None:
                image_ft = str(
                    getattr(cfg.freeze_cfg, "openclip_image_finetune", "frozen")
                ).lower()
                cls.set_openclip_image_finetune(clip_image_encoder, mode=image_ft)
                # Keep eval mode even when partially trainable:
                # we train selected weights, not dropout / patch-dropout behavior.
                clip_image_encoder.eval()

            # OpenCLIP text encoder.
            clip_text_encoder = getattr(core, "clip_text_encoder", None)
            if clip_text_encoder is not None:
                text_ft = str(
                    getattr(cfg.freeze_cfg, "openclip_text_finetune", "frozen")
                ).lower()
                cls.set_openclip_text_finetune(clip_text_encoder, mode=text_ft)
                # Keep eval mode even when partially trainable:
                # we train selected weights, not dropout behavior.
                clip_text_encoder.eval()

            # Encoder refiner is the only module that should be in
            # train mode during training.
            if not cfg.eval_mode:
                core.encoder_refiner.train()

        model.core.prompt_chunk_size = (
            None if cfg.prompt_chunk_size is None else int(cfg.prompt_chunk_size)
        )

        return model

    @classmethod
    def build_hybrid_segmentor(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        raise NotImplementedError(
            "Hybrid task mode is not implemented yet. "
            "The current codebase only supports semantic mode."
        )

    @classmethod
    def build_segmentor(cls, cfg: SegmentorBuildConfig) -> nn.Module:
        if cfg.task_mode == TASK_MODE_SEMANTIC:
            return cls.build_semantic_segmentor(cfg)

        if cfg.task_mode == TASK_MODE_HYBRID:
            return cls.build_hybrid_segmentor(cfg)

        raise ValueError(f"Unsupported task_mode: {cfg.task_mode}")

    @classmethod
    def build_training_components(
        cls,
        cfg: SegmentorBuildConfig,
    ) -> tuple[nn.Module, nn.Module]:
        model = cls.build_segmentor(cfg)
        criterion = cls.build_criterion(cfg)
        return model, criterion

    @classmethod
    def build_trainer_config_from_cfg(
        cls,
        cfg,
        work_dir: str,
    ) -> TrainerConfig:
        train_cfg = cls._require_dict(cfg.train_cfg, "train_cfg")
        train_cfg["save_dir"] = str(work_dir)
        train_cfg["tta_cfg"] = cfg.get("tta_cfg", None)
        train_cfg["eval_cfg"] = cfg.get("eval_cfg", None)
        return TrainerConfig(**train_cfg)

    @classmethod
    def build_checkpoint_manager(
        cls,
        trainer_cfg: TrainerConfig,
    ) -> CheckpointManager:
        checkpoint_cfg = CheckpointManagerConfig(
            save_dir=str(trainer_cfg.save_dir),
            monitor=str(trainer_cfg.monitor),
            mode=str(trainer_cfg.monitor_mode),
            max_keep=int(trainer_cfg.max_keep_ckpts),
        )
        return CheckpointManager(checkpoint_cfg)

    @classmethod
    def build_hooks_from_cfg(cls, cfg) -> list:
        default_hooks = cls._require_dict(cfg.default_hooks, "default_hooks")
        logger_cfg = LoggerHookConfig(
            **cls._require_dict(default_hooks["logger"], "default_hooks.logger")
        )
        hooks = [LoggerHook(logger_cfg)]

        tracking_cfg = cfg.get("experiment_tracking", None)
        if tracking_cfg is None:
            tracking_cfg = {}

        metrics_cfg_raw = tracking_cfg.get("metrics_jsonl", None)
        if metrics_cfg_raw is None:
            metrics_cfg_raw = {}
        metrics_cfg = MetricsJsonlHookConfig(**dict(metrics_cfg_raw))
        if metrics_cfg.enabled:
            hooks.append(MetricsJsonlHook(
                enabled=metrics_cfg.enabled,
                filename=metrics_cfg.filename,
                train_interval=metrics_cfg.train_interval,
                priority=metrics_cfg.priority,
            ))

        wandb_cfg_raw = tracking_cfg.get("wandb", None)
        if wandb_cfg_raw is None:
            wandb_cfg_raw = {}
        wandb_cfg = WandbHookConfig(**dict(wandb_cfg_raw))
        if wandb_cfg.enabled:
            hooks.append(WandbHook(
                enabled=wandb_cfg.enabled,
                project=wandb_cfg.project,
                name=wandb_cfg.name,
                group=wandb_cfg.group,
                tags=wandb_cfg.tags,
                mode=wandb_cfg.mode,
                train_interval=wandb_cfg.train_interval,
                priority=wandb_cfg.priority,
                name_from_config_keys=wandb_cfg.name_from_config_keys,
                name_prefix=wandb_cfg.name_prefix,
            ))

        return hooks

    @classmethod
    def build_visualizer_from_cfg(
        cls,
        cfg,
        work_dir: str,
    ) -> Optional[VisualizationManager]:
        visualization_cfg = cfg.get("visualization", None)
        if visualization_cfg is None:
            return None

        visualizer_cfg = VisualizerConfig(
            **cls._require_dict(visualization_cfg, "visualization")
        )
        if not visualizer_cfg.enabled:
            return None

        save_dir = Path(visualizer_cfg.save_dir)
        if not save_dir.is_absolute():
            visualizer_cfg.save_dir = str(Path(work_dir) / save_dir)

        return VisualizationManager(
            visualizer_cfg,
            eval_cfg=cfg.get("eval_cfg", None),
        )

    @classmethod
    def build_train_runtime_components(
        cls,
        cfg,
        work_dir_override: Optional[str] = None,
    ) -> tuple[
        str,
        TrainerConfig,
        list,
        Optional[VisualizationManager],
        CheckpointManager,
    ]:
        work_dir = cls.resolve_work_dir(
            cfg,
            work_dir_override=work_dir_override,
        )
        trainer_cfg = cls.build_trainer_config_from_cfg(
            cfg,
            work_dir=work_dir,
        )
        return (
            work_dir,
            trainer_cfg,
            cls.build_hooks_from_cfg(cfg),
            cls.build_visualizer_from_cfg(cfg, work_dir=work_dir),
            cls.build_checkpoint_manager(trainer_cfg),
        )


def build_segmentor_model(**kwargs) -> nn.Module:
    cfg = SAM3ModelBuilder.build_config(**kwargs)
    return SAM3ModelBuilder.build_segmentor(cfg)


def build_training_components(**kwargs) -> tuple[nn.Module, nn.Module]:
    cfg = SAM3ModelBuilder.build_config(**kwargs)
    return SAM3ModelBuilder.build_training_components(cfg)


def build_train_runtime_components(
    cfg,
    work_dir_override: Optional[str] = None,
):
    return SAM3ModelBuilder.build_train_runtime_components(
        cfg,
        work_dir_override=work_dir_override,
    )