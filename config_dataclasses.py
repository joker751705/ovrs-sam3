from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Optional


@dataclass
class FreezeConfig:
    train_adapters_only: bool = False
    trainable_modules: list[str] = field(default_factory=list)
    frozen_modules: list[str] = field(default_factory=list)

    # "frozen" | "attention" | "transformer" | "full"
    openclip_text_finetune: str = "frozen"

    # "frozen" | "attention" | "transformer" | "full"
    openclip_image_finetune: str = "frozen"


@dataclass
class OpenCLIPConfig:
    enabled: bool = False
    model_name: str = "ViT-L-14"
    pretrained: Optional[str] = None
    default_output: str = "feat_map"
    image_size: int = 504

    image_intermediate_layers: list[int] = field(default_factory=lambda: [7, 15])

    prompt_templates: list[str] = field(default_factory=lambda: [
        # Basic remote-sensing viewpoints.
        "a remote sensing image of {}.",
        "a satellite image of {}.",
        "an aerial image of {}.",
        "an overhead image of {}.",
        "a top-down view of {}.",
        "a bird's-eye view of {}.",
        "remote sensing imagery showing {}.",
        "an overhead scene containing {}.",

        # Natural caption variants.
        "the remote sensing image shows {}.",
        "the satellite image shows {}.",
        "the aerial image shows {}.",
        "an overhead image in which {} can be seen.",
        "a satellite image in which {} is visible.",
        "an aerial scene with {}.",
        "a geographic scene containing {}.",
        "a view from above showing {}.",

        # Resolution and acquisition platforms.
        "a high-resolution remote sensing image of {}.",
        "a high-resolution satellite image of {}.",
        "a high-resolution aerial image of {}.",
        "a very high-resolution overhead image of {}.",
        "a drone image of {}.",
        "a UAV image of {}.",
        "an orthophoto showing {}.",
        "a wide-area satellite view containing {}.",

        # Region and spatial extent.
        "a close overhead view of {}.",
        "a local overhead view containing {}.",
        "a visible region of {} from above.",
        "the spatial extent of {} in an overhead image.",
        "the visible footprint of {} in an aerial image.",
        "a land-cover area of {} in satellite imagery.",
        "a land-use area of {} in aerial imagery.",
        "a remote sensing region corresponding to {}.",

        # Shape and boundary.
        "the shape of {} in an overhead image.",
        "the outline of {} in satellite imagery.",
        "the boundary of {} in aerial imagery.",
        "the edge pattern of {} in an overhead image.",
        "the geometric structure of {} from above.",
        "the spatial layout of {} in remote sensing imagery.",
        "the visible form of {} in a satellite image.",
        "the surface structure of {} in an aerial image.",

        # Texture and appearance.
        "the texture of {} in satellite imagery.",
        "the fine-grained texture of {} in aerial imagery.",
        "the visual pattern of {} in an overhead image.",
        "the spatial pattern of {} in remote sensing imagery.",
        "the color pattern of {} in an aerial image.",
        "a homogeneous region of {} from above.",
        "a heterogeneous region containing {} from above.",
        "an overhead view where {} is distinguished from its surroundings.",

        # Scale, count, and distribution.
        "a small instance of {} in an aerial image.",
        "a large instance of {} in a satellite image.",
        "multiple instances of {} in overhead imagery.",
        "a group of {} visible from above.",
        "clustered {} in remote sensing imagery.",
        "scattered {} in remote sensing imagery.",
        "densely distributed {} in an overhead image.",
        "sparsely distributed {} in an overhead image.",

        # Spatial morphology and scene context.
        "a compact region of {} in remote sensing imagery.",
        "an elongated region of {} in remote sensing imagery.",
        "a linear structure of {} in an overhead image.",
        "a continuous region of {} viewed from above.",
        "a fragmented region of {} viewed from above.",
        "urban remote sensing imagery containing {}.",
        "rural remote sensing imagery containing {}.",
        "{} within its surrounding landscape in satellite imagery.",
    ])
    normalize_label_for_clip: bool = True

    text_prompt_batch_size: int = 64
    text_prompt_use_checkpoint: bool = True


@dataclass
class EncoderRefinerConfig:
    enabled: bool = True

    fusion_layers: int = 4
    num_heads: int = 8
    dropout: float = 0.1

    hidden_dim: int = 256
    score_embed_dim: int = 256

    refiner_hw: int = 36
    encoder_hw: int = 72

    window_size: int = 12
    shift_size: int = 6

    use_checkpoint: bool = True


@dataclass
class SemanticCriterionConfig:
    ignore_index: int = 255

    mixed_bce_weight: float = 1.0
    final_dice_weight: float = 0.0

    eps: float = 1e-6

    mixed_bce_boundary_width: int = 3


@dataclass
class AdapterConfig:
    class_relative_prob_thd: Optional[float] = None
    class_relative_eps: float = 1e-6


@dataclass
class SegmentorBuildConfig:
    task_mode: str = "semantic"

    bpe_path: Optional[str] = None
    checkpoint_path: Optional[str] = None
    load_from_hf: bool = True
    device: str = "cuda"
    eval_mode: bool = True
    compile: bool = False

    prompt_chunk_size: Optional[int] = None

    freeze_cfg: FreezeConfig = field(default_factory=FreezeConfig)
    openclip_cfg: OpenCLIPConfig = field(default_factory=OpenCLIPConfig)
    encoder_refiner_cfg: EncoderRefinerConfig = field(default_factory=EncoderRefinerConfig)
    criterion_cfg: SemanticCriterionConfig = field(
        default_factory=SemanticCriterionConfig
    )
    adapter_cfg: AdapterConfig = field(default_factory=AdapterConfig)


@dataclass
class TrainerConfig:
    max_iters: int = 10000
    log_window_size: int = 20
    use_amp: bool = True
    grad_clip_norm: Optional[float] = 0.1

    save_dir: str = "./work_dirs/default"
    save_interval: int = 1000
    eval_interval: int = 1000

    # Maximum number of validation batches per validation call.
    # None or <=0 means full validation.
    val_max_iters: Optional[int] = None

    monitor: str = "semantic.miou"
    monitor_mode: str = "max"
    max_keep_ckpts: int = 5

    device: str = "cuda"

    tta_cfg: Optional[Dict] = None
    eval_cfg: Optional[Dict] = None


@dataclass
class CheckpointManagerConfig:
    save_dir: str
    monitor: str = "total_loss"
    mode: str = "min"
    max_keep: int = 5

@dataclass
class LoggerHookConfig:
    interval: int = 20
    val_interval: int = 50
    print_metric_tables: bool = True
    print_per_class_metrics: bool = True
    priority: int = 70


@dataclass
class MetricsJsonlHookConfig:
    enabled: bool = True
    filename: str = "metrics.jsonl"
    train_interval: int = 20
    priority: int = 80


@dataclass
class WandbHookConfig:
    enabled: bool = False
    project: str = "ovrs-sam3"
    name: Optional[str] = None
    group: Optional[str] = None
    tags: list[str] = field(default_factory=list)
    mode: str = "online"
    train_interval: int = 20
    priority: int = 90

    name_from_config_keys: list[str] = field(default_factory=list)
    name_prefix: Optional[str] = None

@dataclass
class VisualizerConfig:
    enabled: bool = False
    save_dir: str = "./visualizations"
    save_stage: str = "val"
    alpha: float = 0.45

    save_original: bool = True
    save_prediction: bool = True
    save_raw_final_prediction: bool = True
    save_ground_truth: bool = True
    save_semantic_prediction: bool = True

    save_score_summary: bool = True
    save_score_heatmaps: bool = True
    heatmap_colormap: str = "turbo"

    save_sam3_direct_segmentation: bool = False
    sam3_direct_seg_threshold: float = 0.5

    vis_prob: float = 0.05
    max_samples_per_epoch: Optional[int] = 50
    vis_seed: int = 42

    image_folder_pattern: str = "image_{image_id:06d}"
    ignore_index: int = 255
