from __future__ import annotations

import argparse
import ast
import hashlib
import json
import random
import re
import sys
from pathlib import Path
from typing import Any, Dict

import numpy as np
import torch

if __package__ in (None, ""):
    repo_root = Path(__file__).resolve().parents[1]
    from importlib import import_module
    import types

    package_name = "_ovrs_sam3_localpkg"
    if package_name not in sys.modules:
        pkg = types.ModuleType(package_name)
        pkg.__path__ = [str(repo_root)]
        sys.modules[package_name] = pkg

    build_dataloader = import_module(f"{package_name}.data.build").build_dataloader
    Config = import_module(f"{package_name}.engine.config").Config
    _opt_mod = import_module(f"{package_name}.engine.optimizer_builder")
    build_optimizer = _opt_mod.build_optimizer
    build_scheduler = _opt_mod.build_scheduler
    _trainer_mod = import_module(f"{package_name}.engine.trainer")
    Trainer = _trainer_mod.Trainer

    _builder_mod = import_module(f"{package_name}.model_builder")
    build_training_components = _builder_mod.build_training_components
    build_train_runtime_components = _builder_mod.build_train_runtime_components
    load_checkpoint_file = import_module(f"{package_name}.engine.checkpoint").load_checkpoint_file
else:
    from ..data.build import build_dataloader
    from ..engine.config import Config
    from ..engine.checkpoint import load_checkpoint_file
    from ..engine.optimizer_builder import build_optimizer, build_scheduler
    from ..engine.trainer import Trainer
    from ..model_builder import (
        build_train_runtime_components,
        build_training_components,
    )


def set_seed(seed: int = 42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


class _DotDict(dict):
    __getattr__ = dict.get
    __setattr__ = dict.__setitem__
    __delattr__ = dict.__delitem__


def _to_dotdict(obj: Any):
    if isinstance(obj, dict):
        return _DotDict({k: _to_dotdict(v) for k, v in obj.items()})
    if isinstance(obj, list):
        return [_to_dotdict(x) for x in obj]
    return obj


def _parse_cfg_value(raw: str):
    text = str(raw).strip()

    lowered = text.lower()
    if lowered == "true":
        return True
    if lowered == "false":
        return False
    if lowered in ("none", "null"):
        return None

    try:
        return ast.literal_eval(text)
    except Exception:
        return text


def _set_by_dot_path(cfg, key: str, value):
    parts = key.split(".")
    if not parts or any(p == "" for p in parts):
        raise ValueError(f"Invalid cfg option key: {key!r}")

    obj = cfg
    for part in parts[:-1]:
        if isinstance(obj, dict):
            if part not in obj:
                raise KeyError(f"Unknown config path: {key!r}, missing {part!r}")
            obj = obj[part]
        else:
            if not hasattr(obj, part):
                raise KeyError(f"Unknown config path: {key!r}, missing {part!r}")
            obj = getattr(obj, part)

    last = parts[-1]
    if isinstance(obj, dict):
        if last not in obj:
            raise KeyError(f"Unknown config key: {key!r}, missing {last!r}")
        obj[last] = value
    else:
        if not hasattr(obj, last):
            raise KeyError(f"Unknown config key: {key!r}, missing {last!r}")
        setattr(obj, last, value)


def apply_cfg_options(cfg, cfg_options):
    if not cfg_options:
        return cfg

    for item in cfg_options:
        if "=" not in item:
            raise ValueError(
                f"Invalid --cfg-options item {item!r}. Expected key=value."
            )
        key, raw_value = item.split("=", 1)
        key = key.strip()
        if not key:
            raise ValueError(f"Invalid empty config key in item {item!r}.")
        value = _parse_cfg_value(raw_value)
        _set_by_dot_path(cfg, key, value)

    return cfg


def _get_by_dot_path(cfg, key: str):
    parts = key.split(".")
    if not parts or any(p == "" for p in parts):
        raise ValueError(f"Invalid dot path: {key!r}")

    obj = cfg
    for part in parts:
        if isinstance(obj, dict):
            if part not in obj:
                raise KeyError(f"Unknown config path {key!r}, missing {part!r}")
            obj = obj[part]
        else:
            if not hasattr(obj, part):
                raise KeyError(f"Unknown config path {key!r}, missing {part!r}")
            obj = getattr(obj, part)
    return obj


def _sanitize_path_token(text: object, max_len: int = 80) -> str:
    text = str(text).strip()
    text = text.replace("/", "-").replace("\\", "-")
    text = re.sub(r"[^A-Za-z0-9_.=-]+", "-", text)
    text = text.strip("-_.")
    if not text:
        text = "empty"
    if len(text) > max_len:
        text = text[:max_len].rstrip("-_.")
    return text


def _short_key_name(key: str) -> str:
    return key.split(".")[-1]


def build_work_dir_suffix_from_cfg(cfg, keys: list[str]) -> str:
    if not keys:
        raise ValueError("--work-dir-suffix-keys requires at least one key.")

    parts = []
    for key in keys:
        value = _get_by_dot_path(cfg, key)
        short_key = _sanitize_path_token(_short_key_name(key), max_len=48)
        value_token = _sanitize_path_token(value, max_len=48)
        parts.append(f"{short_key}-{value_token}")

    return "__".join(parts)


def build_cfg_options_hash(cfg_options) -> str:
    if not cfg_options:
        return "noopts"
    text = "\n".join(str(x) for x in cfg_options)
    return hashlib.sha1(text.encode("utf-8")).hexdigest()[:8]


def maybe_append_work_dir_suffix(
    work_dir: str | None,
    cfg,
    suffix_keys: list[str] | None,
    cfg_options,
    append_hash: bool = False,
) -> str | None:
    if work_dir is None:
        return None
    if not suffix_keys:
        return work_dir

    suffix = build_work_dir_suffix_from_cfg(cfg, suffix_keys)
    if append_hash:
        suffix = f"{suffix}__{build_cfg_options_hash(cfg_options)}"

    return str(Path(work_dir) / suffix)


def _to_builtin(obj):
    if isinstance(obj, dict):
        return {str(k): _to_builtin(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_to_builtin(v) for v in obj]
    return obj


def print_config_as_json(cfg) -> None:
    print(json.dumps(_to_builtin(cfg), indent=2, ensure_ascii=False))


def _unwrap_state_dict(obj: Any) -> Dict[str, torch.Tensor]:
    if not isinstance(obj, dict):
        raise TypeError(f"Unsupported checkpoint type: {type(obj)}")

    if "model" in obj and isinstance(obj["model"], dict):
        return obj["model"]

    if "state_dict" in obj and isinstance(obj["state_dict"], dict):
        return obj["state_dict"]

    if all(isinstance(k, str) for k in obj.keys()):
        return obj

    raise ValueError("Cannot find a valid state_dict in the checkpoint.")


def _strip_prefix_if_present(
    state_dict: Dict[str, torch.Tensor],
    prefix: str,
) -> Dict[str, torch.Tensor]:
    if not state_dict:
        return state_dict

    keys = list(state_dict.keys())
    if all(k.startswith(prefix) for k in keys):
        return {k[len(prefix):]: v for k, v in state_dict.items()}

    return state_dict


def load_model_weights_only(
    model: torch.nn.Module,
    path: str,
    strict: bool = False,
) -> Dict[str, Any]:
    ckpt = load_checkpoint_file(path)
    state_dict = _unwrap_state_dict(ckpt)
    state_dict = _strip_prefix_if_present(state_dict, "module.")

    missing_keys, unexpected_keys = model.load_state_dict(state_dict, strict=strict)

    print(f"Loaded model weights from {path}")
    if len(missing_keys) > 0:
        print(f"Missing keys: {missing_keys}")
    if len(unexpected_keys) > 0:
        print(f"Unexpected keys: {unexpected_keys}")

    return {
        "missing_keys": missing_keys,
        "unexpected_keys": unexpected_keys,
    }


def main():
    parser = argparse.ArgumentParser(
        description="Train/Eval SAM3 semantic segmentor with iter-based training."
    )
    parser.add_argument("config", type=str, help="path to config file")
    parser.add_argument("--work-dir", type=str, default=None)
    parser.add_argument("--resume-from", type=str, default=None)
    parser.add_argument("--load-model-from", type=str, default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--eval-only", action="store_true", help="only run validation")
    parser.add_argument(
        "--eval-iter",
        type=int,
        default=0,
        help="iter id used in eval-only outputs/logging",
    )
    parser.add_argument(
        "--cfg-options",
        nargs="+",
        default=None,
        help=(
            "Override config options, e.g. "
            "train_cfg.max_iters=1000 model.criterion_cfg.final_dice_weight=0.3"
        ),
    )
    parser.add_argument(
        "--print-config",
        action="store_true",
        help="Print merged config after --cfg-options and exit before building model.",
    )
    parser.add_argument(
        "--work-dir-suffix-keys",
        nargs="+",
        default=None,
        help=(
            "Append a deterministic subdirectory to --work-dir using values from "
            "the merged config. Example: "
            "--work-dir-suffix-keys model.freeze_cfg.openclip_text_finetune "
            "model.freeze_cfg.openclip_image_finetune"
        ),
    )
    parser.add_argument(
        "--work-dir-suffix-hash",
        action="store_true",
        help=(
            "Append an 8-char hash of all --cfg-options to the generated work-dir "
            "suffix to avoid collisions."
        ),
    )
    args = parser.parse_args()

    if args.resume_from is not None and args.load_model_from is not None:
        raise ValueError("--resume-from and --load-model-from cannot be used together.")

    cfg = Config.fromfile(args.config)
    cfg = _to_dotdict(cfg)
    cfg = apply_cfg_options(cfg, args.cfg_options)

    if args.print_config:
        print_config_as_json(cfg)
        return

    work_dir_override = maybe_append_work_dir_suffix(
        work_dir=args.work_dir,
        cfg=cfg,
        suffix_keys=args.work_dir_suffix_keys,
        cfg_options=args.cfg_options,
        append_hash=bool(args.work_dir_suffix_hash),
    )

    seed = args.seed if args.seed is not None else int(cfg.get("seed", 42))
    set_seed(seed)

    # ------------------------------------------------------------------
    # Work-dir guard: prevent mixing new training with old checkpoints
    # ------------------------------------------------------------------
    if not args.eval_only and args.resume_from is None:
        work_dir_for_check = Path(work_dir_override or cfg.get("work_dir", ""))
        if work_dir_for_check.is_dir():
            existing = sorted(work_dir_for_check.glob("iter_*.pth"))
            latest = work_dir_for_check / "latest.pth"
            best = work_dir_for_check / "best.pth"
            if existing or latest.exists() or best.exists():
                raise RuntimeError(
                    f"work_dir already contains training checkpoints: {work_dir_for_check}\n"
                    "Use a new --work-dir or explicitly resume with --resume-from."
                )

    model, criterion = build_training_components(**dict(cfg.model))

    if args.load_model_from is not None:
        load_model_weights_only(model=model, path=args.load_model_from, strict=False)

    (
        work_dir,
        trainer_cfg,
        hooks,
        visualizer,
        checkpoint_manager,
    ) = build_train_runtime_components(
        cfg,
        work_dir_override=work_dir_override,
    )

    print(f"Resolved work_dir: {work_dir}")
    Path(work_dir).mkdir(parents=True, exist_ok=True)

    if args.eval_only:
        if cfg.get("val_dataloader") is None:
            raise ValueError("val_dataloader is None, cannot run eval-only mode.")

        print("Building val_dataloader (eval-only)...")
        val_loader = build_dataloader(cfg.val_dataloader, seed=seed)

        trainer = Trainer(
            model=model,
            optimizer=None,
            criterion=criterion,
            train_dataloader=None,
            val_dataloader=val_loader,
            lr_scheduler=None,
            cfg=trainer_cfg,
            hooks=hooks,
            checkpoint_manager=checkpoint_manager,
            visualizer=visualizer,
            raw_cfg_for_logging=_to_builtin(cfg),
        )

        if args.resume_from:
            trainer.resume_from(args.resume_from)
        else:
            trainer.global_iter = int(args.eval_iter)

        trainer.hook_manager.call("before_run", trainer)
        try:
            trainer.val()
        except BaseException as exc:
            trainer.hook_manager.call(
                "on_exception",
                trainer,
                exc,
            )
            raise
        else:
            trainer.hook_manager.call("after_run", trainer)

        return

    print("Building train_dataloader...")
    train_loader = build_dataloader(cfg.train_dataloader, seed=seed)

    print("Building val_dataloader...")
    val_loader = build_dataloader(cfg.val_dataloader, seed=seed) if cfg.get("val_dataloader") else None

    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)

    trainer = Trainer(
        model=model,
        optimizer=optimizer,
        criterion=criterion,
        train_dataloader=train_loader,
        val_dataloader=val_loader,
        lr_scheduler=scheduler,
        cfg=trainer_cfg,
        hooks=hooks,
        checkpoint_manager=checkpoint_manager,
        visualizer=visualizer,
        raw_cfg_for_logging=_to_builtin(cfg),
    )

    if args.resume_from:
        trainer.resume_from(args.resume_from)

    trainer.train()


if __name__ == "__main__":
    main()