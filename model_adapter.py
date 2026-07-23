"""Load a pi0.5 (or SmolVLA) policy for evaluation from a checkpoint in EITHER
of two on-disk layouts, returning a uniform LeRobot policy + pre/post processors
that the sim/real eval loops drive unchanged.

1. **Native LeRobot checkpoint** — a directory with `config.json`, loaded via
   `PreTrainedConfig.from_pretrained` + `make_policy` (the standard path).
2. **Accelerate-style pi0.5 checkpoint** — a directory with `train_config.json`
   + `model.safetensors` (no LeRobot `config.json`), as written by some
   accelerate-based trainers. Its state dict wraps the policy under a `model.`
   prefix and carries extra training-only tensors (an auxiliary discrete-action
   head and in-model normalization buffers) that a LeRobot pi0.5 policy does not
   have. Since the inference architecture (the flow-matching expert) is the same,
   this loads the transformer weights into a LeRobot `PI05Policy`, drops the
   extra tensors, and builds the processors from the dataset stats — no offline
   checkpoint conversion, one runtime.

Callers use `load_policy()` in place of `PreTrainedConfig.from_pretrained` +
`make_policy`.
"""

import json
from pathlib import Path

from lerobot.configs.policies import PreTrainedConfig
from lerobot.configs.types import NormalizationMode
from lerobot.policies.factory import make_policy, make_pre_post_processors
from lerobot.policies.pi05.configuration_pi05 import PI05Config
from safetensors.torch import load_file

# Tensors present in an accelerate-style pi0.5 state dict but absent from a
# LeRobot pi0.5 policy: an auxiliary discrete-action head and in-model
# normalization buffers (LeRobot normalizes in the processor pipeline instead).
_AUX_TENSORS = ("da_head", "discrete_action", "normalize_", "unnormalize_")


def _is_accelerate_ckpt(ckpt: Path) -> bool:
    """True for an accelerate-style checkpoint: train_config.json + model.safetensors
    but no LeRobot config.json (which PreTrainedConfig.from_pretrained requires)."""
    return not (ckpt / "config.json").exists() and (ckpt / "model.safetensors").exists()


def dataset_repo_id(ckpt) -> str:
    """The training dataset's repo_id, read from either checkpoint layout's
    train_config.json — a LeRobot checkpoint stores it at `dataset.repo_id`; an
    accelerate-style checkpoint (which may train on a mixture) at
    `dataset_mixture.datasets[0].repo_id`."""
    tc = json.loads((Path(ckpt) / "train_config.json").read_text())
    if "dataset" in tc:
        return tc["dataset"]["repo_id"]
    return tc["dataset_mixture"]["datasets"][0]["repo_id"]


def load_policy(ckpt, ds_meta, device, *, n_action_steps=None, compile_model=False):
    """Return (policy, preprocessor, postprocessor, cfg) for the checkpoint,
    detecting its layout. `ds_meta` is the LeRobotDatasetMetadata of the dataset
    the policy trained on (its stats drive normalization)."""
    ckpt = Path(ckpt)

    if not _is_accelerate_ckpt(ckpt):
        # Native LeRobot checkpoint.
        cfg = PreTrainedConfig.from_pretrained(str(ckpt))
        cfg.pretrained_path = str(ckpt)
        cfg.device = str(device)
        cfg.compile_model = compile_model
        if n_action_steps is not None:
            cfg.n_action_steps = n_action_steps
        policy = make_policy(cfg, ds_meta=ds_meta, rename_map={})
        pre, post = make_pre_post_processors(
            cfg,
            pretrained_path=str(ckpt),
            preprocessor_overrides={"device_processor": {"device": str(device)}},
        )
        return policy.eval(), pre, post, cfg

    # Accelerate-style pi0.5: build a LeRobot PI05Config from train_config.json,
    # instantiate a LeRobot pi0.5 policy, load its transformer weights.
    p = json.loads((ckpt / "train_config.json").read_text())
    p = p.get("policy", p)
    cfg = PI05Config(
        chunk_size=p["chunk_size"],
        n_action_steps=n_action_steps or p["n_action_steps"],
        max_state_dim=p["max_state_dim"],
        max_action_dim=p["max_action_dim"],
        # Carry the checkpoint's normalization modes — they may differ from the
        # LeRobot pi0.5 default, and must match what the policy trained with for
        # the dataset stats to normalize identically.
        normalization_mapping={
            k: NormalizationMode[v] for k, v in p["normalization_mapping"].items()
        },
    )
    cfg.device = str(device)
    cfg.compile_model = compile_model

    policy = make_policy(cfg, ds_meta=ds_meta, rename_map={})
    sd = load_file(str(ckpt / "model.safetensors"))
    sd = {k: v for k, v in sd.items() if not any(s in k for s in _AUX_TENSORS)}
    _missing, unexpected = policy.load_state_dict(sd, strict=False)
    if unexpected:
        raise RuntimeError(
            f"checkpoint has {len(unexpected)} keys the LeRobot pi0.5 policy does "
            f"not expect (architecture mismatch?): {unexpected[:5]}"
        )
    policy = policy.to(device).eval()

    # Processors from the training dataset's own stats — same dataset, same
    # normalization modes => normalization identical to training.
    pre, post = make_pre_post_processors(
        cfg,
        dataset_stats=ds_meta.stats,
        preprocessor_overrides={"device_processor": {"device": str(device)}},
    )
    return policy, pre, post, cfg
