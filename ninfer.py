#!/usr/bin/env python3
"""Shared NInfer profile, Vast SDK, rental, and instance lifecycle support."""

from __future__ import annotations

import argparse
import collections
import dataclasses
import datetime
import importlib
import json
import os
import re
import shutil
import subprocess
import sys
import threading
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

# The standalone Vast installer keeps its SDK beside the CLI instead of on sys.path.
_vast_paths = list((Path.home() / ".local/share/vastai/current/lib").glob("python*/site-packages"))
_vast_paths += list((Path.home() / ".local/lib").glob("python*/site-packages"))
for _vast_path in _vast_paths:
    if str(_vast_path) not in sys.path:
        sys.path.insert(0, str(_vast_path))

try:
    VastAI = importlib.import_module("vastai").VastAI
except ImportError:
    VastAI = None

ROOT_DIR = Path(__file__).resolve().parent
DEFAULT_IMAGE = (
    "ghcr.io/kobusg/ninfer-sm120a-quant-kv"
    "@sha256:e7f5dbdd202bef2ac8fa48bef28584cae3a6fc6b5f513fef0e49bd280a2e1ae3"
)
DEFAULT_TEMPLATE = "/usr/share/doc/ninfer/froggeric-chat-template.jinja"
DEFAULT_REASONING = "medium"
DEFAULT_SSH_KEY = Path.home() / ".ssh" / "kobus"


@dataclasses.dataclass(frozen=True)
class ModelSpec:
    key: str
    repo: str
    filename: str
    sha256: str
    size_gib: float
    description: str


MODELS: Dict[str, ModelSpec] = {
    "nvfp4-27b": ModelSpec(
        "nvfp4-27b",
        "neroued/Qwen3.6-27B-nvfp4-NInfer",
        "qwen3_6_27b_nvfp4.ninfer",
        "bce5f00d066c0f20f1317bf1fdcb458264cf95837c3b1f3fbec163694627893a",
        17.07,
        "Qwen3.6-27B NVFP4",
    ),
    "int-27b": ModelSpec(
        "int-27b",
        "neroued/Qwen3.6-27B-NInfer",
        "qwen3_6_27b.ninfer",
        "7b51600ffd10632b9660f56085efdd9b751d79733ad32036a652234b64bebe7b",
        16.29,
        "Qwen3.6-27B groupwise-int",
    ),
    "qwen38-27b": ModelSpec(
        "qwen38-27b",
        "neroued/Qwen3.8-27B-NInfer",
        "qwen3_8_27b.ninfer",
        "eec39564993d6e9c7d5e383382a760f093465c9d163ec9a1bd6b80199514bf3e",
        16.96,
        "Qwen3.8-27B groupwise-int",
    ),
    "qwen38-nvfp4": ModelSpec(
        "qwen38-nvfp4",
        "neroued/Qwen3.8-27B-nvfp4-NInfer",
        "qwen3_8_27b_nvfp4.ninfer",
        "bb3360522a06e136e0367f5703414d26272b7285c8a6ab6194135c17dbd81b32",
        20.02,
        "Qwen3.8-27B NVFP4/FP8 official",
    ),
    "qwen38-quasar": ModelSpec(
        "qwen38-quasar",
        "MirkoCovizzi/Qwen3.8-27B-QUASAR-NVFP4-NInfer",
        "qwen3_8_27b_nvfp4.ninfer",
        "931816373707010b03e6e4dcba10f5265c3e820584dacd0eb2c6039e397045cd",
        16.35,
        "Qwen3.8-27B QUASAR NVFP4",
    ),
    "qwen38-nvfp4full": ModelSpec(
        "qwen38-nvfp4full",
        "cometkim/Qwen3.8-27B-nvfp4full-NInfer",
        "qwen3_8_27b_nvfp4full.ninfer",
        "2f59cc27d67cb7acba0ba8a0e0881ac89c1db2b267a60119a696fefa12faf4e7",
        17.07,
        "Qwen3.8-27B nvfp4full",
    ),
    "moe-35b": ModelSpec(
        "moe-35b",
        "neroued/Qwen3.6-35B-A3B-NInfer",
        "qwen3_6_35b_a3b.ninfer",
        "1fb9ea0b5b8561e49d9604115ec89e5d9f2b6f6434e32c37c57fffd480a325d2",
        21.22,
        "Qwen3.6-35B A3B",
    ),
}


@dataclasses.dataclass(frozen=True)
class ProfileConfig:
    name: str
    display_name: str
    provider: str
    target_dir: Path
    kind: str
    default_gpu: str
    default_model: str
    default_disk: int
    default_cuda_min: float
    default_secure_cloud: bool
    default_rank: int
    default_context: int
    default_kv_capacity: int
    default_concurrency: int
    default_prefill_chunk: int
    default_kv_dtype: str
    default_spec: str
    default_draft_tokens: int
    default_vision: bool
    default_vision_tokens: int
    default_chat_template: str
    default_reasoning_effort: str
    default_image: str


PROFILES: Dict[str, ProfileConfig] = {
    "5090": ProfileConfig(
        name="5090",
        display_name="NInfer RTX 5090",
        provider="ninfer5090",
        target_dir=ROOT_DIR / "5090",
        kind="entrypoint",
        default_gpu="RTX 5090",
        default_model="qwen38-nvfp4",
        default_disk=80,
        default_cuda_min=13.1,
        default_secure_cloud=True,
        default_rank=2,
        default_context=524288,
        default_kv_capacity=524288,
        default_concurrency=2,
        default_prefill_chunk=1024,
        default_kv_dtype="rk4v4-e8",
        default_spec="mtp",
        default_draft_tokens=3,
        default_vision=True,
        default_vision_tokens=8192,
        default_chat_template=DEFAULT_TEMPLATE,
        default_reasoning_effort=DEFAULT_REASONING,
        default_image=DEFAULT_IMAGE,
    ),
    "rtx6000pro-21gb": ProfileConfig(
        name="rtx6000pro-21gb",
        display_name="NInfer RTX6000 Pro 21GB",
        provider="ninfer6000pro21gb",
        target_dir=ROOT_DIR / "rtx6000pro-21gb",
        kind="entrypoint",
        default_gpu="RTX PRO 4000",
        default_model="qwen38-27b",
        default_disk=80,
        default_cuda_min=13.1,
        default_secure_cloud=True,
        default_rank=2,
        default_context=145920,
        default_kv_capacity=145920,
        default_concurrency=2,
        default_prefill_chunk=512,
        default_kv_dtype="rk4v4-e8",
        default_spec="mtp",
        default_draft_tokens=3,
        default_vision=True,
        default_vision_tokens=8192,
        default_chat_template=DEFAULT_TEMPLATE,
        default_reasoning_effort=DEFAULT_REASONING,
        default_image=DEFAULT_IMAGE,
    ),
    "3090": ProfileConfig(
        name="3090",
        display_name="NInfer RTX 3090",
        provider="ninfer",
        target_dir=ROOT_DIR / "3090",
        kind="legacy-3090",
        default_gpu="RTX 3090",
        default_model="qwen38-27b",
        default_disk=60,
        default_cuda_min=13.0,
        default_secure_cloud=False,
        default_rank=1,
        default_context=65536,
        default_kv_capacity=65536,
        default_concurrency=4,
        default_prefill_chunk=1024,
        default_kv_dtype="int8",
        default_spec="mtp",
        default_draft_tokens=3,
        default_vision=False,
        default_vision_tokens=0,
        default_chat_template="",
        default_reasoning_effort="",
        default_image="nvidia/cuda:13.1.2-devel-ubuntu24.04",
    ),
}

PROFILE_ALIASES = {
    "5090": "5090",
    "rtx5090": "5090",
    "rtx6000pro": "rtx6000pro-21gb",
    "rtx6000pro-21gb": "rtx6000pro-21gb",
    "6000pro": "rtx6000pro-21gb",
    "21gb": "rtx6000pro-21gb",
    "pro4000": "rtx6000pro-21gb",
    "3090": "3090",
    "rtx3090": "3090",
}


def credential(name: str, profile: Optional[str] = None) -> str:
    """Resolve a credential without requiring a profile-specific context."""
    if os.environ.get(name):
        return os.environ[name]
    candidates = [ROOT_DIR / ".env"]
    if profile:
        candidates.insert(0, PROFILES[PROFILE_ALIASES[profile]].target_dir / ".env")
    candidates.extend(PROFILES[key].target_dir / ".env" for key in PROFILES)
    for path in candidates:
        value = parse_env_file(path).get(name)
        if value:
            return value
    return ""


def infer_profile(info: Dict[str, Any]) -> str:
    """Infer a deployment profile from our label first, then the GPU model."""
    label = str(info.get("label") or "")
    for profile in PROFILES:
        if label.startswith(f"ninfer-{profile}-"):
            return profile

    gpu = str(info.get("gpu_name") or "").upper().replace("_", " ")
    if "5090" in gpu:
        return "5090"
    if "3090" in gpu:
        return "3090"
    if "PRO 4000" in gpu or "PRO 5000" in gpu or "6000 PRO" in gpu:
        return "rtx6000pro-21gb"
    raise ValueError(f"cannot infer a profile from GPU {info.get('gpu_name')!r}; use --profile")


def infer_model(info: Dict[str, Any], profile: str) -> str:
    env = info.get("extra_env") or {}
    if isinstance(env, dict):
        repo = env.get("NINFER_HF_REPO")
        filename = env.get("NINFER_MODEL_FILE")
        for key, spec in MODELS.items():
            if repo == spec.repo and filename == spec.filename:
                return key
    label = str(info.get("label") or "")
    prefix = f"ninfer-{profile}-"
    model = label[len(prefix):] if label.startswith(prefix) else ""
    return model if model in MODELS else PROFILES[profile].default_model


def instance_overrides(info: Dict[str, Any]) -> Dict[str, Any]:
    """Recover non-secret runtime choices from a contract created by ninfer-find."""
    env = info.get("extra_env") or {}
    if not isinstance(env, dict):
        return {}
    result: Dict[str, Any] = {}
    integer_fields = {
        "NINFER_MAX_CONTEXT": "context",
        "NINFER_KV_CAPACITY": "kv_capacity",
        "NINFER_MAX_CONCURRENCY": "concurrency",
        "NINFER_PREFILL_CHUNK": "prefill_chunk",
        "NINFER_DRAFT_TOKENS": "draft_tokens",
        "NINFER_VISION_MAX_TOKENS": "vision_tokens",
    }
    text_fields = {
        "NINFER_KV_DTYPE": "kv_dtype",
        "NINFER_SPEC": "spec",
        "NINFER_CHAT_TEMPLATE": "chat_template",
        "NINFER_REASONING_EFFORT": "reasoning_effort",
    }
    for source, target in integer_fields.items():
        if env.get(source):
            result[target] = int(env[source])
    for source, target in text_fields.items():
        if env.get(source):
            result[target] = env[source]
    if env.get("NINFER_VISION") is not None:
        result["vision"] = str(env["NINFER_VISION"]).lower() in ("1", "true", "yes", "on")
    return result


def context_args(**overrides: Any) -> argparse.Namespace:
    values = {
        "vast_api_key": None,
        "ninfer_api_key": None,
        "ssh_key": None,
        "model": None,
        "gpu_name": None,
        "disk": None,
        "cuda_min": None,
        "secure_cloud": None,
        "rank": None,
        "context": None,
        "kv_capacity": None,
        "concurrency": None,
        "prefill_chunk": None,
        "kv_dtype": None,
        "spec": None,
        "draft_tokens": None,
        "vision": None,
        "vision_tokens": None,
        "chat_template": None,
        "reasoning_effort": None,
        "image": None,
        "instance": None,
    }
    values.update(overrides)
    return argparse.Namespace(**values)


def parse_env_file(path: Path) -> Dict[str, str]:
    if not path.is_file():
        return {}
    values: Dict[str, str] = {}
    for raw in path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip()
        if (value.startswith('"') and value.endswith('"')) or (
            value.startswith("'") and value.endswith("'")
        ):
            value = value[1:-1]
        values[key] = value
    return values


def update_env_file(path: Path, updates: Dict[str, str]) -> None:
    lines = path.read_text(encoding="utf-8").splitlines() if path.is_file() else []
    for key, val in updates.items():
        entry = f"{key}={val}"
        replaced = False
        for i, line in enumerate(lines):
            if re.match(rf"^{re.escape(key)}=", line):
                lines[i] = entry
                replaced = True
                break
        if not replaced:
            lines.append(entry)
    path.write_text("\n".join(lines).rstrip("\n") + "\n", encoding="utf-8")


def read_state(path: Path) -> Optional[str]:
    if not path.is_file():
        return None
    content = path.read_text(encoding="utf-8").strip()
    return content or None


def write_state(path: Path, contract_id: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(f"{contract_id.strip()}\n", encoding="utf-8")


def clear_state(path: Path, contract_id: Optional[str] = None) -> None:
    if not path.is_file():
        return
    current = read_state(path)
    if contract_id is None or current == contract_id:
        path.unlink(missing_ok=True)


def resolve_instance_id(
    vast: "VastClient", explicit: Optional[str], state_path: Path
) -> str:
    if explicit:
        return explicit

    saved = read_state(state_path)
    if saved:
        try:
            if vast.get_instance(saved):
                return saved
        except Exception:
            pass
        clear_state(state_path, saved)

    instances = vast.list_instances()
    if len(instances) == 1:
        instance_id = str(instances[0]["id"])
        write_state(state_path, instance_id)
        return instance_id
    if not instances:
        raise SystemExit("error: no Vast instances found; pass an instance ID or create one")
    ids = ", ".join(str(row.get("id")) for row in instances)
    raise SystemExit(f"error: multiple Vast instances found ({ids}); pass an instance ID")


class VastClient:
    """Vast.ai client using the official vastai Python SDK."""

    def __init__(self, api_key: str):
        self.api_key = api_key
        if VastAI is None:
            raise RuntimeError(
                "vastai Python SDK is not installed or importable. "
                "Install it via 'pip install vastai' or install the vastai CLI."
            )
        self.sdk = VastAI(api_key=api_key) if api_key else VastAI()

    def get_instance(self, instance_id: str) -> Dict[str, Any]:
        res = self.sdk.show_instance(id=int(instance_id))
        return res or {}

    def list_instances(self) -> List[Dict[str, Any]]:
        try:
            return self.sdk.show_instances() or []
        except Exception:
            # Fallback to vastai CLI if SDK endpoint encounters client errors
            try:
                out = subprocess.run(
                    ["vastai", "show", "instances", "--raw"],
                    capture_output=True,
                    text=True,
                    check=True,
                )
                return json.loads(out.stdout) or []
            except Exception:
                return []

    def set_state(self, instance_id: str, state: str) -> None:
        iid = int(instance_id)
        if state == "running":
            self.sdk.start_instance(id=iid)
        elif state == "stopped":
            self.sdk.stop_instance(id=iid)
        else:
            raise ValueError(f"Unsupported state: {state}")

    def delete_instance(self, instance_id: str) -> None:
        self.sdk.destroy_instance(id=int(instance_id))

    def search_offers_cli(
        self,
        gpu_name: str,
        cuda_min: float,
        disk_space: int,
        direct_ports: int = 2,
        secure_cloud: bool = True,
        reliability: float = 0.97,
        limit: int = 100,
        order: str = "dph_total",
    ) -> List[Dict[str, Any]]:
        parts = [
            f"gpu_name={gpu_name.replace(' ', '_')}",
            "num_gpus=1",
            "verified=true",
            "rentable=true",
            f"disk_space>={disk_space}",
            "inet_down>=1000",
            f"reliability>={reliability}",
            f"cuda_vers>={cuda_min}",
        ]
        if direct_ports > 0:
            parts.append(f"direct_port_count>={direct_ports}")
        if secure_cloud:
            parts.append("datacenter=true")

        query_str = " ".join(parts)
        cmd = [
            "vastai",
            "search",
            "offers",
            query_str,
            "-o",
            order,
            "--limit",
            str(limit),
            "--raw",
        ]
        try:
            env = dict(os.environ)
            if self.api_key:
                env["VAST_API_KEY"] = self.api_key
            out = subprocess.run(cmd, env=env, capture_output=True, text=True, check=True)
            return json.loads(out.stdout) or []
        except Exception:
            return []

    def search_offers(
        self,
        query: Dict[str, Any],
        order: str = "dph_total",
        limit: int = 100,
        storage: float = 10.0,
    ) -> List[Dict[str, Any]]:
        try:
            return self.sdk.search_offers_new(
                query=query,
                order=order,
                limit=limit,
                storage=storage,
            ) or []
        except Exception:
            try:
                return self.sdk.search_offers(
                    query=query,
                    order=order,
                    limit=limit,
                    storage=storage,
                ) or []
            except Exception:
                return []

    def create_instance(
        self,
        offer_id: int,
        image: str,
        disk: float,
        label: str,
        env: Dict[str, str],
        runtype: str = "args",
        args_str: str = "",
        cancel_unavail: bool = True,
    ) -> str:
        res = self.sdk.create_instance(
            id=int(offer_id),
            image=image,
            disk=disk,
            label=label,
            env=env,
            runtype=runtype,
            args=args_str,
            cancel_unavail=cancel_unavail,
        )
        if not res.get("success"):
            raise RuntimeError(f"Vast refused offer {offer_id}: {res}")
        contract = res.get("new_contract")
        if not contract:
            raise RuntimeError(f"No contract ID returned: {res}")
        return str(contract)


class ContextManager:
    def __init__(self, profile_name: str, args: argparse.Namespace):
        self.profile = PROFILES[PROFILE_ALIASES[profile_name]]
        self.caller_dir = Path.cwd().resolve()
        self.target_dir = self.profile.target_dir

        self.root_env = parse_env_file(ROOT_DIR / ".env")
        self.prof_env = parse_env_file(self.target_dir / ".env")
        # Also inspect other profile env files so credentials can be shared across directories
        fallback_envs = [
            parse_env_file(ROOT_DIR / "5090" / ".env"),
            parse_env_file(ROOT_DIR / "rtx6000pro-21gb" / ".env"),
            parse_env_file(ROOT_DIR / "3090" / ".env"),
        ]

        def get_cred(key: str) -> str:
            val = (
                getattr(args, key.lower(), None)
                or os.environ.get(key)
                or self.prof_env.get(key)
                or self.root_env.get(key)
            )
            if val:
                return str(val)
            for fb in fallback_envs:
                if fb.get(key):
                    return str(fb[key])
            return ""

        self.vast_api_key = get_cred("VAST_API_KEY")
        self.ninfer_api_key = get_cred("NINFER_API_KEY")
        self.hf_token = get_cred("HF_TOKEN")

        ssh_key_val = (
            getattr(args, "ssh_key", None)
            or os.environ.get("NINFER_SSH_KEY")
            or self.prof_env.get("NINFER_SSH_KEY")
            or self.root_env.get("NINFER_SSH_KEY")
        )
        self.ssh_key = (
            Path(ssh_key_val).expanduser().resolve()
            if ssh_key_val
            else (
                DEFAULT_SSH_KEY
                if DEFAULT_SSH_KEY.exists()
                else Path.home() / ".ssh" / "id_ed25519"
            )
        )

        self.model_key = (
            getattr(args, "model", None)
            or os.environ.get("NINFER_MODEL")
            or self.prof_env.get("NINFER_MODEL")
            or self.profile.default_model
        )
        if self.model_key not in MODELS:
            valid = ", ".join(MODELS.keys())
            raise SystemExit(f"error: unknown model '{self.model_key}' — pick one of: {valid}")
        self.model_spec = MODELS[self.model_key]

        self.gpu_name = (
            getattr(args, "gpu_name", None)
            or os.environ.get("NINFER_GPU_NAME")
            or self.prof_env.get("NINFER_GPU_NAME")
            or self.profile.default_gpu
        )
        self.disk_gb = int(
            getattr(args, "disk", None)
            or os.environ.get("NINFER_DISK")
            or self.prof_env.get("NINFER_DISK")
            or self.profile.default_disk
        )
        self.cuda_min = float(
            getattr(args, "cuda_min", None)
            or os.environ.get("CUDA_MIN")
            or self.prof_env.get("CUDA_MIN")
            or self.profile.default_cuda_min
        )

        sec_cloud = (
            getattr(args, "secure_cloud", None)
            if hasattr(args, "secure_cloud") and args.secure_cloud is not None
            else os.environ.get("NINFER_SECURE_CLOUD")
        )
        if sec_cloud is None:
            sec_cloud = self.prof_env.get("NINFER_SECURE_CLOUD")
        if sec_cloud is None:
            self.secure_cloud = self.profile.default_secure_cloud
        else:
            self.secure_cloud = str(sec_cloud).lower() in ("1", "true", "yes", "on")

        self.offer_rank = int(
            getattr(args, "rank", None)
            or os.environ.get("NINFER_OFFER_RANK")
            or self.prof_env.get("NINFER_OFFER_RANK")
            or self.profile.default_rank
        )

        self.context = int(
            getattr(args, "context", None)
            or os.environ.get("NINFER_MAX_CONTEXT")
            or self.prof_env.get("NINFER_MAX_CONTEXT")
            or self.profile.default_context
        )
        self.kv_capacity = int(
            getattr(args, "kv_capacity", None)
            or os.environ.get("NINFER_KV_CAPACITY")
            or self.prof_env.get("NINFER_KV_CAPACITY")
            or self.profile.default_kv_capacity
        )
        self.concurrency = int(
            getattr(args, "concurrency", None)
            or os.environ.get("NINFER_MAX_CONCURRENCY")
            or self.prof_env.get("NINFER_MAX_CONCURRENCY")
            or self.profile.default_concurrency
        )
        self.prefill_chunk = int(
            getattr(args, "prefill_chunk", None)
            or os.environ.get("NINFER_PREFILL_CHUNK")
            or self.prof_env.get("NINFER_PREFILL_CHUNK")
            or self.profile.default_prefill_chunk
        )
        self.kv_dtype = (
            getattr(args, "kv_dtype", None)
            or os.environ.get("NINFER_KV_DTYPE")
            or self.prof_env.get("NINFER_KV_DTYPE")
            or self.profile.default_kv_dtype
        )
        self.spec_mode = (
            getattr(args, "spec", None)
            or os.environ.get("NINFER_SPEC")
            or self.prof_env.get("NINFER_SPEC")
            or self.profile.default_spec
        )
        self.draft_tokens = int(
            getattr(args, "draft_tokens", None)
            or os.environ.get("NINFER_DRAFT_TOKENS")
            or self.prof_env.get("NINFER_DRAFT_TOKENS")
            or self.profile.default_draft_tokens
        )
        raw_vision = (
            getattr(args, "vision", None)
            if hasattr(args, "vision") and args.vision is not None
            else os.environ.get("NINFER_VISION")
            or self.prof_env.get("NINFER_VISION")
            or ("1" if self.profile.default_vision else "0")
        )
        if raw_vision is None:
            self.vision = self.profile.default_vision
        elif isinstance(raw_vision, (int, bool)):
            self.vision = bool(raw_vision)
        else:
            self.vision = str(raw_vision).strip().lower() in ("1", "true", "yes", "on")
        self.vision_tokens = int(
            getattr(args, "vision_tokens", None)
            or os.environ.get("NINFER_VISION_MAX_TOKENS")
            or self.prof_env.get("NINFER_VISION_MAX_TOKENS")
            or self.profile.default_vision_tokens
        )
        self.chat_template = (
            getattr(args, "chat_template", None)
            or os.environ.get("NINFER_CHAT_TEMPLATE")
            or self.prof_env.get("NINFER_CHAT_TEMPLATE")
            or self.profile.default_chat_template
        )
        self.reasoning_effort = (
            getattr(args, "reasoning_effort", None)
            or os.environ.get("NINFER_REASONING_EFFORT")
            or self.prof_env.get("NINFER_REASONING_EFFORT")
            or self.profile.default_reasoning_effort
        )
        self.image = (
            getattr(args, "image", None)
            or os.environ.get("NINFER_IMAGE")
            or self.prof_env.get("NINFER_IMAGE")
            or self.profile.default_image
        )

        self.state_file = self.target_dir / ".ninfer-instance"
        self.badhosts_file = self.target_dir / ".ninfer-badhosts"

        instance_arg = getattr(args, "instance", None) or os.environ.get("NINFER_INSTANCE")
        self.instance_id = instance_arg or read_state(self.state_file)

        self.vast = VastClient(self.vast_api_key)

    def require_keys(self) -> None:
        self.require_vast_key()
        if not self.ninfer_api_key:
            raise SystemExit(
                f"error: missing NINFER_API_KEY in environment or {self.target_dir / '.env'}"
            )

    def require_vast_key(self) -> None:
        if not self.vast_api_key:
            raise SystemExit(
                f"error: missing VAST_API_KEY in environment or {self.target_dir / '.env'}"
            )

    def require_instance(self) -> str:
        self.require_keys()
        if not self.instance_id:
            raise SystemExit("error: no active instance — run create first")
        return self.instance_id

    def bad_machines(self) -> set[str]:
        if not self.badhosts_file.is_file():
            return set()
        bad = set()
        for line in self.badhosts_file.read_text(encoding="utf-8").splitlines():
            parts = line.strip().split()
            if parts:
                bad.add(parts[0])
        return bad

    def record_bad_machine(self, machine_id: Optional[str], reason: str) -> None:
        if not machine_id:
            return
        bad = self.bad_machines()
        if str(machine_id) not in bad:
            self.badhosts_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.badhosts_file, "a", encoding="utf-8") as f:
                date_str = datetime.datetime.now(datetime.timezone.utc).strftime("%Y-%m-%d")
                clean_reason = " ".join(reason.split())[:90]
                f.write(f"{machine_id}  {date_str}  {clean_reason}\n")
            print(f"  machine {machine_id} recorded in {self.badhosts_file.name} - will be skipped")


def update_opencode(path: Path, provider: str, name: str, url: str, model: str, context: int) -> None:
    data = json.loads(path.read_text(encoding="utf-8"), object_pairs_hook=collections.OrderedDict) if path.exists() else {}
    providers = data.setdefault("provider", collections.OrderedDict())
    entry = providers.setdefault(provider, collections.OrderedDict())
    entry["npm"] = "@ai-sdk/openai-compatible"
    entry["name"] = name
    options = entry.setdefault("options", collections.OrderedDict())
    options["baseURL"] = url
    options["apiKey"] = "{env:NINFER_API_KEY}"
    entry["models"] = collections.OrderedDict([
        (model, collections.OrderedDict([
            ("name", f"{model} [{name}]"),
            ("model", model),
            ("limit", collections.OrderedDict([("context", context), ("output", 8192)])),
            ("modalities", collections.OrderedDict([
                ("input", ["text", "image"]),
                ("output", ["text"]),
            ])),
        ])),
    ])
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
    print(f"  updated:   {path}")
    print(f"             provider {provider}/{model} -> {url}")


def update_harness(path: Path, provider: str, name: str, url: str, model: str, context: int, api_key: str) -> None:
    quoted = lambda value: json.dumps(value, ensure_ascii=True)
    block = [
        f"    {provider}:",
        "      apiKeyEnv: NINFER_API_KEY",
        "      api: openai-completions",
        f"      name: {quoted(name)}",
        f"      baseURL: {quoted(url)}",
        "      compat:",
        "        supportsDeveloperRole: false",
        "        maxTokensField: max_tokens",
        "      models:",
        f"        - id: {quoted(model)}",
        f"          name: {quoted(model + ' [' + name + ']')}",
        f"          contextWindow: {context}",
        "          maxTokens: 8192",
        "          input: [text, image]",
    ]
    lines = path.read_text(encoding="utf-8").splitlines() if path.exists() else []
    indent = lambda line: len(line) - len(line.lstrip(" "))
    meaningful = lambda line: bool(line.strip()) and not line.lstrip().startswith("#")

    def section_end(start: int, level: int) -> int:
        for index in range(start + 1, len(lines)):
            if meaningful(lines[index]) and indent(lines[index]) <= level:
                return index
        return len(lines)

    top = next((i for i, line in enumerate(lines)
                if indent(line) == 0 and line.strip() == "llm-pi-ai:"), None)
    if top is None:
        if lines and lines[-1].strip():
            lines.append("")
        lines.extend(["llm-pi-ai:", "  providers:", *block])
    else:
        top_end = section_end(top, 0)
        providers = next((i for i in range(top + 1, top_end)
                          if indent(lines[i]) == 2 and lines[i].strip() == "providers:"), None)
        if providers is None:
            lines[top_end:top_end] = ["  providers:", *block]
        else:
            providers_end = section_end(providers, 2)
            current = next((i for i in range(providers + 1, providers_end)
                            if indent(lines[i]) == 4 and lines[i].strip() == f"{provider}:"), None)
            if current is None:
                lines[providers_end:providers_end] = block
            else:
                lines[current:section_end(current, 4)] = block

    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    env = path.parent / ".env"
    env_lines = env.read_text(encoding="utf-8").splitlines() if env.exists() else []
    entry = "NINFER_API_KEY=" + api_key
    env_lines = [entry if line.startswith("NINFER_API_KEY=") else line for line in env_lines]
    if not any(line.startswith("NINFER_API_KEY=") for line in env_lines):
        env_lines.append(entry)
    fd = os.open(env, os.O_WRONLY | os.O_CREAT | os.O_TRUNC, 0o600)
    with os.fdopen(fd, "w", encoding="utf-8") as stream:
        stream.write("\n".join(env_lines) + "\n")
    env.chmod(0o600)

    ignore = path.parent / ".gitignore"
    ignored = ignore.read_text(encoding="utf-8").splitlines() if ignore.exists() else []
    if ".env" not in ignored:
        ignored.append(".env")
        ignore.write_text("\n".join(ignored) + "\n", encoding="utf-8")
    print(f"  updated:   {path}")
    print(f"             run with DSH_HOME={path.parent}")


def update_client_configs(ctx: ContextManager, url: str, model_id: str) -> None:
    if not model_id:
        print("  server named no model — project client config skipped")
        return

    update_opencode(
        ctx.caller_dir / "opencode.json",
        ctx.profile.provider,
        ctx.profile.display_name,
        url,
        model_id,
        ctx.context,
    )
    update_harness(
        ctx.caller_dir / ".dsh" / "settings.yaml",
        ctx.profile.provider,
        ctx.profile.display_name,
        url,
        model_id,
        ctx.context,
        ctx.ninfer_api_key,
    )

    target_env = ctx.target_dir / ".env"
    if target_env.is_file():
        update_env_file(target_env, {"NINFER_BASE_URL": url})
        print(f"  updated:   {target_env}")


def wait_for_status(ctx: ContextManager, want: str, limit: int = 900) -> Dict[str, Any]:
    waited = 0
    instance_id = ctx.require_instance()
    while True:
        try:
            info = ctx.vast.get_instance(instance_id)
        except Exception:
            info = {}

        status = info.get("actual_status") or "?"
        msg = " ".join((info.get("status_msg") or "").split())[:300]
        if status == want:
            print(f"  status: {status} ({waited}s)")
            return info

        if "Error" in msg:
            print()
            ctx.record_bad_machine(info.get("machine_id"), msg)
            raise SystemExit(f"error: vast could not start this box: {msg}")

        if waited >= limit:
            print()
            ctx.record_bad_machine(info.get("machine_id"), f"startup timeout: {status} {msg}")
            raise SystemExit(
                f"error: timed out after {limit}s waiting for '{want}' (still '{status}')"
                + (f" — vast says: {msg}" if msg else "")
            )

        print(f"\r  waiting… {status} ({waited}s)   ", end="", flush=True)
        time.sleep(5)
        waited += 5


def get_ssh_target(info: Dict[str, Any], kind: str) -> Tuple[str, str]:
    if kind == "legacy-3090":
        host = info.get("ssh_host") or ""
        port = str(info.get("ssh_port") or "")
        return host.strip(), port.strip()

    ip = (info.get("public_ipaddr") or "").strip()
    ports = info.get("ports") or {}
    p22 = (ports.get("22/tcp") or [{}])[0].get("HostPort") or ""
    return ip, str(p22).strip()


def get_endpoint(info: Dict[str, Any]) -> str:
    ip = (info.get("public_ipaddr") or "").strip()
    ports = info.get("ports") or {}
    p8080 = (ports.get("8080/tcp") or [{}])[0].get("HostPort") or ""
    return f"http://{ip}:{p8080}/v1" if ip and p8080 else ""


def run_ssh(
    ctx: ContextManager,
    cmd: Optional[str] = None,
    interactive: bool = False,
    input_text: Optional[str] = None,
) -> subprocess.CompletedProcess[str]:
    instance_id = ctx.require_instance()
    info = ctx.vast.get_instance(instance_id)
    host, port = get_ssh_target(info, ctx.profile.kind)
    if not host or not port:
        raise SystemExit("error: SSH address not ready yet")

    ssh_cmd = [
        "ssh",
        "-o",
        "BatchMode=yes" if not interactive else "BatchMode=no",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-i",
        str(ctx.ssh_key),
        "-p",
        str(port),
        f"root@{host}",
    ]
    if cmd:
        ssh_cmd.append(cmd)

    if interactive:
        sys.exit(subprocess.call(ssh_cmd))

    return subprocess.run(
        ssh_cmd,
        input=input_text,
        text=True,
        capture_output=True,
        check=False,
    )


def wait_for_ssh(ctx: ContextManager, limit: int = 300) -> None:
    waited = 0
    while True:
        res = run_ssh(ctx, "true")
        if res.returncode == 0:
            print(f"  ssh ready ({waited}s)")
            return
        if waited >= limit:
            print()
            info = ctx.vast.get_instance(ctx.require_instance())
            ctx.record_bad_machine(info.get("machine_id"), "entrypoint SSH timeout")
            raise SystemExit(
                "error: SSH never came up — inspect 'ninfer log', then destroy and re-rent"
            )
        print(f"\r  waiting for ssh… ({waited}s)   ", end="", flush=True)
        time.sleep(5)
        waited += 5


def check_health(url: str, api_key: str) -> int:
    try:
        req = urllib.request.Request(f"{url}/models")
        req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=15) as resp:
            return resp.status
    except urllib.error.HTTPError as e:
        return e.code
    except Exception:
        return 0


def get_served_model(url: str, api_key: str) -> str:
    try:
        req = urllib.request.Request(f"{url}/models")
        req.add_header("Authorization", f"Bearer {api_key}")
        with urllib.request.urlopen(req, timeout=15) as resp:
            data = json.load(resp)
            models = [m.get("id") for m in data.get("data", []) if m.get("id")]
            return models[0] if models else ""
    except Exception:
        return ""


def ensure_serving(ctx: ContextManager) -> None:
    print("\n\033[1mwaiting for the model to load\033[0m")
    waited = 0
    info = ctx.vast.get_instance(ctx.require_instance())
    url = get_endpoint(info)
    if not url:
        raise SystemExit("error: endpoint not assigned yet")

    while True:
        code = check_health(url, ctx.ninfer_api_key)
        if code == 200:
            break

        info = ctx.vast.get_instance(ctx.require_instance())
        if info.get("actual_status") != "running":
            raise SystemExit("error: container exited while provisioning (check: ninfer log)")
        if waited >= 1800:
            raise SystemExit(f"error: server never answered on {url} (check: ninfer log)")

        print(f"\r  loading… ({waited}s)   ", end="", flush=True)
        time.sleep(5)
        waited += 5

    print(f"\n  serving ({waited}s)")
    print("\n\033[1mconfiguring project clients\033[0m")
    model_id = get_served_model(url, ctx.ninfer_api_key) or ctx.model_spec.key
    update_client_configs(ctx, url, model_id)

    rate = info.get("dph_total", 0.0)
    print(f"\n\033[1mready — {url}\033[0m")
    print(f"  model:    {ctx.model_spec.filename} ({ctx.model_spec.size_gib:.2f} GiB, {ctx.model_spec.key})")
    print(f"  opencode: pick {ctx.profile.provider}/{model_id}")
    print(f"  costing ${rate:.4f}/hr; 'ninfer destroy' when you're done")


# --- Command Implementations ---


def offers_for_context(ctx: ContextManager) -> List[Dict[str, Any]]:
    ctx.require_vast_key()
    bad = ctx.bad_machines()
    direct_ports = 0 if ctx.profile.kind == "legacy-3090" else 2
    offers = ctx.vast.search_offers_cli(
        gpu_name=ctx.gpu_name,
        cuda_min=ctx.cuda_min,
        disk_space=ctx.disk_gb + 10,
        direct_ports=direct_ports,
        secure_cloud=ctx.secure_cloud,
        reliability=0.97,
        limit=100,
        order="dph_total",
    )
    return [x for x in offers if str(x.get("machine_id")) not in bad]


def cmd_offers(ctx: ContextManager) -> None:
    valid_offers = offers_for_context(ctx)

    mode_label = "Secure Cloud" if ctx.secure_cloud else "Community / Verified"
    ports_label = "2 direct ports, " if ctx.profile.kind != "legacy-3090" else ""
    print(
        f"\n\033[1mrentable {ctx.gpu_name} ({mode_label}, {ports_label}>= {ctx.disk_gb}+10 GB disk, "
        f">= 1 Gbps, CUDA {ctx.cuda_min}+, reliability >= 0.97)\033[0m"
    )
    print("  %-9s %8s %11s %9s %6s  %s" % ("offer", "gpu/hr", "stor/GB/mo", "down", "rel", "location"))
    for x in valid_offers[:10]:
        print(
            "  %-9s $%.4f  $%9.4f %6.0fMb %6.4f  %s"
            % (
                x.get("id"),
                x.get("dph_total", 0.0),
                x.get("storage_cost", 0.0),
                x.get("inet_down", 0.0),
                x.get("reliability2", 0.0),
                x.get("geolocation", ""),
            )
        )
    if not valid_offers:
        print("  none right now matching the filter")
    else:
        print(f"  automatic selection rank: {ctx.offer_rank}")


def pick_offer(ctx: ContextManager) -> Dict[str, Any]:
    valid = offers_for_context(ctx)
    if len(valid) < ctx.offer_rank:
        raise SystemExit(
            f"error: only {len(valid)} offer(s) match; cannot select rank {ctx.offer_rank}"
        )
    return valid[ctx.offer_rank - 1]


def rent_instance(ctx: ContextManager, offer_id: Optional[int] = None) -> str:
    """Create a contract and return immediately; startup belongs to the root runner."""
    ctx.require_keys()
    if ctx.profile.kind == "legacy-3090":
        raise SystemExit(
            "error: standalone 3090 rental is not supported by the legacy kit; "
            "use 3090/ninfer create"
        )

    chosen: Optional[Dict[str, Any]] = None
    if offer_id is None:
        chosen = pick_offer(ctx)
        offer_id = int(chosen["id"])

    assert offer_id is not None
    if chosen:
        print(
            f"renting offer {offer_id} - ${chosen.get('dph_total', 0.0):.4f}/hr - "
            f"{chosen.get('geolocation', '')}"
        )
    else:
        print(f"renting offer {offer_id}")

    env_vars = {
        "-p 22:22": "1",
        "-p 8080:8080": "1",
        "OPEN_BUTTON_PORT": "8080",
        "NINFER_ENABLE_SSH": "1",
        "NINFER_HF_REPO": ctx.model_spec.repo,
        "NINFER_MODEL_FILE": ctx.model_spec.filename,
        "NINFER_MODEL_SHA256": ctx.model_spec.sha256,
        "NINFER_API_KEY": ctx.ninfer_api_key,
        "NINFER_KV_DTYPE": ctx.kv_dtype,
        "NINFER_MAX_CONTEXT": str(ctx.context),
        "NINFER_KV_CAPACITY": str(ctx.kv_capacity),
        "NINFER_MAX_CONCURRENCY": str(ctx.concurrency),
        "NINFER_PREFILL_CHUNK": str(ctx.prefill_chunk),
        "NINFER_SPEC": ctx.spec_mode,
        "NINFER_DRAFT_TOKENS": str(ctx.draft_tokens),
        "NINFER_VISION": "1" if ctx.vision else "0",
        "NINFER_VISION_MAX_TOKENS": str(ctx.vision_tokens),
        "NINFER_CHAT_TEMPLATE": ctx.chat_template,
        "NINFER_REASONING_EFFORT": ctx.reasoning_effort,
    }
    if ctx.hf_token:
        env_vars["HF_TOKEN"] = ctx.hf_token

    contract_id = ctx.vast.create_instance(
        offer_id=offer_id,
        image=ctx.image,
        disk=float(ctx.disk_gb),
        label=f"ninfer-{ctx.profile.name}-{ctx.model_spec.key}",
        env=env_vars,
        runtype="args",
        args_str="",
        cancel_unavail=True,
    )
    ctx.instance_id = contract_id
    write_state(ctx.state_file, contract_id)
    print(f"instance {contract_id}")
    print(f"next: ./ninfer wait {contract_id}")
    return contract_id


def cmd_status(ctx: ContextManager) -> None:
    if not ctx.instance_id:
        print("\n\033[1mno instance\033[0m")
        print("  select an offer with './ninfer-find PROFILE', then run './ninfer create OFFER_ID --profile PROFILE'")
        return

    ctx.require_keys()
    try:
        info = ctx.vast.get_instance(ctx.instance_id)
    except Exception as e:
        print(f"\n\033[1merror fetching status:\033[0m {e}")
        return

    status = info.get("actual_status", "unknown")
    gpu = info.get("gpu_name", ctx.gpu_name)
    label = info.get("label", "")
    dph = info.get("dph_total", 0.0)
    storage_cost = info.get("storage_total_cost", 0.0)
    disk_used = info.get("disk_usage", 0.0)
    disk_total = info.get("disk_space", 0.0)

    print(f"\n\033[1minstance {ctx.instance_id} — {gpu} — {status}\033[0m")
    print(f"  label:   {label}")
    print(f"  model:   {ctx.model_spec.filename} ({ctx.model_spec.size_gib:.2f} GiB)")
    print(f"  runtime: {ctx.image}")
    print(f"  rate:    ${dph:.4f}/hr running")
    print(f"           ${storage_cost:.4f}/hr stopped (disk only)")
    print(f"  disk:    {disk_used} GB used of {disk_total} GB")

    if status == "running":
        url = get_endpoint(info)
        host, port = get_ssh_target(info, ctx.profile.kind)
        code = check_health(url, ctx.ninfer_api_key) if url else 0
        serving_str = "  (serving)" if code == 200 else "  (not answering yet)"
        print(f"  api:     {url}")
        print(f"  ssh:     root@{host} -p {port}")
        print(f"  health:  HTTP {code}{serving_str}")


def cmd_destroy(ctx: ContextManager) -> None:
    instance_id = ctx.require_instance()
    try:
        info = ctx.vast.get_instance(instance_id)
        status = info.get("actual_status", "unknown")
    except Exception:
        status = "unknown"

    print(f"\n\033[1mdestroying instance {instance_id} (currently {status})\033[0m")
    print(f"  this deletes the disk — the selected model will be re-downloaded next time")
    ctx.vast.delete_instance(instance_id)
    clear_state(ctx.state_file, instance_id)
    print("  \033[1mdestroyed — idle cost is now $0.00/hr\033[0m")


def cmd_up(ctx: ContextManager) -> None:
    instance_id = ctx.require_instance()
    info = ctx.vast.get_instance(instance_id)
    status = info.get("actual_status")
    print(f"\n\033[1m1/3  starting instance {instance_id}\033[0m")
    if status == "running":
        print("  already running")
    else:
        ctx.vast.set_state(instance_id, "running")
        wait_for_status(ctx, "running", limit=300)

    print("\n\033[1m2/3  waiting for ssh\033[0m")
    wait_for_ssh(ctx)
    print("\n\033[1m3/3  server\033[0m")
    ensure_serving(ctx)


def cmd_down(ctx: ContextManager) -> None:
    instance_id = ctx.require_instance()
    info = ctx.vast.get_instance(instance_id)
    status = info.get("actual_status")
    if status != "running":
        print(f"\n\033[1malready {status}\033[0m")
        cmd_status(ctx)
        return

    print(f"\n\033[1mstopping instance {instance_id}\033[0m")
    ctx.vast.set_state(instance_id, "stopped")
    wait_for_status(ctx, "exited", limit=300)
    info = ctx.vast.get_instance(instance_id)
    storage_cost = info.get("storage_total_cost", 0.0)
    print("  \033[1mdown\033[0m")
    print(f"  disk kept: {info.get('disk_usage', 0)} GB of {info.get('disk_space', 0)} GB")
    print(f"  STILL BILLING ${storage_cost:.4f}/hr for storage")
    print("  use 'ninfer destroy' instead to pay nothing at all")


def cmd_ssh(ctx: ContextManager, extra_args: List[str]) -> None:
    instance_id = ctx.require_instance()
    info = ctx.vast.get_instance(instance_id)
    host, port = get_ssh_target(info, ctx.profile.kind)
    if not host or not port:
        raise SystemExit("error: SSH address not ready yet")

    ssh_cmd = [
        "ssh",
        "-o",
        "ConnectTimeout=10",
        "-o",
        "StrictHostKeyChecking=no",
        "-o",
        "UserKnownHostsFile=/dev/null",
        "-o",
        "LogLevel=ERROR",
        "-i",
        str(ctx.ssh_key),
        "-p",
        str(port),
        f"root@{host}",
    ] + extra_args
    sys.exit(subprocess.call(ssh_cmd))


def cmd_log(ctx: ContextManager) -> None:
    run_ssh(ctx, "tail -n 40 -f /root/serve.log", interactive=True)


def cmd_bench(ctx: ContextManager, concurrency_str: str = "1,2", tokens: int = 128) -> None:
    instance_id = ctx.require_instance()
    info = ctx.vast.get_instance(instance_id)
    url = get_endpoint(info)
    model_id = get_served_model(url, ctx.ninfer_api_key) if url else ""
    if not model_id:
        raise SystemExit("error: server is not answering — try 'ninfer status'")

    print(f"\n\033[1mbenchmark — {model_id}, concurrency {concurrency_str}, {tokens} tokens per stream\033[0m")
    print("  measured on the box over localhost; figures are ninfer-serve's own")

    py_script = f"""
import datetime, json, os, re, threading, time, urllib.request

KEY = {json.dumps(ctx.ninfer_api_key)}
MODEL = {json.dumps(model_id)}
MAXNEW = {tokens}
LOG = "/root/serve.log"
PROMPT = ("Write a detailed technical explanation of how paged KV caches work in a "
          "transformer inference engine, covering block allocation, prefix reuse, "
          "eviction and fragmentation. Be thorough and specific.")

def fire(i):
    body = json.dumps({{"model": MODEL, "max_tokens": MAXNEW, "messages":
        [{{"role": "user", "content": "%s (variant %d)" % (PROMPT, i)}}]}}).encode()
    req = urllib.request.Request("http://127.0.0.1:8080/v1/chat/completions", body,
        {{"Authorization": "Bearer " + KEY, "Content-Type": "application/json"}})
    try:
        urllib.request.urlopen(req, timeout=900).read()
    except Exception as e:
        print("    request %d failed: %s" % (i, e))

print("    %-3s %12s %12s %12s %9s" % ("C", "per-stream", "aggregate", "MTP", "accept"))
print("    %-3s %12s %12s %12s %9s" % ("", "tok/s", "tok/s", "tok/round", ""))
for C in [int(x) for x in {json.dumps(concurrency_str)}.split(",")]:
    at = os.path.getsize(LOG)
    ths = [threading.Thread(target=fire, args=(i,)) for i in range(C)]
    for t in ths: t.start()
    for t in ths: t.join()
    time.sleep(3)
    with open(LOG, "rb") as f:
        f.seek(at); new = f.read().decode("utf-8", "replace")
    dec  = [float(x) for x in re.findall(r"\\] done .*? decode=([0-9.]+)tok/s", new)]
    spec = re.findall(r"speculative=mtp ([0-9.]+)tok/round \\(([0-9.]+)%\\)", new)
    done = re.findall(r"^\\[([0-9-]+ [0-9:.]+)\\].*?\\] done .*? gen=([0-9]+) .*? wall=([0-9.]+)s",
                      new, re.M)
    if not dec:
        print("    %-3d  no completed requests appeared in the log" % C)
        continue
    agg = None
    if len(done) >= C:
        rows = []
        for ts, gen, wall in done[-C:]:
            end = datetime.datetime.strptime(ts, "%Y-%m-%d %H:%M:%S.%f")
            rows.append((end - datetime.timedelta(seconds=float(wall)), end, int(gen)))
        span = (max(r[1] for r in rows) - min(r[0] for r in rows)).total_seconds()
        if span > 0:
            agg = sum(r[2] for r in rows) / span
    print("    %-3d %12.1f %12s %12.2f %9.1f%%" % (
        C, sum(dec)/len(dec), "%.1f" % agg if agg else "—",
        sum(float(a) for a, _ in spec)/len(spec) if spec else 0,
        sum(float(b) for _, b in spec)/len(spec) if spec else 0))
print()
print("    per-stream is each request's own decode rate, as ninfer-serve reports it.")
print("    aggregate is total tokens over the wall time the burst actually occupied.")
"""
    res = run_ssh(ctx, "python3 -", input_text=py_script)
    print(res.stdout)
    if res.stderr:
        print(res.stderr, file=sys.stderr)


"""Create and operate Vast NInfer instances."""

import argparse
from pathlib import Path



def add_connection_args(p: argparse.ArgumentParser, require_profile: bool = False) -> None:
    p.add_argument("--profile", required=require_profile, choices=PROFILE_ALIASES)
    p.add_argument("--model", choices=MODELS)
    p.add_argument("--vast-api-key")
    p.add_argument("--ninfer-api-key")
    p.add_argument("--ssh-key")


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    create = sub.add_parser("create", help="Create a configured instance from an offer ID")
    create.add_argument("offer_id", type=int)
    add_connection_args(create, require_profile=True)
    create.add_argument("--disk", type=int)
    create.add_argument("--context", type=int)
    create.add_argument("--kv-capacity", type=int)
    create.add_argument("--concurrency", type=int)
    create.add_argument("--prefill-chunk", type=int)
    create.add_argument("--kv-dtype")
    create.add_argument("--spec")
    create.add_argument("--draft-tokens", type=int)
    create.add_argument("--vision", type=int, choices=(0, 1))
    create.add_argument("--vision-tokens", type=int)
    create.add_argument("--chat-template")
    create.add_argument("--reasoning-effort")
    create.add_argument("--image")

    find = sub.add_parser("find", help="Find best offer ID for a hardware profile")
    find.add_argument("profile", choices=PROFILE_ALIASES, help="Hardware/profile preset")
    find.add_argument("--gpu", dest="gpu_name")
    find.add_argument("--disk", type=int)
    find.add_argument("--rank", type=int)
    find.add_argument("--community", dest="secure_cloud", action="store_false", default=None)
    find.add_argument("--vast-api-key")

    offers = sub.add_parser("offers", help="List viable offers for a hardware profile")
    offers.add_argument("profile", choices=PROFILE_ALIASES, help="Hardware/profile preset")
    offers.add_argument("--gpu", dest="gpu_name")
    offers.add_argument("--disk", type=int)
    offers.add_argument("--rank", type=int)
    offers.add_argument("--community", dest="secure_cloud", action="store_false", default=None)
    offers.add_argument("--vast-api-key")

    for command in ("wait", "status", "up", "down", "stop", "destroy", "rm", "log"):
        command_parser = sub.add_parser(command)
        command_parser.add_argument("instance_id", nargs="?")
        add_connection_args(command_parser)

    ssh = sub.add_parser("ssh")
    ssh.add_argument("instance_id", nargs="?")
    add_connection_args(ssh)
    ssh.add_argument("ssh_args", nargs=argparse.REMAINDER)
    bench = sub.add_parser("bench")
    bench.add_argument("instance_id", nargs="?")
    add_connection_args(bench)
    bench.add_argument("concurrency", nargs="?", default="1,2")
    bench.add_argument("--tokens", type=int, default=128)
    return p


def main() -> None:
    args = parser().parse_args()

    if args.command == "create":
        ctx = ContextManager(args.profile, args)
        ctx.state_file = ctx.caller_dir / ".ninfer-instance"
        rent_instance(ctx, args.offer_id)
        return

    if args.command == "find":
        ctx = ContextManager(args.profile, context_args(**vars(args)))
        ctx.require_vast_key()
        print(pick_offer(ctx)["id"])
        return

    if args.command == "offers":
        ctx = ContextManager(args.profile, context_args(**vars(args)))
        cmd_offers(ctx)
        return

    vast_key = args.vast_api_key or credential("VAST_API_KEY", args.profile)
    if not vast_key:
        raise SystemExit("error: missing VAST_API_KEY")
    vast = VastClient(vast_key)
    args.instance_id = resolve_instance_id(
        vast, args.instance_id, Path.cwd().resolve() / ".ninfer-instance"
    )
    info = vast.get_instance(args.instance_id)
    if not info:
        raise SystemExit(f"error: Vast instance {args.instance_id} was not found")

    profile = PROFILE_ALIASES[args.profile] if args.profile else infer_profile(info)
    model = args.model or infer_model(info, profile)
    runtime = instance_overrides(info)
    runtime.update(
        instance=args.instance_id,
        model=model,
        vast_api_key=vast_key,
        ninfer_api_key=args.ninfer_api_key,
        ssh_key=args.ssh_key,
    )
    ctx = ContextManager(
        profile,
        context_args(**runtime),
    )
    ctx.state_file = ctx.caller_dir / ".ninfer-instance"

    if args.command == "wait":
        wait_for_status(ctx, "running")
        wait_for_ssh(ctx)
        ensure_serving(ctx)
    elif args.command == "status":
        cmd_status(ctx)
    elif args.command == "up":
        cmd_up(ctx)
    elif args.command in ("down", "stop"):
        cmd_down(ctx)
    elif args.command in ("destroy", "rm"):
        cmd_destroy(ctx)
    elif args.command == "ssh":
        cmd_ssh(ctx, args.ssh_args)
    elif args.command == "log":
        cmd_log(ctx)
    elif args.command == "bench":
        cmd_bench(ctx, args.concurrency, args.tokens)


if __name__ == "__main__":
    main()
