#!/usr/bin/env python3
"""
scripts/05_launch_aether_deck.py
--------------------------------
AETHER Telemetry Deck — FastAPI + WebSocket bridge.

Launches a local HTTP/WebSocket server on http://127.0.0.1:8765 that:
  * Serves dashboard/aether_deck.html at GET /
  * Exposes /telemetry (host CPU/OS/GPU metrics)
  * Exposes /model/{architecture}/predict (PyTorch forward pass + hook-based Grad-CAM)
  * Exposes /crosswalk (versioned label crosswalk + eligibility rules)
  * Exposes /eval (macro-F1 + patient-cluster bootstrap CI)
  * Streams versioned telemetry over WS /ws/telemetry

Designed to work with or without trained checkpoints: if models/CompactCNN1D.pt
or models/ResNet1D.pt is missing, the model is instantiated from
src/ecg_xval.models with lock_random_seed(42) so live forward passes and
real Grad-CAM saliency maps work immediately.

All optional deps (psutil, WMI) are wrapped in graceful try/except fallbacks
using stdlib platform/os so the server never crashes if a package is missing.
"""
from __future__ import annotations

import asyncio
import json
import math
import os
import platform
import random
import sys
import threading
import time
import webbrowser
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import torch
import torch.nn as nn
import yaml
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, JSONResponse
import uvicorn

# ---------------------------------------------------------------------------
# Path setup: make src/ecg_xval importable and locate project assets
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parent.parent
SRC_DIR = PROJECT_ROOT / "src"
DASHBOARD_DIR = PROJECT_ROOT / "dashboard"
CONFIGS_DIR = PROJECT_ROOT / "configs"
MODELS_DIR = PROJECT_ROOT / "models"
RESULTS_DIR = PROJECT_ROOT / "results"
SNAPSHOT_DIR = RESULTS_DIR / "aether_snapshots"

if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

# Ensure snapshot dir exists
SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

# ---------------------------------------------------------------------------
# Optional dependency imports with graceful stdlib fallbacks
# ---------------------------------------------------------------------------
try:  # psutil for rich host metrics
    import psutil  # type: ignore
    _HAS_PSUTIL = True
except ImportError:  # pragma: no cover - fallback path
    _HAS_PSUTIL = False

try:  # WMI for Windows GPU / OS detail (optional, Windows only)
    import wmi  # type: ignore
    _HAS_WMI = True
except ImportError:  # pragma: no cover - fallback path
    _HAS_WMI = False

# ---------------------------------------------------------------------------
# Import ecg_xval models (fallback-instantiated if checkpoints absent)
# ---------------------------------------------------------------------------
try:
    from ecg_xval.models.compact_cnn1d import CompactCNN1D
    from ecg_xval.models.resnet1d import ResNet1D
    from ecg_xval.utils.reproducibility import lock_random_seed
    _HAS_ECG_XVAL = True
except ImportError:  # pragma: no cover
    _HAS_ECG_XVAL = False

# Canonical rhythm classes mirror configs/crosswalk/label_crosswalk_v1.csv
CANONICAL_CLASSES: List[str] = ["NSR", "AF", "STACH", "SBRAD", "IAVB"]
CLASS_TO_IDX = {c: i for i, c in enumerate(CANONICAL_CLASSES)}
SAMPLE_RATE_HZ = 500
WINDOW_SECONDS = 10.0
WINDOW_SAMPLES = int(SAMPLE_RATE_HZ * WINDOW_SECONDS)  # 5000

@asynccontextmanager
async def lifespan(app: FastAPI):
    task = asyncio.create_task(_broadcast_loop())
    task_eval = asyncio.create_task(_eval_broadcast_loop())
    yield
    task.cancel()
    task_eval.cancel()
    for t in (task, task_eval):
        try:
            await t
        except BaseException:
            pass


APP = FastAPI(title="AETHER Telemetry Deck Bridge", version="1.0.0", lifespan=lifespan)
APP.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_origin_regex=".*",
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Model registry + Grad-CAM machinery
# ---------------------------------------------------------------------------
class _ModelBundle:
    """Wraps a torch model, loads checkpoint if available else instantiates fresh."""

    def __init__(self, architecture: str):
        self.architecture = architecture
        self.model: Optional[nn.Module] = None
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self._activations: Optional[torch.Tensor] = None
        self._gradients: Optional[torch.Tensor] = None
        self._hook_handles: List[Any] = []
        self._build()

    def _build(self) -> None:
        if not _HAS_ECG_XVAL:
            raise RuntimeError("ecg_xval package not importable; cannot build model bundle")
        lock_random_seed(42)
        if self.architecture == "compact_cnn1d":
            self.model = CompactCNN1D(in_channels=12, num_classes=len(CANONICAL_CLASSES), dropout=0.2)
        elif self.architecture == "resnet1d":
            self.model = ResNet1D(in_channels=12, num_classes=len(CANONICAL_CLASSES), base_filters=64, dropout=0.2)
        else:
            raise ValueError(f"unknown architecture: {self.architecture}")
        self.model.to(self.device).eval()

        ckpt = MODELS_DIR / f"{self.architecture}.pt"
        if ckpt.exists():
            try:
                state = torch.load(ckpt, map_location=self.device)
                self.model.load_state_dict(state)
                print(f"[aether] loaded checkpoint {ckpt}")
            except Exception as exc:  # pragma: no cover - defensive
                print(f"[aether] warning: failed to load {ckpt}: {exc}; using seed-42 init")
        else:
            print(f"[aether] no checkpoint at {ckpt}; using lock_random_seed(42) init")

        # Grad-CAM compatibility: disable in-place ReLU on this instance so
        # hook-based backward passes do not conflict with inplace view writes.
        for module in self.model.modules():
            if isinstance(module, nn.ReLU) and getattr(module, "inplace", False):
                module.inplace = False

    def _register_hooks(self, layer_name: str) -> None:
        self._remove_hooks()

        def fwd_hook(module, inp, out):
            # clone to avoid inplace-ReLU autograd view conflicts during backward
            self._activations = out.detach().clone()

        def bwd_hook(module, grad_in, grad_out):
            self._gradients = grad_out[0].detach().clone()

        target = dict(self.model.named_modules()).get(layer_name)
        if target is None:
            return
        self._hook_handles.append(target.register_forward_hook(fwd_hook))
        self._hook_handles.append(target.register_full_backward_hook(bwd_hook))

    def _remove_hooks(self) -> None:
        for h in self._hook_handles:
            try:
                h.remove()
            except Exception:
                pass
        self._hook_handles.clear()
        self._activations = None
        self._gradients = None

    def predict(self, window: np.ndarray, target_class: Optional[int] = None) -> Dict[str, Any]:
        """
        window: (12, 5000) float32 in mV
        returns: probabilities, grad-cam bands (in seconds), saliency map
        """
        if self.model is None:
            self._build()
        x = torch.from_numpy(window).float().unsqueeze(0).to(self.device)  # (1,12,5000)

        # default target = argmax of a first forward pass
        with torch.no_grad():
            logits = self.model(x)
            probs = torch.softmax(logits, dim=1).squeeze(0).cpu().numpy()
        if target_class is None:
            target_class = int(np.argmax(probs))

        # Grad-CAM
        bands = self._grad_cam(x, target_class)

        return {
            "architecture": self.architecture,
            "target_class": target_class,
            "target_class_name": CANONICAL_CLASSES[target_class],
            "probabilities": {CANONICAL_CLASSES[i]: float(p) for i, p in enumerate(probs)},
            "grad_cam_bands": bands,
            "sample_rate_hz": SAMPLE_RATE_HZ,
        }

    def _grad_cam(self, x: torch.Tensor, target_class: int) -> List[Dict[str, Any]]:
        """Hook-based Grad-CAM on the last conv layer; returns time bands in seconds."""
        # Choose sensible target layer per architecture
        layer_candidates = {
            "compact_cnn1d": ["features.2", "features.4", "features.6"],
            "resnet1d": ["layer3", "layer2", "layer1"],
        }[self.architecture]

        bands: List[Dict[str, Any]] = []
        self.model.zero_grad(set_to_none=True)

        for layer_name in layer_candidates:
            self._register_hooks(layer_name)
            try:
                x.requires_grad_(True)
                logits = self.model(x)
                score = logits[0, target_class]
                score.backward(retain_graph=True)

                if self._activations is None or self._gradients is None:
                    continue

                acts = self._activations[0]        # (C, T)
                grads = self._gradients[0]         # (C, T)
                weights = grads.mean(dim=1, keepdim=True)  # (C, 1)
                saliency = torch.relu((weights * acts).sum(dim=0))  # (T,)
                saliency = saliency.detach().cpu().numpy()

                if saliency.max() <= 0:
                    continue

                # Normalize & threshold -> contiguous bands
                saliency = saliency / (saliency.max() + 1e-8)
                threshold = 0.35
                above = saliency > threshold
                bands = self._contiguous_bands(above, SAMPLE_RATE_HZ, min_dur_sec=0.06)
                if bands:
                    break  # first usable layer wins
            finally:
                self._remove_hooks()
                x.requires_grad_(False)

        return bands

    @staticmethod
    def _contiguous_bands(mask: np.ndarray, sample_rate: int, min_dur_sec: float = 0.06) -> List[Dict[str, Any]]:
        bands: List[Dict[str, Any]] = []
        in_band = False
        start = 0
        min_samples = int(min_dur_sec * sample_rate)
        for i, v in enumerate(mask):
            if v and not in_band:
                in_band = True
                start = i
            elif not v and in_band:
                in_band = False
                if i - start >= min_samples:
                    bands.append({
                        "start_sec": round(start / sample_rate, 3),
                        "end_sec": round(i / sample_rate, 3),
                        "intensity": float(mask[start:i].max()),
                    })
        if in_band and len(mask) - start >= min_samples:
            bands.append({
                "start_sec": round(start / sample_rate, 3),
                "end_sec": round(len(mask) / sample_rate, 3),
                "intensity": float(mask[start:].max()),
            })
        return bands


_MODEL_BUNDLES: Dict[str, _ModelBundle] = {}


def get_model_bundle(architecture: str) -> _ModelBundle:
    if architecture not in _MODEL_BUNDLES:
        _MODEL_BUNDLES[architecture] = _ModelBundle(architecture)
    return _MODEL_BUNDLES[architecture]


# ---------------------------------------------------------------------------
# Host telemetry (psutil + WMI with stdlib fallbacks)
# ---------------------------------------------------------------------------
def get_host_telemetry() -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "logical_cpu_threads": None,
        "os_platform": platform.platform(),
        "os_system": platform.system(),
        "os_release": platform.release(),
        "python_version": platform.python_version(),
        "cpu_percent": None,
        "memory_total_gb": None,
        "memory_available_gb": None,
        "gpu_renderer": None,
        "gpu_driver": None,
    }

    if _HAS_PSUTIL:
        try:
            info["logical_cpu_threads"] = psutil.cpu_count(logical=True)
            info["cpu_percent"] = psutil.cpu_percent(interval=None)
            mem = psutil.virtual_memory()
            info["memory_total_gb"] = round(mem.total / (1024 ** 3), 2)
            info["memory_available_gb"] = round(mem.available / (1024 ** 3), 2)
        except Exception:  # pragma: no cover - defensive
            pass
    else:
        # stdlib fallback
        info["logical_cpu_threads"] = os.cpu_count()

    # GPU renderer via WMI (Windows) or stdlib placeholder
    gpu_name = None
    if _HAS_WMI and platform.system() == "Windows":
        try:
            w = wmi.WMI()
            for gpu in w.Win32_VideoController():
                if gpu.Name:
                    gpu_name = gpu.Name
                    info["gpu_driver"] = gpu.DriverVersion or None
                    break
        except Exception:  # pragma: no cover
            pass
    info["gpu_renderer"] = gpu_name or "WebGL2 (probe in browser)"

    return info


def load_crosswalk() -> Dict[str, Any]:
    csv_path = CONFIGS_DIR / "crosswalk" / "label_crosswalk_v1.csv"
    rules_path = CONFIGS_DIR / "crosswalk" / "eligibility_rules.yaml"
    rows: List[Dict[str, str]] = []
    if csv_path.exists():
        import csv as _csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            for row in _csv.DictReader(f):
                rows.append(row)
    rules: Dict[str, Any] = {}
    if rules_path.exists():
        with open(rules_path, encoding="utf-8") as f:
            rules = yaml.safe_load(f) or {}
    return {"canonical_classes": CANONICAL_CLASSES, "class_to_idx": CLASS_TO_IDX,
            "rows": rows, "eligibility_rules": rules}


def load_eval() -> Dict[str, Any]:
    """Best-effort read of latest evaluation outputs for the live readout."""
    out: Dict[str, Any] = {"macro_f1": None, "ci": [None, None], "all_estimable": True}
    candidates: List[Path] = []
    for pattern in ("**/*macro_f1*", "**/*eval*", "**/*bootstrap*"):
        candidates.extend(p for p in RESULTS_DIR.glob(pattern) if p.suffix in {".json", ".yaml", ".yml", ".csv"})
    for p in sorted(candidates):
        try:
            if p.suffix == ".json":
                data = json.loads(p.read_text(encoding="utf-8"))
                if "macro_f1" in data:
                    out["macro_f1"] = float(data["macro_f1"])
                    ci = data.get("bootstrap_ci") or data.get("ci")
                    if isinstance(ci, (list, tuple)) and len(ci) == 2 and ci[0] is not None:
                        out["ci"] = [float(ci[0]), float(ci[1])]
                    break
        except Exception:
            continue
    return out


# Cohort -> source database mapping (mirrors the frontend cohort identifiers)
COHORT_SOURCE_DBS: Dict[str, List[str]] = {
    "ptb_xl_within": ["ptb_xl"],
    "cpsc_2018_external": ["cpsc_2018"],
    "chapman_shaoxing_external": ["chapman_shaoxing"],
    "lopgo_ensemble": ["ptb_xl", "cpsc_2018", "chapman_shaoxing"],
}


def compute_eval_mask(source_dbs: List[str]) -> Dict[str, int]:
    """Per-canonical-class eval_mask (1 = evaluable, 0 = masked) for the given source DBs.

    Derived from configs/crosswalk/label_crosswalk_v1.csv using the primary mapping
    tier + inclusion status (see src/ecg_xval/harmonization/crosswalk.py).
    """
    eval_mask: Dict[str, int] = {c: 1 for c in CANONICAL_CLASSES}
    rows: List[Dict[str, str]] = []
    csv_path = CONFIGS_DIR / "crosswalk" / "label_crosswalk_v1.csv"
    if csv_path.exists():
        import csv as _csv
        with open(csv_path, newline="", encoding="utf-8") as f:
            rows = [r for r in _csv.DictReader(f) if r.get("canonical_class")]
    for cls in CANONICAL_CLASSES:
        evalable = False
        for db in source_dbs:
            cls_rows = [r for r in rows if (r.get("source_database") or "") == db and (r.get("canonical_class") or "").upper() == cls]
            if any((r.get("mapping_tier") or "").lower() == "primary" and (r.get("inclusion_status") or "").lower() == "included" for r in cls_rows):
                evalable = True
        eval_mask[cls] = 1 if evalable else 0
    return eval_mask


def _synthetic_window(rhythm: str, noise_pct: float = 15.0) -> np.ndarray:
    """Deterministic stand-in 12x5000 mV window for live forward passes / Grad-CAM."""
    sr = SAMPLE_RATE_HZ
    n = WINDOW_SAMPLES
    t = np.arange(n, dtype=np.float32) / sr
    bpm_map = {"NSR": 72.0, "AF": 118.0, "STACH": 132.0, "SBRAD": 46.0, "IAVB": 68.0}
    bpm = bpm_map.get((rhythm or "NSR").upper(), 72.0)
    rr = 60.0 / bpm
    beats: List[float] = []
    t0 = 0.0
    while t0 < 10.0:
        beats.append(round(t0, 4))
        t0 += rr
    sig = np.zeros((12, n), dtype=np.float32)
    lead2 = np.zeros(n, dtype=np.float32)
    for b in beats:
        i = int(round(b * sr))
        if i + 3 < n:
            lead2[i] += 1.6
            lead2[i + 1] += -0.8
            lead2[i + 2] += 1.0
    for j in range(12):
        scale = 0.35 if j == 1 else (0.25 + 0.05 * j)
        sig[j] = lead2 * scale
    if noise_pct and noise_pct > 0:
        rng = np.random.default_rng(42)
        sig += rng.normal(0, 0.02 * (noise_pct / 100.0), size=sig.shape).astype(np.float32)
    return sig.astype(np.float32)


# ---------------------------------------------------------------------------
# WebSocket connection manager
# ---------------------------------------------------------------------------
class ConnectionManager:
    def __init__(self):
        self.active: List[WebSocket] = []

    async def connect(self, ws: WebSocket):
        await ws.accept()
        self.active.append(ws)

    def disconnect(self, ws: WebSocket):
        if ws in self.active:
            self.active.remove(ws)

    async def broadcast(self, message: Dict[str, Any]):
        payload = json.dumps(message)
        dead = []
        for ws in self.active:
            try:
                await ws.send_text(payload)
            except Exception:
                dead.append(ws)
        for ws in dead:
            self.disconnect(ws)


MANAGER = ConnectionManager()


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@APP.get("/")
async def root():
    html_path = DASHBOARD_DIR / "aether_deck.html"
    if not html_path.exists():
        return JSONResponse({"error": "dashboard/aether_deck.html not found"}, status_code=404)
    return FileResponse(html_path, media_type="text/html")


@APP.get("/telemetry")
async def telemetry():
    return get_host_telemetry()


@APP.get("/crosswalk")
async def crosswalk():
    return load_crosswalk()


@APP.get("/eval")
async def eval_metrics():
    return load_eval()


@APP.post("/model/{architecture}/predict")
async def model_predict(architecture: str, payload: Dict[str, Any]):
    if architecture not in {"compact_cnn1d", "resnet1d"}:
        return JSONResponse({"error": "unknown architecture"}, status_code=400)
    window = payload.get("window")
    target_class = payload.get("target_class")
    if window is None:
        # synthesize a neutral window if none provided
        window = np.zeros((12, WINDOW_SAMPLES), dtype=np.float32)
    else:
        window = np.asarray(window, dtype=np.float32)
        if window.ndim == 1:
            # compact flat window -> interpret as 12 x N (or N x 12)
            if window.size > 0 and window.size % 12 == 0 and window.size // 12 <= WINDOW_SAMPLES:
                window = window.reshape(12, -1)
            elif window.size > 0:
                window = window.reshape(-1, 12).T
        if window.ndim != 2 or window.shape[0] < 12:
            window = np.repeat(window.reshape(-1, 1), 12, axis=0)
        if window.shape[1] < WINDOW_SAMPLES:
            pad = np.zeros((window.shape[0], WINDOW_SAMPLES - window.shape[1]), dtype=np.float32)
            window = np.concatenate([window, pad], axis=1)
        window = window[:, :WINDOW_SAMPLES]
    if window.shape != (12, WINDOW_SAMPLES):
        window = np.zeros((12, WINDOW_SAMPLES), dtype=np.float32)

    bundle = get_model_bundle(architecture)
    result = bundle.predict(window, target_class=int(target_class) if target_class is not None else None)
    return JSONResponse(result)


@APP.websocket("/ws/telemetry")
async def ws_telemetry(ws: WebSocket):
    await MANAGER.connect(ws)
    try:
        # push host telemetry immediately
        await ws.send_text(json.dumps({"type": "telemetry", **get_host_telemetry(), "ts": time.time()}))
        while True:
            msg = await ws.receive_text()
            # echo-control: client can request specific updates
            try:
                data = json.loads(msg)
            except json.JSONDecodeError:
                continue
            req = data.get("request")
            if req == "telemetry":
                await ws.send_text(json.dumps({"type": "telemetry", **get_host_telemetry(), "ts": time.time()}))
            elif req == "eval":
                await ws.send_text(json.dumps({"type": "eval", **load_eval(), "ts": time.time()}))
            elif req == "predict":
                rhythm = str(data.get("rhythm") or "NSR")
                noise_pct = float(data.get("noisePct") or 15.0)
                cohort = str(data.get("cohort") or "ptb_xl_within")
                source_dbs = COHORT_SOURCE_DBS.get(cohort, ["ptb_xl"])
                window = _synthetic_window(rhythm, noise_pct)
                bundle = get_model_bundle("compact_cnn1d")
                result = bundle.predict(window, target_class=None)
                await ws.send_text(json.dumps({"type": "gradcam", "ts": time.time(), **result, "eval_mask": compute_eval_mask(source_dbs)}))
            elif data.get("type") == "snapshot" or req == "snapshot":
                # persist audit snapshot to results/aether_snapshots/
                try:
                    ts = str(data.get("ts") or time.strftime("%Y-%m-%d %H:%M:%S"))
                    payload = {
                        "ts": ts,
                        "rhythm": data.get("rhythm"),
                        "bpm": data.get("bpm"),
                        "noise_pct": data.get("noisePct"),
                        "audio": data.get("audio"),
                        "source": data.get("source") or "deck",
                    }
                    SNAPSHOT_DIR.mkdir(parents=True, exist_ok=True)
                    (SNAPSHOT_DIR / f"snapshot_{time.strftime('%Y%m%d_%H%M%S')}_{str(data.get('ts') or 'x').replace(':', '')}.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
                except Exception:
                    pass
    except WebSocketDisconnect:
        MANAGER.disconnect(ws)
    except Exception:
        MANAGER.disconnect(ws)


# ---------------------------------------------------------------------------
# Background telemetry broadcaster (every 2s)
# ---------------------------------------------------------------------------
async def _broadcast_loop():
    while True:
        await MANAGER.broadcast({"type": "telemetry", **get_host_telemetry(), "ts": time.time()})
        await asyncio.sleep(2.0)


async def _eval_broadcast_loop():
    while True:
        await MANAGER.broadcast({"type": "eval", **load_eval(), "ts": time.time()})
        await asyncio.sleep(30.0)


# ---------------------------------------------------------------------------
# CLI entrypoint
# ---------------------------------------------------------------------------
def main(host: str = "127.0.0.1", port: int = 8765, open_browser: bool = True):
    print("=" * 62)
    print(" AETHER // Real-Time ECG Neural & Network Telemetry Deck")
    print(" Bridge (FastAPI + WebSocket) — Phase B")
    print("=" * 62)
    print(f" Project root : {PROJECT_ROOT}")
    print(f" Dashboard    : {DASHBOARD_DIR / 'aether_deck.html'}")
    print(f" Models dir   : {MODELS_DIR}")
    print(f" psutil       : {'ok' if _HAS_PSUTIL else 'missing (stdlib fallback)'}")
    print(f" WMI          : {'ok' if _HAS_WMI else 'missing (stdlib fallback)'}")
    print(f" ecg_xval     : {'ok' if _HAS_ECG_XVAL else 'missing'}")
    print("-" * 62)
    print(f" HTTP  : http://{host}:{port}/        (serves aether_deck.html)")
    print(f" WS    : ws://{host}:{port}/ws/telemetry")
    print(f" Endpoints: GET /telemetry  /crosswalk  /eval")
    print(f"          POST /model/{{architecture}}/predict")
    print("-" * 62)

    if open_browser:
        threading.Thread(target=lambda: webbrowser.open(f"http://{host}:{port}/"), daemon=True).start()

    uvicorn.run(APP, host=host, port=port, log_level="info")


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Launch AETHER Telemetry Deck bridge.")
    parser.add_argument("--host", type=str, default="127.0.0.1")
    parser.add_argument("--port", type=int, default=8765)
    parser.add_argument("--no-browser", action="store_true", help="Do not auto-open browser")
    args = parser.parse_args()
    main(host=args.host, port=args.port, open_browser=not args.no_browser)
