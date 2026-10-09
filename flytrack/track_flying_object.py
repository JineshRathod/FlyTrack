#!/usr/bin/env python
"""
track_flying_object.py  --  give it a video, it tracks the flying vehicle. No telemetry, no ground truth.

    python track_flying_object.py --video clip.mp4 --model fuselage_detector.tflite --out tracked.mp4 --csv track.csv

    # inside Colab / a notebook
    from track_flying_object import track_video
    results = track_video("clip.mp4", model="fuselage_detector.tflite", out="tracked.mp4", csv="track.csv")

What it does
  * Detector adapters: your TFLite model (single-box regressor [ymin,xmin,ymax,xmax], or an SSD-style
    detector -- auto-detected) and a model-free "small bright/dark blob" detector for distant, dot-like targets.
  * The regressor is run on a CROP around where the target should be (so a distant target fills the 224x224
    input the way it did in training), and on the full frame only while searching.
  * Motion-compensated multi-object tracker: global camera motion is estimated and removed, every object gets a
    track, detections are assigned jointly, and the locked subject is protected by gating (motion / size /
    appearance), an ambiguity guard, cannot-link memory and strict re-acquisition rules.
  * Visual-servo outputs per frame (yaw/pitch rate, forward cue) that keep the target centred at constant apparent
    size -- the image-space version of "hold a constant relative filming position".

Honest limits are listed at the bottom of this docstring's twin, README_video_tracker.md.
"""
import argparse
import csv
import itertools
import math
import os
import sys
import threading
import time
from collections import deque
from dataclasses import dataclass, field
from itertools import count
from typing import List, Optional

import cv2
import numpy as np
from scipy.optimize import linear_sum_assignment

try:                                                   # the telemetry HUD (hud.py) is optional
    from hud import LiveHudConfig, LiveTelemetry, draw_live_hud, hud_output_size
except ImportError:
    try:
        from dronefollow.hud import LiveHudConfig, LiveTelemetry, draw_live_hud, hud_output_size
    except ImportError:
        LiveHudConfig = LiveTelemetry = draw_live_hud = hud_output_size = None

INF = 1e6


# ============================================================================ config
@dataclass
class Config:
    # --- pipeline
    work_height: int = 540            # tracking runs at this height; the output video keeps the original size
    detector: str = "auto"            # auto | model | classical | hybrid   (auto: hybrid if a model is given)
    polarity: str = "both"            # blob detector: bright | dark | both
    allow_motion_lock: bool = False   # with a model: also accept steadily moving blobs as the subject (no model vote)

    # --- TFLite model
    input_range: str = "01"           # "01" (x/255), "pm1" (x/127.5-1) or "255" (raw 0..255)
    ssd_score_min: float = 0.3
    roi_target_frac: float = 0.40     # target's long side as a fraction of the model crop
    roi_min_side: int = 64
    conf_min: float = 0.5             # model agreement (IoU on a verification crop) needed to start / restart a lock
    verify_top_k: int = 3
    center_bias: float = 0.3          # with several valid candidates the auto-lock prefers the one nearer the image centre
    min_contrast: float = 1.0         # model boxes sitting on featureless background are discarded
    prefer_blob_box: bool = True      # when model and blob agree, measure with the tight blob box

    # --- blob detector (sizes are fractions of the working frame)
    tophat_frac: float = 0.08
    z_thresh: float = 8.0
    sigma_floor: float = 2.0
    close_kernel: int = 7
    min_area: int = 6
    max_w_frac: float = 0.25
    max_h_frac: float = 0.16
    max_aspect: float = 5.0
    max_blobs: int = 12
    core_frac: float = 0.35           # a blob box is cut back to pixels above this fraction of its peak when a faint tail stretched it

    # --- camera motion
    ego_comp: bool = True
    ego_min_resp: float = 0.05
    ego_max_frac: float = 0.10

    # --- tracker (pixel quantities scale with S = min(W, H) of the working frame)
    meas_sigma: float = 1.5
    meas_sigma_rel: float = 0.04
    q_jerk_frac: float = 0.15
    tau_acc: float = 1.5
    gate_chi2: float = 13.8
    app_min: float = 0.35
    size_tol: float = 2.0
    sigma_size: float = 0.45
    sigma_z: float = 0.6              # contrast consistency: a factor e^0.6 = 1.8 in blob contrast is 1 sigma
    agree_bonus: float = 1.0
    ambiguity_margin: float = 3.0
    short_coast_s: float = 1.0
    confirm_frames: int = 4
    tent_miss: int = 2
    delete_after_s: float = 4.0
    max_tracks: int = 40
    min_disp_frac: float = 0.012      # motion lock: net movement over the confirm window (fraction of S)
    min_straightness: float = 0.6
    lock_z_min: float = 25.0          # motion lock: blob contrast (robust z-score) required
    max_area_change: float = 2.0
    reacq_app_min: float = 0.6
    reacq_confirm: int = 5
    reacq_min_hits: int = 8
    reacq_margin: float = 2.0
    reacq_speed_frac: float = 0.5     # reachable radius grows by this fraction of S per second
    reacq_anywhere_s: float = 2.0     # after this long lost, the subject may reappear anywhere (cuts, long occlusions)
    impostor_radius_frac: float = 0.15
    impostor_app: float = 0.8
    rival_recent_s: float = 1.0
    strict_identity: bool = True

    # --- visual servo (outputs only; offline video does not react to them)
    hfov_deg: float = 60.0
    kp_angle: float = 2.0             # 1/s
    max_rate_dps: float = 90.0
    lead_s: float = 0.15
    k_size: float = 1.5
    k_size_rate: float = 0.8
    desired_h_frac: Optional[float] = None   # None -> keep the apparent size seen at lock

    # --- telemetry HUD on the video (needs hud.py). Everything the camera cannot measure is an explicit assumption.
    hud: bool = False
    hud_alt_m: float = 50.0           # ASSUMED hover altitude of the camera drone (it is also assumed to have zero ground speed)
    target_size_m: float = 12.0       # ASSUMED real length of the target's longest box side (range comes from apparent size)
    cam_heading_deg: float = 0.0      # ASSUMED compass heading of the optical axis
    cam_pitch_deg: float = 0.0        # ASSUMED camera pitch (+ = up)
    hud_dv_scale: float = 15.0        # m/s full scale on the delta-v gauges
    hud_scale: Optional[float] = None # None = auto from the frame size
    hud_layout: str = "auto"          # overlay | dock | auto  (dock puts the HUD in margins around the video so it never covers the picture)
    max_speed_mps: float = 150.0      # estimated speeds above this are shown as '--'


# ============================================================================ geometry helpers
def bw(b): return b[2] - b[0]
def bh(b): return b[3] - b[1]
def bc(b): return (b[0] + b[2]) / 2.0, (b[1] + b[3]) / 2.0
def barea(b): return max(bw(b), 0.0) * max(bh(b), 0.0)


def iou(a, b):
    ix = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    iy = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    i = ix * iy
    u = barea(a) + barea(b) - i
    return i / u if u > 0 else 0.0


def agreement(a, b):
    """How well two boxes describe the same object: mean of IoU and a centre-distance score. A box regressor is
    usually good at the centre and sloppy about the extent, so IoU alone under-rates it."""
    (ax, ay), (bx, by) = bc(a), bc(b)
    cs = max(0.0, 1.0 - math.hypot(ax - bx, ay - by) / (0.6 * max(bw(a), bh(a), 4.0)))
    return 0.5 * (iou(a, b) + cs)


def clamp_box(b, W, H):
    return [float(np.clip(b[0], 0, W)), float(np.clip(b[1], 0, H)), float(np.clip(b[2], 0, W)), float(np.clip(b[3], 0, H))]


def local_contrast(gray, box, pad=0.5):
    """|inside - surrounding ring| / (ring std + 5): is there *something* in this box?"""
    H, W = gray.shape[:2]
    x1, y1 = max(int(box[0]), 0), max(int(box[1]), 0)
    x2, y2 = min(int(math.ceil(box[2])), W), min(int(math.ceil(box[3])), H)
    if x2 - x1 < 2 or y2 - y1 < 2:
        return 0.0
    px, py = int((x2 - x1) * pad) + 2, int((y2 - y1) * pad) + 2
    X1, Y1, X2, Y2 = max(x1 - px, 0), max(y1 - py, 0), min(x2 + px, W), min(y2 + py, H)
    outer, inner = gray[Y1:Y2, X1:X2].astype(np.float64), gray[y1:y2, x1:x2].astype(np.float64)
    n = outer.size - inner.size
    if n <= 0:
        return 0.0
    mean = (outer.sum() - inner.sum()) / n
    var = max((np.square(outer).sum() - np.square(inner).sum()) / n - mean ** 2, 0.0)
    return abs(inner.mean() - mean) / (math.sqrt(var) + 5.0)


def track_sim(a, b):
    """Look-alike measure between two tracks: colour-histogram cosine x size agreement. In grey/IR video every
    bright blob has the same histogram, so size has to carry part of the discrimination."""
    rw, rh = min(a.w, b.w) / max(a.w, b.w, 1e-6), min(a.h, b.h) / max(a.h, b.h, 1e-6)
    return float(a.desc @ b.desc) * math.sqrt(rw * rh)


def describe(img, box):
    """Appearance descriptor: Hellinger-normalised colour histogram of the central part of the box."""
    H, W = img.shape[:2]
    cx, cy = bc(box)
    w, h = max(bw(box) * 0.7, 3.0), max(bh(box) * 0.7, 3.0)
    x1, y1, x2, y2 = int(max(cx - w / 2, 0)), int(max(cy - h / 2, 0)), int(min(cx + w / 2 + 1, W)), int(min(cy + h / 2 + 1, H))
    patch = img[y1:y2, x1:x2]
    if patch.size == 0:
        return np.ones(24) / np.sqrt(24)
    if patch.ndim == 2:
        patch = cv2.merge([patch] * 3)
    hist = np.concatenate([cv2.calcHist([patch], [c], None, [8], [0, 256]).ravel() for c in range(3)])
    hist = np.sqrt(hist / (hist.sum() + 1e-9))
    return hist / (np.linalg.norm(hist) + 1e-9)


# ============================================================================ TFLite model adapter
def _make_interpreter(path):
    try:
        import tensorflow as tf
        return tf.lite.Interpreter(model_path=path, num_threads=2)
    except ImportError:
        pass
    for mod in ("ai_edge_litert.interpreter", "tflite_runtime.interpreter"):
        try:
            return __import__(mod, fromlist=["Interpreter"]).Interpreter(model_path=path, num_threads=2)
        except ImportError:
            continue
    raise RuntimeError("No TFLite runtime found. Install one:  pip install tensorflow   (or ai-edge-litert / tflite-runtime)")


class TFLiteModel:
    """Wraps a TFLite detector. Supported output layouts (auto-detected):
         regressor : one output with 4 values = [ymin, xmin, ymax, xmax], normalised   (your model)
         ssd       : TFLite_Detection_PostProcess style: boxes (1,N,4) + scores (1,N) [+ classes, count]
       `infer(bgr_crop)` returns [(box_norm [ymin,xmin,ymax,xmax], score_or_None), ...]."""

    def __init__(self, path, cfg: Config):
        self.cfg = cfg
        self.net = _make_interpreter(path)
        self.net.allocate_tensors()
        d = self.net.get_input_details()[0]
        self.in_idx, self.in_dtype = d["index"], d["dtype"]
        self.size = (int(d["shape"][2]), int(d["shape"][1]))              # (W, H)
        self.in_q = tuple(d.get("quantization", (0.0, 0)))
        self.outs = self.net.get_output_details()
        shapes = [tuple(int(s) for s in o["shape"]) for o in self.outs]
        if len(self.outs) == 1 and int(np.prod(shapes[0])) == 4:
            self.kind = "regressor"
        elif len(self.outs) >= 2 and any(len(s) == 3 and s[-1] == 4 for s in shapes):
            self.kind = "ssd"
        else:
            raise ValueError(f"Unsupported TFLite output layout {shapes}. Supported: one 4-value box output "
                             f"[ymin,xmin,ymax,xmax], or an SSD post-process head (boxes, scores, ...). "
                             f"Add a parser in TFLiteModel.infer for your layout.")
        self.calls = 0

    def _deq(self, o):
        t = self.net.get_tensor(o["index"])
        sc, zp = o.get("quantization", (0.0, 0))
        return (t.astype(np.float32) - zp) * sc if (t.dtype != np.float32 and sc) else t.astype(np.float32)

    def infer(self, bgr):
        c = self.cfg
        img = cv2.cvtColor(cv2.resize(bgr, self.size, interpolation=cv2.INTER_LINEAR), cv2.COLOR_BGR2RGB)
        if self.in_dtype == np.float32:
            x = img.astype(np.float32)
            x = x / 255.0 if c.input_range == "01" else (x / 127.5 - 1.0 if c.input_range == "pm1" else x)
        else:                                                              # quantised input
            sc, zp = self.in_q
            if sc and c.input_range != "255":
                real = img.astype(np.float32) / 255.0 if c.input_range == "01" else img.astype(np.float32) / 127.5 - 1.0
                info = np.iinfo(self.in_dtype)
                x = np.clip(np.round(real / sc + zp), info.min, info.max).astype(self.in_dtype)
            else:
                x = img.astype(self.in_dtype)
        self.net.set_tensor(self.in_idx, x[None])
        self.net.invoke()
        self.calls += 1
        if self.kind == "regressor":
            y0, x0, y1, x1 = np.clip(self._deq(self.outs[0]).reshape(-1)[:4], 0.0, 1.0)
            return [([float(y0), float(x0), float(y1), float(x1)], None)] if (y1 > y0 and x1 > x0) else []
        arrs = [self._deq(o) for o in self.outs]
        boxes = next(a[0] for a in arrs if a.ndim == 3 and a.shape[-1] == 4)
        scores = None
        for o, a in zip(self.outs, arrs):
            if a.ndim == 2 and "score" in o.get("name", "").lower():
                scores = a[0]
        if scores is None:
            cand = [a[0] for a in arrs if a.ndim == 2 and a.shape[1] == boxes.shape[0]]
            scores = next((a for a in cand if a.max() <= 1.0 and not np.allclose(a, np.round(a))), cand[0] if cand else np.ones(len(boxes)))
        keep = np.argsort(-scores)[:5]
        return [(list(np.clip(boxes[i], 0, 1)), float(scores[i])) for i in keep if scores[i] >= c.ssd_score_min]


# ============================================================================ blobs + camera motion
class BlobDetector:
    """Small compact bright/dark objects: top-hat -> robust z-score -> components -> shape filters.
    Also returns a background image (blobs painted out) used to estimate camera motion."""

    def __init__(self, cfg: Config, shape):
        self.cfg = cfg
        H, W = shape
        k = max(15, int(round(cfg.tophat_frac * min(H, W))) | 1)
        self.k_top = cv2.getStructuringElement(cv2.MORPH_RECT, (k, k))        # separable -> ~20x faster than an ellipse
        self.k_close = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (cfg.close_kernel, cfg.close_kernel))
        self.max_w, self.max_h = cfg.max_w_frac * W, cfg.max_h_frac * H

    def detect(self, gray):
        c = self.cfg
        g = gray.astype(np.float32)
        want_dark = c.polarity in ("dark", "both")
        want_bright = c.polarity in ("bright", "both")
        r = None
        if want_bright:
            r = g - cv2.morphologyEx(gray, cv2.MORPH_OPEN, self.k_top)
        if want_dark:
            rd = cv2.morphologyEx(gray, cv2.MORPH_CLOSE, self.k_top) - g
            r = rd if r is None else np.maximum(r, rd)
        r = cv2.GaussianBlur(r, (3, 3), 0)
        sub = r[::3, ::3]                                                          # robust statistics from a subsample
        med = float(np.median(sub))
        sigma = max(1.4826 * float(np.median(np.abs(sub - med))), c.sigma_floor)
        mask = ((r > med + c.z_thresh * sigma).astype(np.uint8)) * 255
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, self.k_close)
        n, _, st, _ = cv2.connectedComponentsWithStats(mask, connectivity=8)
        out = []
        for k in range(1, n):
            x, y, w, h, a = (int(v) for v in st[k])
            if a < c.min_area or w > self.max_w or h > self.max_h or max(w / h, h / w) > c.max_aspect:
                continue
            sub_r = r[y:y + h, x:x + w]
            peak = float(sub_r.max())
            z = (peak - med) / sigma
            box = [float(x), float(y), float(x + w), float(y + h)]
            ys, xs = np.nonzero(sub_r > med + c.core_frac * (peak - med))             # bright core of the blob
            area = float(len(xs))
            if len(xs):
                cw, ch = xs.max() + 1 - xs.min(), ys.max() + 1 - ys.min()
                if cw < 0.7 * w or ch < 0.7 * h:                                       # a faint tail / contrail stretched the box
                    box = [float(x + xs.min()), float(y + ys.min()), float(x + xs.max() + 1), float(y + ys.max() + 1)]
            out.append(Cand(box, z, "blob", area=area))
        out.sort(key=lambda q: -q.z)
        # background for camera-motion estimation: blobs painted out with a smooth fill (computed at 1/4 resolution)
        blob = cv2.dilate(mask, self.k_close, iterations=2) > 0
        keep = (~blob).astype(np.float32)
        H, W = g.shape
        small = (max(W // 4, 1), max(H // 4, 1))
        num = cv2.GaussianBlur(cv2.resize(g * keep, small, interpolation=cv2.INTER_AREA), (0, 0), 2.5)
        den = cv2.GaussianBlur(cv2.resize(keep, small, interpolation=cv2.INTER_AREA), (0, 0), 2.5)
        smooth = cv2.resize(num / np.maximum(den, 1e-4), (W, H), interpolation=cv2.INTER_LINEAR)
        return out[:c.max_blobs], np.where(blob, smooth, g)


# ============================================================================ tracker
@dataclass
class Cand:
    box: list
    z: float = 0.0
    src: str = "blob"                 # blob | model | both
    conf: Optional[float] = None
    desc: Optional[np.ndarray] = None
    area: float = 0.0                 # blob pixel count above the core threshold (sub-pixel-smooth size cue); 0 = unknown


class KF2:
    """Per-axis [p, v, a] with decaying acceleration; state order [px, vx, ax, py, vy, ay] (stabilised pixels)."""
    Hm = np.array([[1, 0, 0, 0, 0, 0], [0, 0, 0, 1, 0, 0]], dtype=float)

    def __init__(self, z, R, qj, tau, S):
        self.qj, self.tau = qj, tau
        self.x = np.array([z[0], 0, 0, z[1], 0, 0], dtype=float)
        vs, as_ = 0.3 * S, 0.5 * S
        self.P = np.diag([R[0, 0], vs ** 2, as_ ** 2, R[1, 1], vs ** 2, as_ ** 2])

    def copy(self):
        k = KF2.__new__(KF2)
        k.qj, k.tau, k.x, k.P = self.qj, self.tau, self.x.copy(), self.P.copy()
        return k

    def predict(self, dt):
        if dt <= 0:
            return
        rho = math.exp(-dt / self.tau)
        f = np.array([[1, dt, dt * dt / 2], [0, 1, dt], [0, 0, rho]])
        F = np.zeros((6, 6)); F[:3, :3] = f; F[3:, 3:] = f
        qb = self.qj * np.array([[dt**5 / 20, dt**4 / 8, dt**3 / 6], [dt**4 / 8, dt**3 / 3, dt**2 / 2], [dt**3 / 6, dt**2 / 2, dt]])
        Q = np.zeros((6, 6)); Q[:3, :3] = qb; Q[3:, 3:] = qb
        self.x, self.P = F @ self.x, F @ self.P @ F.T + Q

    def gate(self, z, R):
        S = self.Hm @ self.P @ self.Hm.T + R
        nu = z - self.Hm @ self.x
        return float(nu @ np.linalg.solve(S, nu)), S

    def update(self, z, R):
        S = self.Hm @ self.P @ self.Hm.T + R
        K = np.linalg.solve(S, self.Hm @ self.P).T
        self.x = self.x + K @ (z - self.Hm @ self.x)
        IKH = np.eye(6) - K @ self.Hm
        self.P = IKH @ self.P @ IKH.T + K @ R @ K.T

    pos = property(lambda s: s.x[[0, 3]])
    vel = property(lambda s: s.x[[1, 4]])
    pos_sigma = property(lambda s: float(math.sqrt(max(s.P[0, 0], s.P[3, 3]))))


@dataclass
class Track:
    id: int
    kf: KF2
    desc: np.ndarray
    w: float
    h: float
    is_subject: bool = False
    hits: int = 1
    miss: int = 0
    n_app: int = 1
    t_last: float = 0.0
    scale: float = 0.0                # EMA of sqrt(blob area): a size measure that is not quantised to whole pixels
    z: float = 0.0                    # EMA of the blob contrast this track has been matched to (brightness cue)
    pos: deque = field(default_factory=lambda: deque(maxlen=12))
    areas: deque = field(default_factory=lambda: deque(maxlen=12))
    confs: deque = field(default_factory=lambda: deque(maxlen=12))
    zs: deque = field(default_factory=lambda: deque(maxlen=12))


@dataclass
class FrameResult:
    frame: int
    t: float
    status: str                       # SEARCH | TRACKING | COASTING | LOST
    box: Optional[list]               # original-resolution pixels [x1, y1, x2, y2]
    conf: float = 0.0
    src: str = ""
    ego: tuple = (0.0, 0.0)
    vel: tuple = (0.0, 0.0)           # px/s in image coordinates, original resolution
    ambiguous: bool = False
    n_cands: int = 0
    servo: dict = field(default_factory=dict)
    scale_px: float = 0.0             # sqrt(blob area) in original pixels (size cue for range rate)
    vel_stab: tuple = (0.0, 0.0)      # px/s with camera motion removed (angular motion of the target relative to the background)


class VisualServo:
    """Image-space follow commands. Positive yaw = turn right, positive pitch = tilt up, forward in [-1, 1]
    (positive = approach, because the target looks smaller than it did at lock)."""

    def __init__(self, cfg: Config, W, H):
        self.cfg = cfg
        self.f = (W / 2) / math.tan(math.radians(cfg.hfov_deg) / 2)
        self.W, self.H = W, H
        self.h_des = cfg.desired_h_frac * H if cfg.desired_h_frac else None
        self.lh_prev, self.size_rate = None, 0.0

    def step(self, dt, cx, cy, h, vx, vy):
        c = self.cfg
        if self.h_des is None:
            self.h_des = h
        ex, ey = cx + vx * c.lead_s - self.W / 2, cy + vy * c.lead_s - self.H / 2
        ax, ay = math.degrees(math.atan2(ex, self.f)), math.degrees(math.atan2(ey, self.f))
        lh = math.log(max(h, 1.0))
        if self.lh_prev is not None and dt > 0:
            self.size_rate = 0.8 * self.size_rate + 0.2 * (lh - self.lh_prev) / dt
        self.lh_prev = lh
        fwd = c.k_size * (self.h_des - h) / self.h_des - c.k_size_rate * self.size_rate
        return {"err_x_deg": math.degrees(math.atan2(cx - self.W / 2, self.f)), "err_y_deg": math.degrees(math.atan2(cy - self.H / 2, self.f)),
                "yaw_rate_dps": float(np.clip(c.kp_angle * ax, -c.max_rate_dps, c.max_rate_dps)),
                "pitch_rate_dps": float(np.clip(-c.kp_angle * ay, -c.max_rate_dps, c.max_rate_dps)),
                "fwd_cmd": float(np.clip(fwd, -1.0, 1.0))}


class FlyTracker:
    def __init__(self, cfg: Config, shape, model: Optional[TFLiteModel] = None, orig_shape=None):
        self.cfg, self.model = cfg, model
        H, W = shape
        self.W, self.H, self.S = W, H, min(W, H)
        oh, ow = orig_shape or shape
        self.sx, self.sy = ow / W, oh / H                                  # work -> original
        mode = cfg.detector if cfg.detector != "auto" else ("hybrid" if model else "classical")
        if mode in ("model", "hybrid") and model is None:
            raise ValueError("detector='%s' needs a model" % mode)
        self.use_blobs, self.use_model = mode in ("classical", "hybrid"), mode in ("model", "hybrid")
        self.blobs = BlobDetector(cfg, shape)
        self.qj = (cfg.q_jerk_frac * self.S) ** 2
        self.tracks: List[Track] = []
        self.subject: Optional[Track] = None
        self.status, self.t_prev, self.last_match_t = "SEARCH", None, 0.0
        self.cum, self.prev_bg, self.win = np.zeros(2), None, None
        self.excluded, self._ids = set(), count(1)
        self.last_good_pos, self.last_good_vel = np.zeros(2), np.zeros(2)
        self._reacq_id, self._reacq_n = None, 0
        self.servo = VisualServo(cfg, W, H)
        self.init_box = None
        self.events: List[str] = []

    # ---------------------------------------------------------------- camera motion
    def _ego(self, bg):
        """Global translation prev -> cur (full-resolution pixels), estimated at half resolution on the target-suppressed
        background and accepted only if warping the previous frame by it really lowers the frame difference."""
        c = self.cfg
        small = cv2.resize(bg, (max(bg.shape[1] // 2, 8), max(bg.shape[0] // 2, 8)), interpolation=cv2.INTER_AREA)
        prev, self.prev_bg = self.prev_bg, small
        if not c.ego_comp or prev is None:
            return np.zeros(2)
        if self.win is None or self.win.shape != small.shape:
            self.win = cv2.createHanningWindow((small.shape[1], small.shape[0]), cv2.CV_32F)
        (dx, dy), resp = cv2.phaseCorrelate(prev.copy(), small.copy(), self.win)     # NB: mutates its inputs
        if resp < c.ego_min_resp or max(abs(dx), abs(dy)) > c.ego_max_frac * self.S / 2:
            return np.zeros(2)
        m = int(math.ceil(max(abs(dx), abs(dy)))) + 2
        warped = cv2.warpAffine(prev, np.float32([[1, 0, dx], [0, 1, dy]]), (small.shape[1], small.shape[0]), borderMode=cv2.BORDER_REPLICATE)
        sl = (slice(m, -m), slice(m, -m))
        err0, err1 = float(np.mean(np.abs(small[sl] - prev[sl]))), float(np.mean(np.abs(small[sl] - warped[sl])))
        return np.array([2 * dx, 2 * dy]) if err1 < 0.9 * err0 else np.zeros(2)

    # ---------------------------------------------------------------- model on crops
    def _roi(self, box, sigma=0.0):
        c = self.cfg
        cx, cy = bc(box)
        side = int(min(max(c.roi_min_side, max(bw(box), bh(box)) / c.roi_target_frac) + 6.0 * sigma, self.W, self.H))
        x0, y0 = int(np.clip(cx - side / 2, 0, self.W - side)), int(np.clip(cy - side / 2, 0, self.H - side))
        return [x0, y0, x0 + side, y0 + side]

    def _run_model(self, roi, orig):
        x0, y0, x1, y1 = roi
        crop = orig[int(y0 * self.sy):int(y1 * self.sy), int(x0 * self.sx):int(x1 * self.sx)]
        if crop.shape[0] < 8 or crop.shape[1] < 8:
            return []
        out = []
        for (ymin, xmin, ymax, xmax), score in self.model.infer(crop):
            b = [x0 + xmin * (x1 - x0), y0 + ymin * (y1 - y0), x0 + xmax * (x1 - x0), y0 + ymax * (y1 - y0)]
            if bw(b) >= 2 and bh(b) >= 2:
                out.append((b, score))
        return out

    def _verify(self, cands, orig):
        """Model-agreement confidence: crop around the candidate (target ~40 % of the crop) and see whether the
        model puts its box where the candidate is."""
        order = sorted(cands, key=lambda q: (q.src == "blob", -q.z))
        for q in order[:self.cfg.verify_top_k]:
            res = self._run_model(self._roi(q.box), orig)
            q.conf = max([agreement(q.box, b) for b, _ in res], default=0.0)

    # ---------------------------------------------------------------- association
    def _meas(self, q):
        sig = self.cfg.meas_sigma + self.cfg.meas_sigma_rel * max(bw(q.box), bh(q.box))
        return np.array(bc(q.box)) - self.cum, np.eye(2) * sig ** 2

    def _cost(self, tr, q, z, R):
        c = self.cfg
        lim = 5.0 * (tr.kf.pos_sigma + math.sqrt(R[0, 0])) + 2.0                      # cheap superset of the chi-square gate
        if abs(z[0] - tr.kf.x[0]) > lim or abs(z[1] - tr.kf.x[3]) > lim:
            return INF
        d2, Sm = tr.kf.gate(z, R)
        if d2 > c.gate_chi2:
            return INF
        a = float(tr.desc @ q.desc)
        if a < c.app_min:
            return INF
        r = math.sqrt(max(bw(q.box), 1.0) / max(tr.w, 1.0) * max(bh(q.box), 1.0) / max(tr.h, 1.0))
        if not (1.0 / c.size_tol <= r <= c.size_tol):
            return INF
        sig_a = 0.08 + 1.2 / max(8.0, math.sqrt(max(barea(q.box), 1.0)))
        zterm = 0.5 * (math.log(max(q.z, 1.0) / max(tr.z, 1.0)) / c.sigma_z) ** 2 if (q.z > 0 and tr.z > 0) else 0.0
        cost = zterm + 0.5 * d2 + 0.5 * math.log(max(np.linalg.det(Sm), 1e-9)) + 0.5 * ((1 - a) / sig_a) ** 2 + 0.5 * (math.log(r) / c.sigma_size) ** 2
        return cost - (c.agree_bonus if q.src == "both" else 0.0)

    def _pred_box(self, tr):
        cx, cy = tr.kf.pos + self.cum
        return [cx - tr.w / 2, cy - tr.h / 2, cx + tr.w / 2, cy + tr.h / 2]

    # ---------------------------------------------------------------- one frame
    def step(self, frame, orig, t, idx=0) -> FrameResult:
        c = self.cfg
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        blobs, bg = self.blobs.detect(gray)
        shift = self._ego(bg)
        self.cum += shift
        dt = (t - self.t_prev) if self.t_prev is not None else 0.0
        self.t_prev = t
        for tr in self.tracks:
            tr.kf.predict(dt)

        locked = self.subject is not None and self.status in ("TRACKING", "COASTING")
        cands = list(blobs) if self.use_blobs else []
        roi_used = None
        if self.use_model:
            if locked:
                roi_used = self._roi(self._pred_box(self.subject), self.subject.kf.pos_sigma)
            res = self._run_model(roi_used or [0, 0, self.W, self.H], orig)
            for b, _ in res:
                b = clamp_box(b, self.W, self.H)
                if local_contrast(gray, b) >= c.min_contrast:
                    cands.append(Cand(b, 0.0, "model"))
            for m in [q for q in cands if q.src == "model"]:                         # model + blob agreement
                best = max([(agreement(m.box, bq.box), bq) for bq in cands if bq.src == "blob"], key=lambda x: x[0], default=None)
                if best is not None and best[0] >= 0.4:
                    ag, bq = best
                    m.src, m.z, m.conf, m.area = "both", max(m.z, bq.z), ag, bq.area
                    if c.prefer_blob_box:                                            # pixel evidence localises better
                        m.box = bq.box
                    cands.remove(bq)
            if not locked:
                self._verify(cands, orig)
        for q in cands:
            q.desc = describe(frame, q.box)

        # ---- assignment
        subj_active = self.subject is not None and self.status != "LOST"
        active = [tr for tr in self.tracks if (not tr.is_subject) or subj_active]
        C = np.full((len(active), len(cands)), INF)
        meas = [self._meas(q) for q in cands]
        for i, tr in enumerate(active):
            for j, q in enumerate(cands):
                C[i, j] = self._cost(tr, q, meas[j][0], meas[j][1])
        pairs = {}
        if len(active) and len(cands):
            for i, j in zip(*linear_sum_assignment(C)):
                if C[i, j] < INF:
                    pairs[i] = j

        ambiguous, matched_j = False, None
        if subj_active:
            si = next(i for i, tr in enumerate(active) if tr.is_subject)
            fin = np.sort(C[si][C[si] < INF]) if len(cands) else np.array([])
            if len(fin) >= 2 and fin[1] - fin[0] < c.ambiguity_margin:
                ambiguous = True
                pairs.pop(si, None)
            elif si in pairs:
                j = pairs[si]
                rivals = [C[i, j] for i, tr in enumerate(active) if not tr.is_subject and tr.hits >= c.confirm_frames]
                if rivals and min(rivals) < C[si, j] + c.ambiguity_margin:
                    ambiguous = True
                    pairs.pop(si)
        claimed = set(np.where((C < INF).any(axis=0))[0]) if len(active) and len(cands) else set()

        matched = set()
        for i, j in pairs.items():
            tr, q = active[i], cands[j]
            z, R = self._meas(q)
            tr.kf.update(z, R)
            tr.hits, tr.miss, tr.t_last = tr.hits + 1, 0, t
            tr.pos.append(z); tr.areas.append(barea(q.box)); tr.confs.append(q.conf or 0.0); tr.zs.append(q.z)
            if q.z > 0:
                tr.z = q.z if tr.z == 0 else 0.8 * tr.z + 0.2 * q.z
            sc = math.sqrt(q.area) if q.area > 0 else math.sqrt(max(barea(q.box), 1.0))
            tr.scale = sc if tr.scale == 0 else 0.8 * tr.scale + 0.2 * sc
            tr.w, tr.h = 0.7 * tr.w + 0.3 * bw(q.box), 0.7 * tr.h + 0.3 * bh(q.box)
            if (not tr.is_subject) or np.sum(C[i] < INF) == 1:
                tr.n_app += 1
                a = max(0.05, 1.0 / (tr.n_app + 1)) if tr.is_subject else 0.2
                tr.desc = (1 - a) * tr.desc + a * q.desc
                tr.desc /= np.linalg.norm(tr.desc) + 1e-9
            matched.add(i)
            if tr.is_subject:
                matched_j = j
        for i, tr in enumerate(active):
            if i not in matched:
                tr.miss += 1

        if matched_j is not None and not ambiguous:                                  # cannot-link
            for i, j in pairs.items():
                tr = active[i]
                if not tr.is_subject and tr.hits >= c.confirm_frames and j != matched_j:
                    self.excluded.add(tr.id)
        if ambiguous and not c.strict_identity:
            for tr in self.tracks:
                if not tr.is_subject and np.linalg.norm(tr.kf.pos - self.subject.kf.pos) <= c.impostor_radius_frac * self.S:
                    self.excluded.discard(tr.id)
        for tr in self.tracks:
            if tr.id in self.excluded and t - tr.t_last > c.short_coast_s:
                self.excluded.discard(tr.id)

        for j, q in enumerate(cands):                                                # spawn tentatives
            if j in claimed or len(self.tracks) >= c.max_tracks:
                continue
            z, R = self._meas(q)
            tr = Track(next(self._ids), KF2(z, R, self.qj, c.tau_acc, self.S), q.desc.copy(), bw(q.box), bh(q.box), t_last=t, z=q.z,
                       scale=math.sqrt(q.area) if q.area > 0 else math.sqrt(max(barea(q.box), 1.0)))
            tr.pos.append(z); tr.areas.append(barea(q.box)); tr.confs.append(q.conf or 0.0); tr.zs.append(q.z)
            self.tracks.append(tr)

        keep = []
        for tr in self.tracks:
            if tr.is_subject:
                keep.append(tr)
            elif tr.hits >= c.confirm_frames:
                if t - tr.t_last <= c.delete_after_s:
                    keep.append(tr)
            elif tr.miss <= c.tent_miss:
                keep.append(tr)
        self.tracks = keep

        # ---- subject lifecycle
        if self.subject is None:
            if self.init_box is not None and idx >= self.init_box[1]:
                self._lock_box(self.init_box[0], frame, t)
            else:
                self._try_lock(t)
        else:
            s = self.subject
            if matched_j is not None:
                self.status, self.last_match_t = "TRACKING", t
                self.last_good_pos, self.last_good_vel = s.kf.pos.copy(), s.kf.vel.copy()
            elif self.status != "LOST":
                if self.status == "TRACKING":
                    self._flag_impostors()
                lost = t - self.last_match_t > c.short_coast_s
                if lost:
                    self.events.append(f"t={t:.2f}s subject LOST")
                self.status = "LOST" if lost else "COASTING"
            if self.status == "LOST":
                self._try_reacquire(t)

        return self._result(idx, t, matched_j, cands, shift, ambiguous, dt)

    # ---------------------------------------------------------------- lock / re-acquire
    def _promote(self, tr, t, msg):
        tr.is_subject = True
        self.subject, self.status, self.last_match_t = tr, "TRACKING", t
        self.last_good_pos, self.last_good_vel = tr.kf.pos.copy(), tr.kf.vel.copy()
        self.events.append(f"t={t:.2f}s {msg}")

    def _lock_box(self, box, frame, t):
        z = np.array(bc(box)) - self.cum
        R = np.eye(2) * self.cfg.meas_sigma ** 2
        tr = Track(next(self._ids), KF2(z, R, self.qj, self.cfg.tau_acc, self.S), describe(frame, box), bw(box), bh(box), t_last=t, scale=math.sqrt(max(barea(box), 1.0)))
        tr.pos.append(z)
        self.tracks.append(tr)
        self._promote(tr, t, "locked on user-supplied box")

    def _try_lock(self, t):
        c = self.cfg
        ready = []
        for tr in self.tracks:
            if len(tr.pos) < c.confirm_frames or tr.miss > 0:
                continue
            bx = self._pred_box(tr)
            if bx[0] < 3 or bx[1] < 3 or bx[2] > self.W - 3 or bx[3] > self.H - 3:
                continue                                                           # touching the frame edge: not a lock candidate
            pts, confs = list(tr.pos)[-c.confirm_frames:], list(tr.confs)[-c.confirm_frames:]
            ok_model = self.use_model and np.mean(confs) >= c.conf_min and min(confs) >= 0.5 * c.conf_min
            ok_motion, zs = False, list(tr.zs)[-c.confirm_frames:]
            if (not self.use_model) or c.allow_motion_lock:
                disp = float(np.hypot(*(pts[-1] - pts[0])))
                path = sum(float(np.hypot(*(b - a))) for a, b in zip(pts, pts[1:])) + 1e-6
                ar = list(tr.areas)[-c.confirm_frames:]
                ok_motion = (disp >= c.min_disp_frac * self.S and disp / path >= c.min_straightness and max(ar) / max(min(ar), 1.0) <= c.max_area_change
                             and np.mean(zs) >= c.lock_z_min)
            if ok_model or ok_motion:
                score = float(np.mean(confs)) if ok_model else min(float(np.mean(zs)) / 100.0, 1.0)
                ready.append((score - c.center_bias * self._center_dist(tr), tr))
        if ready:
            self._promote(max(ready, key=lambda x: x[0])[1], t, "subject locked")

    def _center_dist(self, tr):
        cx, cy = tr.kf.pos + self.cum
        return math.hypot(cx - self.W / 2, cy - self.H / 2) / (self.S / 2)

    def _flag_impostors(self):
        c, s = self.cfg, self.subject
        for tr in self.tracks:
            if (not tr.is_subject and tr.hits >= c.confirm_frames and tr.miss == 0
                    and np.linalg.norm(tr.kf.pos - s.kf.pos) <= c.impostor_radius_frac * self.S
                    and track_sim(tr, s) >= c.impostor_app):
                self.excluded.add(tr.id)

    def _try_reacquire(self, t):
        c, s = self.cfg, self.subject
        gap = t - self.last_match_t
        scored = []
        for tr in self.tracks:
            if tr.is_subject or tr.hits < c.reacq_min_hits or tr.miss > 0 or tr.id in self.excluded:
                continue
            confs = list(tr.confs)[-c.confirm_frames:]
            if self.use_model and not (len(confs) and np.mean(confs) >= c.conf_min):
                continue
            a = track_sim(tr, s)
            if a < c.reacq_app_min:
                continue
            dist = float(np.linalg.norm(tr.kf.pos - self.last_good_pos))
            reach = c.reacq_speed_frac * self.S * gap + 0.08 * self.S
            if gap < c.reacq_anywhere_s and dist > reach:
                continue
            if any(o is not tr and not o.is_subject and o.hits >= c.confirm_frames and t - o.t_last <= c.rival_recent_s
                   and track_sim(o, tr) >= c.impostor_app for o in self.tracks):
                continue
            sig_a = 0.15
            scored.append((0.5 * ((1 - a) / sig_a) ** 2 + 0.5 * (dist / max(reach, 1.0)) ** 2, tr))
        scored.sort(key=lambda x: x[0])
        if not scored or (len(scored) > 1 and scored[1][0] - scored[0][0] < c.reacq_margin):
            self._reacq_id, self._reacq_n = None, 0
            return
        best = scored[0][1]
        self._reacq_n = self._reacq_n + 1 if best.id == self._reacq_id else 1
        self._reacq_id = best.id
        if self._reacq_n >= c.reacq_confirm:
            s.kf, s.w, s.h, s.miss = best.kf.copy(), best.w, best.h, 0
            self.tracks.remove(best)
            self.status, self.last_match_t = "TRACKING", t
            self._reacq_id, self._reacq_n = None, 0
            self.events.append(f"t={t:.2f}s subject re-acquired after {gap:.1f}s")

    # ---------------------------------------------------------------- output
    def _result(self, idx, t, matched_j, cands, shift, ambiguous, dt):
        status = self.status if self.subject is not None else "SEARCH"
        if self.subject is None or status == "LOST":
            return FrameResult(idx, t, status, None, ego=(shift[0] * self.sx, shift[1] * self.sy), ambiguous=ambiguous, n_cands=len(cands))
        s = self.subject
        b = self._pred_box(s)
        cx, cy = bc(b)
        vel_img = s.kf.vel + (shift / dt if dt > 0 else 0.0)
        servo = self.servo.step(dt, cx, cy, bh(b), vel_img[0], vel_img[1])
        if status == "COASTING":                                   # no measurement: do not command motion from a guess
            servo.update(yaw_rate_dps=0.0, pitch_rate_dps=0.0, fwd_cmd=0.0)
        src = ""
        conf = float(np.mean(list(s.confs)[-3:])) if len(s.confs) else 0.0
        if matched_j is not None:
            src = cands[matched_j].src
        box = [b[0] * self.sx, b[1] * self.sy, b[2] * self.sx, b[3] * self.sy]
        return FrameResult(idx, t, status, box, conf, src, (shift[0] * self.sx, shift[1] * self.sy),
                           (vel_img[0] * self.sx, vel_img[1] * self.sy), ambiguous, len(cands), servo,
                           float(s.scale) * math.sqrt(self.sx * self.sy), (float(s.kf.vel[0]) * self.sx, float(s.kf.vel[1]) * self.sy))


# ============================================================================ drawing
COL = {"TRACKING": (0, 255, 0), "COASTING": (0, 200, 255), "LOST": (0, 0, 255), "SEARCH": (200, 200, 200)}


def draw(img, r: FrameResult, trail, text=True, arrow=True):
    """Lock box + trail (+ a status text block and centre arrow, which the telemetry HUD replaces when it is on)."""
    out = img.copy()
    H, W = out.shape[:2]
    col = COL[r.status]
    if r.box is not None:
        x1, y1, x2, y2 = (int(v) for v in r.box)
        cv2.rectangle(out, (x1, y1), (x2, y2), col, 2)
        cx, cy = (x1 + x2) // 2, (y1 + y2) // 2
        if trail and math.hypot(cx - trail[-1][0], cy - trail[-1][1]) > 0.2 * W:
            trail.clear()                                                       # a jump (cut / re-acquisition): break the trail
        trail.append((cx, cy))
        for a, b in zip(trail, list(trail)[1:]):
            cv2.line(out, a, b, col, 1)
        if arrow:
            cv2.arrowedLine(out, (W // 2, H // 2), (cx, cy), (0, 0, 255), 1, tipLength=0.05)
    if text:
        lines = [f"{r.status}" + ("  (ambiguous: holding)" if r.ambiguous else ""), f"t={r.t:6.2f}s  src={r.src or '-'}  conf={r.conf:.2f}"]
        if r.servo:
            s = r.servo
            lines += [f"yaw {s['yaw_rate_dps']:+6.1f} deg/s   pitch {s['pitch_rate_dps']:+6.1f} deg/s", f"fwd {s['fwd_cmd']:+5.2f}   err ({s['err_x_deg']:+5.1f}, {s['err_y_deg']:+5.1f}) deg"]
        for i, ln in enumerate(lines):
            cv2.putText(out, ln, (12, 26 + 24 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.6, col, 2, cv2.LINE_AA)
        cv2.drawMarker(out, (W // 2, H // 2), (255, 255, 255), cv2.MARKER_CROSS, 16, 1)
    return out


# ============================================================================ live input
def is_live_source(src):
    """Camera index ("0"), or a stream URL (rtsp://, http://, udp://, ...), as opposed to a file path."""
    s = str(src)
    return s.isdigit() or "://" in s


class LatestFrameReader:
    """Reads on a background thread and keeps ONLY the newest frame. If the tracker is slower than the camera, old frames are
    dropped instead of queuing up, so latency stays bounded (this is what makes live use safe). `pace_fps` makes a FILE behave
    like a live camera (frames arrive in real time), which is how real-time behaviour is tested without a camera."""

    def __init__(self, cap, pace_fps=None):
        self.cap, self.pace = cap, pace_fps
        self.cond = threading.Condition()
        self.frame, self.stamp, self.idx, self.taken = None, 0.0, -1, -1
        self.n_read, self.eof, self._stop = 0, False, False
        self.thread = threading.Thread(target=self._run, daemon=True)
        self.thread.start()

    def _run(self):
        t_next = time.monotonic()
        while not self._stop:
            ok, f = self.cap.read()
            now = time.monotonic()
            with self.cond:
                if not ok:
                    self.eof = True
                    self.cond.notify_all()
                    return
                self.frame, self.stamp, self.idx = f, now, self.n_read
                self.n_read += 1
                self.cond.notify_all()
            if self.pace:
                t_next += 1.0 / self.pace
                delay = t_next - time.monotonic()
                if delay > 0:
                    time.sleep(delay)
                else:
                    t_next = time.monotonic()                                   # fell behind: do not try to catch up

    def get(self, timeout=5.0):
        """Newest frame not yet returned -> (frame, capture_time, index); None at end of stream; TimeoutError if the stream stalls."""
        with self.cond:
            end = time.monotonic() + timeout
            while self.idx == self.taken:
                if self.eof:
                    return None
                left = end - time.monotonic()
                if left <= 0:
                    raise TimeoutError("no frame received")
                self.cond.wait(left)
            self.taken = self.idx
            return self.frame, self.stamp, self.idx

    def close(self):
        self._stop = True


def _frames(cap, realtime, pace_fps):
    """Yields (frame, capture_time or None, source_index, frames_dropped_before_this_one)."""
    if not realtime:
        for i in itertools.count():
            ok, f = cap.read()
            if not ok:
                return
            yield f, None, i, 0
    else:
        rd, last = LatestFrameReader(cap, pace_fps), -1
        try:
            while True:
                try:
                    item = rd.get()
                except TimeoutError:
                    print("stream stalled for 5 s - stopping")
                    return
                if item is None:
                    return
                yield item[0], item[1], item[2], item[2] - last - 1
                last = item[2]
        finally:
            rd.close()


class Results(list):
    """List of FrameResult, plus `.stats` (timing / dropped frames) and `.events`."""
    stats: dict = None
    events: list = None


CSV_COLS = ["frame", "t", "status", "x1", "y1", "x2", "y2", "cx", "cy", "w", "h", "conf", "src", "ego_dx", "ego_dy", "vx", "vy",
            "err_x_deg", "err_y_deg", "yaw_rate_dps", "pitch_rate_dps", "fwd_cmd", "ambiguous", "n_cands"]
HUD_COLS = ["range_m_est", "range_rate_mps_est", "subj_speed_mps_est", "subj_heading_deg_est", "dv_along_mps", "dv_cross_mps", "bearing_deg_assumed_hdg",
            "depression_deg", "shot_angle_deg"]


def track_video(video, model=None, out=None, csv_path=None, cfg: Optional[Config] = None, init_box=None, init_frame=0,
                max_frames=None, quiet=False, show=False, realtime=None, **overrides):
    """Track the flying vehicle in `video` (file path, camera index, or stream URL). Returns Results (list of FrameResult).
    model     : path to a .tflite file (or None for the model-free blob detector)
    init_box  : (x, y, w, h) in ORIGINAL pixels to lock on immediately at `init_frame` (skips automatic acquisition)
    show      : open a live preview window (press q / Esc to stop)
    realtime  : treat the source as live: always process the NEWEST frame and drop the ones that arrive while busy.
                Default: True for camera indices and URLs, False for files (True on a file = emulate a live camera)
    overrides : any Config field, e.g. polarity="dark", hud=True, hfov_deg=70, target_size_m=15"""
    cfg = cfg or Config()
    for k, v in overrides.items():
        if not hasattr(cfg, k):
            raise TypeError(f"unknown option {k!r}")
        setattr(cfg, k, v)
    if cfg.hud and LiveTelemetry is None:
        raise ImportError("the telemetry HUD needs hud.py next to track_flying_object.py")
    live_src = is_live_source(video)
    realtime = live_src if realtime is None else realtime
    cap = cv2.VideoCapture(int(video) if str(video).isdigit() else video)
    if not cap.isOpened():
        raise IOError(f"cannot open video source {video}")
    fps = cap.get(cv2.CAP_PROP_FPS)
    fps = fps if fps and fps == fps and 0 < fps < 240 else 30.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not live_src else 0
    frames = _frames(cap, realtime, fps if (realtime and not live_src) else None)
    first = next(frames, None)
    if first is None:
        raise IOError("the source produced no frames")
    OH, OW = first[0].shape[:2]
    WH = min(cfg.work_height, OH)
    WW = int(round(OW * WH / OH))
    tfl = TFLiteModel(model, cfg) if model else None
    trk = FlyTracker(cfg, (WH, WW), tfl, (OH, OW))
    if init_box is not None:
        x, y, w, h = init_box
        trk.init_box = ([x / trk.sx, y / trk.sy, (x + w) / trk.sx, (y + h) / trk.sy], init_frame)
    live = LiveTelemetry(LiveHudConfig(hfov_deg=cfg.hfov_deg, alt_m=cfg.hud_alt_m, target_size_m=cfg.target_size_m, cam_heading_deg=cfg.cam_heading_deg,
                                       cam_pitch_deg=cfg.cam_pitch_deg, dv_scale=cfg.hud_dv_scale, hud_layout=cfg.hud_layout,
                                       max_speed_mps=cfg.max_speed_mps), OW, OH) if cfg.hud else None
    out_size = hud_output_size(OW, OH, cfg.hud_layout, cfg.hud_scale) if live else (OW, OH)
    writer = cv2.VideoWriter(out, cv2.VideoWriter_fourcc(*"mp4v"), fps, out_size) if out else None
    cf = open(csv_path, "w", newline="") if csv_path else None
    cw = csv.writer(cf) if cf else None
    if cw:
        cw.writerow(CSV_COLS + (HUD_COLS if live else []))
    results, trail, k = Results(), deque(maxlen=60), 0
    proc_ms, lat_ms, dropped, last_out, t_start, wall0, win = [], [], 0, None, None, time.time(), "flying-object tracker"
    for orig, stamp, idx, drop in itertools.chain([first], frames):
        if max_frames and k >= max_frames:
            break
        if stamp is None:
            t = idx / fps
        else:
            t_start = stamp if t_start is None else t_start
            t = stamp - t_start
        t0 = time.perf_counter()
        work = cv2.resize(orig, (WW, WH), interpolation=cv2.INTER_AREA) if (WW, WH) != (OW, OH) else orig
        r = trk.step(work, orig, t, idx)
        row = live.update(t, r.status, r.box, r.vel_stab, r.servo, r.ambiguous, r.scale_px) if live else None
        shown = None
        if writer or show:
            shown = draw(orig, r, trail, text=not live, arrow=not live)
            if live:
                marker = ((r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2) if r.box is not None else None
                shown = draw_live_hud(shown, live, cfg.hud_scale, marker, cfg.hud_layout)
        proc_ms.append(1000 * (time.perf_counter() - t0))
        if stamp is not None:
            lat_ms.append(1000 * (time.monotonic() - stamp))
        dropped += drop
        results.append(r)
        if writer:
            for _ in range(drop if (stamp is not None and last_out is not None) else 0):
                writer.write(last_out)                                          # keep playback speed true when frames were dropped
            writer.write(shown)
            last_out = shown
        if show:
            try:
                cv2.imshow(win, shown)
                if cv2.waitKey(1) & 0xFF in (27, ord("q")):
                    break
            except cv2.error:
                print("no display available - continuing without the preview window")
                show = False
        if cw:
            b, sv = r.box, r.servo
            cw.writerow([idx, f"{r.t:.3f}", r.status] + ([f"{v:.1f}" for v in b] + [f"{(b[0]+b[2])/2:.1f}", f"{(b[1]+b[3])/2:.1f}", f"{b[2]-b[0]:.1f}", f"{b[3]-b[1]:.1f}"] if b else [""] * 8)
                         + [f"{r.conf:.3f}", r.src, f"{r.ego[0]:.2f}", f"{r.ego[1]:.2f}", f"{r.vel[0]:.1f}", f"{r.vel[1]:.1f}"]
                         + ([f"{sv[x]:.3f}" for x in ("err_x_deg", "err_y_deg", "yaw_rate_dps", "pitch_rate_dps", "fwd_cmd")] if sv else [""] * 5)
                         + [int(r.ambiguous), r.n_cands]
                         + ([("" if not np.isfinite(row[f]) else f"{row[f]:.2f}") for f in
                             ("rng", "range_rate", "subj_speed", "subj_heading", "dv_along", "dv_cross", "brg", "depr", "shot")] if live else []))
        k += 1
        if not quiet and k % 100 == 0:
            print(f"  frame {idx}" + (f"/{total}" if total else "") + f"  {r.status:9s} ({k / (time.time() - wall0):.1f} fps, {np.mean(proc_ms[-100:]):.1f} ms/frame)")
    cap.release()
    if writer:
        writer.release()
    if cf:
        cf.close()
    if show:
        try:
            cv2.destroyAllWindows()
        except cv2.error:
            pass
    wall = time.time() - wall0
    results.events = trk.events
    results.stats = {"frames_processed": k, "frames_dropped": dropped, "wall_s": wall, "fps": k / max(wall, 1e-9),
                     "ms_per_frame_mean": float(np.mean(proc_ms)) if proc_ms else 0.0, "ms_per_frame_p95": float(np.percentile(proc_ms, 95)) if proc_ms else 0.0,
                     "latency_ms_p50": float(np.percentile(lat_ms, 50)) if lat_ms else None, "latency_ms_p95": float(np.percentile(lat_ms, 95)) if lat_ms else None,
                     "model_calls": tfl.calls if tfl else 0}
    if not quiet:
        n = max(len(results), 1)
        cnt = {s_: sum(1 for r in results if r.status == s_) for s_ in ("TRACKING", "COASTING", "LOST", "SEARCH")}
        st = results.stats
        print(f"\n{k} frames in {wall:.1f}s ({st['fps']:.1f} fps, {st['ms_per_frame_mean']:.1f} ms/frame mean, {st['ms_per_frame_p95']:.1f} p95)"
              + (f" | dropped {dropped}" if realtime else "") + (f" | latency p50/p95 {st['latency_ms_p50']:.0f}/{st['latency_ms_p95']:.0f} ms" if lat_ms else ""))
        print(" | ".join(f"{s_} {100 * v / n:.0f}%" for s_, v in cnt.items()) + (f" | model calls {tfl.calls}" if tfl else ""))
        for e in trk.events:
            print("  ", e)
        if out:
            print("annotated video:", out)
        if csv_path:
            print("track log:", csv_path)
    return results


def main():
    ap = argparse.ArgumentParser(description="Track the flying vehicle in a video, camera or stream (no telemetry needed).")
    ap.add_argument("--video", required=True, help="file path, camera index (0), or stream URL (rtsp://...)")
    ap.add_argument("--model", help="TFLite detector (.tflite). Without it the model-free blob detector is used.")
    ap.add_argument("--out", help="annotated output video (.mp4)")
    ap.add_argument("--csv", help="per-frame track log")
    ap.add_argument("--show", action="store_true", help="live preview window (q / Esc to stop)")
    ap.add_argument("--realtime", action="store_true", help="on a FILE: emulate a live camera (process the newest frame, drop the rest)")
    ap.add_argument("--init-box", help="x,y,w,h in original pixels: lock on this box instead of auto-acquiring")
    ap.add_argument("--init-frame", type=int, default=0)
    ap.add_argument("--detector", default="auto", choices=["auto", "model", "classical", "hybrid"])
    ap.add_argument("--polarity", default="both", choices=["bright", "dark", "both"], help="blob detector: IR white-hot=bright, visible sky=dark")
    ap.add_argument("--input-range", default="01", choices=["01", "pm1", "255"], help="model input scaling (yours: 01)")
    ap.add_argument("--hfov-deg", type=float, default=60.0, help="camera horizontal FOV (angles, range and the HUD depend on it)")
    ap.add_argument("--work-height", type=int, default=540)
    ap.add_argument("--conf-min", type=float, default=0.5)
    ap.add_argument("--allow-motion-lock", action="store_true")
    ap.add_argument("--relaxed", action="store_true", help="recover faster after look-alike passes, may briefly lock the wrong twin")
    ap.add_argument("--max-frames", type=int)
    hg = ap.add_argument_group("telemetry HUD (needs hud.py)")
    hg.add_argument("--hud", action="store_true", help="draw the telemetry HUD on the frames")
    hg.add_argument("--alt-m", type=float, default=50.0, help="ASSUMED hover altitude of the camera drone (m)")
    hg.add_argument("--target-size-m", type=float, default=12.0, help="ASSUMED real length of the target's longest box side (m) -> range")
    hg.add_argument("--cam-heading-deg", type=float, default=0.0, help="ASSUMED compass heading of the optical axis")
    hg.add_argument("--cam-pitch-deg", type=float, default=0.0, help="ASSUMED camera pitch (+ up)")
    hg.add_argument("--dv-scale", type=float, default=15.0, help="m/s at full scale on the delta-v gauges")
    hg.add_argument("--hud-scale", type=float, help="HUD size multiplier (default: automatic)")
    hg.add_argument("--hud-layout", default="auto", choices=["auto", "overlay", "dock"], help="dock = panels in margins around the video (auto for frames < 1280 px wide)")
    hg.add_argument("--max-speed-mps", type=float, default=150.0, help="estimated speeds above this are shown as '--'")
    a = ap.parse_args()
    ib = tuple(float(v) for v in a.init_box.split(",")) if a.init_box else None
    track_video(a.video, a.model, a.out, a.csv, init_box=ib, init_frame=a.init_frame, max_frames=a.max_frames, show=a.show, realtime=True if a.realtime else None,
                detector=a.detector, polarity=a.polarity, input_range=a.input_range, hfov_deg=a.hfov_deg, work_height=a.work_height, conf_min=a.conf_min,
                allow_motion_lock=a.allow_motion_lock, strict_identity=not a.relaxed, hud=a.hud, hud_alt_m=a.alt_m, target_size_m=a.target_size_m,
                cam_heading_deg=a.cam_heading_deg, cam_pitch_deg=a.cam_pitch_deg, hud_dv_scale=a.dv_scale, hud_scale=a.hud_scale,
                hud_layout=a.hud_layout, max_speed_mps=a.max_speed_mps)


if __name__ == "__main__":
    main()
