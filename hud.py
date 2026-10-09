"""OpenCV telemetry HUD: for the simulation (first-person scene) AND for real video (overlay on live frames).

`compute_telemetry(res)` turns a SimResult into per-frame numbers (pure NumPy, unit-tested for sign conventions);
`render_hud_frame` / `render_hud_video` draw a first-person view from the camera drone with the HUD on top.

Everything shown is what the DRONE knows: its own state plus the tracker's estimate of the subject. Ground truth is
used only to draw the scene (the world the camera is looking at).

Sign conventions (also printed on screen)
  delta-v   = drone velocity - subject velocity.  ALONG > 0: drone is faster than the subject along the subject's heading.
              CROSS > 0: drone is drifting to the subject's left.  RANGE RATE > 0: the gap is opening, < 0: closing.
  bearing   = compass degrees from the drone to the subject (0 = N, 90 = E, clockwise).
  shot angle= where the drone sits around the subject, clockwise from the subject's heading (0 front, 90 right, 180 behind).
  gimbal    = PAN is a compass heading, TILT is elevation (negative = looking down).
              CORR = commanded - actual: PAN CORR > 0 turns right (clockwise), TILT CORR > 0 tilts up.
"""
import math
from collections import deque
from dataclasses import dataclass

import cv2
import numpy as np



def cam_axes(yaw, pitch):
    """Camera forward / right / up unit vectors in ENU for a gimbal yaw (0 = east, CCW) and pitch (+ = up)."""
    cy, sy, cp, sp = np.cos(yaw), np.sin(yaw), np.cos(pitch), np.sin(pitch)
    f = np.array([cp * cy, cp * sy, sp])
    r = np.array([sy, -cy, 0.0])
    return f, r, np.cross(r, f)

GREEN, CYAN, AMBER, RED = (110, 255, 130), (255, 225, 80), (0, 190, 255), (80, 80, 255)
WHITE, GREY, DARK = (240, 240, 240), (170, 170, 170), (20, 20, 20)
FONT = cv2.FONT_HERSHEY_SIMPLEX


# ============================================================================ telemetry
def _wrap180(a):
    return (a + 180.0) % 360.0 - 180.0


def compass(yaw_rad):
    """ENU yaw (0 = east, counter-clockwise) -> compass degrees (0 = north, clockwise)."""
    return (90.0 - np.degrees(yaw_rad)) % 360.0


def _ema(x, alpha):
    y = np.copy(x)
    for i in range(1, len(x)):
        y[i] = alpha * x[i] + (1 - alpha) * y[i - 1]
    return y


def compute_telemetry(res, smooth_s=0.15):
    """Per-frame HUD quantities from a SimResult. Subject-dependent entries are NaN while the subject is LOST."""
    L, cfg = res.log, res.cfg
    n, dt = len(L["t"]), cfg.dt
    pd, vd, yaw, pitch = L["cam_p"], L["cam_v"], L["cam_yaw"], L["cam_pitch"]
    ps, vs = L["est_pos"], L["est_vel"].copy()
    vs[:, 2] = 0.0
    valid = np.isin(L["status"], ("TRACKING", "COASTING"))
    names = ["gs", "gs_kmh", "course", "alt", "climb", "subj_speed", "subj_heading", "dv_along", "dv_cross", "dv_vert",
             "dv_mag", "range_rate", "brg", "rng", "rng_h", "depr", "shot", "pan", "tilt", "pan_cmd", "tilt_cmd",
             "pan_err", "tilt_err", "pan_rate", "tilt_rate", "off_u", "off_v", "off_x_deg", "off_y_deg"]
    T = {k: np.full(n, np.nan) for k in names}
    T["t"], T["status"], T["ambiguous"] = L["t"], L["status"], L["ambiguous"]

    gs = np.linalg.norm(vd[:, :2], axis=1)
    T["gs"], T["gs_kmh"], T["alt"], T["climb"] = gs, 3.6 * gs, pd[:, 2], vd[:, 2]
    course = compass(np.arctan2(vd[:, 1], vd[:, 0]))
    T["course"] = course

    T["pan"], T["tilt"] = compass(yaw), np.degrees(pitch)
    T["pan_cmd"], T["tilt_cmd"] = compass(L["g_yaw_cmd"]), np.degrees(L["g_pitch_cmd"])
    T["pan_err"] = -_wrap180(np.degrees(L["g_yaw_cmd"] - yaw))                    # + = turn right (clockwise)
    T["tilt_err"] = np.degrees(L["g_pitch_cmd"] - pitch)                           # + = tilt up
    alpha = dt / (smooth_s + dt)
    T["pan_rate"] = _ema(-np.degrees(np.gradient(np.unwrap(yaw), dt)), alpha)
    T["tilt_rate"] = _ema(np.degrees(np.gradient(pitch, dt)), alpha)

    heading = np.array([1.0, 0.0])
    cam = _cam(cfg)
    for k in range(n):
        if not valid[k]:
            continue
        sp = float(np.linalg.norm(vs[k, :2]))
        if sp > 0.5:
            heading = vs[k, :2] / sp                                              # hold the last heading when (nearly) stopped
        T["subj_speed"][k] = sp
        T["subj_heading"][k] = (90.0 - math.degrees(math.atan2(heading[1], heading[0]))) % 360.0
        dv = vd[k] - vs[k]
        left = np.array([-heading[1], heading[0]])
        T["dv_along"][k], T["dv_cross"][k], T["dv_vert"][k] = dv[:2] @ heading, dv[:2] @ left, dv[2]
        T["dv_mag"][k] = np.linalg.norm(dv)
        r = ps[k] - pd[k]
        rng = float(np.linalg.norm(r))
        rh = float(np.linalg.norm(r[:2]))
        T["rng"][k], T["rng_h"][k] = rng, rh
        T["brg"][k] = (90.0 - math.degrees(math.atan2(r[1], r[0]))) % 360.0
        T["depr"][k] = math.degrees(math.atan2(-r[2], max(rh, 1e-6)))             # + = subject below the horizon
        T["range_rate"][k] = float((r / max(rng, 1e-6)) @ (vs[k] - vd[k]))
        d = -r[:2]                                                                 # subject -> drone
        right = np.array([heading[1], -heading[0]])
        T["shot"][k] = math.degrees(math.atan2(d @ right, d @ heading)) % 360.0
        u, v = _project(ps[k][None], pd[k], yaw[k], pitch[k], cam["F"], cam["cx"], cam["cy"])[0][0]
        T["off_u"][k], T["off_v"][k] = u - cam["cx"], v - cam["cy"]
        T["off_x_deg"][k], T["off_y_deg"][k] = math.degrees(math.atan2(u - cam["cx"], cam["F"])), math.degrees(math.atan2(v - cam["cy"], cam["F"]))
    return T


def _cam(cfg):
    F = (cfg.cam.width / 2) / math.tan(math.radians(cfg.cam.hfov_deg) / 2)
    return {"F": F, "cx": cfg.cam.width / 2, "cy": cfg.cam.height / 2}


# ============================================================================ scene rendering
def _project(P, pos, yaw, pitch, F, cx, cy):
    f, r, u = cam_axes(yaw, pitch)
    d = np.asarray(P, float) - pos
    zc = d @ f
    with np.errstate(divide="ignore", invalid="ignore"):
        pts = np.stack([cx + F * (d @ r) / zc, cy - F * (d @ u) / zc], axis=1)
    return pts, zc


def _background(W, H, F, cy, pitch):
    """Sky above the horizon, hazy ground below it (the camera has no roll, so the horizon is a horizontal line)."""
    vh = cy + F * math.tan(pitch)
    rows = np.arange(H, dtype=np.float32)
    sky_top, sky_hor = np.array([150, 110, 70], np.float32), np.array([235, 215, 185], np.float32)
    gnd_hor, gnd_far = np.array([160, 175, 165], np.float32), np.array([70, 100, 80], np.float32)
    t_sky = np.clip(1 - (vh - rows) / max(vh, 200.0), 0, 1)[:, None]
    t_gnd = np.clip((rows - vh) / 260.0, 0, 1)[:, None]
    col = np.where((rows < vh)[:, None], sky_top + (sky_hor - sky_top) * t_sky, gnd_hor + (gnd_far - gnd_hor) * t_gnd)
    return np.repeat(col[:, None, :], W, axis=1).astype(np.uint8), vh


def _draw_grid(img, pos, yaw, pitch, F, cx, cy, spacing=10.0, reach=140.0):
    H, W = img.shape[:2]
    x0, y0 = math.floor(pos[0] / spacing) * spacing, math.floor(pos[1] / spacing) * spacing
    ticks = np.arange(-reach, reach + spacing, spacing)
    s = np.linspace(-reach, reach, 56)
    for ax in (0, 1):
        for off in ticks:
            line = np.zeros((len(s), 3))
            if ax == 0:
                line[:, 0], line[:, 1] = x0 + off, y0 + s
            else:
                line[:, 0], line[:, 1] = x0 + s, y0 + off
            pts, zc = _project(line, pos, yaw, pitch, F, cx, cy)
            ok = (zc > 0.8) & (np.abs(pts[:, 0]) < 4 * W) & (np.abs(pts[:, 1]) < 4 * H)
            major = int(round(off + (x0 if ax == 0 else y0))) % 50 == 0
            col, th = ((150, 175, 150), 1) if major else ((118, 142, 122), 1)
            for i in range(len(s) - 1):
                if ok[i] and ok[i + 1]:
                    cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), col, th, cv2.LINE_AA)


BOX_L, BOX_W, BOX_H = 4.2, 1.8, 1.5
KIND_COL = {"subject": (90, 190, 70), "vehicle": (40, 110, 225), "bird": (35, 35, 35)}


def _draw_entity(img, ent, p, psi, pos, yaw, pitch, F, cx, cy):
    if ent.kind == "bird":
        pt, zc = _project(p[None], pos, yaw, pitch, F, cx, cy)
        if zc[0] < 0.8:
            return
        x, y = pt[0]
        sz = max(3.0, F * 0.9 / zc[0])
        a, b = (int(x - sz), int(y - 0.35 * sz)), (int(x + sz), int(y - 0.35 * sz))
        c = (int(x), int(y))
        cv2.line(img, a, c, KIND_COL["bird"], 2, cv2.LINE_AA)
        cv2.line(img, c, b, KIND_COL["bird"], 2, cv2.LINE_AA)
        cv2.circle(img, c, max(1, int(sz * 0.18)), KIND_COL["bird"], -1, cv2.LINE_AA)
        return
    sx, sy, sz_ = np.meshgrid([-.5, .5], [-.5, .5], [-.5, .5], indexing="ij")
    loc = np.stack([sx.ravel() * BOX_L, sy.ravel() * BOX_W, sz_.ravel() * BOX_H], axis=1)
    c_, s_ = math.cos(psi), math.sin(psi)
    world = loc @ np.array([[c_, s_, 0], [-s_, c_, 0], [0, 0, 1]]) + p
    pts, zc = _project(world, pos, yaw, pitch, F, cx, cy)
    if (zc < 0.8).any() or np.abs(pts).max() > 6000:
        return
    col = KIND_COL[ent.kind]
    hull = cv2.convexHull(pts.astype(np.int32))
    cv2.fillConvexPoly(img, hull, col, cv2.LINE_AA)
    top = pts[loc[:, 2] > 0].astype(np.int32)
    cv2.fillConvexPoly(img, cv2.convexHull(top), tuple(min(255, int(v * 1.35) + 20) for v in col), cv2.LINE_AA)
    cv2.polylines(img, [hull], True, (20, 20, 20), 1, cv2.LINE_AA)


# ============================================================================ drawing toolkit (resolution independent)
class Ink:
    """Draws in 'design units' (the 960x720 layout) scaled by `u`, so the same HUD is crisp at any frame size."""

    def __init__(self, img, u):
        self.img, self.u = img, u

    def _p(self, p):
        return (int(round(p[0] * self.u)), int(round(p[1] * self.u)))

    def _t(self, th):
        return max(1, int(round(th * self.u)))

    def line(self, a, b, col, th=1):
        cv2.line(self.img, self._p(a), self._p(b), col, self._t(th), cv2.LINE_AA)

    def rect(self, a, b, col, th=1):
        cv2.rectangle(self.img, self._p(a), self._p(b), col, -1 if th < 0 else self._t(th), cv2.LINE_AA)

    def circle(self, c, r, col, th=1):
        cv2.circle(self.img, self._p(c), max(1, int(round(r * self.u))), col, -1 if th < 0 else self._t(th), cv2.LINE_AA)

    def arrow(self, a, b, col, th=2, tip=0.25):
        cv2.arrowedLine(self.img, self._p(a), self._p(b), col, self._t(th), cv2.LINE_AA, tipLength=tip)

    def mark(self, c, col, size=14, th=2, kind=cv2.MARKER_CROSS):
        cv2.drawMarker(self.img, self._p(c), col, kind, max(3, int(size * self.u)), self._t(th), cv2.LINE_AA)

    def poly(self, pts, col, th=1):
        if len(pts) > 1:
            cv2.polylines(self.img, [np.array([self._p(p) for p in pts], np.int32)], False, col, self._t(th), cv2.LINE_AA)

    def text(self, s, org, scale=0.5, col=WHITE, th=1, align="l"):
        """putText with a drop shadow. Hershey fonts are ASCII-only, so the degree sign is drawn as a small circle.
        Returns the width in design units."""
        sc, tp = scale * self.u, self._t(th)
        parts = s.replace("±", "+/-").split("°")
        widths = [cv2.getTextSize(p, FONT, sc, tp)[0][0] for p in parts]
        deg_w = int(round(sc * 11))
        total = sum(widths) + deg_w * (len(parts) - 1)
        x, y = self._p(org)
        if align == "r":
            x -= total
        elif align == "c":
            x -= total // 2
        for i, p in enumerate(parts):
            if p:
                cv2.putText(self.img, p, (x + 1, y + 1), FONT, sc, (0, 0, 0), tp + 1, cv2.LINE_AA)
                cv2.putText(self.img, p, (x, y), FONT, sc, col, tp, cv2.LINE_AA)
            x += widths[i]
            if i < len(parts) - 1:
                r = max(2, int(sc * 4.5))
                c = (x + r + 1, y - int(sc * 17))
                cv2.circle(self.img, (c[0] + 1, c[1] + 1), r, (0, 0, 0), 2, cv2.LINE_AA)
                cv2.circle(self.img, c, r, col, 1, cv2.LINE_AA)
                x += deg_w
        return total / self.u

    def panel(self, x, y, w, h, title=None, alpha=0.52):
        H, W = self.img.shape[:2]
        x0, y0 = self._p((x, y))
        x1, y1 = self._p((x + w, y + h))
        x0, y0, x1, y1 = max(x0, 0), max(y0, 0), min(x1, W), min(y1, H)
        if x1 > x0 and y1 > y0:
            roi = self.img[y0:y1, x0:x1]
            cv2.addWeighted(roi, 1 - alpha, np.full_like(roi, DARK[0]), alpha, 0, dst=roi)
        self.rect((x, y), (x + w, y + h), (110, 140, 110), 1)
        if title:
            self.text(title, (x + 8, y + 16), 0.42, GREEN)


def _fmt(v, spec, unit="", dash="--"):
    return dash if v is None or not np.isfinite(v) else format(v, spec) + unit


def _sev(v, lo, hi):
    return GREEN if v is None or not np.isfinite(v) or abs(v) <= lo else (AMBER if abs(v) <= hi else RED)


def _gauge(ink, x, y, w, val, vmax, col=CYAN):
    ink.line((x, y), (x + w, y), (120, 120, 120), 1)
    ink.line((x + w / 2, y - 5), (x + w / 2, y + 5), (200, 200, 200), 1)
    if val is not None and np.isfinite(val):
        px = x + w / 2 + float(np.clip(val / vmax, -1, 1)) * w / 2
        ink.rect((min(px, x + w / 2), y - 3), (max(px, x + w / 2), y + 3), col, -1)
        ink.circle((px, y), 4, WHITE, 1)


def _compass(ink, c, r, T, k):
    ink.circle(c, r, (150, 170, 150), 1)
    for a in range(0, 360, 30):
        rad, ln = math.radians(a), (7 if a % 90 == 0 else 4)
        ink.line((c[0] + (r - ln) * math.sin(rad), c[1] - (r - ln) * math.cos(rad)), (c[0] + r * math.sin(rad), c[1] - r * math.cos(rad)), (150, 170, 150), 1)
    for a, s in ((0, "N"), (90, "E"), (180, "S"), (270, "W")):
        rad = math.radians(a)
        ink.text(s, (c[0] + (r + 11) * math.sin(rad), c[1] - (r + 11) * math.cos(rad) + 4), 0.4, GREY, align="c")

    def arrow(deg, length, col, th=2):
        if deg is not None and np.isfinite(deg):
            rad = math.radians(deg)
            ink.arrow(c, (c[0] + length * math.sin(rad), c[1] - length * math.cos(rad)), col, th)

    arrow(T["course"][k] if T["gs"][k] > 0.3 else None, r * 0.8, CYAN)                  # where the drone is going
    arrow(T["subj_heading"][k], r * 0.55, AMBER, 1)                                      # where the subject is going
    arrow(T["brg"][k], r * 0.95, GREEN, 2)                                               # where the subject is from here


def _strip(ink, x, y, w, h, title, series, tvec, k, t_win, ymax, legend=()):
    ink.panel(x, y, w, h, title)
    px0, py0, pw, ph = x + 8, y + 24, w - 16, h - 32
    ink.line((px0, py0 + ph / 2), (px0 + pw, py0 + ph / 2), (110, 110, 110), 1)
    t0 = tvec[k] - t_win
    i0 = int(np.searchsorted(tvec, t0))
    for vals, col in series:
        pts = []
        for i in range(i0, k + 1):
            if np.isfinite(vals[i]):
                pts.append((px0 + (tvec[i] - t0) / t_win * pw, py0 + ph / 2 - float(np.clip(vals[i] / ymax, -1, 1)) * ph / 2))
            else:
                ink.poly(pts, col, 1)
                pts = []
        ink.poly(pts, col, 1)
    lx = x + w - 8
    for name, col in reversed(legend):
        lx -= ink.text(name, (lx, y + 16), 0.36, col, align="r") + 6
        ink.line((lx - 14, y + 12), (lx - 2, y + 12), col, 2)
        lx -= 22
    ink.text(f"scale +/-{ymax:g}", (x + 8, y + h - 5), 0.34, GREY)


def _sector(shot):
    if not np.isfinite(shot):
        return ""
    return "in front" if (shot < 45 or shot >= 315) else ("right side" if shot < 135 else ("behind" if shot < 225 else "left side"))


def _cardinal(b):
    return ["N", "NE", "E", "SE", "S", "SW", "W", "NW"][int(((b + 22.5) % 360) // 45)] if np.isfinite(b) else ""


def auto_scale(W, H):
    return float(np.clip(min(W / 960.0, H / 720.0), 0.75, 3.0))


def draw_overlay(img, T, k, scale=None, title="CHASE CAM", subtitle="", notes=None, footer=None, marker=None, dv_scale=5.0, strip_s=6.0, rect=None):
    """Draw the HUD onto `img` (any size) in place, from telemetry arrays `T` at index `k`.
    notes  : {"velocity"|"bearing"|"gimbal": short text} shown on the panel title rows (e.g. what is assumed / estimated)
    footer : one line under the top bar (e.g. the assumptions behind the numbers)
    marker : (x, y) in VIDEO pixels of the tracked subject -> cyan marker + line to the centre + angular offset readout
    rect   : (x0, y0, x1, y1) pixel rectangle that holds the video when the HUD is docked in margins around it (default: whole image)"""
    H, W = img.shape[:2]
    u = scale if scale else auto_scale(W, H)
    ink = Ink(img, u)
    Wd, Hd = W / u, H / u
    notes = notes or {}
    n = np.isfinite
    x0, y0, x1, y1 = rect if rect is not None else (0, 0, W, H)
    cx, cy = (x0 + x1) / 2 / u, (y0 + y1) / 2 / u
    vw, vh = x1 - x0, y1 - y0

    # centre crosshair, subject marker and pointing offset
    ink.mark((cx, cy), WHITE, 22, 1)
    ink.circle((cx, cy), 46, (230, 230, 230), 1)
    if marker is not None and n(marker[0]) and n(marker[1]) and 0 <= marker[0] < vw and 0 <= marker[1] < vh:
        mx, my = (x0 + marker[0]) / u, (y0 + marker[1]) / u
        ink.line((cx, cy), (mx, my), CYAN, 1)
        ink.mark((mx, my), CYAN, 14, 2, cv2.MARKER_TILTED_CROSS)
    if n(T["off_x_deg"][k]):
        ink.text(f"OFFSET {T['off_x_deg'][k]:+.1f}° {T['off_y_deg'][k]:+.1f}°", (cx, cy - 56), 0.42, CYAN, 1, "c")

    # top bar
    ink.panel(0, 0, Wd, 38, alpha=0.6)
    ink.text(f"{title}   t = {T['t'][k]:6.2f} s", (12, 25), 0.55, WHITE)
    status = str(T["status"][k])
    label = status + ("  (HOLD: ambiguous)" if T["ambiguous"][k] else "")
    ink.text(label, (Wd / 2, 25), 0.6, {"TRACKING": GREEN, "COASTING": AMBER, "LOST": RED}.get(status, GREY), 2, "c")
    ink.text(subtitle, (Wd - 12, 25), 0.45, GREY, align="r")
    top = 50
    if footer:
        ink.panel(0, 38, Wd, 18, alpha=0.5)
        ink.text(footer, (Wd / 2, 51), 0.36, AMBER, 1, "c")
        top = 66

    # ---------------- velocity
    x, y, pw, ph = 12, top, 292, 262
    ink.panel(x, y, pw, ph, "VELOCITY")
    if notes.get("velocity"):
        ink.text(notes["velocity"], (x + pw - 8, y + 16), 0.34, AMBER, align="r")
    ink.text("GROUND SPEED", (x + 10, y + 40), 0.4, GREY)
    ink.text(f"{T['gs'][k]:5.1f}", (x + 10, y + 78), 1.25, WHITE, 2)
    ink.text("m/s", (x + 112, y + 78), 0.5, GREY)
    ink.text(f"{T['gs_kmh'][k]:5.1f} km/h", (x + 160, y + 60), 0.5, WHITE)
    ink.text(f"CRS {T['course'][k]:03.0f}°" if T["gs"][k] > 0.3 else "CRS ---", (x + 160, y + 80), 0.5, CYAN)
    ink.text(f"ALT {T['alt'][k]:4.1f} m   CLIMB {T['climb'][k]:+4.1f} m/s", (x + 10, y + 100), 0.42, WHITE)
    ink.line((x + 8, y + 110), (x + pw - 8, y + 110), (90, 120, 90), 1)
    ink.text("SUBJECT", (x + 10, y + 128), 0.4, GREY)
    sub = f"{T['subj_speed'][k]:4.1f} m/s   HDG {T['subj_heading'][k]:03.0f}°" if n(T["subj_speed"][k]) else "-- (no lock)"
    ink.text(sub, (x + 76, y + 128), 0.48, AMBER)
    ink.text("DELTA-V  (drone - subject)", (x + 10, y + 152), 0.4, GREEN)
    lo_, hi_ = 0.2 * dv_scale, 0.6 * dv_scale
    for i, (nm, v) in enumerate((("ALONG", T["dv_along"][k]), ("CROSS", T["dv_cross"][k]))):
        yy = y + 174 + 22 * i
        ink.text(nm, (x + 10, yy), 0.45, GREY)
        ink.text(_fmt(v, "+5.2f", " m/s"), (x + 156, yy), 0.5, _sev(v, lo_, hi_), 1, "r")
        _gauge(ink, x + 168, yy - 4, 112, v, dv_scale)
    yy = y + 174 + 44
    rr = T["range_rate"][k]
    ink.text("RNG RATE", (x + 10, yy), 0.42, GREY)
    ink.text(_fmt(rr, "+5.2f", " m/s"), (x + 178, yy), 0.5, _sev(rr, lo_, hi_), 1, "r")
    ink.text("closing" if n(rr) and rr < -0.2 else ("opening" if n(rr) and rr > 0.2 else ("steady" if n(rr) else "")), (x + 186, yy), 0.45, GREY)
    ink.text(f"|dV| {_fmt(T['dv_mag'][k], '4.1f', ' m/s')}   dVz {_fmt(T['dv_vert'][k], '+4.1f')}", (x + 10, y + 252), 0.4, GREY)

    # ---------------- bearing
    x = Wd - 12 - 292
    ink.panel(x, y, pw, ph, "BEARING TO SUBJECT")
    if notes.get("bearing"):
        ink.text(notes["bearing"], (x + pw - 8, y + 16), 0.34, AMBER, align="r")
    _compass(ink, (x + 146, y + 100), 58, T, k)
    ink.text("drone course", (x + 10, y + 36), 0.34, CYAN)
    ink.text("subject hdg", (x + 10, y + 50), 0.34, AMBER)
    ink.text("to subject", (x + 10, y + 64), 0.34, GREEN)
    b = T["brg"][k]
    ink.text(f"BRG  {b:03.0f}° {_cardinal(b)}" if n(b) else "BRG  ---", (x + 10, y + 188), 0.58, GREEN, 1)
    ink.text(f"RNG  {_fmt(T['rng'][k], '5.1f', ' m')}   (horiz {_fmt(T['rng_h'][k], '4.1f')})", (x + 10, y + 208), 0.45, WHITE)
    ink.text(f"DEPR {_fmt(T['depr'][k], '4.1f', '° below horizon')}", (x + 10, y + 227), 0.45, WHITE)
    sh = T["shot"][k]
    ink.text(f"SHOT {sh:03.0f}°  {_sector(sh)}" if n(sh) else "SHOT  ---", (x + 10, y + 246), 0.45, AMBER)

    # ---------------- gimbal control + strip charts
    by, bh_ = Hd - 168, 156
    gx, gw = 12, 380
    ink.panel(gx, by, gw, bh_, "GIMBAL CONTROL")
    if notes.get("gimbal"):
        ink.text(notes["gimbal"], (gx + gw - 8, by + 16), 0.34, AMBER, align="r")
    for i, (nm, act, cmd, err, rate, vmax, fmt_) in enumerate((
            ("PAN ", T["pan"][k], T["pan_cmd"][k], T["pan_err"][k], T["pan_rate"][k], 20.0, "03.0f"),
            ("TILT", T["tilt"][k], T["tilt_cmd"][k], T["tilt_err"][k], T["tilt_rate"][k], 10.0, "+.0f"))):
        yy = by + 44 + 48 * i
        ink.text(nm, (gx + 10, yy), 0.55, WHITE)
        ink.text(_fmt(act, fmt_, "°"), (gx + 66, yy), 0.62, WHITE, 1)
        ink.text("cmd " + _fmt(cmd, fmt_, "°"), (gx + 160, yy), 0.45, GREY)
        ink.text("rate " + _fmt(rate, "+6.1f", "°/s"), (gx + gw - 10, yy), 0.45, CYAN, 1, "r")
        _gauge(ink, gx + 20, yy + 17, gw - 150, err, vmax)
        ink.text("corr " + _fmt(err, "+5.1f", "°"), (gx + gw - 10, yy + 21), 0.5, _sev(err, 1.5, 6.0), 1, "r")
    ink.text("corr + = turn right / tilt up" + ("   (corr, rate = servo commands)" if notes.get("gimbal") else ""), (gx + 10, by + bh_ - 8), 0.34, GREY)
    sw = (Wd - 24 - gw - 24) / 2
    tv = T["t"]
    _strip(ink, gx + gw + 12, by, sw, bh_, "DELTA-V m/s", [(T["dv_along"], GREEN), (T["dv_cross"], AMBER)], tv, k, strip_s, 0.8 * dv_scale,
           [("along", GREEN), ("cross", AMBER)])
    _strip(ink, gx + gw + 24 + sw, by, sw, bh_, "GIMBAL CORR deg", [(T["pan_err"], CYAN), (T["tilt_err"], AMBER)], tv, k, strip_s, 10.0,
           [("pan", CYAN), ("tilt", AMBER)])
    return img


# ============================================================================ simulation frame = scene + overlay
def render_hud_frame(res, T, k, size=(960, 720), trail_s=3.0, strip_s=6.0):
    cfg, L, world = res.cfg, res.log, res.world
    W, H = size
    s = W / cfg.cam.width
    cam = _cam(cfg)
    F, cx, cy = cam["F"] * s, W / 2, H / 2
    pos, yaw, pitch = L["cam_p"][k], L["cam_yaw"][k], L["cam_pitch"][k]
    t = L["t"][k]
    img, vh = _background(W, H, F, cy, pitch)
    _draw_grid(img, pos, yaw, pitch, F, cx, cy)

    # subject trail (what the drone believes), on the ground
    lo = max(0, k - int(trail_s / cfg.dt))
    trail = L["est_pos"][lo:k + 1].copy()
    ok = np.isin(L["status"][lo:k + 1], ("TRACKING", "COASTING"))
    if ok.sum() > 1:
        trail[:, 2] = 0.05
        pts, zc = _project(trail, pos, yaw, pitch, F, cx, cy)
        for i in range(len(pts) - 1):
            if ok[i] and ok[i + 1] and zc[i] > 0.8 and zc[i + 1] > 0.8:
                cv2.line(img, tuple(pts[i].astype(int)), tuple(pts[i + 1].astype(int)), (120, 255, 150), 2, cv2.LINE_AA)

    # world objects, far to near
    items = []
    for ent, p, _ in world.states(t):
        d = float(np.linalg.norm(p - pos))
        if d < 220:
            items.append((d, ent, p))
    for d, ent, p in sorted(items, key=lambda x: -x[0]):
        _draw_entity(img, ent, p, float(ent.psi[min(k, len(ent.psi) - 1)]), pos, yaw, pitch, F, cx, cy)

    # detector boxes and the tracker's lock
    for d in res.dets[k]:
        u, v, w, h = d.u * s, d.v * s, d.w * s, d.h * s
        cv2.rectangle(img, (int(u - w / 2), int(v - h / 2)), (int(u + w / 2), int(v + h / 2)), (175, 175, 175), 1, cv2.LINE_AA)
    mi = int(L["matched_idx"][k])
    if 0 <= mi < len(res.dets[k]):
        d = res.dets[k][mi]
        x1, y1, x2, y2 = (d.u - d.w / 2) * s - 5, (d.v - d.h / 2) * s - 5, (d.u + d.w / 2) * s + 5, (d.v + d.h / 2) * s + 5
        ln = max(8, int(0.25 * min(x2 - x1, y2 - y1)))
        for (ax, ay, dx, dy) in ((x1, y1, 1, 1), (x2, y1, -1, 1), (x1, y2, 1, -1), (x2, y2, -1, -1)):
            cv2.line(img, (int(ax), int(ay)), (int(ax + dx * ln), int(ay)), GREEN, 2, cv2.LINE_AA)
            cv2.line(img, (int(ax), int(ay)), (int(ax), int(ay + dy * ln)), GREEN, 2, cv2.LINE_AA)
        cv2.putText(img, "LOCK", (int(x1), int(y1) - 6), FONT, 0.45, GREEN, 1, cv2.LINE_AA)

    # pitch ladder (elevation e appears at v = cy - F tan(e - pitch))
    for de in (-30, -20, -10, 0, 10):
        yy = int(cy - F * math.tan(math.radians(de) - pitch))
        if 80 < yy < H - 180:
            cv2.line(img, (int(cx - 40), yy), (int(cx - 18), yy), (220, 255, 220), 1, cv2.LINE_AA)
            cv2.line(img, (int(cx + 18), yy), (int(cx + 40), yy), (220, 255, 220), 1, cv2.LINE_AA)
            cv2.putText(img, f"{de:+d}", (int(cx + 46), yy + 4), FONT, 0.36, (220, 255, 220), 1, cv2.LINE_AA)

    marker = (cx + T["off_u"][k] * s, cy + T["off_v"][k] * s) if np.isfinite(T["off_u"][k]) else None
    draw_overlay(img, T, k, scale=1.0 if W == 960 else None, title="CHASE CAM", subtitle=f"{cfg.scenario} | {cfg.tracker} tracker | seed {cfg.seed}",
                 marker=marker, dv_scale=5.0, strip_s=strip_s)
    return img


def render_hud_video(res, path, size=(960, 720), start=0.0, end=None, skip=1, T=None, progress=True):
    """Write the chase-camera HUD video. `skip` renders every n-th frame (output fps = sim fps / skip)."""
    T = T or compute_telemetry(res)
    n = len(res.log["t"])
    k0, k1 = int(start / res.cfg.dt), n if end is None else min(n, int(end / res.cfg.dt))
    fps = 1.0 / (res.cfg.dt * skip)
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not wr.isOpened():
        raise IOError(f"cannot open {path} for writing")
    for i, k in enumerate(range(k0, k1, skip)):
        wr.write(render_hud_frame(res, T, k, size))
        if progress and i % 200 == 199:
            print(f"  hud frame {k}/{n}")
    wr.release()
    return path


# ============================================================================ REAL VIDEO: telemetry estimated from the image
@dataclass
class LiveHudConfig:
    """Everything the HUD cannot measure from a single camera is an explicit, labelled ASSUMPTION here."""
    hfov_deg: float = 60.0            # camera horizontal field of view (set it to your lens)
    alt_m: float = 50.0               # ASSUMED: the camera drone hovers at this altitude, zero ground speed
    target_size_m: float = 12.0       # ASSUMED: real length of the target's LONGEST box side -> range from apparent size
    cam_heading_deg: float = 0.0      # ASSUMED: compass heading of the optical axis (0 = north)
    cam_pitch_deg: float = 0.0        # ASSUMED: camera pitch, + = up
    window_s: float = 6.0             # history kept for the strip charts
    size_fit_s: float = 1.2           # window for the apparent-size slope (range rate)
    vel_tau_s: float = 0.6            # smoothing time constant of the subject velocity estimate
    min_fit_samples: int = 6
    dv_scale: float = 15.0            # m/s at full scale on the DELTA-V gauges
    min_heading_speed: float = 0.5    # below this the subject heading is held
    scale_tol: float = 0.10           # 1/s: box width and height must scale together to trust a range rate
    max_speed_mps: float = 150.0      # estimates above this are shown as '--' (almost always a size jump, not real speed)
    hud_layout: str = "auto"          # overlay | dock | auto (dock = HUD in margins around the video, for frames narrower than 1280 px)


class LiveTelemetry:
    """Turns what a monocular tracker measures into HUD telemetry, under the assumptions in LiveHudConfig.

    Measured (per frame):  box centre and size, background-stabilised box velocity (px/s), the visual-servo errors/commands.
    Estimated from them:
      range            R = f * target_size / box_size                        (needs target_size_m)
      range rate       dR/dt = -R * d ln(box_size)/dt   (slope of a short regression window)
      tangential vel.  R * (stabilised angular velocity)                      (stabilised = camera/background motion removed)
      subject velocity = radial + tangential, in a world frame built from the ASSUMED camera heading/pitch.
    Assumed: drone hovers (velocity 0, altitude alt_m), so delta-v = -subject velocity. Pan/tilt angles are the assumed pose;
    PAN/TILT corr and rate are the servo's commands to centre the subject (real measurements of the image error)."""

    def __init__(self, cfg: LiveHudConfig, W, H):
        self.cfg, self.W, self.H = cfg, W, H
        self.F = (W / 2) / math.tan(math.radians(cfg.hfov_deg) / 2)
        self.yaw = math.radians(90.0 - cfg.cam_heading_deg)
        self.pitch = math.radians(cfg.cam_pitch_deg)
        self.f_ax, self.r_ax, self.u_ax = cam_axes(self.yaw, self.pitch)
        self.rows, self.size_hist = [], deque()
        self.last_track_t, self.last_c = None, None
        self.vs = None
        self.t_vs = None
        self.heading = np.array([1.0, 0.0])
        self.last = None

    def _range_and_rate(self, t, w, h, tracking, scale=None):
        """Range from the longest box side; range rate from how the box SCALES. Returns (R, rate); rate is None when the box width and
        height are not scaling together (a stretched box, rotor turning, a merged blob: size change that is not distance change)."""
        c = self.cfg
        if tracking:
            self.size_hist.append((t, math.log(max(w, h, 1.0)), math.log(max(w, 1.0)), math.log(max(h, 1.0)), math.log(max(scale or math.sqrt(max(w * h, 1.0)), 1.0))))
        while self.size_hist and t - self.size_hist[0][0] > c.size_fit_s:
            self.size_hist.popleft()
        if len(self.size_hist) < c.min_fit_samples or self.size_hist[-1][0] - self.size_hist[0][0] < 0.75 * c.size_fit_s:      # warm-up
            return None, None
        a = np.array(self.size_hist)
        ts = a[:, 0] - a[-1, 0]
        slope_long, icpt = np.polyfit(ts, a[:, 1], 1)                               # ln(longest side) now, via the fit
        sw, sh = np.polyfit(ts, a[:, 2], 1)[0], np.polyfit(ts, a[:, 3], 1)[0]
        R = float(np.clip(self.F * c.target_size_m / math.exp(icpt), 2.0, 20000.0))
        if abs(sw - sh) > c.scale_tol:
            return R, None
        g = float(np.polyfit(ts, a[:, 4], 1)[0])                                    # d ln(scale)/dt, scale = sqrt(blob area): not pixel-quantised
        return R, float(-R * (0.0 if abs(g) < 0.01 else g))

    def update(self, t, status, box, vel_stab, servo, ambiguous=False, scale_px=None):
        """box = [x1,y1,x2,y2] px (None if no subject); vel_stab = (vx, vy) px/s with camera motion removed;
        servo = {'err_x_deg','err_y_deg','yaw_rate_dps','pitch_rate_dps',...} (may be empty)."""
        c, nan = self.cfg, float("nan")
        row = {k: nan for k in ("subj_speed", "subj_heading", "dv_along", "dv_cross", "dv_vert", "dv_mag", "range_rate", "brg", "rng",
                                "rng_h", "depr", "shot", "pan_err", "tilt_err", "pan_cmd", "tilt_cmd", "pan_rate", "tilt_rate", "off_u",
                                "off_v", "off_x_deg", "off_y_deg", "size_px", "vs_e", "vs_n", "vs_u")}
        row.update(t=t, status=status, ambiguous=int(bool(ambiguous)), gs=0.0, gs_kmh=0.0, course=nan, alt=c.alt_m, climb=0.0,
                   pan=c.cam_heading_deg % 360.0, tilt=c.cam_pitch_deg)
        valid = box is not None and status in ("TRACKING", "COASTING")
        if valid:
            cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
            size_px = max(box[2] - box[0], box[3] - box[1])
            if status == "TRACKING":                                                    # a cut / re-acquisition breaks the size history
                jump = self.last_c is not None and math.hypot(cx - self.last_c[0], cy - self.last_c[1]) > 0.25 * self.W
                resized = bool(self.size_hist) and abs(math.log(max(size_px, 1.0)) - self.size_hist[-1][1]) > math.log(1.4)
                if (self.last_track_t is not None and t - self.last_track_t > 0.4) or jump or resized:
                    self.size_hist.clear()
                    self.vs = None
                self.last_track_t, self.last_c = t, (cx, cy)
            R, rr = self._range_and_rate(t, box[2] - box[0], box[3] - box[1], status == "TRACKING", scale_px)
            ex, ey = cx - self.W / 2, cy - self.H / 2
            d = self.f_ax + (ex / self.F) * self.r_ax + (-ey / self.F) * self.u_ax
            los = d / np.linalg.norm(d)
            row.update(off_u=ex, off_v=ey, size_px=size_px, off_x_deg=math.degrees(math.atan2(ex, self.F)),
                       off_y_deg=math.degrees(math.atan2(ey, self.F)))
            if servo:
                row.update(pan_err=servo["err_x_deg"], tilt_err=-servo["err_y_deg"], pan_rate=servo["yaw_rate_dps"], tilt_rate=servo["pitch_rate_dps"],
                           pan_cmd=(row["pan"] + servo["err_x_deg"]) % 360.0, tilt_cmd=row["tilt"] - servo["err_y_deg"])
            row["brg"] = (90.0 - math.degrees(math.atan2(los[1], los[0]))) % 360.0
            if R is not None and rr is not None:
                row.update(rng=R, rng_h=R * float(np.hypot(los[0], los[1])), depr=math.degrees(math.atan2(-los[2], max(float(np.hypot(los[0], los[1])), 1e-9))))
                w_x, w_y = vel_stab[0] / self.F, vel_stab[1] / self.F                    # angular rates (rad/s), stabilised
                v = rr * los + R * (w_x * self.r_ax - w_y * self.u_ax)                  # subject velocity, ENU, camera at rest
                if float(np.linalg.norm(v)) > c.max_speed_mps:
                    self.rows.append(row)                                                  # implausible: keep range, show no speed
                    return self._finish(row, t)
                row["range_rate"] = rr
                if self.vs is None or t - self.t_vs > 1.0:
                    self.vs = v
                else:
                    self.vs = self.vs + (1 - math.exp(-(t - self.t_vs) / c.vel_tau_s)) * (v - self.vs)
                self.t_vs = t
                vs = self.vs
                sp = float(np.hypot(vs[0], vs[1]))
                if sp > c.min_heading_speed:
                    self.heading = vs[:2] / sp
                h = self.heading
                dv = -vs                                                                 # drone (at rest) minus subject
                row.update(subj_speed=sp, subj_heading=(90.0 - math.degrees(math.atan2(h[1], h[0]))) % 360.0,
                           dv_along=float(dv[:2] @ h), dv_cross=float(dv[:2] @ np.array([-h[1], h[0]])), dv_vert=float(dv[2]),
                           dv_mag=float(np.linalg.norm(dv)), vs_e=vs[0], vs_n=vs[1], vs_u=vs[2])
                to_drone = -R * los[:2]
                row["shot"] = math.degrees(math.atan2(to_drone @ np.array([h[1], -h[0]]), to_drone @ h)) % 360.0
        self.rows.append(row)
        return self._finish(row, t)

    def _finish(self, row, t):
        while len(self.rows) > 2 and row["t"] - self.rows[0]["t"] > self.cfg.window_s + 2.0:
            self.rows.pop(0)
        self.last = row
        return row

    def arrays(self):
        """(T, k): dict of arrays over the retained window (the format draw_overlay expects) and the index of the newest frame."""
        keys = [k for k in self.rows[0] if k != "status"]
        T = {k: np.array([r[k] for r in self.rows], dtype=float) for k in keys}
        T["status"] = np.array([r["status"] for r in self.rows])
        return T, len(self.rows) - 1

    def notes(self):
        c = self.cfg
        return {"velocity": "drone ASSUMED hover | subject EST", "bearing": "HDG ASSUMED, RNG EST", "gimbal": "pose ASSUMED"}

    def footer(self):
        c = self.cfg
        return (f"ESTIMATED TELEMETRY  -  assumed: hover {c.alt_m:.0f} m, target {c.target_size_m:.0f} m, HFOV {c.hfov_deg:.0f}°, "
                f"cam HDG {c.cam_heading_deg:03.0f}° PITCH {c.cam_pitch_deg:+.0f}°")


def _dock_margins(u):
    return int(round(312 * u)), int(round(66 * u)), int(round(176 * u))          # side, top, bottom (pixels)


def resolve_layout(layout, W):
    return ("dock" if W < 1280 else "overlay") if layout == "auto" else layout


def hud_output_size(W, H, layout="auto", scale=None):
    """Size of the frames draw_live_hud returns (the video itself for 'overlay'; the video plus margins for 'dock')."""
    if resolve_layout(layout, W) != "dock":
        return W, H
    side, top, bottom = _dock_margins(scale or auto_scale(W, H))
    return W + 2 * side, H + top + bottom


def draw_live_hud(frame, live: LiveTelemetry, scale=None, box_center=None, layout=None):
    """Draw the HUD from the telemetry accumulated so far. 'overlay' draws on `frame` in place; 'dock' returns a larger canvas with the
    video in the middle and the panels in the margins (so nothing covers the picture). Always use the returned image."""
    H, W = frame.shape[:2]
    layout = resolve_layout(layout or live.cfg.hud_layout, W)
    u = scale or auto_scale(W, H)
    T, k = live.arrays()
    kw = dict(scale=u, title="LIVE", subtitle="estimated from video", notes=live.notes(), footer=live.footer(), marker=box_center, dv_scale=live.cfg.dv_scale)
    if layout == "dock":
        side, top, bottom = _dock_margins(u)
        canvas = np.full((H + top + bottom, W + 2 * side, 3), 16, np.uint8)
        canvas[top:top + H, side:side + W] = frame
        cv2.rectangle(canvas, (side - 1, top - 1), (side + W, top + H), (110, 140, 110), 1, cv2.LINE_AA)
        return draw_overlay(canvas, T, k, rect=(side, top, side + W, top + H), **kw)
    return draw_overlay(frame, T, k, **kw)
