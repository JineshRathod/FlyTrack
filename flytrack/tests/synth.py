"""Synthetic test video with known ground truth: a small helicopter over a panning sky, a look-alike helicopter that
crosses right next to it (and partly covers it), and fast birds. Ground truth is used ONLY for scoring in tests."""
import json
import numpy as np
import cv2


def draw_heli(canvas, cx, cy, L, col, rot=0.0):
    m = np.zeros(canvas.shape[:2], np.uint8)
    ca, sa = np.cos(rot), np.sin(rot)
    P = lambda dx, dy: (int(cx + ca * dx - sa * dy), int(cy + sa * dx + ca * dy))
    cv2.ellipse(m, (int(cx), int(cy)), (max(1, int(L * 0.28)), max(1, int(L * 0.11))), float(np.degrees(rot)), 0, 360, 255, -1)
    cv2.line(m, P(L * 0.2, 0), P(L * 0.62, -L * 0.04), 255, max(1, int(L * 0.045)))
    cv2.line(m, P(L * 0.62, -L * 0.14), P(L * 0.62, L * 0.08), 255, max(1, int(L * 0.03)))
    cv2.line(m, P(-L * 0.55, -L * 0.14), P(L * 0.55, -L * 0.14), 255, max(1, int(L * 0.025)))
    canvas[m > 0] = col
    ys, xs = np.nonzero(m)
    return None if len(xs) == 0 else [float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)]


def make_video(path, seconds=12.0, fps=30, seed=0, W=640, H=480, birds=True, twin=True, pan=True, dark=True, noise=2.0, L0=52.0, growth=3.0):
    rng = np.random.default_rng(seed)
    n = int(seconds * fps)
    CW, CH = 1700, 900
    sky = np.linspace(215, 160, CH)[:, None] * np.ones((1, CW))
    sky += cv2.GaussianBlur(rng.normal(0, 25, (CH, CW)).astype(np.float32), (0, 0), 40) * 2.5
    for _ in range(14):
        cv2.ellipse(sky, (int(rng.integers(0, CW)), int(rng.integers(0, CH))), (int(rng.integers(30, 120)), int(rng.integers(8, 30))),
                    float(rng.integers(0, 180)), 0, 360, float(rng.uniform(185, 235)), -1)
    sky = cv2.GaussianBlur(sky, (0, 0), 4).astype(np.float32)
    if not dark:
        sky = 255 - sky
    col = 25.0 if dark else 235.0

    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    gt = {"fps": fps, "W": W, "H": H, "subject": [], "twin": [], "birds": []}
    bird_plan = [(3.0, 0.0, 330.0, 1), (8.0, 0.0, 230.0, -1), (10.5, 0.4, 280.0, 1)] if birds else []
    for k in range(n):
        t = k / fps
        cam = np.array([200 + (28.0 * t + 12 * np.sin(0.7 * t) if pan else 0.0), 220 + (6.0 * t if pan else 0.0)])
        frame = sky[int(cam[1]):int(cam[1]) + H, int(cam[0]):int(cam[0]) + W].copy()
        # subject (world coordinates), slowly approaching
        sw = np.array([cam[0] + 130 + 22 * t + 40 * np.sin(0.5 * t), cam[1] + 230 - 6 * t + 25 * np.sin(0.8 * t)])
        L = L0 + growth * t
        # twin: crosses the subject's path around t=5.5 s, moving against it, slightly lower
        tw = np.array([cam[0] + 560 - 125 * (t - 0.0), cam[1] + 243 + 4 * np.sin(1.1 * t)]) if twin else None
        scene = frame.copy()
        sub_box = draw_heli(scene, sw[0] - cam[0], sw[1] - cam[1], L, col, 0.05 * np.sin(t))
        tw_box = draw_heli(scene, tw[0] - cam[0], tw[1] - cam[1], L * 0.95, col, -0.04) if twin else None
        b_boxes = []
        for (tb, dy, spd, sgn) in bird_plan:
            bx = sw[0] - cam[0] + sgn * spd * (t - tb) - 0.0
            by = sw[1] - cam[1] - 6 + dy * 60 + 8 * np.sin(6 * t)
            if -20 < bx < W + 20:
                m = np.zeros((H, W), np.uint8)
                cv2.ellipse(m, (int(bx), int(by)), (7, 3), 0, 0, 360, 255, -1)
                scene[m > 0] = col
                ys, xs = np.nonzero(m)
                if len(xs): b_boxes.append([float(xs.min()), float(ys.min()), float(xs.max() + 1), float(ys.max() + 1)])
        img = scene + rng.normal(0, noise, scene.shape)
        img = np.clip(cv2.GaussianBlur(img, (0, 0), 0.8), 0, 255).astype(np.uint8)
        wr.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        gt["subject"].append(sub_box); gt["twin"].append(tw_box); gt["birds"].append(b_boxes)
    wr.release()
    json.dump(gt, open(path + ".gt.json", "w"))
    return gt


def score(results, gt, hit_frac=0.5, min_px=15.0):
    """hit = tracker box centre within max(min_px, hit_frac * GT width) of the subject centre.
    wrong = tracker box centre closer to the twin / a bird than to the subject, while within that distance of it."""
    hits = wrong = vis = locked_frames = 0
    first_lock = None
    for r, sb, tb, bbs in zip(results, gt["subject"], gt["twin"], gt["birds"]):
        if sb is not None and 0 <= (sb[0] + sb[2]) / 2 < gt["W"] and 0 <= (sb[1] + sb[3]) / 2 < gt["H"]:
            vis += 1
        if r.box is None or r.status not in ("TRACKING", "COASTING"):
            continue
        locked_frames += 1
        first_lock = r.frame if first_lock is None else first_lock
        cx, cy = (r.box[0] + r.box[2]) / 2, (r.box[1] + r.box[3]) / 2
        def near(b):
            return b is not None and np.hypot(cx - (b[0] + b[2]) / 2, cy - (b[1] + b[3]) / 2) <= max(min_px, hit_frac * (b[2] - b[0]))
        if near(sb):
            hits += 1
        elif near(tb) or any(near(b) for b in bbs):
            wrong += 1
    return {"visible": vis, "hits": hits, "hit_rate": hits / max(vis, 1), "wrong_object_frames": wrong, "locked_frames": locked_frames, "first_lock_frame": first_lock}


def make_approach_video(path, seconds=6.0, fps=30, W=640, H=480, hfov=60.0, L=12.0, R0=150.0, v_close=10.0, v_tan=0.0, dark=True, seed=0):
    """A target of real length L metres seen by a static camera: range R(t) = R0 - v_close t, crossing sideways at v_tan m/s.
    Apparent size follows the pinhole model exactly, so the HUD's range / speed estimates can be checked against known truth."""
    rng = np.random.default_rng(seed)
    F = (W / 2) / np.tan(np.radians(hfov) / 2)
    sky = np.linspace(215, 165, H)[:, None] * np.ones((1, W)) + cv2.GaussianBlur(rng.normal(0, 8, (H, W)).astype(np.float32), (0, 0), 30) * 3
    sky = (255 - sky) if not dark else sky
    wr = cv2.VideoWriter(path, cv2.VideoWriter_fourcc(*"mp4v"), fps, (W, H))
    truth = []
    for k in range(int(seconds * fps)):
        t = k / fps
        R = R0 - v_close * t
        x_m = v_tan * (t - seconds / 2)                       # lateral offset in metres (0 at mid-clip)
        w = F * L / np.hypot(R, x_m)
        cx = W / 2 + F * x_m / R
        img = sky.copy()
        cv2.ellipse(img, (int(round(cx)), H // 2), (max(2, int(round(w / 2))), max(1, int(round(w * 0.18)))), 0, 0, 360, 25.0 if dark else 235.0, -1)
        img = np.clip(cv2.GaussianBlur(img + rng.normal(0, 1.5, img.shape), (0, 0), 0.8), 0, 255).astype(np.uint8)
        wr.write(cv2.cvtColor(img, cv2.COLOR_GRAY2BGR))
        truth.append({"t": t, "R": float(np.hypot(R, x_m)), "cx": float(cx), "w": float(w)})
    wr.release()
    return truth
