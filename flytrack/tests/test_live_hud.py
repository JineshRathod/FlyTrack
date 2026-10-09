import csv
import os
import sys
import time

import cv2
import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import synth  # noqa: E402
import track_flying_object as T  # noqa: E402
from hud import hud_output_size  # noqa: E402


def read_csv(path):
    return [{k: v for k, v in r.items()} for r in csv.DictReader(open(path))]


def f(r, k):
    return float(r[k]) if r[k] != "" else float("nan")


@pytest.fixture(scope="module")
def closing(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("v") / "closing.mp4")
    truth = synth.make_approach_video(p, seconds=6, R0=150.0, v_close=10.0)
    return p, truth


@pytest.fixture(scope="module")
def crossing(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("v") / "crossing.mp4")
    truth = synth.make_approach_video(p, seconds=6, R0=100.0, v_close=0.0, v_tan=8.0)
    return p, truth


def first_box(path):
    cap = cv2.VideoCapture(path)
    ok, fr = cap.read()
    g = cv2.cvtColor(fr, cv2.COLOR_BGR2GRAY)
    ys, xs = np.nonzero(g < 120)
    return (xs.min(), ys.min(), xs.max() - xs.min(), ys.max() - ys.min())


def test_hud_range_and_closing_speed_match_known_geometry(closing, tmp_path):
    """Measured on this synthetic geometry: range ~5 % low, closing speed ~15-20 % low (blob edges blur the size change)."""
    p, truth = closing
    log = str(tmp_path / "t.csv")
    T.track_video(p, init_box=first_box(p), polarity="dark", hud=True, hfov_deg=60, target_size_m=12.0, csv_path=log, quiet=True, hud_layout="overlay")
    rows = read_csv(log)
    ks = range(60, 175)
    R = np.array([f(rows[k], "range_m_est") for k in ks]); Rt = np.array([truth[k]["R"] for k in ks])
    rr = np.array([f(rows[k], "range_rate_mps_est") for k in ks])
    sp = np.array([f(rows[k], "subj_speed_mps_est") for k in ks])
    assert np.isfinite(rr).mean() > 0.9
    assert np.nanmedian(np.abs(R / Rt - 1)) < 0.12
    assert np.nanmedian(rr) == pytest.approx(-10.0, abs=3.0) and np.nanmedian(rr) < 0          # closing, and it says so
    assert np.nanmedian(sp) == pytest.approx(10.0, abs=3.0)


def test_hud_tangential_speed_from_stabilised_motion(crossing, tmp_path):
    p, truth = crossing
    log = str(tmp_path / "t.csv")
    T.track_video(p, init_box=first_box(p), polarity="dark", hud=True, hfov_deg=60, target_size_m=12.0, csv_path=log, quiet=True, hud_layout="overlay")
    rows = read_csv(log)
    speeds = [f(rows[k], "subj_speed_mps_est") for k in range(90, 150)]
    heads = [f(rows[k], "subj_heading_deg_est") for k in range(90, 150)]
    assert np.nanmedian(speeds) == pytest.approx(8.0, rel=0.2)
    assert np.nanmedian(heads) == pytest.approx(90.0, abs=12.0)          # moving left->right with the camera assumed to face north = east


def test_hover_assumption_makes_delta_v_the_negative_subject_velocity(closing, tmp_path):
    p, _ = closing
    log = str(tmp_path / "t.csv")
    T.track_video(p, init_box=first_box(p), polarity="dark", hud=True, csv_path=log, quiet=True, hud_layout="overlay")
    r = read_csv(log)[150]
    assert f(r, "dv_along_mps") == pytest.approx(-f(r, "subj_speed_mps_est"), abs=1e-6) and abs(f(r, "dv_cross_mps")) < 1e-6


def test_docked_hud_leaves_the_video_untouched_and_matches_the_predicted_size(closing, tmp_path):
    p, _ = closing
    out = str(tmp_path / "o.mp4")
    T.track_video(p, init_box=first_box(p), polarity="dark", hud=True, out=out, max_frames=40, quiet=True, hud_layout="dock")
    cap = cv2.VideoCapture(out)
    W, H = int(cap.get(3)), int(cap.get(4))
    assert (W, H) == hud_output_size(640, 480, "dock")
    ok, fr = cap.read()
    raw = cv2.VideoCapture(p).read()[1]
    side = int(round(312 * 0.75)); top = int(round(66 * 0.75))
    inner = fr[top:top + 480, side:side + 640].astype(float)
    assert np.abs(inner - raw.astype(float)).mean() < 12            # video shown as is (only the lock box / marker drawn on it)


def test_overlay_layout_keeps_the_frame_size(closing, tmp_path):
    p, _ = closing
    out = str(tmp_path / "o.mp4")
    T.track_video(p, init_box=first_box(p), polarity="dark", hud=True, out=out, max_frames=10, quiet=True, hud_layout="overlay")
    cap = cv2.VideoCapture(out)
    assert (int(cap.get(3)), int(cap.get(4))) == (640, 480)


def test_hud_off_is_unchanged_and_still_runs(closing):
    p, _ = closing
    res = T.track_video(p, init_box=first_box(p), polarity="dark", quiet=True)
    assert all(r.status in ("TRACKING", "COASTING") for r in res) and res.stats["frames_dropped"] == 0


# ------------------------------------------------------------------ real-time behaviour
def test_latest_frame_reader_returns_the_newest_frame_and_counts_drops():
    class Cap:
        def __init__(self): self.i = 0
        def read(self):
            if self.i >= 50:
                return False, None
            self.i += 1
            return True, np.full((4, 4, 3), self.i, np.uint8)

    rd = T.LatestFrameReader(Cap())                              # unpaced: the reader runs far ahead of the consumer
    time.sleep(0.2)
    frame, stamp, idx = rd.get()
    assert idx == 49 and frame[0, 0, 0] == 50                    # skipped straight to the newest frame
    assert rd.get() is None                                      # then end of stream
    rd.close()


def test_stalled_stream_raises_timeout():
    class Cap:
        def read(self):
            time.sleep(5)
            return False, None
    rd = T.LatestFrameReader(Cap())
    with pytest.raises(TimeoutError):
        rd.get(timeout=0.2)


@pytest.mark.parametrize("src,live", [("0", True), ("1", True), ("rtsp://cam/stream", True), ("http://x/y.mjpg", True), ("clip.mp4", False), ("/data/a.mp4", False)])
def test_live_source_detection(src, live):
    assert T.is_live_source(src) is live


def test_slow_tracker_on_a_live_source_drops_frames_and_keeps_latency_bounded(closing, monkeypatch):
    p, _ = closing
    orig = T.FlyTracker.step

    def slow(self, *a, **k):
        time.sleep(0.08)                                          # ~12 fps tracker on a 30 fps camera
        return orig(self, *a, **k)

    monkeypatch.setattr(T.FlyTracker, "step", slow)
    res = T.track_video(p, init_box=first_box(p), polarity="dark", realtime=True, max_frames=40, quiet=True)
    st = res.stats
    assert st["frames_dropped"] > 20                              # it skipped ahead instead of queueing
    assert st["latency_ms_p95"] < 300                             # latency stays bounded (a queue would grow to seconds)
    assert 8 < st["fps"] < 20
