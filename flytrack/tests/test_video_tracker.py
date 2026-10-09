import csv
import json
import os
import sys

import numpy as np
import pytest

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, ".."))
sys.path.insert(0, HERE)
import synth  # noqa: E402
from track_flying_object import Config, TFLiteModel, track_video  # noqa: E402

MODEL = os.path.join(HERE, "assets", "toy_regressor.tflite")      # TOY model, only to exercise the pipeline


@pytest.fixture(scope="session")
def two_heli(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("v") / "two.mp4")
    gt = synth.make_video(p, seconds=12, seed=0)                 # subject + crossing look-alike + 3 birds + camera pan
    sb = gt["subject"][0]
    return p, gt, (sb[0], sb[1], sb[2] - sb[0], sb[3] - sb[1])


@pytest.fixture(scope="session")
def one_heli(tmp_path_factory):
    p = str(tmp_path_factory.mktemp("v") / "one.mp4")
    return p, synth.make_video(p, seconds=12, seed=3, twin=False)


def test_model_free_tracking_survives_lookalike_and_birds(two_heli):
    p, gt, ib = two_heli
    res = track_video(p, init_box=ib, polarity="dark", quiet=True)
    s = synth.score(res, gt)
    assert s["hit_rate"] > 0.95 and s["wrong_object_frames"] == 0


def test_tflite_hybrid_tracking_survives_lookalike_and_birds(two_heli):
    p, gt, ib = two_heli
    res = track_video(p, model=MODEL, init_box=ib, polarity="dark", quiet=True)
    s = synth.score(res, gt)
    assert s["hit_rate"] > 0.95 and s["wrong_object_frames"] == 0


def test_tflite_hybrid_locks_automatically(one_heli):
    p, gt = one_heli
    res = track_video(p, model=MODEL, polarity="dark", quiet=True)
    s = synth.score(res, gt)
    assert s["hit_rate"] > 0.95 and s["wrong_object_frames"] == 0 and s["first_lock_frame"] <= 10


def test_outputs_video_and_csv(one_heli, tmp_path):
    p, _ = one_heli
    out, log = str(tmp_path / "o.mp4"), str(tmp_path / "o.csv")
    res = track_video(p, out=out, csv_path=log, init_box=(95, 220, 70, 20), polarity="dark", max_frames=60, quiet=True)
    assert os.path.getsize(out) > 1000
    rows = list(csv.DictReader(open(log)))
    assert len(rows) == len(res) == 60
    assert {"status", "cx", "cy", "yaw_rate_dps", "pitch_rate_dps", "fwd_cmd", "conf"} <= set(rows[0])


def test_servo_commands_point_toward_the_target(one_heli):
    p, _ = one_heli
    res = track_video(p, init_box=(95, 220, 70, 20), polarity="dark", max_frames=40, quiet=True)
    r = res[20]
    cx = (r.box[0] + r.box[2]) / 2
    assert cx < 320 and r.servo["yaw_rate_dps"] < 0 and r.servo["err_x_deg"] < 0       # target left of centre -> turn left


def test_unsupported_model_layout_is_rejected_clearly(tmp_path, monkeypatch):
    import track_flying_object as T

    class Fake:
        def allocate_tensors(self): pass
        def get_input_details(self): return [{"index": 0, "dtype": np.float32, "shape": [1, 224, 224, 3]}]
        def get_output_details(self): return [{"index": 1, "shape": [1, 10], "dtype": np.float32}]

    monkeypatch.setattr(T, "_make_interpreter", lambda path: Fake())
    with pytest.raises(ValueError, match="Unsupported TFLite output layout"):
        TFLiteModel("x.tflite", Config())


def test_ssd_style_outputs_are_parsed(monkeypatch):
    import track_flying_object as T

    class Fake:
        def allocate_tensors(self): pass
        def invoke(self): pass
        def set_tensor(self, i, x): self.x = x
        def get_input_details(self): return [{"index": 0, "dtype": np.uint8, "shape": [1, 300, 300, 3], "quantization": (0.0, 0)}]
        def get_output_details(self):
            return [{"index": 1, "shape": [1, 3, 4], "dtype": np.float32, "name": "boxes"},
                    {"index": 2, "shape": [1, 3], "dtype": np.float32, "name": "classes"},
                    {"index": 3, "shape": [1, 3], "dtype": np.float32, "name": "scores"}]
        def get_tensor(self, i):
            return {1: np.array([[[.1, .2, .3, .4], [.5, .5, .9, .9], [0, 0, .1, .1]]], np.float32),
                    2: np.array([[0, 0, 1]], np.float32), 3: np.array([[.9, .2, .6]], np.float32)}[i]

    monkeypatch.setattr(T, "_make_interpreter", lambda path: Fake())
    m = TFLiteModel("x.tflite", Config(input_range="255"))
    out = m.infer(np.zeros((50, 60, 3), np.uint8))
    assert m.kind == "ssd" and len(out) == 2 and abs(out[0][1] - 0.9) < 1e-6 and out[0][0] == pytest.approx([.1, .2, .3, .4])


def test_regressor_contract_matches_documented_format():
    m = TFLiteModel(MODEL, Config())
    assert m.kind == "regressor" and m.size == (224, 224)
    (box, score), = m.infer(np.full((300, 400, 3), 128, np.uint8))
    assert len(box) == 4 and score is None and 0 <= box[0] <= box[2] <= 1 and 0 <= box[1] <= box[3] <= 1
