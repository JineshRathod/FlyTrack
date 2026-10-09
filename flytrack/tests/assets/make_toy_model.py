"""Trains a TOY single-box regressor with the same I/O contract as the user's model:
   input  (1,224,224,3) float32 in [0,1]  ->  output (1,4) sigmoid [ymin, xmin, ymax, xmax] (normalised).
   Synthetic helicopter-like silhouettes on synthetic skies. FOR TESTING THE TRACKING PIPELINE ONLY."""
import sys, numpy as np, cv2
import tensorflow as tf
rng = np.random.default_rng(0)
S = 224

def sky(rng):
    g = np.linspace(rng.uniform(80, 220), rng.uniform(80, 220), S)[:, None] * np.ones((1, S))
    img = g + cv2.GaussianBlur(rng.normal(0, 25, (S, S)).astype(np.float32), (0, 0), 20) * 2
    for _ in range(rng.integers(0, 4)):                                  # clouds / clutter blobs
        cv2.ellipse(img, (int(rng.integers(0, S)), int(rng.integers(0, S))), (int(rng.integers(10, 50)), int(rng.integers(4, 18))),
                    float(rng.integers(0, 180)), 0, 360, float(rng.uniform(60, 240)), -1)
    if rng.random() < 0.4:                                               # horizon band
        y = int(rng.integers(int(S * 0.6), S)); img[y:] = rng.uniform(20, 120)
    img = cv2.GaussianBlur(img, (0, 0), 1.2) + rng.normal(0, 3, img.shape)
    return np.clip(img, 0, 255).astype(np.float32)

def draw_heli(img, cx, cy, L, col, rot):
    m = np.zeros((S, S), np.uint8)
    ang = rot
    ca, sa = np.cos(ang), np.sin(ang)
    def P(dx, dy): return (int(cx + ca * dx - sa * dy), int(cy + sa * dx + ca * dy))
    cv2.ellipse(m, (int(cx), int(cy)), (int(L * 0.28), int(L * 0.11)), float(np.degrees(ang)), 0, 360, 255, -1)       # body
    cv2.line(m, P(L * 0.2, 0), P(L * 0.62, -L * 0.04), 255, max(1, int(L * 0.045)))                                   # tail boom
    cv2.line(m, P(L * 0.62, -L * 0.14), P(L * 0.62, L * 0.08), 255, max(1, int(L * 0.03)))                            # tail rotor
    cv2.line(m, P(-L * 0.55, -L * 0.14), P(L * 0.55, -L * 0.14), 255, max(1, int(L * 0.025)))                         # main rotor
    ys, xs = np.nonzero(m)
    if len(xs) == 0: return None
    img[m > 0] = col
    return [ys.min() / S, xs.min() / S, (ys.max() + 1) / S, (xs.max() + 1) / S]

def sample(rng):
    img = sky(rng)
    L = rng.uniform(0.35, 0.8) * S
    cx, cy = rng.uniform(0.3, 0.7) * S, rng.uniform(0.3, 0.7) * S
    col = float(rng.choice([rng.uniform(0, 40), rng.uniform(215, 255)]))
    if rng.random() < 0.5: col = float(rng.uniform(0, 40)) if img.mean() > 120 else float(rng.uniform(215, 255))
    box = draw_heli(img, cx, cy, L, col, rng.uniform(-0.25, 0.25))
    if box is None or min(box[:2]) < 0 or max(box[2:]) > 1: return sample(rng)
    img = cv2.GaussianBlur(img, (0, 0), 0.8)
    return np.repeat(np.clip(img, 0, 255)[..., None], 3, axis=2) / 255.0, np.array(box, np.float32)

class Seq(tf.keras.utils.Sequence):
    def __init__(self, n, bs): self.n, self.bs = n, bs
    def __len__(self): return self.n
    def __getitem__(self, i):
        r = np.random.default_rng(i + 12345 * (i % 7))
        xs, ys = zip(*[sample(r) for _ in range(self.bs)])
        return np.stack(xs).astype(np.float32), np.stack(ys)

if __name__ == "__main__":
    steps = int(sys.argv[1]) if len(sys.argv) > 1 else 1200
    inp = tf.keras.Input((S, S, 3))
    x = inp
    for f in (16, 32, 48, 64, 96):
        x = tf.keras.layers.Conv2D(f, 3, strides=2, padding="same", activation="relu")(x)
        x = tf.keras.layers.Conv2D(f, 3, padding="same", activation="relu")(x)
    x = tf.keras.layers.Flatten()(x)
    x = tf.keras.layers.Dense(128, activation="relu")(x)
    x = tf.keras.layers.Dropout(0.1)(x)
    out = tf.keras.layers.Dense(4, activation="sigmoid")(x)
    model = tf.keras.Model(inp, out)
    model.compile(optimizer=tf.keras.optimizers.Adam(2e-3), loss=tf.keras.losses.Huber())
    EP = int(sys.argv[2]) if len(sys.argv) > 2 else 6
    model.fit(Seq(steps // EP, 32), epochs=EP, verbose=2)
    # quick IoU check on fresh samples
    xs, ys = Seq(4, 64)[999]
    p = model.predict(xs, verbose=0)
    def iou(a, b):
        iy = max(0, min(a[2], b[2]) - max(a[0], b[0])); ix = max(0, min(a[3], b[3]) - max(a[1], b[1]))
        i = ix * iy; u = (a[2]-a[0])*(a[3]-a[1]) + (b[2]-b[0])*(b[3]-b[1]) - i; return i / u
    ious = [iou(a, b) for a, b in zip(p, ys)]
    print("held-out mean IoU %.3f, median %.3f" % (np.mean(ious), np.median(ious)))
    conv = tf.lite.TFLiteConverter.from_keras_model(model)
    open("toy_regressor.tflite", "wb").write(conv.convert())
    print("saved toy_regressor.tflite")
