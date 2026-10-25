"""
Multimodal Attendance & Engagement Agent  (Skill Wallet Capstone Project 5)

Privacy-first, 100% offline classroom analytics:
  * Tab 1  Presence-based attendance  (MediaPipe / OpenCV face tracking, sustained-presence rule)
  * Tab 2  Acoustic interaction       (Librosa/NumPy spectral energy + voice-activity heuristics)
  * Tab 3  Multimodal fusion          (visual attentiveness + audio interaction -> Engagement Index 0-100)
  * Tab 4  Teaching suggestion engine (heuristic, non-intrusive pedagogical prompts)
  * Tab 5  Session analytics & report (KPIs + downloadable report)

Privacy by design: no frame or audio is stored, nothing is sent over a network, tracking IDs are
anonymous, and engagement is reported for the class as a whole, never per individual.
"""
from __future__ import annotations

import json
import math
import os
import tempfile
import time
from datetime import datetime

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

try:
    import cv2
except Exception:  # pragma: no cover
    cv2 = None
try:
    import mediapipe as mp
except Exception:  # pragma: no cover
    mp = None
try:
    import librosa
except Exception:  # pragma: no cover
    librosa = None

st.set_page_config(page_title="Multimodal Attendance & Engagement Agent", page_icon="🎓", layout="wide")

# ----------------------------------------------------------------------------- design tokens
INK, TEAL, AMBER, CORAL, SLATE, MIST = "#16324F", "#1F8A8A", "#D99A00", "#CF4F3A", "#5B6B7B", "#EAF0F4"
AUDIO_CLASSES = ["Silence", "Instructor Lecture", "Student Discussion", "Ambient Distraction"]
AUDIO_COLORS = {"Silence": "#C9D3DB", "Instructor Lecture": INK, "Student Discussion": TEAL, "Ambient Distraction": CORAL}
AUDIO_SCORE = {"Silence": 0.35, "Instructor Lecture": 0.70, "Student Discussion": 1.00, "Ambient Distraction": 0.0}

st.markdown(
    f"""
<style>
html, body, [class*="css"] {{ font-family: "Segoe UI", system-ui, -apple-system, "Helvetica Neue", Arial, sans-serif; }}
.block-container {{ padding-top: 1.6rem; max-width: 1280px; }}
h1, h2, h3 {{ color: {INK}; letter-spacing: -0.01em; }}
[data-testid="stMetric"] {{ background: {MIST}; border-left: 4px solid {TEAL}; padding: .7rem .9rem; border-radius: 4px; }}
[data-testid="stMetricLabel"] {{ color: {SLATE}; }}
.stTabs [data-baseweb="tab-list"] {{ gap: 4px; border-bottom: 2px solid {MIST}; }}
.stTabs [data-baseweb="tab"] {{ padding: .5rem 1rem; font-weight: 600; color: {SLATE}; }}
.stTabs [aria-selected="true"] {{ color: {INK}; }}
.banner {{ background: {INK}; color: #fff; padding: 1.1rem 1.4rem; border-radius: 6px; margin-bottom: 1rem; }}
.banner h1 {{ color: #fff; margin: 0 0 .2rem 0; font-size: 1.7rem; }}
.banner p {{ margin: 0; color: #C9D8E6; }}
.advice {{ border-left: 5px solid {AMBER}; background: #FFF8E6; padding: .7rem 1rem; margin: .5rem 0; border-radius: 3px; color: #3A2E00; }}
.advice.tip {{ border-color: {TEAL}; background: #E8F5F5; color: #0E3B3B; }}
.advice.positive {{ border-color: #3C9A5F; background: #E9F6EE; color: #123B22; }}
.advice small {{ color: {SLATE}; }}
</style>
<div class="banner"><h1>Multimodal Attendance &amp; Engagement Agent</h1>
<p>Offline classroom analytics for the whole class, not individuals. Nothing leaves this device.</p></div>
""",
    unsafe_allow_html=True,
)

plt.rcParams.update({"axes.spines.top": False, "axes.spines.right": False, "axes.edgecolor": "#9AA8B5",
                     "axes.labelcolor": SLATE, "xtick.color": SLATE, "ytick.color": SLATE, "font.size": 9})


# ----------------------------------------------------------------------------- helpers
def fmt_t(s: float) -> str:
    s = int(max(0, round(float(s))))
    return f"{s // 60:02d}:{s % 60:02d}"


def fmt_dur(s: float) -> str:
    s = int(max(0, round(float(s))))
    return f"{s // 60}m {s % 60:02d}s"


def write_tmp(data: bytes, suffix: str) -> str:
    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as f:
        f.write(data)
        return f.name


def find_runs(mask: np.ndarray):
    """Maximal runs of True -> list of (start_idx, end_idx_exclusive)."""
    runs, start = [], None
    for i, m in enumerate(mask):
        if m and start is None:
            start = i
        if start is not None and (not m or i == len(mask) - 1):
            runs.append((start, i if not m else i + 1))
            start = None
    return runs


def top_idx(y: np.ndarray, k: int, sep: int, largest: bool = True):
    order = np.argsort(-y if largest else y)
    chosen = []
    for i in order:
        if all(abs(int(i) - j) >= sep for j in chosen):
            chosen.append(int(i))
        if len(chosen) == k:
            break
    return chosen


# ----------------------------------------------------------------------------- EPIC 2: vision
class FaceDetector:
    """MediaPipe face detection (with eye/nose keypoints for head pose); OpenCV Haar fallback."""

    def __init__(self, min_conf: float = 0.5):
        self.backend, self._mp, self._haar = "none", None, None
        if mp is not None and cv2 is not None:
            try:
                self._mp = mp.solutions.face_detection.FaceDetection(model_selection=1, min_detection_confidence=min_conf)
                self.backend = "MediaPipe"
            except Exception:
                self._mp = None
        if self.backend == "none" and cv2 is not None:
            self._haar = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")
            self.backend = "OpenCV Haar (fallback)"

    def detect(self, bgr: np.ndarray) -> list[dict]:
        out: list[dict] = []
        if self._mp is not None:
            res = self._mp.process(cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB))
            for d in res.detections or []:
                bb, kp = d.location_data.relative_bounding_box, d.location_data.relative_keypoints
                yaw = roll = 0.0
                if len(kp) >= 3:
                    re, le, nose = kp[0], kp[1], kp[2]
                    eye_d = max(1e-6, math.hypot(le.x - re.x, le.y - re.y))
                    yaw = (nose.x - (re.x + le.x) / 2) / eye_d  # ~0 when facing the front
                    ang = math.degrees(math.atan2(le.y - re.y, le.x - re.x))
                    roll = abs((ang + 90) % 180 - 90)  # head tilt in degrees
                out.append({"cx": bb.xmin + bb.width / 2, "cy": bb.ymin + bb.height / 2,
                            "w": bb.width, "h": bb.height, "yaw": float(yaw), "roll": float(roll)})
        elif self._haar is not None:
            h, w = bgr.shape[:2]
            gray = cv2.equalizeHist(cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY))
            for (x, y, fw, fh) in self._haar.detectMultiScale(gray, 1.1, 6, minSize=(24, 24)):
                # A frontal-cascade hit implies a roughly forward-facing head.
                out.append({"cx": (x + fw / 2) / w, "cy": (y + fh / 2) / h, "w": fw / w, "h": fh / h, "yaw": 0.0, "roll": 0.0})
        return out


@st.cache_resource(show_spinner=False)
def get_detector(min_conf: float) -> FaceDetector:
    return FaceDetector(min_conf)


class CentroidTracker:
    """Greedy nearest-centroid tracker assigning anonymous, stable IDs."""

    def __init__(self, max_dist: float = 0.10, max_gap: float = 8.0):
        self.max_dist, self.max_gap, self.tracks, self.next_id = max_dist, max_gap, {}, 1

    def update(self, t: float, dets: list[dict]) -> list[int]:
        active = [(i, v) for i, v in self.tracks.items() if t - v[2] <= self.max_gap]
        pairs = sorted((math.hypot(d["cx"] - v[0], d["cy"] - v[1]), di, i)
                       for di, d in enumerate(dets) for i, v in active)
        ids, used_t = [None] * len(dets), set()
        for dist, di, i in pairs:
            if dist > self.max_dist:
                break
            if ids[di] is None and i not in used_t:
                ids[di] = i
                used_t.add(i)
        for di, d in enumerate(dets):
            if ids[di] is None:
                ids[di] = self.next_id
                self.next_id += 1
            self.tracks[ids[di]] = (d["cx"], d["cy"], t)
        return ids


REC_COLS = ["t", "track", "cx", "cy", "w", "h", "yaw", "roll"]


def video_frames(path: str, sample_fps: float, max_frames: int):
    cap = cv2.VideoCapture(path)
    if not cap.isOpened():
        raise ValueError("The video file could not be opened. Try an MP4 (H.264) file.")
    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
    step = max(1, int(round(fps / sample_fps)))
    if total > 0:
        step = max(step, math.ceil(total / max_frames))
    meta = {"duration": total / fps if total > 0 else None, "dt": step / fps,
            "hint": math.ceil(total / step) if total > 0 else None}

    def gen():
        i = 0
        try:
            while cap.grab():
                if i % step == 0:
                    ok, fr = cap.retrieve()
                    if ok:
                        yield i / fps, fr
                i += 1
        finally:
            cap.release()

    return gen(), meta


def webcam_frames(seconds: float, sample_fps: float):
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        raise RuntimeError("No local webcam found. Live capture works only when the app runs on a machine with a camera.")
    t0, nxt = time.time(), 0.0
    try:
        while time.time() - t0 < seconds:
            ok, fr = cap.read()
            if not ok:
                break
            t = time.time() - t0
            if t >= nxt:
                yield t, fr
                nxt += 1.0 / sample_fps
    finally:
        cap.release()


def detect_stream(frames, detector: FaceDetector, hint=None, progress=None):
    tracker, rows, n, last_t = CentroidTracker(), [], 0, 0.0
    for t, fr in frames:
        if fr.shape[1] > 800:
            s = 800 / fr.shape[1]
            fr = cv2.resize(fr, None, fx=s, fy=s)
        dets = detector.detect(fr)
        for d, i in zip(dets, tracker.update(t, dets)):
            rows.append({"t": t, "track": i, **d})
        n, last_t = n + 1, t
        if progress is not None and hint:
            progress.progress(min(1.0, n / hint))
    return pd.DataFrame(rows, columns=REC_COLS), n, last_t


def add_posture(rec: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """Forward head pose + upright posture (face dropping below the person's own median seated height)."""
    rec = rec.copy()
    if rec.empty:
        rec["attentive"] = pd.Series(dtype=bool)
        return rec
    med = rec.groupby("track")["cy"].transform("median")
    rec["forward"] = (rec["yaw"].abs() < cfg["yaw_max"]) & (rec["roll"] < 25)
    rec["upright"] = ((rec["cy"] - med) / rec["h"].clip(lower=1e-3)) < cfg["slump"]
    rec["attentive"] = rec["forward"] & rec["upright"]
    return rec


def build_attendance(rec: pd.DataFrame, duration: float, dt: float, threshold: float, snapshot: bool) -> pd.DataFrame:
    rows = []
    for tid, g in rec.groupby("track"):
        seen = len(g) * dt
        ratio = 1.0 if snapshot or duration <= 0 else min(1.0, seen / duration)
        present = snapshot or ratio >= threshold
        note = "Single-frame snapshot" if snapshot else ("Transient pass-through" if ratio < 0.15 else "")
        rows.append({"Student ID": f"S-{int(tid):03d}", "First Seen": fmt_t(g["t"].min()), "Last Seen": fmt_t(g["t"].max()),
                     "Total Duration": fmt_dur(seen) if not snapshot else "n/a", "Presence %": round(ratio * 100, 1),
                     "Status": "Present" if present else "Absent", "Note": note})
    return pd.DataFrame(rows, columns=["Student ID", "First Seen", "Last Seen", "Total Duration", "Presence %", "Status", "Note"])


# ----------------------------------------------------------------------------- EPIC 3: audio
def load_audio(data: bytes, suffix: str, sr: int = 16000, max_s: int = 3600) -> np.ndarray:
    if librosa is None:
        raise RuntimeError("librosa is not installed.")
    path = write_tmp(data, suffix)
    try:
        y, _ = librosa.load(path, sr=sr, mono=True, duration=max_s)
    finally:
        os.remove(path)
    return y


def _window_features(seg: np.ndarray, sr: int) -> dict:
    rms = float(np.sqrt(np.mean(seg ** 2) + 1e-12))
    spec = np.abs(np.fft.rfft(seg * np.hanning(len(seg)))) ** 2 + 1e-12
    freqs = np.fft.rfftfreq(len(seg), 1 / sr)
    speech_ratio = float(spec[(freqs >= 300) & (freqs <= 3400)].sum() / spec.sum())
    try:
        flat = float(np.mean(librosa.feature.spectral_flatness(y=seg)))
    except Exception:
        flat = float(np.exp(np.mean(np.log(spec))) / np.mean(spec))
    sub_n = max(1, int(0.05 * sr))
    k = len(seg) // sub_n
    sub = np.sqrt(np.mean(seg[: k * sub_n].reshape(k, sub_n) ** 2, axis=1) + 1e-12)
    return {"rms": rms, "rms_db": 20 * math.log10(rms), "speech_ratio": speech_ratio, "flatness": flat, "_sub": sub}


def analyze_audio(y: np.ndarray, sr: int, cfg: dict) -> pd.DataFrame:
    n = int(cfg["win_s"] * sr)
    if len(y) < n:
        raise ValueError("The audio is shorter than one analysis window.")
    feats = [_window_features(y[k * n:(k + 1) * n], sr) for k in range(len(y) // n)]
    f = pd.DataFrame(feats)
    f.insert(0, "t", np.arange(len(f)) * cfg["win_s"])
    floor = max(float(np.percentile(f["rms"], 5)), 1e-5)
    med = float(np.median(f["rms"]))
    f["activity"] = [float(np.mean(s > floor * 2.5)) for s in f["_sub"]]
    silent = ((f["rms"] < floor * cfg["sil_factor"]) & (f["rms"] < 0.35 * med)) | (f["rms_db"] < -55)
    speechy = (~silent) & (f["speech_ratio"] >= cfg["speech_min"]) & (f["flatness"] <= cfg["flat_max"])
    speech_rms = float(f.loc[speechy, "rms"].median()) if speechy.any() else 0.0
    lecture = speechy & (f["activity"] >= 0.6) & (f["rms"] >= 0.9 * speech_rms)
    label = np.full(len(f), "Ambient Distraction", dtype=object)
    label[speechy.to_numpy()] = "Student Discussion"
    label[lecture.to_numpy()] = "Instructor Lecture"
    label[silent.to_numpy()] = "Silence"
    f["label"] = label
    return f.drop(columns=["_sub"])


# ----------------------------------------------------------------------------- EPIC 4: fusion
def fuse(rec: pd.DataFrame | None, aud: pd.DataFrame | None, duration: float, cfg: dict) -> pd.DataFrame:
    step = cfg["step"]
    duration = max(duration, step)
    rows = []
    for t0 in np.arange(0, duration, step):
        t1 = t0 + step
        v = rec[(rec["t"] >= t0) & (rec["t"] < t1)] if rec is not None and len(rec) else None
        a = aud[(aud["t"] >= t0) & (aud["t"] < t1)] if aud is not None and len(aud) else None
        vis = float(v["attentive"].mean()) if v is not None and len(v) else np.nan
        heads = float(v.groupby("t")["track"].nunique().mean()) if v is not None and len(v) else np.nan
        au = float(a["label"].map(AUDIO_SCORE).mean()) if a is not None and len(a) else np.nan
        dom = a["label"].mode().iloc[0] if a is not None and len(a) else ""
        num = den = 0.0
        if not np.isnan(vis):
            num, den = num + cfg["w_v"] * vis, den + cfg["w_v"]
        if not np.isnan(au):
            num, den = num + (1 - cfg["w_v"]) * au, den + (1 - cfg["w_v"])
        rows.append({"t": float(t0), "visual": vis, "headcount": heads, "audio": au, "dominant": dom,
                     "raw": 100 * num / den if den > 0 else np.nan})
    tl = pd.DataFrame(rows)
    tl["engagement"] = tl["raw"].interpolate(limit_direction="both").rolling(cfg["smooth"], min_periods=1).mean().clip(0, 100)
    return tl


# ----------------------------------------------------------------------------- EPIC 5: advisory agent
def generate_advisories(tl: pd.DataFrame, cfg: dict) -> list[dict]:
    step, sustain = cfg["step"], cfg["sustain_min"] * 60
    t, eng = tl["t"].to_numpy(), tl["engagement"].to_numpy()
    vis, dom = tl["visual"].to_numpy(), tl["dominant"].fillna("").to_numpy()
    adv: list[dict] = []

    def runs(mask, min_dur):
        return [(s, e) for s, e in find_runs(np.asarray(mask)) if (e - s) * step >= min_dur]

    def add(kind, s, e, title, msg, min_dur):
        adv.append({"issued": float(t[s] + min_dur), "start": float(t[s]), "end": float(t[e - 1] + step),
                    "kind": kind, "title": title, "message": msg})

    for s, e in runs(np.nan_to_num(eng, nan=100) < cfg["low"], sustain):
        seg = pd.Series(dom[s:e])
        main = seg[seg != ""].mode()
        main = main.iloc[0] if len(main) else ""
        mins = (t[e - 1] + step - t[s]) / 60
        span = f"{fmt_t(t[s])}–{fmt_t(t[e - 1] + step)}"
        head = f"Collective engagement dropped below {cfg['low']}% for {mins:.1f} minutes ({span}) — "
        tail = {
            "Ambient Distraction": "background noise is high; try a quiet signal, then a structured pair activity.",
            "Silence": "the room has been quiet; try a think-pair-share or a low-stakes question to re-activate participation.",
            "Student Discussion": "discussion is lively but attention is slipping; refocus with a summary question or rotate roles.",
        }.get(main, "recommend launching a quick poll, interactive discussion, or a 2-minute stretch break.")
        add("alert", s, e, "Sustained low engagement", head + tail, sustain)

    for s, e in runs(dom == "Instructor Lecture", cfg["lecture_max"] * 60):
        mins = (t[e - 1] + step - t[s]) / 60
        add("tip", s, e, "Long uninterrupted lecture",
            f"Lecture has run for {mins:.0f} minutes without student interaction ({fmt_t(t[s])}–{fmt_t(t[e - 1] + step)}) — "
            "consider a check-for-understanding question or a turn-and-talk.", cfg["lecture_max"] * 60)

    noise_min = max(cfg["sustain_min"] / 2, 0.25) * 60
    for s, e in runs(dom == "Ambient Distraction", noise_min):
        add("tip", s, e, "Elevated classroom noise",
            f"Ambient noise dominated for {(t[e - 1] + step - t[s]) / 60:.1f} minutes ({fmt_t(t[s])}) — "
            "pause for a short reset and restate the next task clearly.", noise_min)

    for s, e in runs(np.nan_to_num(vis, nan=1) < 0.5, sustain):
        add("tip", s, e, "Posture and attention drift",
            f"Fewer than half of visible learners are facing forward and upright ({fmt_t(t[s])}–{fmt_t(t[e - 1] + step)}) — "
            "a short stretch or change of activity can help.", sustain)

    for s, e in runs(np.nan_to_num(eng, nan=0) >= cfg["high"], sustain):
        add("positive", s, e, "Strong momentum",
            f"Engagement stayed above {cfg['high']}% from {fmt_t(t[s])} to {fmt_t(t[e - 1] + step)} — "
            "note which activity worked and reuse it.", sustain)

    return sorted(adv, key=lambda a: a["issued"])


# ----------------------------------------------------------------------------- demo data
def make_demo(seed: int = 7, minutes: int = 40, n_students: int = 24):
    rng = np.random.default_rng(seed)
    dur, dt = minutes * 60.0, 5.0
    ts = np.arange(0, dur, dt)

    def att_p(t):
        m, p = t / 60, 0.9
        for a, b, d in [(14, 22, 0.5), (33, 39, 0.45)]:
            if a <= m < b:
                p -= d * math.sin(math.pi * (m - a) / (b - a))
        return p

    rows = []
    span = {i: (0, dur) for i in range(1, n_students + 1)}
    span[21], span[22] = (0, 18 * 60), (0, 18 * 60)           # leave early -> ~45% presence
    span[n_students + 1] = (9 * 60, 9 * 60 + 15)               # hallway pass-through
    span[n_students + 2] = (9 * 60 + 5, 9 * 60 + 20)           # hallway pass-through
    span[n_students + 3] = (30 * 60, dur)                      # very late arrival
    for tid, (a, b) in span.items():
        base_cy, base_cx = rng.uniform(0.25, 0.8), rng.uniform(0.05, 0.95)
        for t in ts[(ts >= a) & (ts < b)]:
            if rng.random() < 0.03:
                continue
            p = att_p(t)
            fwd, up = rng.random() < p, rng.random() < min(1, p + 0.05)
            rows.append({"t": t, "track": tid, "cx": base_cx, "cy": base_cy + rng.normal(0, 0.004) + (0.06 if not up else 0),
                         "w": 0.08, "h": 0.12, "yaw": 0.0 if fwd else 0.9, "roll": 5.0})
    rec = pd.DataFrame(rows, columns=REC_COLS)

    phases = [(0, 14, [.05, .8, .1, .05]), (14, 22, [.1, .5, .05, .35]), (22, 29, [.03, .12, .8, .05]),
              (29, 33, [.05, .8, .1, .05]), (33, 39, [.25, .25, .05, .45]), (39, 40, [.05, .1, .8, .05])]
    wt = np.arange(0, dur, 2.0)
    labels = []
    for t in wt:
        pr = next(p for a, b, p in phases if a <= t / 60 < b)
        labels.append(rng.choice(AUDIO_CLASSES, p=pr))
    prof = {"Silence": (-62, 3, .30, .6), "Instructor Lecture": (-24, 3, .70, .05),
            "Student Discussion": (-30, 4, .60, .08), "Ambient Distraction": (-26, 4, .25, .40)}
    aud = pd.DataFrame({"t": wt, "label": labels})
    aud["rms_db"] = [rng.normal(prof[l][0], prof[l][1]) for l in labels]
    aud["speech_ratio"] = [np.clip(rng.normal(prof[l][2], .05), 0, 1) for l in labels]
    aud["flatness"] = [np.clip(rng.normal(prof[l][3], .03), 0, 1) for l in labels]
    return ({"rec": rec, "duration": dur, "dt": dt, "snapshot": False, "backend": "Synthetic demo"},
            {"df": aud, "duration": dur, "source": "Synthetic demo"})


# ----------------------------------------------------------------------------- plots
def plot_engagement(tl, cfg, clock, adv):
    fig, (ax, ax2) = plt.subplots(2, 1, figsize=(10, 5.2), sharex=True, gridspec_kw={"height_ratios": [2.2, 1]})
    x, y = tl["t"] / 60, tl["engagement"].to_numpy()
    past = (tl["t"] <= clock).to_numpy()
    ax.axhspan(0, cfg["low"], color=CORAL, alpha=0.07)
    ax.plot(x, y, color=INK, alpha=0.2, lw=1.4)
    ax.plot(x[past], y[past], color=INK, lw=2.2)
    if past.sum() > 3:
        yp, xp = y[past], x[past].to_numpy()
        for i in top_idx(yp, 2, 6, True):
            ax.scatter(xp[i], yp[i], color=TEAL, zorder=5)
            ax.annotate(f"peak {yp[i]:.0f}", (xp[i], yp[i]), textcoords="offset points", xytext=(0, 8), ha="center", color=TEAL, fontsize=8)
        for i in top_idx(yp, 2, 6, False):
            ax.scatter(xp[i], yp[i], color=CORAL, zorder=5)
            ax.annotate(f"dip {yp[i]:.0f}", (xp[i], yp[i]), textcoords="offset points", xytext=(0, -14), ha="center", color=CORAL, fontsize=8)
    for a in adv:
        if a["issued"] <= clock and a["kind"] != "positive":
            ax.axvline(a["issued"] / 60, color=AMBER, ls=":", lw=1.2)
    ax.axvline(clock / 60, color=SLATE, lw=0.8)
    ax.set_ylim(0, 105)
    ax.set_ylabel("Engagement Index")
    ax.set_title("Class engagement over time (dotted amber lines = advisories issued)", loc="left", color=INK, fontsize=10)
    ax2.plot(x, tl["visual"] * 100, color=INK, lw=1.4, label="Visual attentiveness %")
    ax2.plot(x, tl["audio"] * 100, color=TEAL, lw=1.4, label="Audio interaction %")
    ax2.set_ylim(0, 105)
    ax2.set_xlabel("Minutes")
    ax2.legend(frameon=False, loc="lower left", ncol=2, fontsize=8)
    fig.tight_layout()
    return fig


def plot_audio(df, win_s):
    fig, (a1, a2) = plt.subplots(2, 1, figsize=(10, 3.8), sharex=True, gridspec_kw={"height_ratios": [2, 0.6]})
    x = df["t"] / 60
    a1.plot(x, df["rms_db"], color=INK, lw=1)
    a1.set_ylabel("Level (dB)")
    a1.set_title("Spectral energy and classified audio state", loc="left", color=INK, fontsize=10)
    a2.bar(x, 1, width=win_s / 60, align="edge", color=[AUDIO_COLORS[l] for l in df["label"]])
    a2.set_yticks([])
    a2.set_xlabel("Minutes")
    a2.spines["left"].set_visible(False)
    handles = [plt.Rectangle((0, 0), 1, 1, color=AUDIO_COLORS[c]) for c in AUDIO_CLASSES]
    a1.legend(handles, AUDIO_CLASSES, frameon=False, ncol=4, fontsize=8, loc="lower center", bbox_to_anchor=(0.5, 1.12))
    fig.tight_layout()
    return fig


def plot_presence(att, threshold):
    d = att.sort_values("Presence %").tail(40)
    fig, ax = plt.subplots(figsize=(6, max(2.5, 0.22 * len(d) + 1)))
    ax.barh(d["Student ID"], d["Presence %"], color=[TEAL if s == "Present" else "#B9C4CE" for s in d["Status"]])
    ax.axvline(threshold * 100, color=CORAL, ls="--", lw=1.2)
    ax.text(threshold * 100 + 1, 0, f"{threshold:.0%} rule", color=CORAL, fontsize=8, va="bottom")
    ax.set_xlim(0, 105)
    ax.set_xlabel("Presence (% of session)")
    fig.tight_layout()
    return fig


# ----------------------------------------------------------------------------- sidebar / config
S = st.session_state
S.setdefault("visual", None)
S.setdefault("audio", None)

with st.sidebar:
    st.header("Session controls")
    c1, c2 = st.columns(2)
    if c1.button("Load demo"):
        S["visual"], S["audio"] = make_demo()
        st.rerun()
    if c2.button("Clear"):
        S["visual"] = S["audio"] = None
        st.rerun()
    st.caption("Anonymous track IDs. No frames or audio are stored. No network calls. Engagement is class-level only.")
    with st.expander("Attendance and vision", expanded=True):
        presence = st.slider("Sustained presence threshold (% of session)", 30, 100, 70) / 100
        sample_fps = st.slider("Frames analysed per second", 0.2, 5.0, 1.0, 0.2)
        max_frames = st.number_input("Max frames per video", 50, 5000, 600, 50)
        yaw_max = st.slider("Head-turn tolerance", 0.2, 0.9, 0.45, 0.05, help="Lower is stricter about facing forward.")
        slump = st.slider("Slump sensitivity", 0.1, 0.8, 0.35, 0.05, help="Face drop, in face heights, counted as slumping.")
    with st.expander("Audio classification"):
        win_s = st.slider("Window length (s)", 1.0, 5.0, 2.0, 0.5)
        sil_factor = st.slider("Silence margin", 1.0, 4.0, 1.8, 0.1)
        speech_min = st.slider("Min speech-band energy ratio", 0.2, 0.8, 0.45, 0.05)
        flat_max = st.slider("Max spectral flatness for speech", 0.05, 0.6, 0.2, 0.05)
    with st.expander("Fusion and advisories"):
        step = st.slider("Fusion time step (s)", 5, 60, 10, 5)
        w_v = st.slider("Weight on visual attentiveness (%)", 10, 90, 60, 5) / 100
        smooth = st.slider("Smoothing (steps)", 1, 9, 3)
        low = st.slider("Low engagement threshold", 20, 70, 45)
        high = st.slider("High engagement threshold", 60, 95, 75)
        sustain_min = st.number_input("Sustain before advising (minutes)", 0.25, 30.0, 5.0, 0.25)
        lecture_max = st.number_input("Max uninterrupted lecture (minutes)", 2.0, 60.0, 10.0, 1.0)

cfg = dict(presence=presence, yaw_max=yaw_max, slump=slump, win_s=win_s, sil_factor=sil_factor, speech_min=speech_min,
           flat_max=flat_max, step=step, w_v=w_v, smooth=smooth, low=low, high=high, sustain_min=sustain_min, lecture_max=lecture_max)

# ----------------------------------------------------------------------------- derived data
V, A = S["visual"], S["audio"]
rec = att = tl = None
adv: list[dict] = []
if V is not None and len(V["rec"]):
    rec = add_posture(V["rec"], cfg)
    att = build_attendance(rec, V["duration"], V["dt"], presence, V["snapshot"])
duration = max(V["duration"] if V else 0.0, A["duration"] if A else 0.0)
if rec is not None or A is not None:
    tl = fuse(rec, A["df"] if A else None, duration, cfg)
    adv = generate_advisories(tl, cfg)

tab1, tab2, tab3, tab4, tab5 = st.tabs(["Attendance", "Audio and noise", "Engagement fusion", "Teaching suggestions", "Session report"])

# ----------------------------------------------------------------------------- TAB 1
with tab1:
    st.subheader("Presence-based attendance")
    st.caption("A learner is marked Present only when detected for at least the sustained-presence share of the session, "
               "so a hallway pass-through never counts.")
    if cv2 is None:
        st.error("OpenCV is not installed. Add opencv-python-headless to requirements.txt.")
    else:
        src = st.radio("Source", ["Upload video", "Upload photo", "Webcam snapshot", "Local webcam recording"], horizontal=True)
        det = get_detector(0.5)
        st.caption(f"Face detector: {det.backend}")
        try:
            if src == "Upload video":
                f = st.file_uploader("Classroom video", type=["mp4", "mov", "avi", "mkv", "webm"])
                if f and st.button("Analyse video", type="primary"):
                    path = write_tmp(f.getvalue(), os.path.splitext(f.name)[1] or ".mp4")
                    bar = st.progress(0.0, "Tracking faces…")
                    try:
                        frames, meta = video_frames(path, sample_fps, int(max_frames))
                        df, n, last_t = detect_stream(frames, det, meta["hint"], bar)
                    finally:
                        os.remove(path)
                    dur = meta["duration"] or (last_t + meta["dt"])
                    S["visual"] = {"rec": df, "duration": dur, "dt": meta["dt"], "snapshot": False, "backend": det.backend}
                    st.rerun()
            elif src in ("Upload photo", "Webcam snapshot"):
                f = st.file_uploader("Classroom photo", type=["jpg", "jpeg", "png"]) if src == "Upload photo" else st.camera_input("Take a snapshot")
                if f and st.button("Analyse photo", type="primary"):
                    img = cv2.imdecode(np.frombuffer(f.getvalue(), np.uint8), cv2.IMREAD_COLOR)
                    if img is None:
                        raise ValueError("The image could not be decoded.")
                    df, _, _ = detect_stream([(0.0, img)], det)
                    S["visual"] = {"rec": df, "duration": 0.0, "dt": 0.0, "snapshot": True, "backend": det.backend}
                    st.rerun()
            else:
                secs = st.number_input("Recording length (seconds)", 5, 600, 30, 5)
                if st.button("Start recording", type="primary"):
                    with st.spinner("Recording from the local webcam…"):
                        df, n, last_t = detect_stream(webcam_frames(secs, sample_fps), det)
                    S["visual"] = {"rec": df, "duration": float(secs), "dt": 1.0 / sample_fps, "snapshot": False, "backend": det.backend}
                    st.rerun()
        except Exception as e:
            st.error(f"Could not process the visual input: {e}")

    if V is not None and att is None:
        st.warning("No faces were detected. Try better lighting, a higher frame rate, or a clearer view of the class.")
    if att is not None:
        n_present = int((att["Status"] == "Present").sum())
        participants = int((att["Presence %"] >= 15).sum()) if not V["snapshot"] else len(att)
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Tracks detected", len(att))
        k2.metric("Present", n_present)
        k3.metric("Absent / transient", len(att) - n_present)
        k4.metric("Attendance rate", f"{(n_present / participants * 100) if participants else 0:.0f}%")
        left, right = st.columns([3, 2])
        with left:
            st.dataframe(att, hide_index=True)
            st.download_button("Download attendance register (CSV)", att.to_csv(index=False).encode(), "attendance_register.csv", "text/csv")
        with right:
            if not V["snapshot"]:
                fig = plot_presence(att, presence)
                st.pyplot(fig)
                plt.close(fig)
    elif V is None:
        st.info("Upload a classroom video or photo, or load the demo from the sidebar.")

# ----------------------------------------------------------------------------- TAB 2
with tab2:
    st.subheader("Acoustic interaction and noise dynamics")
    st.caption("Each window is classified as Silence, Instructor Lecture, Student Discussion or Ambient Distraction using "
               "loudness, speech-band energy, spectral flatness and pause patterns. Audio is processed in memory only.")
    f = st.file_uploader("Audio or video file", type=["wav", "mp3", "ogg", "flac", "m4a", "mp4", "mov", "mkv", "webm"], key="aud_up")
    if f and st.button("Analyse audio", type="primary"):
        try:
            with st.spinner("Analysing audio…"):
                y = load_audio(f.getvalue(), os.path.splitext(f.name)[1] or ".wav")
                df = analyze_audio(y, 16000, cfg)
            S["audio"] = {"df": df, "duration": len(y) / 16000, "source": f.name}
            st.rerun()
        except Exception as e:
            st.error(f"Could not process the audio: {e}. For video files, ffmpeg must be available (see packages.txt).")
    if A is None:
        st.info("Upload a recording or load the demo from the sidebar.")
    else:
        # Re-classify live when thresholds change (real recordings keep features; demo keeps its labels)
        df = A["df"]
        mix = df["label"].value_counts(normalize=True).reindex(AUDIO_CLASSES).fillna(0) * 100
        cols = st.columns(4)
        for c, name in zip(cols, AUDIO_CLASSES):
            c.metric(name, f"{mix[name]:.0f}%")
        fig = plot_audio(df, df["t"].diff().median() if len(df) > 1 else cfg["win_s"])
        st.pyplot(fig)
        plt.close(fig)
        with st.expander("Window-level features"):
            st.dataframe(df.round(3), hide_index=True)

# ----------------------------------------------------------------------------- TAB 3
with tab3:
    st.subheader("Multimodal fusion and Engagement Index")
    if tl is None:
        st.info("Add attendance video or audio first. Either modality alone works; both give the best picture.")
    else:
        used = [n for n, ok in (("visual attentiveness", rec is not None), ("audio interaction", A is not None)) if ok]
        st.caption(f"Fused from: {' + '.join(used)}. Index = weighted blend of the share of visible learners facing forward and "
                   f"upright, and the interaction value of the audio state, scaled to 0–100. Both streams are assumed to start together.")
        clock = st.slider("Session clock (replay as if live)", 0.0, float(tl["t"].max() + step), float(tl["t"].max() + step), float(step), key="clock")
        k1, k2, k3, k4 = st.columns(4)
        k1.metric("Average engagement", f"{tl['engagement'].mean():.0f}")
        k2.metric("Peak", f"{tl['engagement'].max():.0f}", fmt_t(tl.loc[tl['engagement'].idxmax(), 't']))
        k3.metric("Lowest dip", f"{tl['engagement'].min():.0f}", fmt_t(tl.loc[tl['engagement'].idxmin(), 't']))
        k4.metric("Time below threshold", f"{(tl['engagement'] < low).mean() * 100:.0f}%")
        fig = plot_engagement(tl, cfg, clock, adv)
        st.pyplot(fig)
        plt.close(fig)
        show = tl.copy()
        show["t"] = show["t"].map(fmt_t)
        show = show.drop(columns=["raw"]).round(2)
        with st.expander("Fused time-step matrix"):
            st.dataframe(show, hide_index=True)
            st.download_button("Download timeline (CSV)", show.to_csv(index=False).encode(), "engagement_timeline.csv", "text/csv")

# ----------------------------------------------------------------------------- TAB 4
with tab4:
    st.subheader("Teaching suggestion engine")
    st.caption("Prompts are for the educator only, tied to class-wide patterns, and never name or single out a learner.")
    if tl is None:
        st.info("Suggestions appear once engagement has been computed.")
    else:
        clock = S.get("clock", float(tl["t"].max() + step))
        live = [a for a in adv if a["issued"] <= clock]
        st.markdown(f"**Session clock {fmt_t(clock)}** — {len(live)} of {len(adv)} advisories issued so far. Move the clock in the Engagement tab to replay.")
        if not live:
            st.success("No advisories at this point. The class looks on track.")
        for a in reversed(live):
            cls = {"alert": "advice", "tip": "advice tip", "positive": "advice positive"}[a["kind"]]
            st.markdown(f'<div class="{cls}"><b>{a["title"]}</b> <small>issued at {fmt_t(a["issued"])}</small><br>{a["message"]}</div>',
                        unsafe_allow_html=True)

# ----------------------------------------------------------------------------- TAB 5
def build_report() -> tuple[str, dict]:
    k = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M"), "duration": fmt_dur(duration)}
    if att is not None:
        pres = int((att["Status"] == "Present").sum())
        parts = int((att["Presence %"] >= 15).sum()) if not V["snapshot"] else len(att)
        k.update(present=pres, participants=parts, attendance_rate_pct=round(pres / parts * 100, 1) if parts else 0)
    if tl is not None:
        e = tl["engagement"]
        k.update(avg_engagement=round(float(e.mean()), 1), peak=f"{e.max():.0f} at {fmt_t(tl.loc[e.idxmax(), 't'])}",
                 lowest=f"{e.min():.0f} at {fmt_t(tl.loc[e.idxmin(), 't'])}",
                 pct_time_high=round(float((e >= high).mean() * 100), 1), pct_time_low=round(float((e < low).mean() * 100), 1),
                 advisories=len(adv))
    if A is not None:
        mix = A["df"]["label"].value_counts(normalize=True) * 100
        k["audio_mix_pct"] = {c: round(float(mix.get(c, 0)), 1) for c in AUDIO_CLASSES}
        k["participation_rate_pct"] = k["audio_mix_pct"]["Student Discussion"]
    md = [f"# Session summary report", f"Generated {k['generated']} · Session length {k['duration']}", "", "## Key indicators"]
    for key, val in k.items():
        if key not in ("generated", "duration") and not isinstance(val, dict):
            md.append(f"- {key.replace('_', ' ').capitalize()}: {val}")
    if "audio_mix_pct" in k:
        md += ["", "## Audio interaction mix"] + [f"- {c}: {v}%" for c, v in k["audio_mix_pct"].items()]
    if att is not None:
        md += ["", "## Attendance register", "| Student ID | First seen | Last seen | Duration | Presence % | Status |", "|---|---|---|---|---|---|"]
        md += [f"| {r['Student ID']} | {r['First Seen']} | {r['Last Seen']} | {r['Total Duration']} | {r['Presence %']} | {r['Status']} |"
               for _, r in att.iterrows()]
    md += ["", "## Teaching advisories"] + ([f"- [{fmt_t(a['issued'])}] **{a['title']}**: {a['message']}" for a in adv] or ["- None issued."])
    md += ["", "---", "Privacy note: processed fully offline; no frames or audio retained; metrics describe the class collectively."]
    return "\n".join(md), k


with tab5:
    st.subheader("Session analytics and educator report")
    if tl is None and att is None:
        st.info("Run an analysis or load the demo to see the report.")
    else:
        report_md, kpis = build_report()
        c = st.columns(4)
        c[0].metric("Session length", kpis["duration"])
        c[1].metric("Attendance", f"{kpis['attendance_rate_pct']:.0f}%" if "attendance_rate_pct" in kpis else "n/a")
        c[2].metric("Participation", f"{kpis['participation_rate_pct']:.0f}%" if "participation_rate_pct" in kpis else "n/a",
                    help="Share of audio windows classified as student discussion.")
        c[3].metric("Avg engagement", f"{kpis['avg_engagement']:.0f}" if "avg_engagement" in kpis else "n/a")
        if tl is not None:
            c = st.columns(3)
            c[0].metric("Time at high engagement", f"{kpis['pct_time_high']:.0f}%")
            c[1].metric("Time at low engagement", f"{kpis['pct_time_low']:.0f}%")
            c[2].metric("Advisories issued", kpis["advisories"])
            fig, ax = plt.subplots(figsize=(10, 2.8))
            ax.fill_between(tl["t"] / 60, tl["engagement"], color=TEAL, alpha=0.25)
            ax.plot(tl["t"] / 60, tl["engagement"], color=INK, lw=1.8)
            ax.axhline(low, color=CORAL, ls="--", lw=1)
            ax.set_ylim(0, 100)
            ax.set_xlabel("Minutes")
            ax.set_ylabel("Engagement")
            fig.tight_layout()
            st.pyplot(fig)
            plt.close(fig)
        if adv:
            st.markdown("**Advisories this session**")
            st.dataframe(pd.DataFrame([{"Issued": fmt_t(a["issued"]), "Type": a["title"], "Advice": a["message"]} for a in adv]), hide_index=True)
        d1, d2 = st.columns(2)
        d1.download_button("Download report (Markdown)", report_md.encode(), "session_report.md", "text/markdown")
        d2.download_button("Download KPIs (JSON)", json.dumps(kpis, indent=2).encode(), "session_kpis.json", "application/json")
        with st.expander("Preview report"):
            st.markdown(report_md)
