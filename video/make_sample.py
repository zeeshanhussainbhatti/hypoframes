"""Render the sample clip: cover intro + the welcome spread, with animated
characters, subtitles, voice-over and background music.

Usage:  python3 video/make_sample.py
Needs:  ffmpeg, festival (+ festvox-us-slt-hts), numpy, opencv, Pillow.
Assets: video/assets/{cover,page06,page07}.png extracted from the book PDF.
"""
import math
import os
import subprocess
import wave

import cv2
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = os.path.dirname(os.path.abspath(__file__))
ASSETS = os.path.join(HERE, "assets")
OUT = os.path.join(HERE, "out")
os.makedirs(OUT, exist_ok=True)

W, H, FPS = 1920, 1080, 30
SR = 44100
INTRO = 3.4          # seconds of cover intro before the voice starts
GAP = 0.35           # pause between captions
TAIL = 1.3           # hold after the last line

CAPTIONS = [
    ("Welcome to these pages,", "To hear the story told"),
    ("Of how America came to be—", "A land both brave and bold."),
    ("Many voices guide you", "In every page you turn;"),
    ("Each poem holds a message", "To help you read and learn."),
]


# ----------------------------------------------------------------- voice
def synth_voice():
    """One wav per caption (female HTS voice), returns [(start, end)], track."""
    clips = []
    for i, lines in enumerate(CAPTIONS):
        text = " ".join(lines).replace("—", ",").replace(";", ",")
        txt = os.path.join(OUT, f"line{i}.txt")
        wav = os.path.join(OUT, f"line{i}.wav")
        with open(txt, "w") as f:
            f.write(text + "\n")
        subprocess.run(
            ["text2wave", "-eval",
             "(begin (voice_cmu_us_slt_arctic_hts) (Parameter.set 'Duration_Stretch 1.12))",
             txt, "-o", wav], check=True)
        # resample + gentle warmth so it sits softly over the music
        soft = os.path.join(OUT, f"line{i}_soft.wav")
        subprocess.run(
            ["ffmpeg", "-y", "-v", "error", "-i", wav, "-af",
             "highpass=f=90,equalizer=f=3200:t=q:w=1:g=-3,lowpass=f=7500,"
             "aecho=0.8:0.5:40:0.12,loudnorm=I=-18:TP=-2",
             "-ar", str(SR), "-ac", "1", soft], check=True)
        with wave.open(soft) as w:
            data = np.frombuffer(w.readframes(w.getnframes()), dtype=np.int16)
        clips.append(data.astype(np.float32) / 32768.0)

    times, t = [], INTRO
    for c in clips:
        d = len(c) / SR
        times.append((t, t + d))
        t += d + GAP
    total = t - GAP + TAIL
    track = np.zeros(int(total * SR) + SR, dtype=np.float32)
    for c, (s, _) in zip(clips, times):
        a = int(s * SR)
        track[a:a + len(c)] += c
    return times, track[: int(total * SR)], total


# ----------------------------------------------------------------- music
def synth_music(total):
    """Original soft music-box lullaby: arpeggios over a warm pad."""
    n = int(total * SR)
    out = np.zeros(n + SR * 12, dtype=np.float32)
    beat = 60 / 76
    note = lambda semis: 261.63 * 2 ** (semis / 12)
    chords = [(0, 4, 7), (-3, 0, 4), (-7, -3, 0), (-5, -1, 2)]  # C Am F G
    pattern = [0, 1, 2, 1, 0 + 12, 2, 1, 2]

    def bell(freq, dur=2.2, amp=0.2):
        tt = np.arange(int(dur * SR)) / SR
        env = np.exp(-tt * 2.6)
        s = (np.sin(2 * np.pi * freq * tt)
             + 0.35 * np.sin(2 * np.pi * 2 * freq * tt) * np.exp(-tt * 4)
             + 0.12 * np.sin(2 * np.pi * 3.01 * freq * tt) * np.exp(-tt * 7))
        return (amp * env * s).astype(np.float32)

    def pad(freqs, dur):
        tt = np.arange(int(dur * SR)) / SR
        env = np.minimum(1, tt / 0.9) * np.minimum(1, (dur - tt) / 0.9)
        s = sum(np.sin(2 * np.pi * f * tt) + 0.5 * np.sin(2 * np.pi * f * 1.003 * tt)
                for f in freqs)
        return (0.035 * env * s).astype(np.float32)

    bar, t = 0, 0.0
    while t < total:
        ch = chords[bar % 4]
        p = pad([note(s - 12) for s in ch], beat * 4 + 0.8)
        a = int(t * SR)
        out[a:a + len(p)] += p
        for k, idx in enumerate(pattern):
            semis = ch[idx % 12 % 3] + (12 if idx >= 12 else 0) + 12
            b = bell(note(semis), amp=0.17 if k % 2 else 0.22)
            a = int((t + k * beat / 2) * SR)
            out[a:a + len(b)] += b
        t += beat * 4
        bar += 1
    out = out[:n]
    fade = int(1.5 * SR)
    out[:fade] *= np.linspace(0, 1, fade)
    out[-fade:] *= np.linspace(1, 0, fade)
    return out


def write_wav(path, data):
    data = np.clip(data, -1, 1)
    with wave.open(path, "w") as w:
        w.setnchannels(1)
        w.setsampwidth(2)
        w.setframerate(SR)
        w.writeframes((data * 32767).astype(np.int16).tobytes())


# ----------------------------------------------------------------- scene
def ease(x):
    x = min(1.0, max(0.0, x))
    return x * x * (3 - 2 * x)


class Spread:
    """Pages 6+7 stitched; characters are moved with a smooth warp field."""

    # head centre x, y, radius (spread pixels)
    HEADS = [
        (680, 330, 95), (575, 570, 95), (780, 540, 90),
        (913, 270, 90), (1093, 320, 90), (1273, 370, 90), (1433, 370, 85),
        (1623, 420, 85), (953, 570, 90), (1118, 600, 100), (1353, 610, 100),
        (1603, 620, 90),
    ]
    FLAG = (1418, 55, 1545, 152)      # x0, y0, x1, y1
    DOVE = (1183, 65, 62)
    BOOK = (1010, 800)

    def __init__(self):
        left = cv2.imread(os.path.join(ASSETS, "page06.png"))
        right = cv2.imread(os.path.join(ASSETS, "page07.png"))
        self.img = np.concatenate([left, right], axis=1)
        self.h, self.w = self.img.shape[:2]
        ys, xs = np.mgrid[0:self.h, 0:self.w].astype(np.float32)
        self.xs, self.ys = xs, ys
        rng = np.random.default_rng(7)
        self.heads = []
        for (cx, cy, r) in self.HEADS:
            wgt = np.exp(-((xs - cx) ** 2 + (ys - cy) ** 2) / (2 * (r * 0.95) ** 2))
            self.heads.append(dict(
                cx=cx, cy=cy, r=r, w=wgt,
                f=rng.uniform(0.22, 0.38), ph=rng.uniform(0, 6.28),
                f2=rng.uniform(0.5, 0.8), ph2=rng.uniform(0, 6.28),
                amp=rng.uniform(1.8, 2.8)))
        x0, y0, x1, y1 = self.FLAG
        mx = np.clip((xs - x0) / 12, 0, 1) * np.clip((x1 + 14 - xs) / 14, 0, 1)
        my = np.clip((ys - y0 + 10) / 10, 0, 1) * np.clip((y1 + 12 - ys) / 12, 0, 1)
        self.flag_mask = mx * my * np.clip((xs - x0) / (x1 - x0), 0, 1)
        dx, dy, dr = self.DOVE
        self.dove_w = np.exp(-((xs - dx) ** 2 + (ys - dy) ** 2) / (2 * (dr * 0.8) ** 2))
        bx, by = self.BOOK
        self.glow = np.exp(-(((xs - bx) / 330) ** 2 + ((ys - by) / 210) ** 2))
        self.sparks = [dict(x=rng.uniform(bx - 200, bx + 230), t0=rng.uniform(-4, 16),
                            life=rng.uniform(3.0, 4.6), sway=rng.uniform(10, 34),
                            ph=rng.uniform(0, 6.28), size=rng.uniform(1.6, 3.2),
                            rise=rng.uniform(260, 420))
                       for _ in range(70)]

    def frame(self, t):
        xs, ys = self.xs, self.ys
        ddx = np.zeros_like(xs)
        ddy = np.zeros_like(ys)
        # heads: tilt around the neck + small bob, each child on its own rhythm
        for hd in self.heads:
            ang = math.radians(hd["amp"]) * math.sin(2 * math.pi * hd["f"] * t + hd["ph"])
            bob = 3.2 * math.sin(2 * math.pi * hd["f2"] * t + hd["ph2"])
            px, py = hd["cx"], hd["cy"] + hd["r"] * 1.1
            rx, ry = xs - px, ys - py
            ca, sa = math.cos(ang), math.sin(ang)
            ddx += hd["w"] * ((ca * rx - sa * ry) - rx)
            ddy += hd["w"] * ((sa * rx + ca * ry) - ry + bob)
        # flag waving from the pole outwards
        x0 = self.FLAG[0]
        wave_y = 6.5 * np.sin((xs - x0) * 0.085 - t * 5.2)
        ddy += self.flag_mask * wave_y
        ddx += self.flag_mask * 2.0 * np.sin((xs - x0) * 0.06 - t * 4.1)
        # dove: wing flap (vertical squash) + hover
        cx, cy, _ = self.DOVE
        flap = 0.22 * math.sin(2 * math.pi * 2.3 * t)
        ddy += self.dove_w * ((ys - cy) * flap + 7 * math.sin(2 * math.pi * 0.45 * t))
        ddx += self.dove_w * 9 * math.sin(2 * math.pi * 0.2 * t)

        out = cv2.remap(self.img, xs - ddx, ys - ddy, cv2.INTER_LINEAR,
                        borderMode=cv2.BORDER_REFLECT).astype(np.float32)
        # glowing book + rising sparkles
        pulse = 0.10 + 0.07 * math.sin(2 * math.pi * 0.45 * t)
        light = self.glow[..., None] * pulse * np.array([150, 225, 255], np.float32)
        layer = np.zeros((self.h, self.w), np.float32)
        for s in self.sparks:
            age = (t - s["t0"]) % (s["life"] + 1.2)
            if age > s["life"]:
                continue
            k = age / s["life"]
            x = s["x"] + s["sway"] * math.sin(2 * math.pi * 0.35 * age + s["ph"])
            y = self.BOOK[1] - 30 - s["rise"] * k
            a = math.sin(math.pi * k) * (0.75 + 0.25 * math.sin(11 * age + s["ph"]))
            cv2.circle(layer, (int(x), int(y)), int(round(s["size"])), float(a), -1, cv2.LINE_AA)
        layer = cv2.GaussianBlur(layer, (0, 0), 2.2) * 1.9 + cv2.GaussianBlur(layer, (0, 0), 7) * 1.2
        out += light + layer[..., None] * np.array([170, 235, 255], np.float32)
        return np.clip(out, 0, 255).astype(np.uint8)

    def camera(self, img, k):
        """Slow crane move: landmarks first, then down to the children and book."""
        e = ease(k)
        vw = 1240 + (1090 - 1240) * e
        vh = vw * H / W
        cx = 1104 + (1010 - 1104) * e
        cy = 372 + (648 - 372) * e
        m = np.float32([[W / vw, 0, -(cx - vw / 2) * W / vw],
                        [0, H / vh, -(cy - vh / 2) * H / vh]])
        f = cv2.warpAffine(img, m, (W, H), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REFLECT)
        blur = cv2.GaussianBlur(f, (0, 0), 1.6)
        return cv2.addWeighted(f, 1.35, blur, -0.35, 0)


class Cover:
    """Book cover swinging in as a 3D card over a blurred copy of itself."""

    def __init__(self):
        self.img = cv2.imread(os.path.join(ASSETS, "cover.png"))
        bg = cv2.resize(self.img, (W, int(W * self.img.shape[0] / self.img.shape[1])),
                        interpolation=cv2.INTER_CUBIC)
        y = (bg.shape[0] - H) // 3
        bg = cv2.GaussianBlur(bg[y:y + H], (0, 0), 26)
        self.bg = (bg.astype(np.float32) * 0.62).astype(np.uint8)
        ch = 960
        cw = int(ch * self.img.shape[1] / self.img.shape[0])
        self.card = cv2.resize(self.img, (cw, ch), interpolation=cv2.INTER_CUBIC)
        self.cw, self.ch = cw, ch

    def frame(self, t):
        k = ease(t / 1.6)
        ang = math.radians(38 * (1 - k) + 3 * math.sin(t * 0.9))
        scale = 0.86 + 0.10 * ease(t / INTRO) + 0.02 * k
        hw, hh = self.cw / 2 * scale, self.ch / 2 * scale
        d = 1700.0
        pts = []
        for sx, sy in [(-1, -1), (1, -1), (1, 1), (-1, 1)]:
            x, z = sx * hw * math.cos(ang), sx * hw * math.sin(ang)
            p = d / (d + z)
            pts.append([W / 2 + x * p, H / 2 + 6 * math.sin(t * 1.3) + sy * hh * p])
        src = np.float32([[0, 0], [self.cw, 0], [self.cw, self.ch], [0, self.ch]])
        m = cv2.getPerspectiveTransform(src, np.float32(pts))
        card = cv2.warpPerspective(self.card, m, (W, H), flags=cv2.INTER_CUBIC)
        mask = cv2.warpPerspective(np.full((self.ch, self.cw), 255, np.uint8), m, (W, H))
        shadow = cv2.GaussianBlur(np.roll(mask, (22, 14), axis=(0, 1)), (0, 0), 24)
        out = self.bg.astype(np.float32) * (1 - 0.55 * shadow[..., None] / 255.0)
        a = cv2.GaussianBlur(mask, (0, 0), 0.8)[..., None] / 255.0
        out = out * (1 - a) + card.astype(np.float32) * a
        out *= min(1.0, t / 0.5)                       # fade from black
        return np.clip(out, 0, 255).astype(np.uint8)


# -------------------------------------------------------------- subtitles
FONT = ImageFont.truetype(os.path.join(ASSETS, "nunito-latin-800-normal.woff"), 54)


def caption_layer(lines):
    img = Image.new("RGBA", (W, H), (0, 0, 0, 0))
    d = ImageDraw.Draw(img)
    widths = [d.textlength(s, font=FONT) for s in lines]
    lh, pad_x, pad_y = 70, 44, 22
    bw, bh = max(widths) + pad_x * 2, lh * len(lines) + pad_y * 2
    x0, y0 = (W - bw) / 2, H - 70 - bh
    d.rounded_rectangle([x0, y0, x0 + bw, y0 + bh], radius=34, fill=(22, 26, 48, 172))
    for i, (s, wd) in enumerate(zip(lines, widths)):
        d.text(((W - wd) / 2, y0 + pad_y + i * lh - 4), s, font=FONT, fill=(255, 255, 255, 255),
               stroke_width=2, stroke_fill=(22, 26, 48, 255))
    arr = np.asarray(img).astype(np.float32)
    return arr[..., [2, 1, 0]], arr[..., 3:4] / 255.0


def main():
    times, voice, total = synth_voice()
    music = synth_music(total)
    # duck the music a little under the voice
    env = np.abs(voice)
    win = int(0.25 * SR)
    env = np.convolve(env, np.ones(win) / win, mode="same")
    duck = 1 - 0.5 * np.clip(env * 14, 0, 1)
    write_wav(os.path.join(OUT, "audio.wav"), voice * 1.0 + music * duck * 0.7)

    cover, spread = Cover(), Spread()
    caps = [caption_layer(c) for c in CAPTIONS]
    nframes = int(total * FPS)
    xfade = 0.7
    out_path = os.path.join(OUT, "sample.mp4")
    ff = subprocess.Popen(
        ["ffmpeg", "-y", "-v", "error", "-f", "rawvideo", "-pix_fmt", "bgr24",
         "-s", f"{W}x{H}", "-r", str(FPS), "-i", "-",
         "-i", os.path.join(OUT, "audio.wav"),
         "-c:v", "libx264", "-preset", "medium", "-crf", "20", "-pix_fmt", "yuv420p",
         "-c:a", "aac", "-b:a", "192k", "-shortest", "-movflags", "+faststart", out_path],
        stdin=subprocess.PIPE)
    t_scene = INTRO - xfade                      # spread starts here
    for i in range(nframes):
        t = i / FPS
        if t < INTRO:
            f = cover.frame(t)
        if t >= t_scene:
            ts = t - t_scene
            s = spread.camera(spread.frame(ts), ts / (total - t_scene))
            if t < INTRO:
                a = ease((t - t_scene) / xfade)
                f = cv2.addWeighted(f, 1 - a, s, a, 0)
            else:
                f = s
        for (st, en), (rgb, alpha) in zip(times, caps):
            a = min(1, (t - st + 0.15) / 0.25, (en + 0.3 - t) / 0.25)
            if a > 0:
                f = (f * (1 - alpha * a) + rgb * alpha * a).astype(np.uint8)
        if total - t < 0.8:                      # fade out
            f = (f * ((total - t) / 0.8)).astype(np.uint8)
        ff.stdin.write(f.tobytes())
        if i % 90 == 0:
            print(f"frame {i}/{nframes}", flush=True)
    ff.stdin.close()
    ff.wait()
    print("wrote", out_path, f"{total:.1f}s")


if __name__ == "__main__":
    main()
