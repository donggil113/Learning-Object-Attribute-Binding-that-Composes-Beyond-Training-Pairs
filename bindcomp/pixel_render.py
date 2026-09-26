"""Minimal RGB renderer for the scene-graph world (numpy + Pillow).

Every vocabulary item has a distinct pixel realization; nothing is dropped or
substituted:
  shape    cube (front/top/side faces), sphere, cylinder (body + caps),
           cone (triangle + base ellipse)
  color    fixed RGB palette, modulated by shading
  material rubber = matte diffuse shading; metal = high-contrast shading, a
           dark reflection band and a white specular highlight
  slot     three horizontal columns (left/center/right); position jitter never
           leaves the column, so ``left_of`` in the scene graph is exactly the
           x-order of the objects in the image.
Nuisance per render seed: position and size jitter, background gray level, and
pixel noise. Base and edited scenes use the same function.

``decode`` is an audit-only, hand-written pixel decoder (no learning, no
metadata input). It recovers (slot, shape, color, material) from an image and
is used to show the renderer preserves the scene semantics. It must never be
used as a model input.
"""

from __future__ import annotations

import colorsys
import random

import numpy as np
from PIL import Image, ImageDraw

from . import vocab

SIZE = 224
SLOT_X = (37, 112, 187)
SLOT_EDGES = (0, 75, 150, SIZE)
CENTER_Y = 124
PALETTE = {
    "red": (205, 40, 40),
    "blue": (45, 75, 215),
    "green": (40, 165, 65),
    "yellow": (230, 205, 40),
    "purple": (145, 60, 185),
}
SPECULAR_AT = {"cube": (-0.35, -0.1), "sphere": (-0.35, -0.35), "cylinder": (-0.35, -0.1), "cone": (-0.15, 0.25)}
MAX_JITTER = 6
RADIUS_RANGE = (21, 25)


def _face_masks(shape, cx, cy, r):
    """Boolean masks (full canvas) of each face of one object, with a shade multiplier."""
    faces = []

    def poly(points, mult):
        im = Image.new("L", (SIZE, SIZE), 0)
        ImageDraw.Draw(im).polygon([(float(x), float(y)) for x, y in points], fill=255)
        faces.append((np.asarray(im) > 0, mult))

    def ellipse(box, mult):
        im = Image.new("L", (SIZE, SIZE), 0)
        ImageDraw.Draw(im).ellipse([float(v) for v in box], fill=255)
        faces.append((np.asarray(im) > 0, mult))

    if shape == "cube":
        d = 0.38 * r
        poly([(cx - r, cy - r + d), (cx + r - d, cy - r + d), (cx + r - d, cy + r), (cx - r, cy + r)], 1.0)
        poly([(cx - r, cy - r + d), (cx - r + d, cy - r), (cx + r, cy - r), (cx + r - d, cy - r + d)], 1.25)
        poly([(cx + r - d, cy - r + d), (cx + r, cy - r), (cx + r, cy + r - d), (cx + r - d, cy + r)], 0.7)
    elif shape == "sphere":
        ellipse([cx - r, cy - r, cx + r, cy + r], 1.0)
    elif shape == "cylinder":
        w, e = 0.8 * r, 0.3 * r
        poly([(cx - w, cy - r + e), (cx + w, cy - r + e), (cx + w, cy + r - e), (cx - w, cy + r - e)], 1.0)
        ellipse([cx - w, cy + r - 2 * e, cx + w, cy + r], 1.0)
        ellipse([cx - w, cy - r, cx + w, cy - r + 2 * e], 1.25)
    elif shape == "cone":
        w, e = 0.85 * r, 0.3 * r
        ellipse([cx - w, cy + r - 2 * e, cx + w, cy + r], 1.0)
        poly([(cx, cy - r), (cx + w, cy + r - e), (cx - w, cy + r - e)], 1.0)
    else:  # unsupported shapes must fail loudly, never be substituted
        raise ValueError(f"renderer does not support shape {shape!r}")
    return faces


def _shade_field(shape, material, cx, cy, r):
    ys, xs = np.mgrid[0:SIZE, 0:SIZE].astype(np.float64)
    if shape == "sphere":
        dist = np.hypot(xs - (cx - 0.35 * r), ys - (cy - 0.35 * r)) / (1.6 * r)
        base = 1.1 - 0.55 * np.clip(dist, 0, 1)
    else:
        t = np.clip((xs - (cx - r)) / (2 * r), 0, 1)
        base = 1.05 - 0.35 * t
    if material == "metal":
        base = 0.35 + 1.1 * (base - 0.55)  # higher contrast
        band = (ys > cy + 0.05 * r) & (ys < cy + 0.3 * r)
        base = np.where(band, base * 0.55, base)
        # Highlight centre chosen per shape so it always lies inside the silhouette
        # (a fixed upper-left spot fell outside the narrow top of the cone).
        hx, hy = SPECULAR_AT[shape]
        spec = np.exp(-((xs - (cx + hx * r)) ** 2 + (ys - (cy + hy * r)) ** 2) / (2 * (0.2 * r) ** 2))
    elif material == "rubber":
        spec = np.zeros_like(base)
    else:
        raise ValueError(f"renderer does not support material {material!r}")
    return base, spec


def render(scene, render_seed, return_masks=False):
    """Render ``scene`` to a 224x224 RGB PIL image; optionally also per-object masks (audit only)."""
    rng = random.Random(f"pixel-render:{render_seed}")
    g0 = 165 + rng.uniform(-12, 12)
    ys = np.arange(SIZE, dtype=np.float64)[:, None]
    bg = g0 + 18 * (0.5 - ys / SIZE)
    img = np.repeat(np.repeat(bg, SIZE, axis=1)[:, :, None], 3, axis=2)
    masks = {}
    for o in scene.objects:
        if o.color not in PALETTE:
            raise ValueError(f"renderer does not support color {o.color!r}")
        cx = SLOT_X[o.slot] + rng.uniform(-MAX_JITTER, MAX_JITTER)
        cy = CENTER_Y + rng.uniform(-MAX_JITTER, MAX_JITTER)
        r = rng.uniform(*RADIUS_RANGE)
        base, spec = _shade_field(o.shape, o.material, cx, cy, r)
        rgb = np.array(PALETTE[o.color], dtype=np.float64)
        full = np.zeros((SIZE, SIZE), dtype=bool)
        for m, mult in _face_masks(o.shape, cx, cy, r):
            col = np.clip(rgb[None, None, :] * (base * mult)[:, :, None], 0, 255)
            col = col * (1 - spec[:, :, None]) + 255.0 * spec[:, :, None]
            img[m] = col[m]
            full |= m
        masks[o.oid] = full
    nrng = np.random.default_rng(rng.getrandbits(63))
    img = np.clip(img + nrng.normal(0, 2.0, img.shape), 0, 255).astype(np.uint8)
    pil = Image.fromarray(img, "RGB")
    return (pil, masks) if return_masks else pil


# ---------------------------------------------------------------------------
# Audit-only decoder
# ---------------------------------------------------------------------------

_TEMPLATES = None


def _normalized_silhouette(mask):
    ys, xs = np.nonzero(mask)
    crop = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    im = Image.fromarray((crop * 255).astype(np.uint8)).resize((24, 24), Image.BILINEAR)
    return np.asarray(im) > 127


def _templates():
    global _TEMPLATES
    if _TEMPLATES is None:
        _TEMPLATES = {}
        for s in vocab.SHAPES:
            m = np.zeros((SIZE, SIZE), dtype=bool)
            for f, _ in _face_masks(s, 112, 112, 23):
                m |= f
            _TEMPLATES[s] = _normalized_silhouette(m)
    return _TEMPLATES


def _hue_name(rgb):
    h, s, v = colorsys.rgb_to_hsv(*(np.asarray(rgb) / 255.0))
    best, dist = None, 9.0
    for name, ref in PALETTE.items():
        hr = colorsys.rgb_to_hsv(*(np.asarray(ref) / 255.0))[0]
        d = min(abs(h - hr), 1 - abs(h - hr))
        if d < dist:
            best, dist = name, d
    return best


def decode(pil_image):
    """Recover [(slot, shape, color, material)] from pixels only (audit tool)."""
    a = np.asarray(pil_image).astype(np.float64)
    bg = np.median(np.concatenate([a[:, :3], a[:, -3:]], axis=1), axis=1)  # per-row background
    fg = np.abs(a - bg[:, None, :]).max(axis=2) > 45
    out = []
    for slot in range(vocab.N_SLOTS):
        m = np.zeros_like(fg)
        m[:, SLOT_EDGES[slot]:SLOT_EDGES[slot + 1]] = fg[:, SLOT_EDGES[slot]:SLOT_EDGES[slot + 1]]
        if m.sum() < 200:
            continue
        px = a[m]
        white = (px.min(axis=1) > 225).mean()
        sat = px.max(axis=1) - px.min(axis=1)
        colored = px[(sat > 60) & (px.min(axis=1) <= 225)]
        color = _hue_name(np.median(colored, axis=0)) if len(colored) else None
        sil = _normalized_silhouette(m)
        ious = {s: (sil & t).sum() / max(1, (sil | t).sum()) for s, t in _templates().items()}
        shape = max(ious, key=ious.get)
        out.append((slot, shape, color, "metal" if white > 0.005 else "rubber"))
    return out


def scene_signature(scene):
    return sorted((o.slot, o.shape, o.color, o.material) for o in scene.objects)
