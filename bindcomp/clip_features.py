"""Frozen CLIP features computed from rendered pixels and raw caption strings ONLY.

Inputs are type-checked: ``encode_images`` accepts PIL images and
``encode_texts`` accepts strings. No scene graph, attribute assignment, object
crop, parser output, scene ID or edit ID can reach the encoder through this
module.

Tokens (fixed before any run, configs/encoder_pixel_v1.json):
  image: final-layer class token + 49 patch tokens after ln_post, each @ visual.proj
  text : final-layer per-token states after ln_final from SOT to EOT, each @ text_projection
The pooled outputs equal open_clip's own encode_image / encode_text (checked).
"""

from __future__ import annotations

import json
from pathlib import Path

import open_clip
import torch
from PIL import Image

from .manifest import sha256_file

PROVENANCE = "pixels (rendered RGB) + raw caption strings"


class ClipEncoder:
    def __init__(self, cfg, root, verify_sha=True):
        torch.set_num_threads(cfg["torch_threads"])
        path = Path(root) / cfg["local_path"]
        if verify_sha and sha256_file(path) != cfg["sha256"]:
            raise RuntimeError(f"checkpoint sha256 mismatch: {path}")
        self.cfg = cfg
        self.model = open_clip.create_model(cfg["arch"], pretrained=str(path)).eval()
        for p in self.model.parameters():
            p.requires_grad_(False)
        pp = cfg["preprocess"]
        self.preprocess = open_clip.image_transform(pp["image_size"], is_train=False, mean=pp["mean"], std=pp["std"])
        self.tokenizer = open_clip.get_tokenizer(cfg["arch"])
        self.model.visual.output_tokens = True

    @torch.no_grad()
    def encode_images(self, images, batch=16):
        if not all(isinstance(im, Image.Image) for im in images):
            raise TypeError("encode_images accepts PIL images only")
        pooled, tokens = [], []
        v = self.model.visual
        for i in range(0, len(images), batch):
            x = torch.stack([self.preprocess(im) for im in images[i:i + batch]])
            p, t = v(x)  # pooled already projected; patch tokens after ln_post, unprojected
            pooled.append(p)
            tokens.append(torch.cat([p[:, None, :], t @ v.proj], dim=1))
        return torch.cat(pooled), torch.cat(tokens)

    @torch.no_grad()
    def encode_texts(self, texts):
        if not all(isinstance(t, str) for t in texts):
            raise TypeError("encode_texts accepts caption strings only")
        m = self.model
        ids = self.tokenizer(texts)
        x = m.token_embedding(ids) + m.positional_embedding
        x = m.transformer(x, attn_mask=m.attn_mask)
        x = m.ln_final(x) @ m.text_projection
        eot = ids.argmax(dim=-1)
        pooled = x[torch.arange(len(texts)), eot]
        if (eot >= ids.shape[1] - 1).any():
            raise ValueError("caption reached the context length; EOT may be truncated")
        return pooled, [x[i, : int(eot[i]) + 1] for i in range(len(texts))]

    def describe(self):
        return json.loads(json.dumps(self.cfg))
