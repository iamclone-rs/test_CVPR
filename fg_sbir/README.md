# FG-ZS-SBIR

This package implements the paper's fine-grained zero-shot SBIR architecture with:

- one shared CLIP image encoder
- one common visual prompt
- LayerNorm-only tuning on the image encoder
- hard triplet loss
- CLIP text-encoder classification loss
- patch-shuffling loss

`f-divergence` is intentionally omitted.

Expected dataset layout:

```text
data_dir/
  photo/
    category_a/
    category_b/
  sketch/
    category_a/
    category_b/
```

The dataset parser matches sketches to photos by filename stem. It first tries exact stem matches, then common
Sketchy-style variants such as `photo_id-1.png`, `photo_id-2.png`, etc.

Run:

```bash
python -m fg_sbir.train --data_dir /kaggle/input/datasets/b20dccn616nguynhutun/sketchy-fg
```
