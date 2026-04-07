"""Vendored MR-MT3 — pretrained MT3 source code + checkpoint.

Origin: https://github.com/kunato/mr-mt3 (commit pinned in vendor.md).
We vendor the model code (``models/``, ``contrib/``) and the pretrained
checkpoint (``pretrained/mt3.pth``, tracked via git-lfs) so the transcribe
service is self-contained — no out-of-tree paths, no per-developer setup.

Internal imports inside ``contrib/`` were rewritten from the upstream
``from contrib import …`` form to package-relative ``from . import …`` so
the modules import cleanly under ``backend.vendor.mr_mt3``.
"""
