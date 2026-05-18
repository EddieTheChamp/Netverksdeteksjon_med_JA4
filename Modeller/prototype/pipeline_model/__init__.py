# ─────────────────────────────────────────────────────────────────────────────
# pipeline_model/__init__.py
#
# Makes this folder a Python package so you can import from it like:
#   from pipeline_model.pipeline import classify
#   from pipeline_model import database_lookup
#
# FILES IN THIS FOLDER:
#   models.py          — shared dataclasses (input/output types)
#   database_lookup.py — reads & caches the local fingerprint database
#   ja4_parser.py      — breaks JA4 strings into ML features
#   pipeline.py        — the full 3-stage classifier + decision engine
# ─────────────────────────────────────────────────────────────────────────────
