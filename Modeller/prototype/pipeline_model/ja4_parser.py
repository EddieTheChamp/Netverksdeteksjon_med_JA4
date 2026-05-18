"""Parse JA4 fingerprint strings into flat feature dictionaries.

The Random Forest model cannot work with raw strings like
"t13d1516h2_8daaf6152771_02713d6af862" directly, so this file breaks
those strings apart into individual numeric and categorical features.

JA4 STRING FORMAT:
  "<prefix10chars>_<ciphers_hash>_<extensions_hash>"

  The 10-char prefix encodes:
    [0]     protocol        t=TLS, q=QUIC, d=DTLS …
    [1:3]   TLS version     13=TLS1.3, 12=TLS1.2 …
    [3]     SNI indicator   d=domain, i=IP address
    [4:6]   num_ciphers     zero-padded count of cipher suites offered
    [6:8]   num_extensions  zero-padded count of extensions offered
    [8:10]  ALPN            first two chars of the ALPN protocol (h2, h1, 00 …)

  If the long "ja4_string" variant is available (comma-separated raw values
  instead of hashes), individual ciphers / extensions become binary features.
"""

from __future__ import annotations
from typing import Any

from .models import DatabaseRecord, Observation

# Sentinel value used when a feature field is missing / unparseable.
MISSING = "__missing__"

# Features the RF model treats as categories (label-encoded, not numeric).
CATEGORICAL_FEATURES = [
    "protocol", "tls_version", "sni_indicator", "alpn",
    "ja4ts",
]
# Features the RF model treats as plain numbers.
NUMERIC_FEATURES = ["num_ciphers", "num_extensions", "has_ja4s", "has_ja4ts"]

# Combined ordered list used to build the pandas DataFrame sent to the model.
FEATURE_COLUMNS = CATEGORICAL_FEATURES + NUMERIC_FEATURES


# ─────────────────────────────────────────────────────────────────────────────
# JA4 PARSER
# ─────────────────────────────────────────────────────────────────────────────

def parse_ja4(ja4: str | None, ja4_string: str | None = None) -> dict[str, Any]:
    """Parse one JA4 hash (or long string) into a flat feature dictionary.

    Uses ja4_string (the raw comma-separated form) when available because it
    lets us create one binary feature per cipher / extension ("cipher_1301": 1).
    Falls back to the short hash and extracts only what the prefix encodes.
    """
    # Provide safe defaults for all features so callers always get a full dict.
    defaults: dict[str, Any] = {
        "protocol": MISSING, "tls_version": MISSING,
        "sni_indicator": MISSING, "num_ciphers": 0,
        "num_extensions": 0, "alpn": MISSING,
    }
    # Prefer the long string; fall back to the short hash.
    text = (ja4_string or ja4 or "").strip().lower()
    if not text or "_" not in text:
        return defaults   # nothing to parse → all defaults

    parts  = text.split("_")
    prefix = parts[0].ljust(10, "0")   # pad to 10 chars so slicing is safe

    features = {
        "protocol":       prefix[0]        or MISSING,
        "tls_version":    prefix[1:3]      or MISSING,
        "sni_indicator":  prefix[3]        or MISSING,
        # Convert the zero-padded digit pairs to actual integers.
        "num_ciphers":    int(prefix[4:6])  if prefix[4:6].isdigit()  else 0,
        "num_extensions": int(prefix[6:8])  if prefix[6:8].isdigit()  else 0,
        "alpn":           prefix[8:10]      or MISSING,
    }

    # If we have the long comma-separated form, also create one binary feature
    # per cipher and extension (e.g. "cipher_1301": 1, "ext_0017": 1).
    # This gives the RF richer signal than just the counts.
    if ja4_string and len(parts) >= 3:
        for c in parts[1].split(","):
            if c:
                features[f"cipher_{c}"] = 1
        for e in parts[2].split(","):
            if e:
                features[f"ext_{e}"] = 1
        if len(parts) >= 4:   # signature algorithms (optional 4th field)
            for s in parts[3].split(","):
                if s:
                    features[f"sig_{s}"] = 1

    return features


def parse_ja4s(ja4s_string: str | None) -> dict[str, Any]:
    """Parse the server-side JA4S long string into binary cipher/extension features.

    JA4S captures what the server advertised back — its chosen cipher,
    extensions, etc.  Used as additional features when available.
    """
    features: dict[str, Any] = {}
    if not ja4s_string:
        return features
    text  = ja4s_string.strip().lower()
    parts = text.split("_")
    if len(parts) >= 2:
        for c in parts[1].split(","):
            if c:
                features[f"ja4s_cipher_{c}"] = 1
    if len(parts) >= 3:
        for e in parts[2].split(","):
            if e:
                features[f"ja4s_ext_{e}"] = 1
    return features


# ─────────────────────────────────────────────────────────────────────────────
# OBSERVATION → FEATURE DICT  (used at inference time)
# ─────────────────────────────────────────────────────────────────────────────

def observation_to_features(obs: Observation) -> dict[str, Any]:
    """Convert an Observation into the full flat feature dict consumed by the RF.

    Merges JA4 features, optional JA4S features, and binary presence flags
    for each fingerprint type (has_ja4s, has_ja4t, has_ja4ts).
    """
    features = parse_ja4(obs.ja4, obs.ja4_string)

    # Merge server-side features if the JA4S long string is available.
    if obs.ja4s_string:
        features.update(parse_ja4s(obs.ja4s_string))

    features.update({
        "ja4ts": (obs.ja4ts or "").strip().lower() or MISSING,
        # Binary flags: did this connection include a JA4S fingerprint at all?
        "has_ja4s":  1 if obs.ja4s  else 0,
        "has_ja4ts": 1 if obs.ja4ts else 0,
    })
    return features


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE RECORD → TRAINING SAMPLE  (used when training the RF)
# ─────────────────────────────────────────────────────────────────────────────

def record_to_training_sample(record: DatabaseRecord) -> dict[str, Any] | None:
    """Convert one DatabaseRecord into an RF training sample dict.

    Returns None for unlabeled records (no application name) so callers can
    simply filter with: [s for s in map(record_to_training_sample, records) if s]

    The returned dict has all the parsed features PLUS:
      label    — the target application name
      category — the target category
      weight   — the record's count (seen N times → N times as influential)
    """
    if not record.application:
        return None  # skip — no ground-truth label to train on
    obs = Observation(
        observation_id="train",
        ja4=record.ja4, ja4s=record.ja4s,
        ja4_string=record.ja4_string, ja4s_string=record.ja4s_string,
        ja4ts=record.ja4ts,
        source="training",
    )
    sample = observation_to_features(obs)
    sample["label"]    = record.application
    sample["category"] = record.category
    sample["weight"]   = max(record.count, 1)   # at least weight 1
    return sample


def records_to_training_samples(records: list[DatabaseRecord]) -> list[dict[str, Any]]:
    """Batch-convert a list of records into RF training samples (skips unlabeled)."""
    return [s for r in records if (s := record_to_training_sample(r)) is not None]
