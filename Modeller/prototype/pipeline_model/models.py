"""Shared dataclasses for the JA4+ pipeline.

This file defines the data types that flow between all pipeline stages.
Think of these as the "shapes" of data — they describe what fields exist
and what types they hold, but contain no classification logic themselves.

HOW DATA FLOWS:
  1. You hand in fingerprint strings → packed into an Observation
  2. The pipeline stages produce raw dicts → combined into a FinalDecision
  3. Everything is wrapped up in a ClassificationResult for the caller
"""

from __future__ import annotations
from dataclasses import asdict, dataclass, field
from typing import Any


# ─────────────────────────────────────────────────────────────────────────────
# INPUT TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class Observation:
    """One fingerprint observation to classify.

    This is the input container. Wrap one network connection's JA4 fingerprints
    in this before passing it to the pipeline stages.

    Fields:
      observation_id — any string so you can trace results back to a row/packet
      ja4            — the short JA4 hash (e.g. "t13d1516h2_8daaf6152771_02713d6af862")
      ja4s           — server-side JA4 hash
      ja4_string     — the long human-readable JA4 string (more features for RF)
      ja4s_string    — the long human-readable JA4S string
      ja4t           — JA4 TCP timing fingerprint
      ja4ts          — JA4 TCP server timing fingerprint
      true_application / true_category — ground-truth label (only known during eval)
    """
    observation_id: str
    ja4:              str | None = None
    ja4s:             str | None = None
    ja4_string:       str | None = None
    ja4s_string:      str | None = None
    ja4t:             str | None = None
    ja4ts:            str | None = None
    true_application: str | None = None
    true_category:    str | None = None
    source:           str | None = None
    raw_record: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


# ─────────────────────────────────────────────────────────────────────────────
# DATABASE ROW TYPE
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class DatabaseRecord:
    """One normalized row from the local fingerprint database.

    When database_lookup.py reads the JSON file it converts each raw dict
    into this dataclass so the rest of the code always works with typed fields
    instead of raw dicts with inconsistent key names.

    Fields:
      ja4 / ja4s / ja4t / ja4ts   — the fingerprint hashes
      ja4_string / ja4s_string     — long readable form
      application                  — the app that produced this fingerprint
                                     (e.g. "chrome.exe", "teams.exe")
      category                     — broad group (browser, messaging, system…)
      count                        — how many times this fingerprint was seen
                                     (used to weight ambiguous matches)
      metadata                     — any extra fields from the JSON
    """
    ja4:          str | None = None
    ja4s:         str | None = None
    ja4_string:   str | None = None
    ja4s_string:  str | None = None
    ja4t:         str | None = None
    ja4ts:        str | None = None
    application:  str | None = None
    category:     str | None = None
    count:        int = 1
    metadata: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# INTERMEDIATE TYPE (used inside egenlagd_lookup, not the final answer)
# ─────────────────────────────────────────────────────────────────────────────

@dataclass
class CandidateMatch:
    """One candidate application from an exact-match lookup.

    A single fingerprint may appear in the database for multiple applications.
    Each application that matched gets wrapped in one of these.
    """
    application:             str | None
    category:                str | None
    occurrences_in_database: int
    probability_percent:     float
    metadata: dict[str, Any] = field(default_factory=dict)


# ─────────────────────────────────────────────────────────────────────────────
# CATEGORY INFERENCE
# ─────────────────────────────────────────────────────────────────────────────

# Simple keyword rules used when the database row has no explicit category.
# We check if any keyword appears inside the application name string.
_CATEGORY_RULES: list[tuple[str, tuple[str, ...]]] = [
    ("malware",           ("cobalt", "sliver", "meterpreter", "beacon", "empire", "metasploit")),
    ("security",          ("defender", "antivirus", "msmpeng", "smartscreen", "sentinel",
                           "crowdstrike", "falcon", "carbonblack", "sophos", "symantec",
                           "mcafee", "kaspersky", "eset", "avast", "avg", "bitdefender")),
    ("browser",           ("chrome.exe", "chrome", "msedge.exe", "msedge", "edgewebview",
                           "webview2", "firefox.exe", "firefox", "opera.exe", "opera",
                           "brave.exe", "brave", "microsoftedgeupdate", "code.exe")),
    ("microsoft_office",  ("outlook.exe", "outlook", "winword.exe", "winword",
                           "excel.exe", "excel", "powerpnt.exe", "powerpnt",
                           "onenote.exe", "onenote", "officec2rclient",
                           "filecoauth", "m365copilot")),
    ("messaging",         ("teams.exe", "teams", "ms-teams", "discord.exe", "discord",
                           "slack.exe", "slack", "telegram.exe", "telegram",
                           "skype.exe", "skype", "signal.exe", "signal")),
    ("system",            ("svchost.exe", "svchost", "lsass.exe", "lsass",
                           "services.exe", "services", "wininit.exe", "wininit",
                           "explorer.exe", "explorer", "backgroundtaskhost",
                           "wermgr.exe", "wermgr", "pwsh.exe", "powershell",
                           "tailscaled", "onedrive", "jotta", "unknown process")),
]


def infer_category(application: str | None, explicit_category: str | None = None) -> str:
    """Return a normalized category string for an application name.

    First priority: use the explicit_category if the database row already has one.
    Fallback: scan the application name against _CATEGORY_RULES keyword lists.
    If nothing matches, return "unknown".
    """
    # If the DB row already has a category, clean it up and use that directly.
    if explicit_category:
        return explicit_category.strip().lower().replace("-", "_").replace(" ", "_")
    if not application:
        return "unknown"
    app = application.strip().lower()
    # Walk the rules in order; return the first category whose keywords match.
    for category, patterns in _CATEGORY_RULES:
        if any(p in app for p in patterns):
            return category
    return "unknown"


# ─────────────────────────────────────────────────────────────────────────────
# OUTPUT TYPES
# ─────────────────────────────────────────────────────────────────────────────

@dataclass(slots=True)
class FinalDecision:
    """Final explanation produced by the decision engine (decision.py).

    This is the internal result before wrapping into ClassificationResult.

    Fields:
      application_prediction  — the predicted app name (or None if unsure)
      category_prediction     — broad category (or None)
      application_confidence  — "high" / "medium" / "low" / "none"
      category_confidence     — "high" / "medium" / "low"
      decision_source         — which stage made the final call
                                e.g. "egenlagd_exact", "random_forest",
                                     "ambiguous_resolved_by_rf",
                                     "category_fallback_local", "unknown"
      reasoning               — human-readable explanation of the decision
    """
    application_prediction: str | None
    category_prediction:    str | None
    application_confidence: str   # high | medium | low | none
    category_confidence:    str   # high | medium | low
    decision_source:        str
    reasoning:              str

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


@dataclass(slots=True)
class ClassificationResult:
    """Full per-observation output for reporting / evaluation.

    This is what classify() in pipeline.py returns to the outside world.

    Fields:
      observation_id          — matches the id you passed in
      ja4                     — the fingerprint hash you classified
      true_application        — the real answer (only set during eval, else None)
      true_category           — same
      predicted_application   — our best guess
      predicted_category      — our best category guess
      is_correct              — whether predicted == true (None at query time)
      confidence              — "high" / "medium" / "low" / "none"
      decision_source         — which stage decided (see FinalDecision above)
      reasoning               — plain-English explanation
      model_details           — raw dicts from each stage for debugging
    """
    observation_id:        str
    ja4:                   str | None
    true_application:      str | None
    true_category:         str | None
    predicted_application: str | None
    predicted_category:    str | None
    is_correct:            bool | None
    confidence:            str
    decision_source:       str
    reasoning:             str
    # Raw outputs from each sub-classifier; useful for debugging and eval scripts.
    model_details: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)
