"""Shared deterministic life/liberty urgency signals used by scoring and review."""

import re
from typing import Any


_PATTERNS = [
    re.compile(r"\bhabeas\s+corpus\b", re.I),
    re.compile(r"\b(?:anticipatory|regular|interim|default|statutory)?\s*bail(?:\s+(?:application|petition|plea))?\b", re.I),
    re.compile(r"\bunder[- ]?trial\s+(?:detention|custody|incarceration|prisoner)\b", re.I),
    re.compile(r"\b(?:illegal|unlawful)\s+detention\b", re.I),
    re.compile(r"\b(?:preventive|preventative)\s+detention\b", re.I),
    re.compile(r"\b(?:in|under)\s+(?:police|judicial)\s+custody\b", re.I),
    re.compile(r"\bremand(?:ed)?\b", re.I),
    re.compile(r"\bcustodial\s+(?:violence|torture|death|assault)\b", re.I),
    re.compile(r"\b(?:police|custody|custodial)\s+torture\b", re.I),
    re.compile(r"\b(?:immediate|imminent|physical)\s+threat\s+to\s+(?:life|safety|person)\b", re.I),
    re.compile(r"\bthreat\s+to\s+life\b|\blife\s+in\s+danger\b|\bimminent\s+danger\b", re.I),
    re.compile(r"\b(?:execution|death\s+penalty|capital\s+punishment)\b", re.I),
    re.compile(r"\bjuvenile\s+(?:detention|custody|confinement)\b", re.I),
]
_NEGATION = re.compile(r"\b(?:no|not|without|never|neither)\b[^.!?;:]{0,48}$", re.I)


def find_urgency_evidence(text: Any) -> list[dict[str, Any]]:
    """Return non-negated matched spans with character offsets for auditability."""
    source = str(text or "")
    matches = []
    for pattern in _PATTERNS:
        for match in pattern.finditer(source):
            prefix = source[max(0, match.start() - 60):match.start()]
            if _NEGATION.search(prefix):
                continue
            matches.append({
                "text": match.group(0),
                "start": match.start(),
                "end": match.end(),
            })
    matches.sort(key=lambda item: (item["start"], item["end"]))
    deduplicated = []
    for item in matches:
        if deduplicated and item["start"] < deduplicated[-1]["end"]:
            continue
        deduplicated.append(item)
    return deduplicated

