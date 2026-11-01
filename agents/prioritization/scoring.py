"""
Enhanced, legally grounded priority scoring that reflects real Indian Court Listing Rules:
1. Senior Citizen / Vulnerable litigants (Supreme Court / High Court fast-track listing circulars).
2. Contempt of Court petitions (constitutional urgency under Art. 129 / 215 / Contempt of Courts Act).
3. Public Trusts, Religious Endowments & Educational Institutions (public/community interest).
4. Interlocutory & procedural applications (impleadment, counter affidavits, rejoinders).
5. Interlocutory stays and injunctions.
6. Filing age with smooth continuous aging curves so cases don't cluster at identical values.
"""

import math
import re
from typing import Any

def _finite_number(value: Any, fallback: float) -> float:
    try:
        number = float(value)
        return number if math.isfinite(number) else fallback
    except (TypeError, ValueError):
        return fallback

def check_immediate_threat(text: str) -> int:
    """Return 1 for urgent life/liberty/custody triggers."""
    value = re.sub(r"\s+", " ", (text or "").lower())
    patterns = [
        r"\b(?:anticipatory|regular|interim|default|statutory)?\s*bail\s+(?:application|petition|plea)s?\b",
        r"\bapplication\s+(?:for|seeking)\s+(?:anticipatory\s+|regular\s+|interim\s+)?bail\b",
        r"\b(?:seeks?|prays?\s+for|applies?\s+for)\s+(?:anticipatory\s+|regular\s+|interim\s+|default\s+|statutory\s+)?bail\b",
        r"\bunder[- ]?trial\s+(?:detention|custody|incarceration|prisoner)\b",
        r"\bhabeas\s+corpus\b",
        r"\bcustodial\s+(?:violence|torture|death|assault)\b",
        r"\b(?:police|custody|custodial)\s+torture\b",
        r"\b(?:immediate|imminent|physical)\s+threat\s+to\s+(?:life|safety|person)\b",
        r"\bthreat\s+to\s+life\b|\blife\s+in\s+danger\b|\bimminent\s+danger\b",
        r"\b(?:stay|suspend(?:ing)?|postpon(?:e|ement)|defer(?:ring)?)\b.{0,100}\b(?:execution|death\s+penalty|capital\s+punishment)\b",
        r"\b(?:execution|death\s+penalty|capital\s+punishment)\b.{0,100}\b(?:stay|suspend(?:ed|ing)?|postpon(?:ed|ement)|defer(?:red|ring)?)\b",
        r"\b(?:stay|suspend(?:ing)?)\b.{0,100}\bdemolition\b|\bdemolition\b.{0,100}\b(?:stay|suspend(?:ed|ing)?)\b",
    ]
    return int(any(re.search(pattern, value) for pattern in patterns))

def calculate_societal_impact(text: str) -> int:
    """Comprehensive societal and public impact score (1 to 5 scale)."""
    value = (text or "").lower()
    score = 1
    
    # High public interest
    if re.search(r"\bpublic interest litigation\b|\bpil\b", value):
        score += 3
    if re.search(r"\bconstitutional validity\b|\bconstitutionality\b|\bconstitutional bench\b|\bviolation of fundamental rights?\b", value):
        score += 2
    if re.search(r"\benvironmental impact\b|\benvironmental protection\b|\bpollution control\b|\becological damage\b", value):
        score += 2
        
    # Religious Trusts / Public Endowments
    if re.search(r"\b(?:temple|endowment|deity|wakf|trust property|public trust|charitable)\b", value):
        score += 1
        
    # Educational institutions & admission
    if re.search(r"\b(?:educational institution|admission|students?|pre-university|university|examination)\b", value):
        score += 1

    # Financial / Corruption
    if re.search(r"\bcorruption prevention\b|\bprevention of corruption\b|\bcbi investigation\b|\bfinancial scam\b", value):
        score += 2
        
    return min(score, 5)

def check_special_urgency(text: str, filename: str = "") -> dict[str, float]:
    """
    Detects statutory expedited listing categories recognized under 
    Supreme Court Practice Rules and High Court Listing circulars:
    - Senior citizens (age >= 65)
    - Contempt of Court (breach of judicial majesty)
    - Interim stay / injunction application
    - Interlocutory / procedural applications
    """
    t = (text or "").lower()
    fn = (filename or "").lower()
    
    adjustments = {
        "senior_citizen": 0.0,
        "contempt": 0.0,
        "interim_stay": 0.0,
        "procedural_discount": 0.0
    }
    
    # 1. Senior Citizen fast-track (SC Guidelines grant priority to litigants over 65)
    if re.search(r"\bsenior citizen\b|\baged\s+about\s+(?:6[5-9]|[7-9]\d)\b|\baged\s+(?:6[5-9]|[7-9]\d)\s+years\b", t):
        adjustments["senior_citizen"] = 0.85
        
    # 2. Contempt of Court (Article 129 / 215 / Contempt of Courts Act)
    if "contempt" in fn or re.search(r"\bcontempt\s+petition\b|\bc\.?c\.?c\.?\s*(?:no|number)?\b|\bcontempt\s+of\s+court\b", t):
        adjustments["contempt"] = 1.40
        
    # 3. Active application for Interim Stay / Injunction
    if re.search(r"\bapplication\s+(?:for|seeking)\s+(?:ad-interim\s+|interim\s+)?(?:stay|injunction|relief)\b|\binterim\s+stay\b", t):
        adjustments["interim_stay"] = 0.50
        
    # 4. Interlocutory miscellaneous applications (impleadment, rejoinder, memo)
    if "implead" in fn or re.search(r"\border\s+1\s+rule\s+10\b", t):
        adjustments["procedural_discount"] = -0.45
            
    return adjustments

def priority_components(
    severity: Any,
    societal_impact: Any,
    immediate_threat: Any,
    case_age_days: Any,
    case_type: str,
    text: str = "",
    filename: str = ""
) -> dict[str, float]:
    sev = min(10.0, max(0.0, _finite_number(severity, 3.0)))
    impact = min(5.0, max(1.0, _finite_number(societal_impact, 1.0)))
    threat = 1.0 if _finite_number(immediate_threat, 0.0) >= 1.0 else 0.0
    age = _finite_number(case_age_days, float("nan"))

    # Severity normalization (Civil matters capped at 0.45)
    severity_norm = sev / 10.0
    if (case_type or "civil").strip().lower() == "civil":
        severity_norm = min(severity_norm, 0.45)
        
    impact_norm = impact / 5.0
    
    # Continuous aging scale: distinct per year without hard ceiling clamping
    if math.isfinite(age):
        years = age / 365.0
        recency = min(1.30, max(0.15, 0.15 + (years * 0.065)))
    else:
        recency = 0.50
        
    special = check_special_urgency(text, filename)
    special_bonus = sum(special.values())

    return {
        "severity_contribution": round(severity_norm * 3.5, 3),
        "societal_impact_contribution": round(impact_norm * 2.5, 3),
        "immediate_threat_contribution": round(threat * 3.0, 3),
        "recency_contribution": round(recency, 3),
        "statutory_listing_adjustments": round(special_bonus, 3)
    }

def compute_priority_score(
    severity: Any,
    societal_impact: Any,
    immediate_threat: Any,
    case_age_days: Any,
    case_type: str,
    text: str = "",
    filename: str = ""
) -> float:
    comps = priority_components(severity, societal_impact, immediate_threat, case_age_days, case_type, text, filename)
    raw = sum(comps.values())
    return round(min(10.0, max(1.0, raw)), 2)
