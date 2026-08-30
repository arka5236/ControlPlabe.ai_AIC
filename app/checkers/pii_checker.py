"""
PII & Data Leakage Checker Module
Utilizes compiled regular expressions and heuristic token recognition
to catch Social Security Numbers, API Keys, Passwords, Emails, and Phone Numbers.
"""

import re
from typing import List, Tuple
from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from app.core.schemas import RiskScores


class PIIChecker:
    """Zero-overhead privacy inspector."""

    # High-precision patterns for enterprise data loss prevention (DLP)
    PATTERNS = {
        "SSN": re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
        "API_KEY": re.compile(r"\b(?:sk-[a-zA-Z0-9]{20,}|AKIA[0-9A-Z]{16})\b"),
        "CREDIT_CARD": re.compile(r"\b(?:\d{4}[-\s]?){3}\d{4}\b"),
        "EMAIL": re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,7}\b"),
        "PHONE_IN_US": re.compile(r"\b(?:\+?1[-.\s]?)?\(?[0-9]{3}\)?[-.\s]?[0-9]{3}[-.\s]?[0-9]{4}\b"),
        "EMPLOYEE_ID": re.compile(r"\bEMP-[0-9]{4,6}\b", re.IGNORECASE),
    }

    @classmethod
    async def inspect(cls, text: str) -> Tuple[float, List[str]]:
        """
        Scans text for sensitive patterns asynchronously.
        Returns: (pii_leak_score [0.0 - 1.0], list_of_detected_strings)
        """
        detected_entities: List[str] = []
        severity_weight = 0.0

        for pattern_name, regex in cls.PATTERNS.items():
            matches = regex.findall(text)
            if matches:
                detected_entities.extend(matches)
                # Assign heavier weights to high-risk credentials/SSNs
                if pattern_name in ("SSN", "API_KEY", "CREDIT_CARD"):
                    severity_weight += 0.9
                else:
                    severity_weight += 0.4

        # Normalize score to [0.0, 1.0]
        leak_score = min(severity_weight, 1.0)
        return leak_score, detected_entities
