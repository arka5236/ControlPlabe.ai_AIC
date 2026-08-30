

"""
Policy & Decision Engine for ControlPlane.ai
Configures risk weighting, tolerance thresholds, and action resolution
per operational environment (Customer-facing, Internal Copilot, Regulated).
"""

from typing import Tuple, Dict, Any
from pathlib import Path
import sys

# Support both package-style execution (`python -m app.core.policy`) and
# direct script execution (`python app/core/policy.py`). Try a relative
# import first (works when executed as package), otherwise add the project
# root to `sys.path` and import the package-absolute path.
try:
    from .schemas import AppContext, ActionVerdict, RiskScores
except Exception:
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from app.core.schemas import AppContext, ActionVerdict, RiskScores


class PolicyEngine:
    """
    Evaluates raw checker scores against configurable risk matrices
    to deliver deterministic, policy-aware operational verdicts.
    """

    # Policy definitions: Weights (w_p, w_h, w_t) and thresholds
    PROFILES: Dict[AppContext, Dict[str, Any]] = {
        AppContext.CUSTOMER_FACING: {
            "weights": {"pii": 0.45, "hallucination": 0.30, "toxicity": 0.25},
            "hard_block_pii": True,
            "block_threshold": 0.70,
            "edit_threshold": 0.40,
            "escalate_threshold": 0.55,
        },
        AppContext.INTERNAL_COPILOT: {
            "weights": {"pii": 0.35, "hallucination": 0.45, "toxicity": 0.20},
            "hard_block_pii": False,  # Internal users can receive masked/edited tokens
            "block_threshold": 0.85,
            "edit_threshold": 0.35,
            "escalate_threshold": 0.65,
        },
        AppContext.REGULATED_WORKFLOW: {
            "weights": {"pii": 0.40, "hallucination": 0.40, "toxicity": 0.20},
            "hard_block_pii": True,
            "block_threshold": 0.60,
            "edit_threshold": 0.30,
            "escalate_threshold": 0.40,  # Highly conservative: Escalate early to human reviewer
        }
    }

    FALLBACK_BLOCK_MESSAGE = (
        "This response has been blocked by ControlPlane.ai enterprise guardrails "
        "due to high risk or data privacy compliance violations."
    )

    HALLUCINATION_DISCLAIMER = (
        "\n\n[ControlPlane Notice: This generated output contains unverified claims "
        "not fully corroborated by internal source documents.]"
    )

    @classmethod
    def evaluate(
        cls,
        context: AppContext,
        scores: RiskScores,
        raw_text: str
    ) -> Tuple[ActionVerdict, float, str, str | None]:
        """
        Computes composite risk and routes to ALLOW, EDIT, ESCALATE, or BLOCK.
        
        Returns:
            Tuple[ActionVerdict, composite_risk, delivered_text, escalation_reason]
        """
        config = cls.PROFILES.get(context, cls.PROFILES[AppContext.CUSTOMER_FACING])
        weights = config["weights"]

        # 1. Calculate Composite Risk Index R
        composite_risk = (
            (weights["pii"] * scores.pii_leak_score) +
            (weights["hallucination"] * scores.hallucination_score) +
            (weights["toxicity"] * scores.toxicity_score)
        )
        composite_risk = round(min(max(composite_risk, 0.0), 1.0), 4)

        # 2. Critical Safety Checks (Hard PII / Extreme Toxicity)
        if config["hard_block_pii"] and scores.pii_leak_score >= 0.8:
            return ActionVerdict.BLOCK, composite_risk, cls.FALLBACK_BLOCK_MESSAGE, "Severe PII pattern detected in customer-facing context."

        if scores.toxicity_score >= 0.85:
            return ActionVerdict.BLOCK, composite_risk, cls.FALLBACK_BLOCK_MESSAGE, "High toxicity or safety breach detected."

        # 3. Dynamic Threshold Tiers
        if composite_risk >= config["block_threshold"]:
            return ActionVerdict.BLOCK, composite_risk, cls.FALLBACK_BLOCK_MESSAGE, f"Composite risk score ({composite_risk}) exceeded block threshold."

        if composite_risk >= config["escalate_threshold"]:
            # Route to human queue, but allow response with warning or pause stream
            delivered = raw_text + "\n\n[Status: Flagged for asynchronous human compliance audit.]"
            return ActionVerdict.ESCALATE, composite_risk, delivered, f"Risk score ({composite_risk}) triggered mandatory human audit."

        if composite_risk >= config["edit_threshold"] or scores.hallucination_score >= 0.45:
            # Inline mitigation: sanitize or add grounding disclaimer
            sanitized = cls._sanitize_entities(raw_text, scores.flagged_entities)
            if scores.hallucination_score >= 0.45:
                sanitized += cls.HALLUCINATION_DISCLAIMER
            return ActionVerdict.EDIT, composite_risk, sanitized, None

        # 4. Clean Path
        return ActionVerdict.ALLOW, composite_risk, raw_text, None

    @staticmethod
    def _sanitize_entities(text: str, entities: list[str]) -> str:
        """Replaces identified PII tokens with sanitized placeholders."""
        sanitized = text
        for item in entities:
            sanitized = sanitized.replace(item, "[REDACTED_PII]")
        return sanitized
