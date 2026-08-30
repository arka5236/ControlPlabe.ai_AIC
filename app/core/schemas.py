##Defines strong data contracts for client requests, inspection results, and action verdicts.

"""
Schemas Definition Module for ControlPlane.ai
Defines the Pydantic data contracts for incoming proxy requests,
checker outputs, telemetry logs, and final routing decisions.
"""

from typing import List, Optional, Dict, Any
from enum import Enum
from pydantic import BaseModel, Field


class AppContext(str, Enum):
    """Supported risk profiles mapped to application use-cases."""
    CUSTOMER_FACING = "customer_facing"
    INTERNAL_COPILOT = "internal_copilot"
    REGULATED_WORKFLOW = "regulated_workflow"


class ActionVerdict(str, Enum):
    """Dynamic routing actions taken by the ControlPlane composite engine."""
    ALLOW = "ALLOW"
    EDIT = "EDIT"
    ESCALATE = "ESCALATE"
    BLOCK = "BLOCK"


class ProxyRequest(BaseModel):
    """Incoming request payload forwarded to the ControlPlane proxy."""
    prompt: str = Field(..., description="The original user prompt.")
    generated_text: str = Field(..., description="The LLM-generated response candidate to evaluate.")
    context_documents: Optional[List[str]] = Field(
        default_factory=list,
        description="Retrieved source context chunks (used for faithfulness validation)."
    )
    app_context: AppContext = Field(
        default=AppContext.CUSTOMER_FACING,
        description="Execution profile determining risk tolerance and latency thresholds."
    )


class RiskScores(BaseModel):
    """Individual normalized risk metrics [0.0 - 1.0]."""
    pii_leak_score: float = Field(default=0.0, ge=0.0, le=1.0)
    hallucination_score: float = Field(default=0.0, ge=0.0, le=1.0)
    toxicity_score: float = Field(default=0.0, ge=0.0, le=1.0)
    flagged_entities: List[str] = Field(default_factory=list)


class InspectionVerdict(BaseModel):
    """Complete evaluation report returned by ControlPlane Checker."""
    action: ActionVerdict
    composite_risk: float = Field(..., ge=0.0, le=1.0)
    risk_breakdown: RiskScores
    delivered_text: str = Field(..., description="Original, sanitized, or fallback response text.")
    latency_overhead_ms: float = Field(..., description="Inspection latency in milliseconds.")
    escalation_reason: Optional[str] = None

class FeedbackPayload(BaseModel):
    thread_id : str
    feedback_value : int # 1 for thums up and 1 for thumbs down
    correction:Optional[str] = None
