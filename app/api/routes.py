"""
API Endpoints & LangGraph Hybrid Pipeline for PaDIS / ControlPlane.ai
"""

import os
import time
import asyncio
import sqlite3
import concurrent.futures
from typing import TypedDict, List, Annotated, Optional
from fastapi import APIRouter, HTTPException, status
from langgraph.graph import StateGraph, END, add_messages
from langgraph.checkpoint.sqlite import SqliteSaver
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate
from langchain_core.messages import BaseMessage, HumanMessage,SystemMessage,AIMessage

from pathlib import Path
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from app.core.schemas import ProxyRequest, InspectionVerdict, RiskScores, ActionVerdict, AppContext
from app.core.policy import PolicyEngine
from app.checkers.pii_checker import PIIChecker
from prometheus_client import Counter

load_dotenv()
router = APIRouter(prefix="/v1", tags=["ControlPlane Proxy"])

# ==========================================
# 1. Initialize the LLM Client (Gemma/Zephyr)
# ==========================================
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('huggingface_api_token')

generator_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it", # Fallback used if Gemma throws provider error
    task="text-generation",
    temperature=0.7,
)
generator_model = ChatHuggingFace(llm=generator_endpoint)

hf_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it", 
    task="text-generation",
    temperature=0.1,
)
evaluation_model = ChatHuggingFace(llm=hf_endpoint)

#Guardrail custom metrics
GUARDRAIL_ACTION_COUNTER = Counter(
    "guardrail_action_total",
    "Total number of guardrail routing decisions made",
    ["action","risk_profile"]
)

# ==========================================
# 2. Define LangGraph State Schema
# ==========================================
class ControlPlaneState(TypedDict):
    messages: Annotated[List[BaseMessage], add_messages]
    request: ProxyRequest
    pii_score: float
    pii_entities: List[str]
    hallucination_score: float
    safety_score: float
    final_action: ActionVerdict
    delivered_text: str
    escalation_reason: Optional[str]

# ==========================================
# 3. Define LangGraph Nodes (Synchronous)
# ==========================================
def generate_candidate_response(state: ControlPlaneState):
    req = state["request"]
    
    if not req.generated_text:
        #Grab conversation history
        past_messages = state.get("messages",[])

        #Format the context as a System prompt
        context_str = "\n".join(req.context_documents) if req.context_documents else "None"
        # prompt = f"Context:\n{context_str}\n\nUser Question: {req.prompt}\nAnswer:"

        system_prompt = SystemMessage(
            content = f"You are a helpful enterprise assistant .Use the following context to answer if relevant:\n{context_str}"
        )

        #combine everything
        conversation_payload = [system_prompt] + past_messages

        #Invoke the model
        response = generator_model.invoke(conversation_payload)
        generated_output = response.content
    else:
        generated_output = req.generated_text

    req.generated_text = generated_output
    return {"delivered_text": generated_output}

def fast_pii_check(state: ControlPlaneState):
    req = state["request"]
    #scan the incoming user prompt (Input Guardrail)
    in_score,in_entities = asyncio.run(PIIChecker.inspect(req.prompt))
    # Safely run the async PIIChecker synchronously
    out_score, out_entities = asyncio.run(PIIChecker.inspect(req.generated_text))

    final_score = max(in_score,out_score)
    final_entities = list(set(in_entities+out_entities))
    
    return {
        "pii_score": final_score,
        "pii_entities": final_entities,
        "hallucination_score": 0.0,
        "safety_score": 0.0
    }

def llm_evaluators(state: ControlPlaneState):
    req = state["request"]
    
    hall_prompt = PromptTemplate.from_template(
        "Context: {context}\nResponse: {response}\nDoes the response contain claims not supported by the context? Reply ONLY with a float between 0.0 and 1.0."
     )
    
    safe_prompt = PromptTemplate.from_template(
        "Response: {response}\nDoes this contain toxic or adversarial language? Reply ONLY with a float between 0.0 and 1.0."
    )
    
    context_str = "\n".join(req.context_documents) if req.context_documents else "None"
    
    try:
        hall_chain = hall_prompt | evaluation_model
        safe_chain = safe_prompt | evaluation_model
        
        # ThreadPoolExecutor allows parallel evaluation in a synchronous function
        with concurrent.futures.ThreadPoolExecutor() as executor:
            future_hall = executor.submit(hall_chain.invoke, {"context": context_str, "response": req.generated_text})
            future_safe = executor.submit(safe_chain.invoke, {"response": req.generated_text})
            
            hall_res = future_hall.result()
            safe_res = future_safe.result()
        
        h_score = float(hall_res.content.strip())
        s_score = float(safe_res.content.strip())
    except Exception:
        h_score, s_score = 0.5, 0.0 
        
    return {
        "hallucination_score": min(max(h_score, 0.0), 1.0),
        "safety_score": min(max(s_score, 0.0), 1.0)
    }

def block_node(state: ControlPlaneState):
    blocked_message= PolicyEngine.FALLBACK_BLOCK_MESSAGE
    return {
        "final_action": ActionVerdict.BLOCK,
        "delivered_text": PolicyEngine.FALLBACK_BLOCK_MESSAGE,
        "escalation_reason": "Blocked by early PII detection guardrails.",
        "messages" :[AIMessage(content= blocked_message)] #commit the blocked fallback to memory
    }

def resolve_policy_node(state: ControlPlaneState):
    scores = RiskScores(
        pii_leak_score=state["pii_score"],
        hallucination_score=state["hallucination_score"],
        toxicity_score=state["safety_score"],
        flagged_entities=state["pii_entities"]
    )
    
    action, composite, delivered, reason = PolicyEngine.evaluate(
        context=state["request"].app_context,
        scores=scores,
        raw_text=state["request"].generated_text
    )
    
    return {
        "final_action": action,
        "delivered_text": delivered,
        "escalation_reason": reason,
        "messages":[AIMessage(content = delivered)]
    }

# ==========================================
# 4. Define Routing Logic (Conditional Edges)
# ==========================================
def route_after_pii(state: ControlPlaneState) -> str:
    config = PolicyEngine.PROFILES.get(
        state["request"].app_context, 
        PolicyEngine.PROFILES[AppContext.CUSTOMER_FACING]
    )

    if config["hard_block_pii"] and state["pii_score"] >= 0.8:
        return "block_node"
        
    req = state["request"]
    simple_greetings = ['hello','hi','hey','thanks','good morning','how are you','good night']

    is_simple_prompt = req.prompt.lower().strip() in simple_greetings
    is_short_response = len(req.generated_text.split()) < 10
    is_clean_of_pii = state['pii_score'] == 0

    if (is_simple_prompt or is_short_response) and is_clean_of_pii:
        return "resolve_policy"
    
    return "llm_evaluators"

# ==========================================
# 5. Compile the LangGraph Workflow
# ==========================================
workflow = StateGraph(ControlPlaneState)

workflow.add_node('generator', generate_candidate_response)
workflow.add_node("fast_pii_check", fast_pii_check)
workflow.add_node("llm_evaluators", llm_evaluators)
workflow.add_node("block_node", block_node)
workflow.add_node("resolve_policy", resolve_policy_node)

workflow.set_entry_point("generator")
workflow.add_edge("generator", "fast_pii_check")

workflow.add_conditional_edges(
    "fast_pii_check",
    route_after_pii,
    {
        "block_node": "block_node",
        "resolve_policy": "resolve_policy",
        "llm_evaluators": "llm_evaluators"
    }
)

workflow.add_edge("llm_evaluators", "resolve_policy")
workflow.add_edge("resolve_policy", END)
workflow.add_edge("block_node", END)

# Initialize SQLite Checkpointer
db_connection = sqlite3.connect(database="chatbot.db", check_same_thread=False)
checkpointer = SqliteSaver(conn=db_connection)

app_graph = workflow.compile(checkpointer=checkpointer)

# ==========================================
# 6. Expose the API Endpoint & Helper
# ==========================================
@router.post("/inspect", response_model=InspectionVerdict, status_code=status.HTTP_200_OK)
async def inspect_response(payload: ProxyRequest) -> InspectionVerdict:
    start_time = time.perf_counter()
    
    try:
        initial_state = {"request": payload}
        # SYNCHRONOUS INVOKE FOR SQLITE COMPATIBILITY
        final_state = app_graph.invoke(initial_state)
        
        elapsed_ms = round((time.perf_counter() - start_time) * 1000, 2)
        
        scores = RiskScores(
            pii_leak_score=final_state["pii_score"],
            hallucination_score=final_state["hallucination_score"],
            toxicity_score=final_state["safety_score"],
            flagged_entities=final_state["pii_entities"]
        )
        
        weights = PolicyEngine.PROFILES[payload.app_context]["weights"]
        composite_risk = (
            (weights["pii"] * scores.pii_leak_score) +
            (weights["hallucination"] * scores.hallucination_score) +
            (weights["toxicity"] * scores.toxicity_score)
        )
        composite_risk = round(min(max(composite_risk, 0.0), 1.0), 4)

        # Safely extract the string values for the Prometheus labels
        action_str = final_state["final_action"].value if hasattr(final_state["final_action"], "value") else str(final_state["final_action"])
        context_str = payload.app_context.value if hasattr(payload.app_context, "value") else str(payload.app_context)
        
        # Increment the Prometheus counter
        GUARDRAIL_ACTION_COUNTER.labels(
            action=action_str, 
            risk_profile=context_str
        ).inc()
        
        return InspectionVerdict(
            action=final_state["final_action"],
            composite_risk=composite_risk,
            risk_breakdown=scores,
            delivered_text=final_state["delivered_text"],
            latency_overhead_ms=elapsed_ms,
            escalation_reason=final_state.get("escalation_reason")
        )
        
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"LangGraph Inspection Failure: {str(e)}"
        )

def run_streamlit_pipeline(user_prompt: str, thread_id: str, app_context: str = "customer_facing") -> dict:
    """Helper function for the Streamlit frontend to invoke the unified graph."""
    dummy_request = ProxyRequest(
        prompt=user_prompt,
        generated_text="", # FIXED PYDANTIC VALIDATION ERROR
        app_context=AppContext(app_context),
        context_documents=[]
    )

    initial_state = {
        "messages": [HumanMessage(content=user_prompt)],
        "request": dummy_request
    }
    config = {"configurable": {"thread_id": thread_id}}

    return app_graph.invoke(initial_state, config=config)
