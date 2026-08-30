"""
graph_nodes.py
--------------
Contains all the synchronous LangGraph nodes and LLM initializations for the PaDIS pipeline.
"""

import os
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[2]))
import asyncio
import concurrent.futures
from dotenv import load_dotenv

from langchain_huggingface import ChatHuggingFace, HuggingFaceEndpoint
from langchain_core.prompts import PromptTemplate

from app.core.schemas import RiskScores, ActionVerdict, AppContext
from app.core.policy import PolicyEngine
from app.checkers.pii_checker import PIIChecker

load_dotenv()
os.environ['HUGGINGFACEHUB_API_TOKEN'] = os.getenv('huggingface_api_token')

# ==========================================
# 1. Initialize the LLMs
# ==========================================
generator_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-2b-it", 
    task="text-generation",
    temperature=0.7,
)
generator_model = ChatHuggingFace(llm=generator_endpoint)

hf_endpoint = HuggingFaceEndpoint(
    repo_id="google/gemma-4-31B-it", # Or mistral/zephyr if you switched them
    task="text-generation",
    temperature=0.1,
)
evaluation_model = ChatHuggingFace(llm=hf_endpoint)

# ==========================================
# 2. Define the Nodes
# ==========================================
def generate_candidate_response(state):
    """Stage 0: Primary AI generates the candidate response."""
    req = state["request"]
    
    if not req.generated_text:
        context_str = "\n".join(req.context_documents) if req.context_documents else "None"
        prompt = f"Context:\n{context_str}\n\nUser Question: {req.prompt}\nAnswer:"
        
        response = generator_model.invoke(prompt)
        generated_output = response.content
    else:
        generated_output = req.generated_text

    req.generated_text = generated_output
    return {"delivered_text": generated_output}


def fast_pii_check(state):
    """Stage 1: Lightning-fast regex scan."""
    req = state["request"]
    pii_score, pii_entities = asyncio.run(PIIChecker.inspect(req.generated_text))
    
    return {
        "pii_score": pii_score,
        "pii_entities": pii_entities,
        "hallucination_score": 0.0,
        "safety_score": 0.0
    }


def llm_evaluators(state):
    """Stage 2: LLM-as-a-Judge for complex constraints."""
    req = state["request"]
    
    hall_prompt = PromptTemplate.from_template(
        "Context: {context}\nResponse: {response}\nDoes the response contain claims not supported by the context? Reply ONLY with a float between 0.0 and 1.0."
    )
    safe_prompt = PromptTemplate.from_template(
        "Response: {response}\nDoes this contain toxic language? Reply ONLY with a float between 0.0 and 1.0."
    )
    
    context_str = "\n".join(req.context_documents) if req.context_documents else "None"
    
    try:
        hall_chain = hall_prompt | evaluation_model
        safe_chain = safe_prompt | evaluation_model
        
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


def block_node(state):
    """A terminal node executed when severe risk mandates an immediate block."""
    return {
        "final_action": ActionVerdict.BLOCK,
        "delivered_text": PolicyEngine.FALLBACK_BLOCK_MESSAGE,
        "escalation_reason": "Blocked by early PII detection guardrails."
    }


def resolve_policy_node(state):
    """Applies the enterprise policy matrix to calculate final composite risk."""
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
        "escalation_reason": reason
    }


def route_after_pii(state) -> str:
    """Conditional Edge logic."""
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
