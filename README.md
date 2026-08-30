# ControlPlabe.ai_AIC

# 🛡️  ControlPlane: Enterprise AI Guardrail Gateway

ControlPlane is a dual-stage, context-aware proxy built to secure Large Language Models (LLMs) in enterprise environments. It dynamically routes traffic based on real-time risk assessments, ensuring zero Data Loss (DLP), strict hallucination monitoring, and full Human-in-the-Loop (HITL) audit trails.

## 🌟 Key Features

*   **Dynamic Two-Stage Routing:** 
    *   *Stage 1:* Ultra-low latency regex PII scanning (~5ms) for immediate blocking of severe data leaks.
    *   *Stage 2:* Asynchronous LLM-as-a-judge evaluators (~500ms) for nuanced hallucination and toxicity detection.
*   **Context-Aware Policy Engine:** Adjusts strictness dynamically based on the application context (`customer_facing`, `internal_copilot`, or `regulated_workflow`).
*   **Stateful Audit Trails:** Built on LangGraph and SQLite to maintain immutable records of all user prompts, LLM generations, and proxy interventions.
*   **Human-in-the-Loop (HITL):** Built-in feedback APIs tied directly to the conversation state for RLHF (Reinforcement Learning from Human Feedback) and continuous model alignment.

## 🏗️ Architecture

1.  **Streamlit Frontend:** A stateful, chat-based UI equipped with real-time telemetry dashboards showing latency, Composite Risk (R), and intercepted PII.
2.  **FastAPI Backend (Proxy):** Exposes a stateless `/v1/inspect` endpoint for integration into any existing microservice architecture.
3.  **LangGraph Workflow:** Manages the DAG (Directed Acyclic Graph) of evaluators, conditional routing edges, and SQLite checkpointer memory.

## 🚀 Getting Started

### Prerequisites
*   Python 3.10+
*   HuggingFace API Token (for LLM generation and evaluation)

### Installation
1. Clone the repository and navigate to the root directory.
2. Create and activate a virtual environment:
   ```bash
   python -m venv .myenv
   source .myenv/bin/activate  # Or .myenv\Scripts\activate on Windows
Install dependencies:

Bash
pip install -r requirements.txt
Set your environment variable:

Bash
# Add this to a .env file or export it directly
HUGGINGFACEHUB_API_TOKEN="your_token_here"
Running the Application
To launch the unified Streamlit interface (Monolith mode):

Bash
python -m streamlit run streamlit_app.py
The application will be available at http://localhost:8501.

🗄️ Database & Memory Management
ControlPlane uses a local SQLite instance (chatbot.db) for zero-configuration deployments. It automatically provisions two critical tables:

LangGraph Checkpoints: Saves the complete BaseMessage history of every thread for multi-turn conversational memory.

Human Feedback (human_feedback): Stores timestamped 👍/👎 actions linked to the specific thread_id for offline anomaly review.

## 🛠️ Future Roadmap
Semantic Anomaly Detection: Implement local HuggingFaceEmbeddings to cluster and identify adversarial jailbreak vectors pre-generation.

Retrieval-Based Verification: Connect FAISS/ChromaDB to the llm_evaluators node to ground hallucination scores in verifiable source documents.
