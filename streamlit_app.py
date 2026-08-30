import time
import uuid
import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage

# MUST be the very first Streamlit command
st.set_page_config(page_title="ControlPanel.ai", page_icon="🛡️", layout="wide")

# Adjust this import path based on your exact folder structure
from app.api.routes import app_graph, checkpointer, run_streamlit_pipeline
from app.core.policy import PolicyEngine
from app.core.schemas import AppContext
from app.api.feedback import save_feedback_sync

# ==========================================
# Utility Functions
# ==========================================
def generate_thread_id(): 
    return str(uuid.uuid4())[:8] # Shortened for cleaner UI

def reset_chat():
    new_id = generate_thread_id()
    st.session_state['thread_id'] = new_id
    add_thread(new_id)
    st.session_state['message_history'] = []
    st.session_state['latest_telemetry'] = None

def add_thread(thread_id):
    if thread_id not in st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(thread_id)

def retrieve_all_threads():
    """Fetches unique persisted conversation IDs from the SQLite checkpointer."""
    threads = set()
    for checkpoint in checkpointer.list(None):
        t_id = checkpoint.config.get("configurable", {}).get("thread_id")
        if t_id:
            threads.add(t_id)
    return sorted(list(threads))

def load_conversation(thread_id):
    state = app_graph.get_state(config={'configurable': {'thread_id': thread_id}})
    values = getattr(state, 'values', {}) or {}
    return values.get('messages', [])



# ==========================================
# Session Setup
# ==========================================
if 'message_history' not in st.session_state:
    st.session_state['message_history'] = []

if 'thread_id' not in st.session_state:
    st.session_state['thread_id'] = generate_thread_id()

if 'chat_threads' not in st.session_state:
    st.session_state['chat_threads'] = retrieve_all_threads()
    if not st.session_state['chat_threads']:
        st.session_state['chat_threads'].append(st.session_state['thread_id'])

#Autoload history on refresh of defaulting to empty
if 'message_histtory' not in st.session_state or not st.session_state['message_history']:
    raw_msgs = load_conversation(st.session_state['thread_id'])
    loaded_history = []

    for msg in raw_msgs:
        role = 'user' if isinstance(msg,HumanMessage) else 'assistant'
        loaded_history.append({'role':role,'content' : msg.content})
    st.session_state['message_history'] = loaded_history

if 'latest_telemetry' not in st.session_state:
    st.session_state['latest_telemetry'] = None

add_thread(st.session_state['thread_id'])

# ==========================================
# Sidebar UI
# ==========================================
st.sidebar.title('🛡️ ControlPlane.ai')

# Problem Statement Req: Context-Aware Routing Profiles
app_context_selection = st.sidebar.selectbox(
    "Risk Profile",
    options=["customer_facing", "internal_copilot", "regulated_workflow"],
    format_func=lambda x: x.replace("_", " ").title()
)
st.session_state['app_context'] = app_context_selection

st.sidebar.divider()

if st.sidebar.button('➕ New Chat', use_container_width=True, type="primary"):
    reset_chat()
    st.rerun()

st.sidebar.header('My Conversations')

# Display persisted SQLite threads
for t_id in st.session_state['chat_threads']:
    is_active = " (Active)" if t_id == st.session_state['thread_id'] else ""
    if st.sidebar.button(f"💬 Thread {t_id}{is_active}", use_container_width=True):
        st.session_state['thread_id'] = t_id
        messages = load_conversation(t_id)

        temp_messages = []
        for message in messages:
            role = 'user' if isinstance(message, HumanMessage) else 'assistant'
            temp_messages.append({'role': role, 'content': message.content})

        st.session_state['message_history'] = temp_messages
        st.session_state['latest_telemetry'] = None
        st.rerun()

# ==========================================
# Main UI & Telemetry Dashboard
# ==========================================
st.subheader(f"Active Session: `{st.session_state['thread_id']}`")

# Render Guardrail Telemetry (Problem Statement: Audit & Governance)
if st.session_state['latest_telemetry']:
    t = st.session_state['latest_telemetry']
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Action Verdict", t["action"], delta="Secured" if t["action"] in ["ALLOW", "EDIT"] else "Interrupted", delta_color="normal")
    c2.metric("Composite Risk (R)", f"{t['composite_risk']:.2f}")
    c3.metric("PII Score", f"{t['pii_score']:.2f}")
    c4.metric("Latency", f"{t['latency_ms']:.1f} ms")
    
    if t.get("reason"):
        st.warning(f"**Policy Trigger:** {t['reason']}")
    st.divider()

# Load the conversation history
for m in st.session_state['message_history']:
    with st.chat_message(m['role']):
        st.markdown(m['content'])

# ==========================================
# Chat Input & Graph Execution
# ==========================================
user_input = st.chat_input('Type your prompt here...')

if user_input:
    # 1. Add and display user message
    st.session_state['message_history'].append({'role': 'user', 'content': user_input})
    with st.chat_message('user'):
        st.markdown(user_input)

    # 2. Execute Graph & Track Latency using the routes.py helper
    start_time = time.perf_counter()
    with st.spinner("Inspecting via PaDIS ControlPlane..."):
        final_state = run_streamlit_pipeline(
            user_prompt=user_input,
            thread_id=st.session_state['thread_id'],
            app_context=st.session_state['app_context']
        )
    latency_ms = (time.perf_counter() - start_time) * 1000

    # 3. Calculate Composite Risk for Telemetry Display
    weights = PolicyEngine.PROFILES[AppContext(st.session_state['app_context'])]["weights"]
    raw_composite = (
        (weights["pii"] * final_state.get("pii_score", 0.0)) +
        (weights["hallucination"] * final_state.get("hallucination_score", 0.0)) +
        (weights["toxicity"] * final_state.get("safety_score", 0.0))
    )
    composite_risk = round(min(max(raw_composite, 0.0), 1.0), 4)

    # 4. Extract Results
    delivered_text = final_state.get("delivered_text", "Error processing response.")
    
    # Safely extract the final action enum value if it exists
    action_val = final_state.get("final_action", "ALLOW")
    if hasattr(action_val, "value"):
        action_val = action_val.value

    st.session_state['latest_telemetry'] = {
        "action": action_val,
        "composite_risk": composite_risk,
        "pii_score": final_state.get("pii_score", 0.0),
        "latency_ms": latency_ms,
        "reason": final_state.get("escalation_reason")
    }

    # 5. Stream the Output
    with st.chat_message('assistant'):
        def stream_generator(text):
            for word in text.split(" "):
                yield word + " "
                time.sleep(0.03)
                
        ai_message = st.write_stream(stream_generator(delivered_text))
        st.session_state['message_history'].append({'role': 'assistant', 'content': ai_message})
        
    st.rerun()

#Human in the loop Feedback UI
if(len(st.session_state['message_history'])>0 and st.session_state['message_history'][-1]['role'] == 'assistant'):
    st.caption("Help us improve the Guardrail:")

    #Render thumbs up/down buttons
    feedback_val = st.feedback("thumbs",key = f"fb_{len(st.session_state['message_history'])}")

    if feedback_val is not None:
        db_val = 1 if feedback_val ==1 else -1
        save_feedback_sync(st.session_state['thread_id'], db_val)
        st.toast("✅ Feedback successfully logged for audit review!", icon="🔒")
