# streamlit_app.py
# Complete AI Engineering Tutor — no FastAPI needed
# Layout: Left = input + controls | Right = chat output + KG subpanel

import streamlit as st
import time
import random
from streamlit_agraph import agraph, Node, Edge, Config

st.set_page_config(
    page_title="MOSAICurriculum",
    layout="wide",
    initial_sidebar_state="collapsed"
)

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #F8FAFC;
    color: #1E293B;
}
#MainMenu, footer { visibility: hidden; }
# header { visibility: hidden; }
[data-testid="stSidebarCollapsedControl"] { visibility: visible !important; }
header [data-testid="stSidebarCollapsedControl"],
section[data-testid="collapsedControl"] { visibility: visible !important; display: flex !important; }
.stDeployButton { display: none; }
.stApp {
    background: linear-gradient(135deg, #F0FDF4 0%, #F0F9FF 50%, #FAFAFA 100%);
}

.tutor-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.6rem;
    font-weight: 800;
    background: linear-gradient(90deg, #059669, #0284C7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.tutor-subtitle {
    font-size: 0.65rem;
    color: #94A3B8;
    letter-spacing: 0.15em;
    text-transform: uppercase;
}
.message-user {
    background: linear-gradient(135deg, #DCFCE7, #DBEAFE);
    border: 1px solid #BBF7D0;
    border-radius: 10px 10px 0 10px;
    padding: 0.7rem 0.9rem;
    margin-bottom: 0.6rem;
    font-size: 0.82rem;
    color: #065F46;
}
.message-assistant {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 10px 10px 10px 0;
    padding: 0.9rem 1rem;
    margin-bottom: 0.6rem;
    font-size: 0.84rem;
    color: #1E293B;
    white-space: pre-wrap;
    line-height: 1.6;
    box-shadow: 0 1px 4px rgba(0,0,0,0.04);
}
.agent-tag {
    display: inline-block;
    font-size: 0.58rem;
    font-weight: 700;
    letter-spacing: 0.12em;
    text-transform: uppercase;
    padding: 0.12rem 0.45rem;
    border-radius: 4px;
    margin-bottom: 0.3rem;
}
.tag-solver     { background: #DBEAFE; color: #1D4ED8; border: 1px solid #93C5FD; }
.tag-assessment { background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
.tag-feedback   { background: #F3E8FF; color: #6B21A8; border: 1px solid #D8B4FE; }
.tag-system     { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
.panel-header {
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.18em;
    text-transform: uppercase;
    color: #94A3B8;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
    margin-top: 0.2rem;
}
.progress-bar-bg   { background:#E2E8F0; border-radius:4px; height:5px; margin-top:0.4rem; }
.progress-bar-fill { background:linear-gradient(90deg,#059669,#0284C7); border-radius:4px; height:5px; }
.status-dot     { display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:0.3rem; }
.status-online  { background:#059669; box-shadow:0 0 5px #059669; }
.status-offline { background:#CBD5E1; }
.stTextInput input {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #1E293B !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stTextInput input:focus { border-color: #059669 !important; }
.stTextArea textarea {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #1E293B !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.82rem !important;
}
.stButton button {
    background: linear-gradient(135deg, #F0FDF4, #DCFCE7) !important;
    border: 1px solid #86EFAC !important;
    color: #065F46 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.72rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
}
.stButton button:hover { box-shadow: 0 2px 8px rgba(5,150,105,0.15) !important; }
hr { border-color: #F1F5F9 !important; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-track { background: transparent; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# Load components
# ─────────────────────────────────────────────────────
@st.cache_resource(ttl=1800)
def load_components():
    from llm_client import LLMClient
    from memory.letta_client import LettaClient
    from rag.embedder import BGEEmbedder
    from rag.retriever import RAGRetriever
    from kg.neo4j_client import Neo4jClient
    from agents.solver_agent import SolverAgent
    from agents.assessment_agent import AssessmentAgent
    from agents.feedback_agent import FeedbackAgent
    from agents.orchestrator import Orchestrator

    llm          = LLMClient()
    embedder     = BGEEmbedder()
    retriever    = RAGRetriever(embedder)
    neo4j        = Neo4jClient()
    letta        = LettaClient()
    solver       = SolverAgent(llm, retriever, neo4j, letta)
    assessment   = AssessmentAgent(llm, retriever, neo4j, letta)
    feedback     = FeedbackAgent(llm, retriever, neo4j, letta)
    orchestrator = Orchestrator(solver, assessment, feedback, neo4j, letta)
    return {
        "llm": llm, "retriever": retriever, "neo4j": neo4j, "letta": letta,
        "solver": solver, "assessment": assessment,
        "feedback": feedback, "orchestrator": orchestrator,
    }

try:
    components        = load_components()
    COMPONENTS_LOADED = True
    LOAD_ERROR        = ""
except Exception as e:
    COMPONENTS_LOADED = False
    LOAD_ERROR        = str(e)
    components        = {}

if COMPONENTS_LOADED and not st.session_state.get("ingestion_done"):
    with st.spinner("Checking knowledge base..."):
        from rag.fetch_docs import run_ingestion
        run_ingestion()
    st.session_state.ingestion_done = True

# ─────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────
STATUS_COLORS = {
    "grey": "#9CA3AF", "blue": "#3B82F6", "yellow": "#F59E0B",
    "green": "#10B981", "red": "#EF4444", "orange": "#F97316",
}
STATUS_LABELS = {
    "grey": "Not reached", "blue": "Learning now", "yellow": "Assessed",
    "green": "Mastered ✓", "red": "Needs review", "orange": "Prereq gap",
}
AGENT_TAGS = {
    "Solver":     ("SOLVER",   "tag-solver"),
    "Assessment": ("ASSESS",   "tag-assessment"),
    "Feedback":   ("FEEDBACK", "tag-feedback"),
    "System":     ("SYSTEM",   "tag-system"),
}

# ─────────────────────────────────────────────────────
# Session state
# ─────────────────────────────────────────────────────
for key, val in {
    "messages": [],
    "student_id": f"student_{random.randint(1000, 9999)}",
    "current_concept": None, "current_question": None,
    "kg_data": None, "kg_visible": False, "last_kg_refresh": 0,
    "response_style": "Balanced", "difficulty_override": "Auto",
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────
# Agent helpers
# ─────────────────────────────────────────────────────
def call_chat(message: str) -> dict:
    if not COMPONENTS_LOADED:
        return {"response": f"Error: {LOAD_ERROR}", "agent": "System"}
    try:
        orch     = components["orchestrator"]
        response = orch.route(student_id=st.session_state.student_id, message=message)
        return {"response": response, "agent": orch.last_agent_used}
    except Exception as e:
        return {"response": f"Agent error: {e}", "agent": "System"}

def call_get_question(concept: str) -> dict:
    if not COMPONENTS_LOADED:
        return {}
    try:
        return components["assessment"].generate_question(
            student_id=st.session_state.student_id, concept=concept)
    except Exception as e:
        st.error(f"Question error: {e}")
        return {}

def call_evaluate(concept, question, answer, expected) -> dict:
    if not COMPONENTS_LOADED:
        return {}
    try:
        result = components["assessment"].evaluate_answer(
            student_id=st.session_state.student_id, concept=concept,
            question=question, student_answer=answer, expected_points=expected)
        fb = components["feedback"].give_feedback(
            student_id=st.session_state.student_id, concept=concept,
            question=question, student_answer=answer, assessment_result=result)
        return {
            "score": result["score"], "passed": result["passed"],
            "feedback": fb["feedback_text"],
            "what_was_right": fb["what_was_right"],
            "what_was_wrong": fb["what_was_wrong"],
            "next_action": fb["next_action"],
            "re_teach_focus": fb["re_teach_focus"],
        }
    except Exception as e:
        st.error(f"Evaluation error: {e}")
        return {}

def get_kg_data() -> dict:
    if not COMPONENTS_LOADED:
        return {"elements": {"nodes": [], "edges": []}, "node_count": 0, "visible": False}
    try:
        return components["neo4j"].to_cytoscape_json()
    except Exception:
        return {"elements": {"nodes": [], "edges": []}, "node_count": 0, "visible": False}

def get_progress() -> dict | None:
    if not COMPONENTS_LOADED:
        return None
    try:
        letta    = components["letta"]
        neo4j    = components["neo4j"]
        mastered = letta.get_mastered_concepts(st.session_state.student_id)
        core     = letta.read_core_memory(st.session_state.student_id)
        total    = neo4j.get_node_count()
        return {
            "current_level":    core.get("current_level", "beginner"),
            "current_topic":    core.get("current_topic", ""),
            "mastered_count":   len(mastered),
            "total_concepts":   total,
            "progress_percent": round(len(mastered) / total * 100) if total > 0 else 0,
        }
    except Exception:
        return None

# ─────────────────────────────────────────────────────
# Render helpers
# ─────────────────────────────────────────────────────
def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(
            f'<div class="message-user">💬 {msg["content"]}</div>',
            unsafe_allow_html=True)
    else:
        tag, cls = AGENT_TAGS.get(msg.get("agent", "System"), ("SYSTEM", "tag-system"))
        st.markdown(
            f'<span class="agent-tag {cls}">{tag}</span>'
            f'<div class="message-assistant">{msg["content"]}</div>',
            unsafe_allow_html=True)

def render_kg(kg_data: dict, height: int = 380):
    elements   = kg_data.get("elements", {})
    nodes_data = elements.get("nodes", [])
    edges_data = elements.get("edges", [])

    if len(nodes_data) <= 1:
        st.caption("⏳ Ask a question to start building the graph...")
        return

    nodes = []
    for n in nodes_data:
        d      = n["data"]
        status = d.get("status", "grey")
        size   = {"beginner": 14, "intermediate": 18, "advanced": 22}.get(
                 d.get("difficulty", "intermediate"), 18)
        nodes.append(Node(
            id=d["id"], label=d["label"], size=size,
            color=STATUS_COLORS.get(status, "#9CA3AF"),
            title=f"{d['label']} · {STATUS_LABELS.get(status, status)} · {d.get('difficulty','')}",
            font={"color": "#1E293B", "size": 11, "face": "JetBrains Mono"}
        ))

    edge_colors = {
        "REQUIRES": "#EF4444", "BUILDS_ON": "#3B82F6",
        "PART_OF": "#10B981", "USED_IN": "#F59E0B", "RELATED_TO": "#94A3B8",
    }
    edges = [
        Edge(
            source=e["data"]["source"], target=e["data"]["target"],
            label=e["data"].get("relationship", "").replace("_", " ").lower(),
            color=edge_colors.get(e["data"].get("relationship", "RELATED_TO"), "#94A3B8"),
            arrows="to"
        ) for e in edges_data
    ]

    agraph(nodes=nodes, edges=edges, config=Config(
        width="100%", height=height, directed=True, physics=True,
        hierarchical=False, nodeHighlightBehavior=True,
        highlightColor="#059669",
        d3={"gravity": -250, "linkLength": 110}
    ))

# ─────────────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────────────
hc1, hc2 = st.columns([5, 1])
with hc1:
    st.markdown('<div class="tutor-title">MOSAICurriculum</div>', unsafe_allow_html=True)
    st.markdown('<div class="tutor-subtitle">Memory-Orchestrated Symbolic Agent Intelligent Curriculum</div>', unsafe_allow_html=True)
with hc2:
    color = "#059669" if COMPONENTS_LOADED else "#EF4444"
    label = "● Ready" if COMPONENTS_LOADED else "● Error"
    st.markdown(
        f'<div style="text-align:right;padding-top:0.8rem;font-size:0.7rem;font-weight:700;color:{color}">{label}</div>',
        unsafe_allow_html=True)

if not COMPONENTS_LOADED:
    st.error(f"Failed to load: {LOAD_ERROR}")
    st.stop()

st.markdown("---")

# ─────────────────────────────────────────────────────
# SIDEBAR — Knowledge Graph
# ─────────────────────────────────────────────────────
with st.sidebar:
    st.markdown('<div class="panel-header">Knowledge Graph</div>', unsafe_allow_html=True)

    now = time.time()
    if now - st.session_state.last_kg_refresh > 5:
        kg                               = get_kg_data()
        st.session_state.kg_data         = kg
        st.session_state.kg_visible      = kg.get("visible", False)
        st.session_state.last_kg_refresh = now

    node_count = st.session_state.kg_data.get("node_count", 0) if st.session_state.kg_data else 0
    st.caption(f"🕸️ {node_count} concepts indexed")

    if st.session_state.kg_visible and st.session_state.kg_data:
        leg_cols = st.columns(2)
        for i, (status, slabel) in enumerate(STATUS_LABELS.items()):
            with leg_cols[i % 2]:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.25rem;margin-bottom:0.4rem">'
                    f'<div style="width:8px;height:8px;border-radius:50%;'
                    f'background:{STATUS_COLORS[status]};flex-shrink:0"></div>'
                    f'<span style="font-size:0.57rem;color:#64748B;white-space:nowrap">{slabel}</span></div>',
                    unsafe_allow_html=True)

        render_kg(st.session_state.kg_data, height=420)
        st.caption("💡 Right-click → Save image as... to export PNG")
    else:
        st.markdown("""
        <div style="text-align:center;padding:2rem 1rem;color:#94A3B8">
            <div style="font-size:1.8rem">🕸️</div>
            <div style="font-size:0.72rem;margin-top:0.5rem">
                Graph appears after 2+ concepts are indexed
            </div>
        </div>""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# MAIN LAYOUT
# ─────────────────────────────────────────────────────
col_left, col_right = st.columns([1, 1.8], gap="large")

# ══════════════════════════════════════════════════════
# LEFT — Input & Controls
# ══════════════════════════════════════════════════════
with col_left:
    tab_chat, tab_assess, tab_settings, tab_eval = st.tabs(["💬 Chat", "📝 Assessment", "⚙️ Settings", "🧪 Evaluation"])

    # ── CHAT ──
    with tab_chat:
        st.markdown('<div class="panel-header">Ask a question</div>', unsafe_allow_html=True)

        ic, bc = st.columns([4, 1])
        with ic:
            user_input = st.text_input(
                "msg", placeholder="e.g. explain gradient descent...",
                label_visibility="collapsed", key="chat_input")
        with bc:
            send = st.button("Send →", use_container_width=True, key="send_btn")

        if send and user_input:
            st.session_state.messages.append({"role": "user", "content": user_input})
            with st.spinner("Thinking..."):
                r = call_chat(user_input)
            st.session_state.messages.append({
                "role": "assistant", "content": r["response"],
                "agent": r.get("agent", "Solver")})
            st.rerun()

        st.markdown("---")
        st.markdown('<div style="font-size:0.6rem;color:#94A3B8;letter-spacing:0.12em;text-transform:uppercase;margin-bottom:0.5rem">Quick topics</div>', unsafe_allow_html=True)

        quick_prompts = [
            "Explain gradient descent", "What is backpropagation?",
            "How do transformers work?", "Explain overfitting",
            "What is RAG?", "Explain embeddings",
        ]
        qc1, qc2 = st.columns(2)
        for i, prompt in enumerate(quick_prompts):
            col = qc1 if i % 2 == 0 else qc2
            with col:
                if st.button(prompt, key=f"qp_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": prompt})
                    with st.spinner("Thinking..."):
                        r = call_chat(prompt)
                    st.session_state.messages.append({
                        "role": "assistant", "content": r["response"],
                        "agent": r.get("agent", "Solver")})
                    st.rerun()

    # ── ASSESSMENT ──
    with tab_assess:
        st.markdown('<div class="panel-header">Test your understanding</div>', unsafe_allow_html=True)

        concept_input = st.text_input(
            "Concept", placeholder="e.g. gradient descent...", key="assess_concept")

        if st.button("Get Question →", key="get_q", use_container_width=True):
            if concept_input:
                with st.spinner("Generating..."):
                    q = call_get_question(concept_input)
                if q:
                    st.session_state.current_question = q
                    st.session_state.current_concept  = concept_input
            else:
                st.warning("Enter a concept first.")

        if st.session_state.current_question:
            q = st.session_state.current_question
            st.markdown("---")
            st.markdown(
                f'<span class="agent-tag tag-assessment">QUESTION</span>'
                f'<div class="message-assistant">{q["question"]}</div>',
                unsafe_allow_html=True)
            st.caption(f"Type: {q.get('question_type','general')} · {q.get('concept','')}")

            answer = st.text_area(
                "Your answer", placeholder="Type your answer...",
                height=120, key="ans")

            if st.button("Submit →", key="submit", use_container_width=True):
                if answer:
                    with st.spinner("Evaluating..."):
                        result = call_evaluate(
                            q.get("concept", concept_input),
                            q["question"], answer,
                            q.get("expected_answer_points", []))
                    if result:
                        score  = result.get("score", 0)
                        passed = result.get("passed", False)
                        color  = "#059669" if passed else "#EF4444"
                        badge  = "✓ PASSED" if passed else "✗ FAILED"

                        st.markdown(
                            f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;'
                            f'border-radius:8px;padding:0.8rem;margin:0.6rem 0">'
                            f'<div style="font-family:Syne,sans-serif;font-size:1.8rem;'
                            f'font-weight:800;color:{color}">{score} '
                            f'<span style="font-size:0.8rem">{badge}</span></div>'
                            f'<div style="font-size:0.7rem;color:#059669;margin-top:0.3rem">'
                            f'✅ {" · ".join(result.get("what_was_right", []))}</div>'
                            f'<div style="font-size:0.7rem;color:#EF4444;margin-top:0.2rem">'
                            f'❌ {" · ".join(result.get("what_was_wrong", []))}</div></div>',
                            unsafe_allow_html=True)

                        st.markdown(
                            f'<span class="agent-tag tag-feedback">FEEDBACK</span>'
                            f'<div class="message-assistant">{result.get("feedback","")}</div>',
                            unsafe_allow_html=True)

                        next_action = result.get("next_action", "")
                        re_teach    = result.get("re_teach_focus", "")

                        if next_action == "advance":
                            st.success("✓ Ready to advance!")
                        elif next_action == "re_teach" and re_teach:
                            st.warning(f"↩ Review: **{re_teach}**")
                            if st.button(f"Re-explain {re_teach}", key="reteach"):
                                st.session_state.messages.append(
                                    {"role": "user", "content": f"Re-explain {re_teach}"})
                                with st.spinner("Re-teaching..."):
                                    r = call_chat(f"Re-explain {re_teach}")
                                st.session_state.messages.append({
                                    "role": "assistant", "content": r["response"],
                                    "agent": "Solver"})
                                st.session_state.current_question = None
                                st.rerun()
                        elif next_action == "practice_more":
                            st.info("Keep practising before moving on.")

                        if st.button("Next Question →", key="next_q"):
                            st.session_state.current_question = None
                            st.rerun()
                else:
                    st.warning("Write your answer first.")

    # ── SETTINGS ──
    with tab_settings:
        st.markdown('<div class="panel-header">Configuration</div>', unsafe_allow_html=True)

        st.caption(f"Your session ID: `{st.session_state.student_id}`")
        st.caption("Each session gets a unique ID. Use the same ID on different devices to share progress.")
        new_id = st.text_input(
            "Override Student ID",
            value=st.session_state.student_id,
            label_visibility="collapsed",
            placeholder="student_1234"
        )
        if new_id != st.session_state.student_id:
            st.session_state.student_id = new_id

        st.markdown("---")
        st.markdown('<div class="panel-header">Response style</div>', unsafe_allow_html=True)
        st.session_state.response_style = st.selectbox(
            "How detailed should responses be?",
            ["Concise", "Balanced", "Detailed"],
            index=["Concise", "Balanced", "Detailed"].index(st.session_state.response_style),
            label_visibility="collapsed"
        )

        st.markdown("---")
        st.markdown('<div class="panel-header">Difficulty level</div>', unsafe_allow_html=True)
        st.session_state.difficulty_override = st.selectbox(
            "Override auto-detected level",
            ["Auto", "Beginner", "Intermediate", "Advanced"],
            index=["Auto", "Beginner", "Intermediate", "Advanced"].index(st.session_state.difficulty_override),
            label_visibility="collapsed"
        )
        st.caption("Auto uses your learning history to set the level.")

        st.markdown("---")
        st.markdown('<div class="panel-header">Export</div>', unsafe_allow_html=True)
        if st.session_state.messages:
            lines = []
            for msg in st.session_state.messages:
                role = "You" if msg["role"] == "user" else msg.get("agent", "Tutor")
                lines.append(f"[{role}]\n{msg['content']}\n")
            chat_text = "\n".join(lines)
            st.download_button(
                label="⬇ Download chat as .txt",
                data=chat_text,
                file_name="mosaic_chat.txt",
                mime="text/plain",
                use_container_width=True
            )
        else:
            st.caption("No chat history to export yet.")

        st.markdown("---")
        st.markdown('<div class="panel-header">Tools & Danger Zone</div>', unsafe_allow_html=True)

        b1, b2, b3 = st.columns(3)
        with b1:
            check_rag  = st.button("🔍 RAG Status",    use_container_width=True)
        with b2:
            clear_pine = st.button("🗑 Clear Pinecone", use_container_width=True)
        with b3:
            debug_pdf  = st.button("🔬 Debug PDF",      use_container_width=True)

        if check_rag:
            try:
                retriever = components["retriever"]
                stats = retriever.index.describe_index_stats()
                namespaces = stats.get("namespaces", {})
                st.write("**Pinecone Index Stats:**")
                if namespaces:
                    for ns, data in namespaces.items():
                        st.write(f"- `{ns}`: {data.get('vector_count', 0)} vectors")
                st.write("**Sources & chunk counts:**")
                dummy = [0.0] * 384
                results = retriever.index.query(
                    vector=dummy, top_k=10000,
                    namespace="knowledge_base", include_metadata=True
                )
                from collections import Counter
                source_counts = Counter(m["metadata"].get("source", "") for m in results["matches"])
                for source, count in sorted(source_counts.items()):
                    st.write(f"- `{source}`: {count} chunks")
            except Exception as e:
                st.error(f"RAG check failed: {e}")

        if clear_pine:
            try:
                retriever = components["retriever"]
                retriever.index.delete(delete_all=True, namespace="knowledge_base")
                st.success("Pinecone cleared — reboot app to re-ingest")
            except Exception as e:
                st.error(f"Clear failed: {e}")

        if debug_pdf:
            try:
                from pypdf import PdfReader
                pdf_path = "docs/FODS Question bank.pdf"
                reader = PdfReader(pdf_path)
                st.write(f"**Pages found:** {len(reader.pages)}")
                total_text = ""
                for page in reader.pages:
                    text = page.extract_text()
                    if text:
                        total_text += text
                words = total_text.split()
                st.write(f"**Total words extracted:** {len(words)}")
                st.write(f"**Expected chunks:** ~{len(words) // 350}")
                st.code(total_text[:200])
            except Exception as e:
                st.error(f"Debug failed: {e}")

# ══════════════════════════════════════════════════════
# RIGHT — Chat output
# ══════════════════════════════════════════════════════
with col_right:

    st.markdown('<div class="panel-header">Conversation</div>', unsafe_allow_html=True)

    if not st.session_state.messages:
        st.markdown(
            '<div style="padding:2rem 0;color:#94A3B8;font-size:0.78rem;text-align:center">'
            'Ask a question to get started.</div>',
            unsafe_allow_html=True)
    else:
        messages = st.session_state.messages

        # Handle unpaired last message (user sent, no reply yet)
        if len(messages) % 2 == 1:
            render_message(messages[-1])

        # Render pairs newest first, user above assistant within each pair
        for i in range(len(messages) - 2, -1, -2):
            render_message(messages[i])    # user prompt
            render_message(messages[i+1]) # assistant reply

    # ── EVALUATION ──
    with tab_eval:
        st.markdown('<div class="panel-header">RAGAs Evaluation</div>', unsafe_allow_html=True)
        st.caption("Evaluates RAG pipeline quality using 4 RAGAs metrics. Uses Groq as judge — no extra cost.")

        st.markdown("---")
        st.markdown('<div class="panel-header">Metrics explained</div>', unsafe_allow_html=True)
        st.markdown("""
        <div style="font-size:0.72rem;color:#475569;line-height:2">
        <b>Faithfulness</b> — does the answer stay grounded in retrieved chunks?<br>
        <b>Answer Relevancy</b> — does the answer actually address the question asked?<br>
        <b>Context Precision</b> — are the retrieved chunks relevant to the question?<br>
        <b>Context Recall</b> — do the chunks contain enough info to answer correctly?
        </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        n_questions = st.slider(
            "Number of questions to evaluate",
            min_value=5, max_value=40, value=10, step=5,
            help="More questions = more accurate results but takes longer to run"
        )

        unit_filter = st.selectbox(
            "Filter by unit",
            ["All Units", "Unit I", "Unit II", "Unit III", "Unit IV"],
            label_visibility="collapsed"
        )

        if st.button("▶ Run RAGAs Evaluation", use_container_width=True):
            try:
                import traceback

                # ── Load questions from evaluation/test_dataset.py ──
                import sys
                sys.path.append(".")
                from evaluation.test_dataset import DATASET as ALL_QUESTIONS


                # Apply unit filter
                all_questions = ALL_QUESTIONS
                if unit_filter != "All Units":
                    all_questions = [q for q in all_questions if q["unit"] == unit_filter]

                if not all_questions:
                    st.warning(f"No questions found for {unit_filter}.")
                    st.stop()

                questions_sample = all_questions[:n_questions]
                retriever        = components["retriever"]
                solver           = components["solver"]

                # ── Query Solver + Pinecone for each question ──
                st.info(f"Querying Solver and Pinecone for {len(questions_sample)} questions...")
                progress_bar = st.progress(0)
                status_text  = st.empty()

                eval_questions    = []
                eval_answers      = []
                eval_contexts     = []
                eval_ground_truth = []

                for idx, item in enumerate(questions_sample):
                    q = item["question"]
                    status_text.caption(f"Processing: {q[:60]}...")

                    # Get answer from Solver Agent
                    try:
                        solver_result = solver.explain(
                            student_id=st.session_state.student_id,
                            concept=q,
                            message=q
                        )
                        answer = solver_result if isinstance(solver_result, str) else solver_result.get("response", "")
                    except Exception:
                        answer = ""

                    # Get contexts from Pinecone
                    try:
                        chunks   = retriever.retrieve_for_solver(q)
                        contexts = [c["text"] for c in chunks] if chunks else ["no context retrieved"]
                    except Exception:
                        contexts = ["no context retrieved"]

                    eval_questions.append(q)
                    eval_answers.append(answer)
                    eval_contexts.append(contexts)
                    eval_ground_truth.append(item["ground_truth"])
                    progress_bar.progress((idx + 1) / len(questions_sample))

                status_text.empty()

                # ── Build RAGAs dataset ──
                from datasets import Dataset as HFDataset
                ragas_data = HFDataset.from_dict({
                    "question":    eval_questions,
                    "answer":      eval_answers,
                    "contexts":    eval_contexts,
                    "ground_truth": eval_ground_truth,
                })

                # ── Set up Groq as judge ──
                import os
                from ragas import evaluate
                from ragas.metrics import (
                    faithfulness,
                    answer_relevancy,
                    context_precision,
                    context_recall,
                )
                from ragas.llms import LangchainLLMWrapper
                from langchain_groq import ChatGroq

                groq_api_key = st.secrets.get("GROQ_API_KEY", os.getenv("GROQ_API_KEY", ""))
                groq_llm     = ChatGroq(
                    model="llama-3.3-70b-versatile",
                    api_key=groq_api_key,
                    temperature=0
                )
                ragas_llm = LangchainLLMWrapper(groq_llm)

                # Use BGE embedder already in the project — avoids OpenAI dependency
                from ragas.embeddings import BaseRagasEmbeddings
                from langchain_core.embeddings import Embeddings as LCEmbeddings

                class BGELangchainWrapper(LCEmbeddings):
                    """Wraps our existing BGEEmbedder for RAGAs compatibility."""
                    def __init__(self, bge_embedder):
                        self._embedder = bge_embedder
                    def embed_documents(self, texts):
                        return self._embedder.embed_documents(texts)
                    def embed_query(self, text):
                        return self._embedder.embed_query(text)

                from ragas.embeddings import LangchainEmbeddingsWrapper
                bge_embedder   = components["embedder"]
                ragas_embedder = LangchainEmbeddingsWrapper(BGELangchainWrapper(bge_embedder))

                st.info("Running RAGAs scoring with Groq as judge and BGE as embedder...")
                results = evaluate(
                    ragas_data,
                    metrics=[faithfulness, answer_relevancy, context_precision, context_recall],
                    llm=ragas_llm,
                    embeddings=ragas_embedder
                )

                # ══════════════════════════════════════════════════
                # OVERALL SCORES
                # ══════════════════════════════════════════════════
                st.markdown("---")
                st.markdown('<div class="panel-header">Overall Scores</div>', unsafe_allow_html=True)

                scores = {
                    "Faithfulness":      round(float(results["faithfulness"]), 3),
                    "Answer Relevancy":  round(float(results["answer_relevancy"]), 3),
                    "Context Precision": round(float(results["context_precision"]), 3),
                    "Context Recall":    round(float(results["context_recall"]), 3),
                }
                avg_score = round(sum(scores.values()) / len(scores), 3)

                def score_color(s):
                    return "#059669" if s >= 0.7 else "#F59E0B" if s >= 0.5 else "#EF4444"

                sc1, sc2, sc3, sc4 = st.columns(4)
                for col, (metric, score) in zip([sc1, sc2, sc3, sc4], scores.items()):
                    with col:
                        st.markdown(
                            f'<div style="background:#F8FAFC;border:1px solid #E2E8F0;border-radius:8px;'
                            f'padding:0.7rem;text-align:center">'
                            f'<div style="font-size:1.5rem;font-weight:800;color:{score_color(score)}">{score}</div>'
                            f'<div style="font-size:0.58rem;color:#64748B;margin-top:0.2rem;line-height:1.4">{metric}</div>'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                st.markdown(
                    f'<div style="background:#F0FDF4;border:1px solid #86EFAC;border-radius:8px;'
                    f'padding:0.6rem;text-align:center;margin-top:0.6rem">'
                    f'<span style="font-size:0.7rem;font-weight:700;color:#065F46">AVERAGE SCORE: </span>'
                    f'<span style="font-size:1.1rem;font-weight:800;color:{score_color(avg_score)}">{avg_score}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )

                # ── Score guide ──
                st.markdown("""
                <div style="font-size:0.65rem;color:#94A3B8;margin-top:0.5rem;text-align:center">
                <span style="color:#059669">≥ 0.7 Good</span> &nbsp;·&nbsp;
                <span style="color:#F59E0B">0.5–0.69 Acceptable</span> &nbsp;·&nbsp;
                <span style="color:#EF4444">&lt; 0.5 Needs improvement</span>
                </div>
                """, unsafe_allow_html=True)

                # ══════════════════════════════════════════════════
                # PER QUESTION BREAKDOWN
                # ══════════════════════════════════════════════════
                st.markdown("---")
                st.markdown('<div class="panel-header">Per Question Breakdown</div>', unsafe_allow_html=True)

                import pandas as pd
                df_results = results.to_pandas()

                df_display = pd.DataFrame({
                    "Question":      eval_questions,
                    "Faithfulness":  df_results["faithfulness"].round(3),
                    "Ans Relevancy": df_results["answer_relevancy"].round(3),
                    "Ctx Precision": df_results["context_precision"].round(3),
                    "Ctx Recall":    df_results["context_recall"].round(3),
                    "Model Answer":  eval_answers,
                    "Ground Truth":  eval_ground_truth,
                })

                st.dataframe(
                    df_display,
                    use_container_width=True,
                    height=360,
                    column_config={
                        "Question":      st.column_config.TextColumn(width="large"),
                        "Faithfulness":  st.column_config.NumberColumn(format="%.3f"),
                        "Ans Relevancy": st.column_config.NumberColumn(format="%.3f"),
                        "Ctx Precision": st.column_config.NumberColumn(format="%.3f"),
                        "Ctx Recall":    st.column_config.NumberColumn(format="%.3f"),
                        "Model Answer":  st.column_config.TextColumn(width="large"),
                        "Ground Truth":  st.column_config.TextColumn(width="large"),
                    }
                )

                # ══════════════════════════════════════════════════
                # WEAKEST QUESTIONS
                # ══════════════════════════════════════════════════
                st.markdown("---")
                st.markdown('<div class="panel-header">Weakest Questions</div>', unsafe_allow_html=True)
                st.caption("Bottom 5 questions by average score across all 4 metrics.")

                df_display["Avg Score"] = df_display[
                    ["Faithfulness", "Ans Relevancy", "Ctx Precision", "Ctx Recall"]
                ].mean(axis=1).round(3)

                weakest = df_display.nsmallest(5, "Avg Score")[
                    ["Question", "Avg Score", "Faithfulness", "Ans Relevancy", "Ctx Precision", "Ctx Recall"]
                ]
                st.dataframe(weakest, use_container_width=True)

                # ══════════════════════════════════════════════════
                # DOWNLOAD
                # ══════════════════════════════════════════════════
                st.markdown("---")
                csv = df_display.to_csv(index=False)
                st.download_button(
                    label="⬇ Download full results as .csv",
                    data=csv,
                    file_name="mosaic_ragas_results.csv",
                    mime="text/csv",
                    use_container_width=True
                )

            except ImportError as e:
                st.error(f"Missing package: {e}")
                st.caption("Add these to requirements.txt: ragas, langchain-groq, datasets")
            except Exception as e:
                st.error(f"Evaluation failed: {e}")
                st.code(traceback.format_exc())

