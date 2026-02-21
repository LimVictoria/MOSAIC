# streamlit_app.py
# Complete AI Engineering Tutor — no FastAPI needed
# All agents called directly from Streamlit
# Run: streamlit run streamlit_app.py

import streamlit as st
import json
import time
import os
from streamlit_agraph import agraph, Node, Edge, Config

# ─────────────────────────────────────────────────────
# Page config
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Engineering Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────────────────
# CSS
# ─────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;800&display=swap');

html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #F8FAFC;
    color: #1E293B;
}
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }
.stApp {
    background: linear-gradient(135deg, #F0FDF4 0%, #F0F9FF 50%, #FAFAFA 100%);
}
.tutor-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #059669, #0284C7);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
}
.tutor-subtitle {
    font-size: 0.7rem;
    color: #94A3B8;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.1rem;
}
.message-user {
    background: linear-gradient(135deg, #DCFCE7, #DBEAFE);
    border: 1px solid #BBF7D0;
    border-radius: 12px 12px 0 12px;
    padding: 0.8rem 1rem;
    margin-left: 2rem;
    margin-bottom: 0.8rem;
    font-size: 0.85rem;
    color: #065F46;
}
.message-assistant {
    background: linear-gradient(135deg, #F0F9FF, #EFF6FF);
    border: 1px solid #BAE6FD;
    border-radius: 12px 12px 12px 0;
    padding: 0.8rem 1rem;
    margin-right: 2rem;
    margin-bottom: 0.8rem;
    font-size: 0.85rem;
    color: #1E3A5F;
    white-space: pre-wrap;
}
.agent-tag {
    display: inline-block;
    font-size: 0.6rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    padding: 0.15rem 0.5rem;
    border-radius: 4px;
    margin-bottom: 0.4rem;
}
.tag-solver     { background: #DBEAFE; color: #1D4ED8; border: 1px solid #93C5FD; }
.tag-assessment { background: #FEF3C7; color: #92400E; border: 1px solid #FCD34D; }
.tag-feedback   { background: #F3E8FF; color: #6B21A8; border: 1px solid #D8B4FE; }
.tag-system     { background: #DCFCE7; color: #166534; border: 1px solid #86EFAC; }
.section-header {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #94A3B8;
    border-bottom: 1px solid #E2E8F0;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}
.progress-container {
    background: #FFFFFF;
    border: 1px solid #E2E8F0;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
    box-shadow: 0 1px 3px rgba(0,0,0,0.05);
}
.progress-bar-bg   { background:#E2E8F0; border-radius:4px; height:6px; margin-top:0.5rem; }
.progress-bar-fill { background:linear-gradient(90deg,#059669,#0284C7); border-radius:4px; height:6px; }
.status-dot    { display:inline-block; width:6px; height:6px; border-radius:50%; margin-right:0.3rem; }
.status-online { background:#059669; box-shadow:0 0 6px #059669; }
.status-offline{ background:#CBD5E1; }
.stTextInput input {
    background: #FFFFFF !important;
    border: 1px solid #E2E8F0 !important;
    border-radius: 8px !important;
    color: #1E293B !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}
.stTextInput input:focus { border-color: #059669 !important; }
.stButton button {
    background: linear-gradient(135deg, #DCFCE7, #D1FAE5) !important;
    border: 1px solid #059669 !important;
    color: #065F46 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
}
.stButton button:hover { box-shadow: 0 0 12px rgba(5,150,105,0.2) !important; }
hr { border-color: #E2E8F0 !important; }
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #F8FAFC; }
::-webkit-scrollbar-thumb { background: #CBD5E1; border-radius: 2px; }
</style>
""", unsafe_allow_html=True)

# DEBUG — remove after fixing
import streamlit as st
st.write("NEO4J_URI reading as:", st.secrets.get("NEO4J_URI", "NOT FOUND"))


# ─────────────────────────────────────────────────────
# Load all components once — cached across sessions
# No FastAPI — agents called directly
# ─────────────────────────────────────────────────────

@st.cache_resource
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
        "llm":          llm,
        "retriever":    retriever,
        "neo4j":        neo4j,
        "letta":        letta,
        "solver":       solver,
        "assessment":   assessment,
        "feedback":     feedback,
        "orchestrator": orchestrator,
    }


try:
    components        = load_components()
    COMPONENTS_LOADED = True
    LOAD_ERROR        = ""
except Exception as e:
    COMPONENTS_LOADED = False
    LOAD_ERROR        = str(e)
    components        = {}

# ─────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────

STATUS_COLORS = {
    "grey":   "#374151",
    "blue":   "#1D4ED8",
    "yellow": "#D97706",
    "green":  "#059669",
    "red":    "#DC2626",
    "orange": "#EA580C",
}
STATUS_LABELS = {
    "grey":   "Not reached",
    "blue":   "Learning now",
    "yellow": "Being assessed",
    "green":  "Mastered ✓",
    "red":    "Needs review",
    "orange": "Prereq gap",
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
    "messages":          [],
    "student_id":        "student_001",
    "current_concept":   None,
    "current_question":  None,
    "kg_data":           None,
    "kg_visible":        False,
    "last_kg_refresh":   0,
    "progress":          None,
}.items():
    if key not in st.session_state:
        st.session_state[key] = val

# ─────────────────────────────────────────────────────
# Direct agent calls — no HTTP requests needed
# ─────────────────────────────────────────────────────

def call_chat(message: str) -> dict:
    if not COMPONENTS_LOADED:
        return {"response": f"Error loading components: {LOAD_ERROR}", "agent": "System"}
    try:
        orch     = components["orchestrator"]
        response = orch.route(student_id=st.session_state.student_id, message=message)
        return   {"response": response, "agent": orch.last_agent_used}
    except Exception as e:
        return   {"response": f"Agent error: {e}", "agent": "System"}


def call_get_question(concept: str) -> dict:
    if not COMPONENTS_LOADED:
        return {}
    try:
        return components["assessment"].generate_question(
            student_id=st.session_state.student_id,
            concept=concept
        )
    except Exception as e:
        st.error(f"Question generation error: {e}")
        return {}


def call_evaluate(concept: str, question: str, answer: str, expected: list) -> dict:
    if not COMPONENTS_LOADED:
        return {}
    try:
        result = components["assessment"].evaluate_answer(
            student_id=st.session_state.student_id,
            concept=concept,
            question=question,
            student_answer=answer,
            expected_points=expected
        )
        fb = components["feedback"].give_feedback(
            student_id=st.session_state.student_id,
            concept=concept,
            question=question,
            student_answer=answer,
            assessment_result=result
        )
        return {
            "score":          result["score"],
            "passed":         result["passed"],
            "feedback":       fb["feedback_text"],
            "what_was_right": fb["what_was_right"],
            "what_was_wrong": fb["what_was_wrong"],
            "next_action":    fb["next_action"],
            "re_teach_focus": fb["re_teach_focus"]
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
        letta       = components["letta"]
        neo4j       = components["neo4j"]
        mastered    = letta.get_mastered_concepts(st.session_state.student_id)
        core        = letta.read_core_memory(st.session_state.student_id)
        total_nodes = neo4j.get_node_count()
        return {
            "current_level":    core.get("current_level", "beginner"),
            "current_topic":    core.get("current_topic", ""),
            "mastered_count":   len(mastered),
            "total_concepts":   total_nodes,
            "progress_percent": round(len(mastered) / total_nodes * 100)
                                if total_nodes > 0 else 0,
        }
    except Exception:
        return None

# ─────────────────────────────────────────────────────
# KG render
# ─────────────────────────────────────────────────────

def render_kg(kg_data: dict):
    elements   = kg_data.get("elements", {})
    nodes_data = elements.get("nodes", [])
    edges_data = elements.get("edges", [])

    if len(nodes_data) <= 1:
        st.caption("⏳ Waiting for more concepts to be indexed...")
        return

    nodes = []
    for n in nodes_data:
        d      = n["data"]
        status = d.get("status", "grey")
        size   = {"beginner": 15, "intermediate": 20, "advanced": 25}.get(
                 d.get("difficulty", "intermediate"), 20)
        nodes.append(Node(
            id=d["id"], label=d["label"], size=size,
            color=STATUS_COLORS.get(status, "#374151"),
            title=f"{d['label']}\n{STATUS_LABELS.get(status, status)}\n{d.get('difficulty','')}\n{d.get('topic_area','')}",
            font={"color": "#E2E8F0", "size": 10, "face": "JetBrains Mono"}
        ))

    edge_colors = {"REQUIRES": "#EF4444", "BUILDS_ON": "#3B82F6",
                   "PART_OF": "#10B981", "USED_IN": "#F59E0B", "RELATED_TO": "#6B7280"}
    edges = [
        Edge(
            source=e["data"]["source"], target=e["data"]["target"],
            label=e["data"].get("relationship","").replace("_"," ").lower(),
            color=edge_colors.get(e["data"].get("relationship","RELATED_TO"), "#6B7280"),
            arrows="to"
        ) for e in edges_data
    ]

    agraph(nodes=nodes, edges=edges, config=Config(
        width=420, height=460, directed=True, physics=True,
        hierarchical=False, nodeHighlightBehavior=True,
        highlightColor="#00FF94",
        d3={"gravity": -300, "linkLength": 120}
    ))

# ─────────────────────────────────────────────────────
# Message render
# ─────────────────────────────────────────────────────

def render_message(msg: dict):
    if msg["role"] == "user":
        st.markdown(f'<div class="message-user">{msg["content"]}</div>', unsafe_allow_html=True)
    else:
        tag, cls = AGENT_TAGS.get(msg.get("agent","System"), ("SYSTEM","tag-system"))
        st.markdown(
            f'<span class="agent-tag {cls}">{tag}</span>'
            f'<div class="message-assistant">{msg["content"]}</div>',
            unsafe_allow_html=True
        )

# ─────────────────────────────────────────────────────
# Layout
# ─────────────────────────────────────────────────────

# Header
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown('<div class="tutor-title">AI Engineering Tutor</div>', unsafe_allow_html=True)
    st.markdown('<div class="tutor-subtitle">Multi-Agent Learning System</div>', unsafe_allow_html=True)
with col_status:
    dot   = "status-online" if COMPONENTS_LOADED else "status-offline"
    label = "Ready" if COMPONENTS_LOADED else "Error"
    st.markdown(
        f'<div style="text-align:right;padding-top:0.6rem;font-size:0.7rem;color:#6B7280">'
        f'<span class="status-dot {dot}"></span>{label}</div>',
        unsafe_allow_html=True
    )

if not COMPONENTS_LOADED:
    st.error(f"Failed to load: {LOAD_ERROR}")
    st.info("Check that Neo4j, Letta, and your LLM (Groq API key or Ollama) are running and your .env / Streamlit secrets are set.")
    st.stop()

st.markdown("---")

col_kg, col_chat = st.columns([1, 1.6], gap="large")

# ── LEFT — KG ────────────────────────────────────────
with col_kg:
    st.markdown('<div class="section-header">Knowledge Map</div>', unsafe_allow_html=True)

    now = time.time()
    if now - st.session_state.last_kg_refresh > 5:
        kg                               = get_kg_data()
        st.session_state.kg_data         = kg
        st.session_state.kg_visible      = kg.get("visible", False)
        st.session_state.last_kg_refresh = now

    if st.session_state.kg_visible and st.session_state.kg_data:
        # Legend
        lcols = st.columns(3)
        for i, (status, label) in enumerate(STATUS_LABELS.items()):
            with lcols[i % 3]:
                st.markdown(
                    f'<div style="display:flex;align-items:center;gap:0.3rem;margin-bottom:0.3rem">'
                    f'<div style="width:8px;height:8px;border-radius:50%;background:{STATUS_COLORS[status]};flex-shrink:0"></div>'
                    f'<span style="font-size:0.6rem;color:#6B7280">{label}</span></div>',
                    unsafe_allow_html=True
                )
        render_kg(st.session_state.kg_data)
        st.caption(f"📊 {st.session_state.kg_data.get('node_count',0)} concepts indexed")
    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;border:1px dashed #1F2937;border-radius:8px">
            <div style="font-size:2rem">🕸️</div>
            <div style="font-family:'Syne',sans-serif;font-size:0.9rem;color:#4B5563;margin-top:0.5rem">
                Building knowledge graph...
            </div>
            <div style="font-size:0.7rem;color:#374151;margin-top:0.3rem">
                Appears when 2+ concepts indexed
            </div>
        </div>""", unsafe_allow_html=True)

    st.markdown("---")
    st.markdown('<div class="section-header">Progress</div>', unsafe_allow_html=True)

    progress = get_progress()
    if progress:
        pct = progress.get("progress_percent", 0)
        st.markdown(
            f'<div class="progress-container">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#6B7280">'
            f'<span>Mastered {progress["mastered_count"]}/{progress["total_concepts"]} concepts</span>'
            f'<span style="color:#00FF94;font-weight:700">{pct}%</span></div>'
            f'<div class="progress-bar-bg"><div class="progress-bar-fill" style="width:{pct}%"></div></div>'
            f'</div>', unsafe_allow_html=True
        )
        m1, m2 = st.columns(2)
        with m1: st.metric("Level", progress.get("current_level","beginner").title())
        with m2: st.metric("Topic", progress.get("current_topic","") or "—")
    else:
        st.caption("Progress appears after first interaction")

# ── RIGHT — Chat ──────────────────────────────────────
with col_chat:
    tab_chat, tab_assess, tab_settings = st.tabs(["💬  Chat", "📝  Assessment", "⚙️  Settings"])

    # Chat
    with tab_chat:
        if not st.session_state.messages:
            st.markdown("""
            <div style="text-align:center;padding:2rem;color:#374151">
                <div style="font-size:1.5rem">🧠</div>
                <div style="font-family:'Syne',sans-serif;color:#4B5563;font-size:0.9rem;margin-top:0.5rem">
                    Start by asking about any AI engineering concept
                </div>
                <div style="font-size:0.7rem;color:#374151;margin-top:0.8rem">
                    Try: "Explain gradient descent" or "What is backpropagation?"
                </div>
            </div>""", unsafe_allow_html=True)
        else:
            for msg in st.session_state.messages:
                render_message(msg)

        st.markdown("---")

        # Quick prompts
        st.markdown('<div style="font-size:0.65rem;color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem">Quick start</div>', unsafe_allow_html=True)
        qcols = st.columns(3)
        for i, prompt in enumerate(["Explain gradient descent","What is backpropagation?",
                                     "How do transformers work?","Explain overfitting",
                                     "What is RAG?","Explain embeddings"]):
            with qcols[i % 3]:
                if st.button(prompt, key=f"qp_{i}", use_container_width=True):
                    st.session_state.messages.append({"role":"user","content":prompt})
                    with st.spinner("Solver thinking..."):
                        r = call_chat(prompt)
                    st.session_state.messages.append({"role":"assistant","content":r["response"],"agent":r.get("agent","Solver")})
                    st.rerun()

        st.markdown("---")
        ic, bc = st.columns([5, 1])
        with ic:
            user_input = st.text_input("Message", placeholder="Ask a question...",
                                       label_visibility="collapsed", key="chat_input")
        with bc:
            send = st.button("Send →", use_container_width=True, key="send_btn")

        if send and user_input:
            st.session_state.messages.append({"role":"user","content":user_input})
            with st.spinner("Agent processing..."):
                r = call_chat(user_input)
            st.session_state.messages.append({"role":"assistant","content":r["response"],"agent":r.get("agent","Solver")})
            st.rerun()

        if st.session_state.messages:
            if st.button("Clear chat", key="clear"):
                st.session_state.messages = []
                st.rerun()

    # Assessment
    with tab_assess:
        st.markdown('<div class="section-header">Test Your Understanding</div>', unsafe_allow_html=True)

        concept_input = st.text_input("Concept", placeholder="e.g. gradient descent...", key="assess_concept")

        if st.button("Get Question →", key="get_q"):
            if concept_input:
                with st.spinner("Generating question..."):
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
                f'<span class="agent-tag tag-assessment">ASSESSMENT</span>'
                f'<div class="message-assistant">{q["question"]}</div>',
                unsafe_allow_html=True
            )
            st.caption(f"Type: {q.get('question_type','general')} | Concept: {q.get('concept','')}")

            answer = st.text_area("Your answer", placeholder="Type your answer...", height=150, key="ans")

            if st.button("Submit →", key="submit"):
                if answer:
                    with st.spinner("Evaluating..."):
                        result = call_evaluate(
                            q.get("concept", concept_input),
                            q["question"], answer,
                            q.get("expected_answer_points", [])
                        )
                    if result:
                        st.markdown("---")
                        score  = result.get("score", 0)
                        passed = result.get("passed", False)
                        color  = "#00FF94" if passed else "#EF4444"

                        st.markdown(
                            f'<div style="display:flex;align-items:center;gap:1rem;background:#0F1A0F;'
                            f'border:1px solid #166534;border-radius:8px;padding:0.8rem 1rem;margin-bottom:1rem">'
                            f'<div><div style="font-family:Syne,sans-serif;font-size:2rem;font-weight:800;color:{color}">{score}</div>'
                            f'<div style="font-size:0.7rem;color:#4B5563">{"PASSED" if passed else "FAILED"}</div></div>'
                            f'<div style="flex:1"><div style="font-size:0.7rem;color:#6B7280;margin-bottom:0.2rem">What was right</div>'
                            f'<div style="font-size:0.75rem;color:#A7F3D0">{"<br>".join(result.get("what_was_right",[]))}</div></div>'
                            f'<div style="flex:1"><div style="font-size:0.7rem;color:#6B7280;margin-bottom:0.2rem">What was wrong</div>'
                            f'<div style="font-size:0.75rem;color:#FECACA">{"<br>".join(result.get("what_was_wrong",[]))}</div></div>'
                            f'</div>', unsafe_allow_html=True
                        )

                        st.markdown(
                            f'<span class="agent-tag tag-feedback">FEEDBACK</span>'
                            f'<div class="message-assistant">{result.get("feedback","")}</div>',
                            unsafe_allow_html=True
                        )

                        next_action = result.get("next_action","")
                        re_teach    = result.get("re_teach_focus","")

                        if next_action == "advance":
                            st.success("✓ Ready to advance!")
                        elif next_action == "re_teach" and re_teach:
                            st.warning(f"↩ Feedback Agent suggests reviewing: **{re_teach}**")
                            if st.button(f"Re-explain {re_teach}", key="reteach"):
                                st.session_state.messages.append({"role":"user","content":f"Re-explain {re_teach}"})
                                with st.spinner("Solver re-teaching..."):
                                    r = call_chat(f"Re-explain {re_teach}")
                                st.session_state.messages.append({"role":"assistant","content":r["response"],"agent":"Solver"})
                                st.session_state.current_question = None
                                st.rerun()
                        elif next_action == "practice_more":
                            st.info("Practice more before advancing.")

                        if st.button("Get Another Question →", key="next_q"):
                            st.session_state.current_question = None
                            st.rerun()
                else:
                    st.warning("Write your answer first.")

    # Settings
    with tab_settings:
        st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)
        new_id = st.text_input("Student ID", value=st.session_state.student_id)
        if new_id != st.session_state.student_id:
            st.session_state.student_id = new_id

        st.markdown("---")
        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)
        for label, active in {
            "Components loaded": COMPONENTS_LOADED,
            "Knowledge Graph":   st.session_state.kg_visible,
            "Chat history":      len(st.session_state.messages) > 0
        }.items():
            dot   = "status-online" if active else "status-offline"
            color = "#00FF94" if active else "#4B5563"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;font-size:0.75rem;color:{color};margin-bottom:0.4rem">'
                f'<span class="status-dot {dot}"></span>{label}</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown('<div class="section-header">Agents</div>', unsafe_allow_html=True)
        for agent, (tag, cls, desc) in {
            "Solver Agent":     ("SOLVER",   "tag-solver",     "Explains concepts step by step"),
            "Assessment Agent": ("ASSESS",   "tag-assessment", "Tests your understanding"),
            "Feedback Agent":   ("FEEDBACK", "tag-feedback",   "Diagnoses what went wrong"),
            "KG Builder":       ("KG-BUILD", "tag-system",     "Builds knowledge graph"),
        }.items():
            st.markdown(
                f'<div style="margin-bottom:0.6rem"><span class="agent-tag {cls}">{tag}</span> '
                f'<span style="font-size:0.7rem;color:#6B7280">{desc}</span></div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        if st.toggle("Auto-refresh KG every 5s", value=False):
            time.sleep(5)
            st.rerun()
