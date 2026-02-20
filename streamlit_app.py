# streamlit_app.py
# Complete Streamlit frontend for AI Engineering Tutor
# Run with: streamlit run streamlit_app.py

import streamlit as st
import requests
import json
import time
from streamlit_agraph import agraph, Node, Edge, Config

# ─────────────────────────────────────────────────────
# Page configuration
# ─────────────────────────────────────────────────────

st.set_page_config(
    page_title="AI Engineering Tutor",
    page_icon="🧠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────────────
# Custom CSS — dark technical aesthetic
# ─────────────────────────────────────────────────────

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=JetBrains+Mono:wght@300;400;500;700&family=Syne:wght@400;600;800&display=swap');

/* Global */
html, body, [class*="css"] {
    font-family: 'JetBrains Mono', monospace;
    background-color: #0A0A0F;
    color: #E2E8F0;
}

/* Hide Streamlit default elements */
#MainMenu, footer, header { visibility: hidden; }
.stDeployButton { display: none; }

/* Main background */
.stApp {
    background: linear-gradient(135deg, #0A0A0F 0%, #0F0F1A 50%, #0A0F0A 100%);
}

/* Title */
.tutor-title {
    font-family: 'Syne', sans-serif;
    font-size: 1.8rem;
    font-weight: 800;
    background: linear-gradient(90deg, #00FF94, #00D4FF);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    letter-spacing: -0.02em;
    margin-bottom: 0;
}

.tutor-subtitle {
    font-size: 0.7rem;
    color: #4B5563;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    margin-top: 0.1rem;
}

/* Chat messages */
.message-wrapper {
    margin-bottom: 1rem;
}

.message-user {
    background: linear-gradient(135deg, #1A2F1A, #0F1F2F);
    border: 1px solid #1E3A1E;
    border-radius: 12px 12px 0 12px;
    padding: 0.8rem 1rem;
    margin-left: 2rem;
    font-size: 0.85rem;
    color: #A7F3D0;
}

.message-assistant {
    background: linear-gradient(135deg, #0F1F2F, #1A1A2F);
    border: 1px solid #1E2A3E;
    border-radius: 12px 12px 12px 0;
    padding: 0.8rem 1rem;
    margin-right: 2rem;
    font-size: 0.85rem;
    color: #BFDBFE;
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

.tag-solver    { background: #1E3A5F; color: #60A5FA; border: 1px solid #2563EB; }
.tag-assessment{ background: #3B1F0A; color: #FCD34D; border: 1px solid #D97706; }
.tag-feedback  { background: #1F0A3B; color: #C084FC; border: 1px solid #7C3AED; }
.tag-system    { background: #0F2010; color: #4ADE80; border: 1px solid #16A34A; }

/* Score display */
.score-display {
    display: flex;
    align-items: center;
    gap: 1rem;
    background: #0F1A0F;
    border: 1px solid #166534;
    border-radius: 8px;
    padding: 0.8rem 1rem;
    margin-top: 0.5rem;
}

.score-number {
    font-family: 'Syne', sans-serif;
    font-size: 2rem;
    font-weight: 800;
    color: #00FF94;
}

.score-label {
    font-size: 0.7rem;
    color: #4B5563;
    text-transform: uppercase;
    letter-spacing: 0.1em;
}

/* Progress bar */
.progress-container {
    background: #111827;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 1rem;
    margin-bottom: 1rem;
}

.progress-bar-bg {
    background: #1F2937;
    border-radius: 4px;
    height: 6px;
    margin-top: 0.5rem;
}

.progress-bar-fill {
    background: linear-gradient(90deg, #00FF94, #00D4FF);
    border-radius: 4px;
    height: 6px;
    transition: width 0.5s ease;
}

/* KG Legend */
.kg-legend {
    display: flex;
    flex-wrap: wrap;
    gap: 0.5rem;
    margin-bottom: 0.8rem;
}

.legend-item {
    display: flex;
    align-items: center;
    gap: 0.3rem;
    font-size: 0.65rem;
    color: #6B7280;
}

.legend-dot {
    width: 8px;
    height: 8px;
    border-radius: 50%;
    flex-shrink: 0;
}

/* Concept info panel */
.concept-panel {
    background: #0F1117;
    border: 1px solid #1F2937;
    border-radius: 8px;
    padding: 0.8rem;
    margin-top: 0.5rem;
}

.concept-name {
    font-family: 'Syne', sans-serif;
    font-size: 1rem;
    font-weight: 700;
    color: #F9FAFB;
}

.concept-meta {
    display: flex;
    gap: 0.5rem;
    margin-top: 0.3rem;
    flex-wrap: wrap;
}

.meta-badge {
    font-size: 0.6rem;
    padding: 0.15rem 0.4rem;
    border-radius: 3px;
    text-transform: uppercase;
    letter-spacing: 0.05em;
}

/* Input area */
.stTextInput input {
    background: #111827 !important;
    border: 1px solid #1F2937 !important;
    border-radius: 8px !important;
    color: #E2E8F0 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.85rem !important;
}

.stTextInput input:focus {
    border-color: #00FF94 !important;
    box-shadow: 0 0 0 1px #00FF94 !important;
}

/* Buttons */
.stButton button {
    background: linear-gradient(135deg, #064E3B, #065F46) !important;
    border: 1px solid #00FF94 !important;
    color: #00FF94 !important;
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    font-weight: 700 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
    border-radius: 6px !important;
    transition: all 0.2s !important;
}

.stButton button:hover {
    background: linear-gradient(135deg, #065F46, #047857) !important;
    box-shadow: 0 0 12px rgba(0, 255, 148, 0.3) !important;
}

/* Sidebar */
.css-1d391kg, [data-testid="stSidebar"] {
    background: #0A0A0F !important;
    border-right: 1px solid #1F2937 !important;
}

/* Section headers */
.section-header {
    font-size: 0.65rem;
    font-weight: 700;
    letter-spacing: 0.15em;
    text-transform: uppercase;
    color: #4B5563;
    border-bottom: 1px solid #1F2937;
    padding-bottom: 0.4rem;
    margin-bottom: 0.8rem;
}

/* Status indicators */
.status-dot {
    display: inline-block;
    width: 6px;
    height: 6px;
    border-radius: 50%;
    margin-right: 0.3rem;
}
.status-online { background: #00FF94; box-shadow: 0 0 6px #00FF94; }
.status-offline { background: #4B5563; }

/* Tabs */
.stTabs [data-baseweb="tab-list"] {
    background: #0F1117 !important;
    border-bottom: 1px solid #1F2937 !important;
}

.stTabs [data-baseweb="tab"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #4B5563 !important;
    letter-spacing: 0.1em !important;
    text-transform: uppercase !important;
}

.stTabs [aria-selected="true"] {
    color: #00FF94 !important;
    border-bottom-color: #00FF94 !important;
}

/* Expander */
.streamlit-expanderHeader {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #6B7280 !important;
    background: #0F1117 !important;
    border: 1px solid #1F2937 !important;
    border-radius: 6px !important;
}

/* Metric */
[data-testid="metric-container"] {
    background: #0F1117 !important;
    border: 1px solid #1F2937 !important;
    border-radius: 8px !important;
    padding: 0.8rem !important;
}

/* Divider */
hr { border-color: #1F2937 !important; }

/* Scrollbar */
::-webkit-scrollbar { width: 4px; }
::-webkit-scrollbar-track { background: #0A0A0F; }
::-webkit-scrollbar-thumb { background: #1F2937; border-radius: 2px; }
::-webkit-scrollbar-thumb:hover { background: #374151; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────────────
# Constants
# ─────────────────────────────────────────────────────

API_BASE = "http://localhost:8000"

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
    "Solver":     ("SOLVER",     "tag-solver"),
    "Assessment": ("ASSESS",     "tag-assessment"),
    "Feedback":   ("FEEDBACK",   "tag-feedback"),
    "System":     ("SYSTEM",     "tag-system"),
}

# ─────────────────────────────────────────────────────
# Session state initialization
# ─────────────────────────────────────────────────────

def init_session():
    defaults = {
        "messages": [],
        "student_id": "student_001",
        "current_concept": None,
        "assessment_active": False,
        "current_question": None,
        "progress": None,
        "kg_visible": False,
        "kg_data": None,
        "last_kg_refresh": 0,
        "api_online": False,
    }
    for key, val in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = val

init_session()

# ─────────────────────────────────────────────────────
# API helpers
# ─────────────────────────────────────────────────────

def check_api():
    try:
        r = requests.get(f"{API_BASE}/api/kg/status", timeout=2)
        return r.status_code == 200
    except Exception:
        return False


def fetch_kg():
    try:
        r = requests.get(f"{API_BASE}/api/kg/graph", timeout=5)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def fetch_kg_status():
    try:
        r = requests.get(f"{API_BASE}/api/kg/status", timeout=3)
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return {"visible": False, "node_count": 0}


def fetch_progress():
    try:
        r = requests.get(
            f"{API_BASE}/api/progress/{st.session_state.student_id}",
            timeout=5
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def send_chat(message: str):
    try:
        r = requests.post(
            f"{API_BASE}/api/chat",
            json={
                "student_id": st.session_state.student_id,
                "message": message
            },
            timeout=60
        )
        if r.status_code == 200:
            return r.json()
    except Exception as e:
        return {"response": f"Connection error: {e}", "agent": "System"}
    return None


def get_question(concept: str):
    try:
        r = requests.post(
            f"{API_BASE}/api/assessment/question",
            params={
                "student_id": st.session_state.student_id,
                "concept": concept
            },
            timeout=30
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None


def submit_answer(concept: str, question: str, answer: str, expected: list):
    try:
        r = requests.post(
            f"{API_BASE}/api/assessment/evaluate",
            json={
                "student_id": st.session_state.student_id,
                "concept": concept,
                "question": question,
                "answer": answer,
                "expected_points": expected
            },
            timeout=60
        )
        if r.status_code == 200:
            return r.json()
    except Exception:
        pass
    return None

# ─────────────────────────────────────────────────────
# KG Visualization with streamlit-agraph
# ─────────────────────────────────────────────────────

def render_kg(kg_data: dict):
    """Render knowledge graph using streamlit-agraph."""

    if not kg_data or not kg_data.get("elements"):
        st.info("Knowledge graph is building...")
        return

    elements = kg_data["elements"]
    nodes_data = elements.get("nodes", [])
    edges_data = elements.get("edges", [])

    if len(nodes_data) <= 1:
        st.caption("⏳ Waiting for more concepts to be indexed...")
        return

    # Build agraph nodes
    nodes = []
    for n in nodes_data:
        d = n["data"]
        status = d.get("status", "grey")
        color = STATUS_COLORS.get(status, "#374151")

        # Size by difficulty
        difficulty = d.get("difficulty", "intermediate")
        size = {"beginner": 15, "intermediate": 20, "advanced": 25}.get(difficulty, 20)

        nodes.append(Node(
            id=d["id"],
            label=d["label"],
            size=size,
            color=color,
            title=f"{d['label']}\n{STATUS_LABELS.get(status, status)}\nDifficulty: {difficulty}\nTopic: {d.get('topic_area', '')}",
            font={"color": "#E2E8F0", "size": 10, "face": "JetBrains Mono"}
        ))

    # Build agraph edges
    edges = []
    for e in edges_data:
        d = e["data"]
        rel = d.get("relationship", "RELATED_TO")
        edge_colors = {
            "REQUIRES":   "#EF4444",
            "BUILDS_ON":  "#3B82F6",
            "PART_OF":    "#10B981",
            "USED_IN":    "#F59E0B",
            "RELATED_TO": "#6B7280",
        }
        edges.append(Edge(
            source=d["source"],
            target=d["target"],
            label=rel.replace("_", " ").lower(),
            color=edge_colors.get(rel, "#6B7280"),
            arrows="to",
            font={"size": 8, "color": "#6B7280"}
        ))

    # Agraph config
    config = Config(
        width=400,
        height=450,
        directed=True,
        physics=True,
        hierarchical=False,
        nodeHighlightBehavior=True,
        highlightColor="#00FF94",
        collapsible=False,
        node={"labelProperty": "label"},
        link={"labelProperty": "label", "renderLabel": True},
        d3={"gravity": -300, "linkLength": 120}
    )

    agraph(nodes=nodes, edges=edges, config=config)


# ─────────────────────────────────────────────────────
# Message rendering
# ─────────────────────────────────────────────────────

def render_message(msg: dict):
    role = msg["role"]
    content = msg["content"]
    agent = msg.get("agent", "System")

    if role == "user":
        st.markdown(
            f'<div class="message-wrapper">'
            f'<div class="message-user">{content}</div>'
            f'</div>',
            unsafe_allow_html=True
        )
    else:
        tag_text, tag_class = AGENT_TAGS.get(agent, ("SYSTEM", "tag-system"))
        st.markdown(
            f'<div class="message-wrapper">'
            f'<span class="agent-tag {tag_class}">{tag_text}</span>'
            f'<div class="message-assistant">{content}</div>'
            f'</div>',
            unsafe_allow_html=True
        )


def render_score(result: dict):
    score = result.get("score", 0)
    passed = result.get("passed", False)
    color = "#00FF94" if passed else "#EF4444"
    label = "PASSED" if passed else "FAILED"

    st.markdown(
        f'<div class="score-display">'
        f'<div>'
        f'<div class="score-number" style="color:{color}">{score}</div>'
        f'<div class="score-label">{label}</div>'
        f'</div>'
        f'<div style="flex:1">'
        f'<div style="font-size:0.75rem;color:#9CA3AF;margin-bottom:0.3rem">What was right</div>'
        f'<div style="font-size:0.75rem;color:#A7F3D0">{"<br>".join(result.get("what_was_right", []))}</div>'
        f'</div>'
        f'<div style="flex:1">'
        f'<div style="font-size:0.75rem;color:#9CA3AF;margin-bottom:0.3rem">What was wrong</div>'
        f'<div style="font-size:0.75rem;color:#FECACA">{"<br>".join(result.get("what_was_wrong", []))}</div>'
        f'</div>'
        f'</div>',
        unsafe_allow_html=True
    )


# ─────────────────────────────────────────────────────
# Main Layout
# ─────────────────────────────────────────────────────

# Check API status
st.session_state.api_online = check_api()

# Header
col_title, col_status = st.columns([4, 1])
with col_title:
    st.markdown('<div class="tutor-title">AI Engineering Tutor</div>', unsafe_allow_html=True)
    st.markdown('<div class="tutor-subtitle">Multi-Agent Learning System</div>', unsafe_allow_html=True)

with col_status:
    status_class = "status-online" if st.session_state.api_online else "status-offline"
    status_text = "Online" if st.session_state.api_online else "Offline"
    st.markdown(
        f'<div style="text-align:right;padding-top:0.5rem;font-size:0.7rem;color:#6B7280;">'
        f'<span class="status-dot {status_class}"></span>{status_text}</div>',
        unsafe_allow_html=True
    )

st.markdown("---")

# Main columns — KG left, Chat right
col_kg, col_chat = st.columns([1, 1.6], gap="large")

# ──────────────────────────────────
# LEFT COLUMN — Knowledge Map
# ──────────────────────────────────
with col_kg:
    st.markdown('<div class="section-header">Knowledge Map</div>', unsafe_allow_html=True)

    # Refresh KG every 5 seconds
    now = time.time()
    if now - st.session_state.last_kg_refresh > 5:
        kg_status = fetch_kg_status()
        st.session_state.kg_visible = kg_status.get("visible", False)
        node_count = kg_status.get("node_count", 0)

        if st.session_state.kg_visible:
            st.session_state.kg_data = fetch_kg()

        st.session_state.last_kg_refresh = now

    # Show KG or loading state
    if st.session_state.kg_visible and st.session_state.kg_data:
        # Legend
        st.markdown('<div class="kg-legend">', unsafe_allow_html=True)
        legend_cols = st.columns(3)
        for i, (status, label) in enumerate(STATUS_LABELS.items()):
            with legend_cols[i % 3]:
                st.markdown(
                    f'<div class="legend-item">'
                    f'<div class="legend-dot" style="background:{STATUS_COLORS[status]}"></div>'
                    f'<span style="font-size:0.6rem;color:#6B7280">{label}</span>'
                    f'</div>',
                    unsafe_allow_html=True
                )
        st.markdown('</div>', unsafe_allow_html=True)

        # Render graph
        render_kg(st.session_state.kg_data)

        # Node count
        node_count = st.session_state.kg_data.get("node_count", 0)
        st.caption(f"📊 {node_count} concepts indexed")

        # Edge legend
        with st.expander("Edge types"):
            st.markdown("""
            <div style="font-size:0.65rem;color:#6B7280">
            <div>🔴 <b>REQUIRES</b> — must know first</div>
            <div>🔵 <b>BUILDS ON</b> — extends concept</div>
            <div>🟢 <b>PART OF</b> — component of</div>
            <div>🟡 <b>USED IN</b> — applied in</div>
            <div>⚪ <b>RELATED TO</b> — connected</div>
            </div>
            """, unsafe_allow_html=True)

    else:
        st.markdown("""
        <div style="text-align:center;padding:3rem 1rem;color:#374151;border:1px dashed #1F2937;border-radius:8px">
            <div style="font-size:2rem;margin-bottom:0.5rem">🕸️</div>
            <div style="font-family:'Syne',sans-serif;font-size:0.9rem;color:#4B5563">
                Building knowledge graph...
            </div>
            <div style="font-size:0.7rem;color:#374151;margin-top:0.3rem">
                Will appear when 2+ concepts indexed
            </div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("---")

    # Progress section
    st.markdown('<div class="section-header">Progress</div>', unsafe_allow_html=True)

    # Refresh progress
    if st.session_state.api_online:
        progress = fetch_progress()
        if progress:
            st.session_state.progress = progress

    if st.session_state.progress:
        p = st.session_state.progress
        pct = p.get("progress_percent", 0)
        mastered = p.get("mastered_count", 0)
        total = p.get("total_concepts", 0)
        level = p.get("current_level", "beginner")
        topic = p.get("current_topic", "")

        # Progress bar
        st.markdown(
            f'<div class="progress-container">'
            f'<div style="display:flex;justify-content:space-between;font-size:0.7rem;color:#6B7280">'
            f'<span>Mastered {mastered}/{total} concepts</span>'
            f'<span style="color:#00FF94;font-weight:700">{pct}%</span>'
            f'</div>'
            f'<div class="progress-bar-bg">'
            f'<div class="progress-bar-fill" style="width:{pct}%"></div>'
            f'</div>'
            f'</div>',
            unsafe_allow_html=True
        )

        # Metrics
        m1, m2 = st.columns(2)
        with m1:
            st.metric("Level", level.title())
        with m2:
            st.metric("Current Topic", topic or "—")
    else:
        st.caption("Progress will appear after first interaction")


# ──────────────────────────────────
# RIGHT COLUMN — Chat Interface
# ──────────────────────────────────
with col_chat:
    tab_chat, tab_assess, tab_settings = st.tabs([
        "💬  Chat", "📝  Assessment", "⚙️  Settings"
    ])

    # ── Chat Tab ──
    with tab_chat:
        # Messages container
        messages_container = st.container()

        with messages_container:
            if not st.session_state.messages:
                st.markdown("""
                <div style="text-align:center;padding:2rem;color:#374151">
                    <div style="font-size:1.5rem;margin-bottom:0.5rem">🧠</div>
                    <div style="font-family:'Syne',sans-serif;color:#4B5563;font-size:0.9rem">
                        Start by asking about any AI engineering concept
                    </div>
                    <div style="font-size:0.7rem;color:#374151;margin-top:1rem">
                        Try: "Explain gradient descent" or "What is backpropagation?"
                    </div>
                </div>
                """, unsafe_allow_html=True)
            else:
                for msg in st.session_state.messages:
                    render_message(msg)

        st.markdown("---")

        # Quick prompts
        st.markdown('<div style="font-size:0.65rem;color:#4B5563;letter-spacing:0.1em;text-transform:uppercase;margin-bottom:0.4rem">Quick start</div>', unsafe_allow_html=True)
        qp_cols = st.columns(3)
        quick_prompts = [
            "Explain gradient descent",
            "What is backpropagation?",
            "How do transformers work?",
            "Explain overfitting",
            "What is RAG?",
            "Explain embeddings"
        ]
        for i, prompt in enumerate(quick_prompts):
            with qp_cols[i % 3]:
                if st.button(prompt, key=f"qp_{i}", use_container_width=True):
                    st.session_state.messages.append(
                        {"role": "user", "content": prompt}
                    )
                    with st.spinner("Solver thinking..."):
                        result = send_chat(prompt)
                    if result:
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["response"],
                            "agent": result.get("agent", "Solver")
                        })
                    st.rerun()

        st.markdown("---")

        # Input area
        input_col, btn_col = st.columns([5, 1])
        with input_col:
            user_input = st.text_input(
                "Message",
                placeholder="Ask a question or request an explanation...",
                label_visibility="collapsed",
                key="chat_input"
            )
        with btn_col:
            send_clicked = st.button("Send →", use_container_width=True, key="send_btn")

        # Handle send
        if (send_clicked or user_input) and user_input:
            st.session_state.messages.append(
                {"role": "user", "content": user_input}
            )

            with st.spinner("Agent processing..."):
                result = send_chat(user_input)

            if result:
                st.session_state.messages.append({
                    "role": "assistant",
                    "content": result["response"],
                    "agent": result.get("agent", "Solver")
                })

                # Update concept tracking
                if result.get("agent") == "Solver":
                    st.session_state.current_concept = user_input

            st.rerun()

        # Clear chat button
        if st.session_state.messages:
            if st.button("Clear chat", key="clear_chat"):
                st.session_state.messages = []
                st.rerun()

    # ── Assessment Tab ──
    with tab_assess:
        st.markdown('<div class="section-header">Test Your Understanding</div>', unsafe_allow_html=True)

        # Concept input
        assess_concept = st.text_input(
            "Concept to be assessed on",
            placeholder="e.g. gradient descent, backpropagation...",
            key="assess_concept"
        )

        get_q_btn = st.button("Get Question →", key="get_question")

        if get_q_btn and assess_concept:
            with st.spinner("Assessment Agent generating question..."):
                q_data = get_question(assess_concept)

            if q_data:
                st.session_state.current_question = q_data
                st.session_state.current_concept = assess_concept

        # Show current question
        if st.session_state.current_question:
            q = st.session_state.current_question
            st.markdown("---")

            # Question display
            st.markdown(
                f'<div class="message-assistant">'
                f'<span class="agent-tag tag-assessment">ASSESSMENT</span><br>'
                f'{q["question"]}'
                f'</div>',
                unsafe_allow_html=True
            )

            st.caption(f"Type: {q.get('question_type', 'general')} | Concept: {q.get('concept', '')}")

            # Answer input
            answer = st.text_area(
                "Your answer",
                placeholder="Type your answer here...",
                height=150,
                key="answer_input"
            )

            submit_btn = st.button("Submit Answer →", key="submit_answer")

            if submit_btn and answer:
                with st.spinner("Assessment + Feedback Agents evaluating..."):
                    result = submit_answer(
                        concept=q.get("concept", assess_concept),
                        question=q["question"],
                        answer=answer,
                        expected=q.get("expected_answer_points", [])
                    )

                if result:
                    st.markdown("---")
                    st.markdown('<div class="section-header">Results</div>', unsafe_allow_html=True)

                    # Score display
                    render_score(result)

                    st.markdown("---")

                    # Feedback display
                    st.markdown(
                        f'<div class="message-wrapper">'
                        f'<span class="agent-tag tag-feedback">FEEDBACK</span>'
                        f'<div class="message-assistant">{result.get("feedback", "")}</div>'
                        f'</div>',
                        unsafe_allow_html=True
                    )

                    # Next action
                    next_action = result.get("next_action", "")
                    re_teach = result.get("re_teach_focus", "")

                    if next_action == "advance":
                        st.success("✓ Ready to advance to the next concept!")
                    elif next_action == "re_teach" and re_teach:
                        st.warning(f"↩ Feedback Agent suggests reviewing: **{re_teach}**")
                        if st.button(f"Ask Solver to re-explain {re_teach}", key="reteach_btn"):
                            msg = f"Please re-explain {re_teach} with a focus on {result.get('what_was_wrong', [''])[0]}"
                            st.session_state.messages.append({"role": "user", "content": msg})
                            with st.spinner("Solver re-teaching..."):
                                solver_result = send_chat(msg)
                            if solver_result:
                                st.session_state.messages.append({
                                    "role": "assistant",
                                    "content": solver_result["response"],
                                    "agent": "Solver"
                                })
                            st.session_state.current_question = None
                            st.rerun()
                    elif next_action == "practice_more":
                        st.info("Practice more on this concept before advancing.")

                    # Get another question button
                    if st.button("Get Another Question →", key="next_q"):
                        st.session_state.current_question = None
                        st.rerun()

    # ── Settings Tab ──
    with tab_settings:
        st.markdown('<div class="section-header">Configuration</div>', unsafe_allow_html=True)

        st.text_input(
            "Student ID",
            value=st.session_state.student_id,
            key="student_id_input",
            on_change=lambda: setattr(
                st.session_state, "student_id",
                st.session_state.student_id_input
            )
        )

        st.text_input(
            "API Base URL",
            value=API_BASE,
            key="api_url_input",
            disabled=True
        )

        st.markdown("---")
        st.markdown('<div class="section-header">System Status</div>', unsafe_allow_html=True)

        status_items = {
            "FastAPI Backend": st.session_state.api_online,
            "Knowledge Graph": st.session_state.kg_visible,
            "Chat Active": len(st.session_state.messages) > 0,
        }

        for label, active in status_items.items():
            dot = "status-online" if active else "status-offline"
            text_color = "#00FF94" if active else "#4B5563"
            st.markdown(
                f'<div style="display:flex;align-items:center;gap:0.5rem;'
                f'font-size:0.75rem;color:{text_color};margin-bottom:0.4rem">'
                f'<span class="status-dot {dot}"></span>{label}'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")
        st.markdown('<div class="section-header">Agents</div>', unsafe_allow_html=True)

        agents_info = {
            "Solver Agent":     ("SOLVER",     "tag-solver",     "Explains concepts step by step"),
            "Assessment Agent": ("ASSESS",     "tag-assessment", "Tests your understanding"),
            "Feedback Agent":   ("FEEDBACK",   "tag-feedback",   "Diagnoses what went wrong"),
            "KG Builder":       ("KG-BUILD",   "tag-system",     "Builds knowledge graph (background)"),
        }

        for agent, (tag, cls, desc) in agents_info.items():
            st.markdown(
                f'<div style="margin-bottom:0.6rem">'
                f'<span class="agent-tag {cls}">{tag}</span> '
                f'<span style="font-size:0.7rem;color:#6B7280">{desc}</span>'
                f'</div>',
                unsafe_allow_html=True
            )

        st.markdown("---")

        # Auto-refresh toggle
        auto_refresh = st.toggle("Auto-refresh KG every 5s", value=True)
        if auto_refresh:
            time.sleep(5)
            st.rerun()
