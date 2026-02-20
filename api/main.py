# api/main.py
# FastAPI backend — all endpoints for Streamlit to call

import asyncio
from fastapi import FastAPI, WebSocket
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

from config import NEO4J_URI
from llm_client import LLMClient
from memory.letta_client import LettaClient
from rag.embedder import BGEEmbedder
from rag.retriever import RAGRetriever
from kg.neo4j_client import Neo4jClient
from agents.solver_agent import SolverAgent
from agents.assessment_agent import AssessmentAgent
from agents.feedback_agent import FeedbackAgent
from agents.orchestrator import Orchestrator

app = FastAPI(title="AI Engineering Tutor API")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"]
)

# ── Initialize all shared components once at startup ──
embedder     = BGEEmbedder()
retriever    = RAGRetriever(embedder)
neo4j        = Neo4jClient()
letta        = LettaClient()
llm          = LLMClient()

# ── One shared Letta memory, three distinct agents ──
solver       = SolverAgent(llm, retriever, neo4j, letta)
assessment   = AssessmentAgent(llm, retriever, neo4j, letta)
feedback     = FeedbackAgent(llm, retriever, neo4j, letta)
orchestrator = Orchestrator(solver, assessment, feedback, neo4j, letta)


# ── Request / Response models ──────────────────────

class ChatRequest(BaseModel):
    student_id: str
    message:    str
    session_id: str = None

class AnswerRequest(BaseModel):
    student_id:      str
    concept:         str
    question:        str
    answer:          str
    expected_points: list = []


# ── Chat ───────────────────────────────────────────

@app.post("/api/chat")
async def chat(request: ChatRequest):
    """
    Main chat endpoint.
    Streamlit sends every student message here.
    Orchestrator routes it to the right agent.
    """
    response = orchestrator.route(
        student_id=request.student_id,
        message=request.message
    )
    return {
        "response": response,
        "agent":    orchestrator.last_agent_used
    }


# ── Assessment ─────────────────────────────────────

@app.post("/api/assessment/question")
async def get_question(student_id: str, concept: str):
    """Generate one assessment question for a concept."""
    return assessment.generate_question(student_id, concept)


@app.post("/api/assessment/evaluate")
async def evaluate_answer(request: AnswerRequest):
    """
    Evaluate a student answer.
    Runs Assessment Agent then Feedback Agent.
    Returns score + full feedback in one response.
    """
    result = assessment.evaluate_answer(
        student_id=request.student_id,
        concept=request.concept,
        question=request.question,
        student_answer=request.answer,
        expected_points=request.expected_points
    )

    fb = feedback.give_feedback(
        student_id=request.student_id,
        concept=request.concept,
        question=request.question,
        student_answer=request.answer,
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


# ── Knowledge Graph ────────────────────────────────

@app.get("/api/kg/status")
async def kg_status():
    """
    Check if KG should be visible.
    Streamlit polls this every 5 seconds.
    Returns visible=True when node_count > 1.
    """
    node_count = neo4j.get_node_count()
    return {
        "visible":    node_count > 1,
        "node_count": node_count
    }


@app.get("/api/kg/graph")
async def kg_graph():
    """
    Full KG in Cytoscape-compatible JSON format.
    Called by Streamlit when rendering the knowledge map.
    """
    return neo4j.to_cytoscape_json()


@app.post("/api/kg/node/update")
async def update_node(concept_name: str, status: str):
    """Manually update a node status. Used for debugging."""
    neo4j.update_node_status(concept_name, status)
    return {"success": True}


# ── Progress ───────────────────────────────────────

@app.get("/api/progress/{student_id}")
async def get_progress(student_id: str):
    """
    Student learning progress summary.
    Shown in the Streamlit sidebar progress bar.
    """
    mastered     = letta.get_mastered_concepts(student_id)
    core_memory  = letta.read_core_memory(student_id)
    total_nodes  = neo4j.get_node_count()

    return {
        "current_level":     core_memory.get("current_level", "beginner"),
        "current_topic":     core_memory.get("current_topic", ""),
        "mastered_count":    len(mastered),
        "total_concepts":    total_nodes,
        "progress_percent":  round(len(mastered) / total_nodes * 100)
                             if total_nodes > 0 else 0,
        "mastered_concepts": mastered
    }


# ── WebSocket — real-time KG updates ──────────────

@app.websocket("/ws/kg/{student_id}")
async def kg_websocket(websocket: WebSocket, student_id: str):
    """
    WebSocket for live KG node color updates.
    Streamlit connects here to receive updates
    without needing to poll.
    """
    await websocket.accept()
    last_count = 0
    try:
        while True:
            current_count = neo4j.get_node_count()
            if current_count != last_count:
                kg_data = neo4j.to_cytoscape_json()
                await websocket.send_json(kg_data)
                last_count = current_count
            await asyncio.sleep(2)
    except Exception:
        pass
