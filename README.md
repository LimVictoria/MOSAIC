# MOSAIC

Multi-agent AI tutor for learning AI engineering concepts.
Built with LLaMA 3.1 70B, Letta memory, RAG (ChromaDB + BGE-large), Neo4j knowledge graph, and Streamlit.
Access @ https://mosaicurriculum.streamlit.app/

---

## Project Structure

```
ai_tutor/
│
├── streamlit_app.py          ← run this to use the tutor
│
├── config.py                 ← all settings (reads from .env)
├── llm_client.py             ← LLaMA via Ollama or Groq API
├── requirements.txt          ← Python dependencies
├── streamlit_requirements.txt← Streamlit specific
├── .env.example              ← copy to .env and fill in keys
├── .gitignore
│
├── agents/
│   ├── solver_agent.py       ← explains concepts step by step
│   ├── assessment_agent.py   ← tests understanding, gives score
│   ├── feedback_agent.py     ← diagnoses right/wrong, decides next step
│   ├── kg_builder_agent.py   ← builds KG from documents (background)
│   └── orchestrator.py       ← LangGraph routing between agents
│
├── memory/
│   └── letta_client.py       ← ONE shared Letta agent per student
│
├── rag/
│   ├── embedder.py           ← BGE-large-en-v1.5
│   ├── retriever.py          ← ChromaDB queries
│   └── ingest.py             ← load docs into ChromaDB
│
├── kg/
│   └── neo4j_client.py       ← Neo4j + Cytoscape JSON export
│
├── api/
│   └── main.py               ← FastAPI backend
│
└── scripts/
    ├── collect_documents.py  ← download free docs + ingest into ChromaDB
    └── build_kg.py           ← run KG Builder Agent to populate Neo4j
```

---

## How the agents work

```
Student message
      ↓
Orchestrator (LangGraph)
      ↓
Solver Agent        — explains using RAG + KG + Letta memory
      ↓
Assessment Agent    — generates question, scores answer
      ↓
Feedback Agent      — diagnoses right/wrong, updates KG colors
      ↓
Decision: re-teach (→ Solver) or advance (→ next concept)
```

All three agents share **one Letta memory agent per student**.
The LLM inside Letta autonomously decides what to remember.

---

## First-time setup

### Step 1 — Copy environment file
```bash
cp .env.example .env
# Edit .env with your actual values
```

### Step 2 — Install dependencies
```bash
pip install -r requirements.txt
pip install -r streamlit_requirements.txt
```

### Step 3 — Start Neo4j
```bash
docker run --name neo4j \
  -p 7474:7474 -p 7687:7687 \
  -e NEO4J_AUTH=neo4j/password \
  neo4j:latest
```

### Step 4 — Start Letta memory server
```bash
pip install letta
letta server
# Runs at http://localhost:8283
```

### Step 5 — Start LLM
```bash
# Option A: Local GPU (needs 40GB VRAM)
ollama pull llama3.1:70b
ollama serve

# Option B: No GPU — use Groq (free at console.groq.com)
# Set LLM_PROVIDER=groq and GROQ_API_KEY in .env
```

### Step 6 — Collect documents and build KG
```bash
python scripts/collect_documents.py   # downloads + ingests docs
python scripts/build_kg.py            # populates Neo4j
```

### Step 7 — Run the application
```bash
# Terminal 1 — backend
uvicorn api.main:app --reload --port 8000

# Terminal 2 — frontend
streamlit run streamlit_app.py
```

Open **http://localhost:8501** to start learning.

---

## Using the tutor

The Streamlit interface has three tabs:

**Chat tab** — talk to the Solver Agent
```
"Explain gradient descent"
"What is backpropagation?"
"How do transformers work?"
```

**Assessment tab** — test your understanding
```
1. Enter a concept name
2. Click "Get Question"
3. Write your answer
4. Submit — see score + Feedback Agent diagnosis
5. Feedback Agent decides: re-teach or advance
```

**Settings tab** — system status and agent info

The **Knowledge Map** (left panel) appears automatically once 2+ concepts are indexed.
Node colors update live as you learn:

| Color  | Meaning               |
|--------|-----------------------|
| Grey   | Not yet studied       |
| Blue   | Currently studying    |
| Yellow | Being assessed        |
| Green  | Mastered              |
| Red    | Needs review          |
| Orange | Prerequisite gap      |

---

## Without a GPU

Set these in your `.env`:
```
LLM_PROVIDER=groq
GROQ_API_KEY=your_key_here
```

Get a free key at **console.groq.com** — Groq runs LLaMA 3.1 70B very fast on their hardware.
