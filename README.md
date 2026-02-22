# MOSAICurriculum

**Memory-Orchestrated Symbolic Agent Intelligent Curriculum**

Memory → Letta Cloud persistent memory  Orchestrated → LangGraph Orchestrator routing all messages  Symbolic → Neo4j Knowledge Graph (symbolic representation of concepts)  Agent → Multi-agent system (Solver, Assessment, Feedback, KG Builder)  Intelligent → LLM-powered reasoning via Groq/llama  Curriculum → The tutoring and learning domain␠␠

Multi-agent AI tutor for learning Data Science and AI/ML concepts.
Built with LLaMA 3.3 70B, Letta Cloud memory, RAG (Pinecone + BGE-small), Neo4j knowledge graph, and Streamlit.

Access @ https://mosaicurriculum.streamlit.app/

---

## Project Structure

```
mosaic/
│
├── streamlit_app.py          ← run this to use the tutor (no backend needed)
│
├── config.py                 ← all settings (reads from Streamlit secrets or .env)
├── llm_client.py             ← LLaMA 3.3 70B via Groq API (Ollama fallback)
├── requirements.txt          ← Python dependencies
├── .env.example              ← copy to .env and fill in keys
├── .gitignore
│
├── agents/
│   ├── solver_agent.py       ← explains concepts step by step
│   ├── assessment_agent.py   ← tests understanding, gives score
│   ├── feedback_agent.py     ← diagnoses right/wrong, decides next step
│   ├── kg_builder_agent.py   ← builds KG from conversations (real-time)
│   └── orchestrator.py       ← LangGraph chat-first routing between agents
│
├── memory/
│   └── letta_client.py       ← Letta Cloud persistent memory per student
│
├── rag/
│   ├── embedder.py           ← BAAI/bge-small-en-v1.5 (384 dimensions)
│   ├── retriever.py          ← Pinecone queries
│   ├── ingest.py             ← chunks documents and upserts to Pinecone
│   └── fetch_docs.py         ← scans docs/ folder and incrementally ingests new files
│
├── kg/
│   └── neo4j_client.py       ← Neo4j AuraDB + Cytoscape JSON export
│
└── docs/                     ← drop PDF/HTML/TXT/MD files here to add to RAG
    └── FODS Question bank.pdf
```

---

## How the agents work

```
Student message
      ↓
Orchestrator (LangGraph) — chat-first routing
      ↓ brief answer + "want to know more?"
      ↓ (user says yes)
Solver Agent        — explains using RAG (Pinecone) + KG + Letta memory
      ↓
Assessment Agent    — generates question, scores answer 0-100
      ↓
Feedback Agent      — diagnoses right/wrong, updates KG colors, decides next step
      ↓
Decision: re-teach (→ Solver) or advance (→ next concept)
```

All three agents share **one Letta Cloud memory agent per student**, keyed by student ID.

### Orchestrator routing priority
```
1. Assessment keywords (test me, quiz me) → redirect to Assessment Tab
2. Pending concept followup (user said yes) → Solver full lesson
3. Casual chat keywords (hi, thanks) → friendly chat
4. LLM classifier → brief answer or chat
```

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
```

### Step 3 — Set up required services

**Neo4j AuraDB** (free cloud instance)
- Create at https://neo4j.com/cloud/aura/
- Copy URI, username, password to secrets

**Letta Cloud** (persistent memory)
- Sign up at https://app.letta.com
- Copy API key to secrets

**Groq API** (LLM inference)
- Sign up at https://console.groq.com
- Copy API key to secrets — free tier available

**Pinecone** (vector database)
- Sign up at https://app.pinecone.io
- Create index named `mosaicurriculum`, dimension `384`, metric `cosine`, region `us-east-1`
- Copy API key to secrets

### Step 4 — Add documents to RAG
```
Drop any PDF, HTML, TXT, or MD files into the docs/ folder.
The app will automatically ingest them into Pinecone on first startup.
Topic area is inferred from filename keywords:
  pandas, numpy, matplotlib → python_data_science
  stats, probability        → statistics_probability
  eda, wrangling, cleaning  → data_wrangling_eda
  anything else             → general
```

### Step 5 — Run locally (optional)
```bash
streamlit run streamlit_app.py
```
Only needed if running locally for development. The live app is already deployed at https://mosaicurriculum.streamlit.app/

---

## Deploying to Streamlit Cloud

Add these secrets in your Streamlit Cloud dashboard:

```toml
GROQ_API_KEY = "your_groq_key"
LETTA_API_KEY = "your_letta_key"
NEO4J_URI = "neo4j+s://xxxxxxxx.databases.neo4j.io"
NEO4J_USER = "neo4j"
NEO4J_PASSWORD = "your_password"
PINECONE_API_KEY = "your_pinecone_key"
```

On first deploy, the app will automatically download and ingest all files in `docs/` into Pinecone. Subsequent deploys skip already-ingested files instantly.

---

## Using the tutor

The Streamlit interface has three tabs on the left panel:

**💬 Chat tab** — talk to the Solver Agent
```
"Explain gradient descent"
"What is a DataFrame?"
"How does K-Means clustering work?"
```
The tutor gives a brief answer first, then offers a deeper explanation if you want one.

**📝 Assessment tab** — test your understanding
```
1. Enter a concept name (e.g. "pandas DataFrame")
2. Click "Get Question"
3. Write your answer
4. Submit — see score + Feedback Agent diagnosis
5. Feedback Agent decides: re-teach or advance
```

**⚙️ Settings tab** — configuration and tools
- Your session ID is auto-generated (`student_XXXX`) — use the same ID on different devices to share progress
- Response style: Concise / Balanced / Detailed
- Difficulty override: Auto / Beginner / Intermediate / Advanced
- Export chat history as `.txt`
- RAG tools: Check RAG Status, Clear Pinecone, Debug PDF

The **Knowledge Graph** (sidebar) appears automatically once 2+ concepts are indexed.
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

## Key design decisions

| Decision | Reason |
|----------|--------|
| Pinecone instead of ChromaDB | ChromaDB is local — data wiped on every Streamlit Cloud redeploy. Pinecone is persistent cloud storage |
| Incremental ingestion | Only new files are processed on startup — safe to redeploy without re-ingesting everything |
| Chat-first routing | Prevents the tutor from launching into full lessons unprompted — student opts in |
| Auto student ID | Each session gets a unique ID so multiple students can use the same deployment independently |
| No FastAPI backend | Streamlit directly calls agents — simpler deployment, no separate backend process needed |
