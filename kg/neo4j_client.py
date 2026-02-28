# kg/neo4j_client.py
# Neo4j connection and core operations
# Supports two KGs in the same database:
#   1. FODS Curriculum KG  — Topic + Technique nodes (student learning progress)
#   2. Time Series KG      — PipelineStage, Model, Concept, EvalMetric, etc.

from neo4j import GraphDatabase
from config import KG_VISIBLE_THRESHOLD, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class Neo4jClient:
    """
    Core Neo4j client.

    FODS Curriculum KG (Topic / Technique):
      Status: grey, blue, yellow, green, red, orange
      mastered_at = ISO timestamp when first reached green
      updated_at  = ISO timestamp of last status change

    Time Series KG (PipelineStage, Model, Concept, EvalMetric, etc.):
      Read-only display — no student status tracking on these nodes.
    """

    def __init__(self):
        import streamlit as st

        uri      = st.secrets.get("NEO4J_URI", "")
        user     = st.secrets.get("NEO4J_USER", "neo4j")
        password = st.secrets.get("NEO4J_PASSWORD", "")

        print("NEO4J URI:", uri)
        print("NEO4J USER:", user)
        print("NEO4J PASSWORD SET:", bool(password))

        try:
            self.driver = GraphDatabase.driver(uri, auth=(user, password))
            self.driver.verify_connectivity()
            print(f"Neo4j connected: {uri}")
        except Exception as e:
            raise RuntimeError(f"Neo4j failed to connect: {e}")

    def close(self):
        if self.driver:
            self.driver.close()

    def query(self, cypher: str, params: dict = None) -> list:
        if self.driver is None:
            return []
        try:
            with self.driver.session() as session:
                result = session.run(cypher, params or {})
                return [record.data() for record in result]
        except Exception as e:
            print(f"Query error: {e}")
            return []

    # ── Status updates ─────────────────────────────

    def update_topic_status(self, topic_name: str, status: str):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MATCH (t:Topic {name: $name})
        SET t.status = $status,
            t.updated_at = $now
        WITH t
        WHERE $status = 'green' AND t.mastered_at IS NULL
        SET t.mastered_at = $now
        RETURN t
        """
        self.query(cypher, {"name": topic_name, "status": status, "now": now})

    def update_technique_status(self, technique_name: str, status: str):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MATCH (t:Technique {name: $name})
        WHERE coalesce(t.kg, 'fods') = 'fods'
        SET t.status = $status,
            t.updated_at = $now
        WITH t
        WHERE $status = 'green' AND t.mastered_at IS NULL
        SET t.mastered_at = $now
        RETURN t
        """
        self.query(cypher, {"name": technique_name, "status": status, "now": now})

    def update_node_status(self, name: str, status: str):
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MATCH (n)
        WHERE (n:Topic OR n:Technique)
          AND n.name = $name
          AND coalesce(n.kg, 'fods') = 'fods'
        SET n.status = $status,
            n.updated_at = $now
        WITH n
        WHERE $status = 'green' AND n.mastered_at IS NULL
        SET n.mastered_at = $now
        RETURN n
        """
        self.query(cypher, {"name": name, "status": status, "now": now})

    # ── Prerequisite checks ────────────────────────

    def get_prerequisites(self, topic_name: str) -> list[str]:
        cypher = """
        MATCH (t:Topic {name: $name})-[:PREREQUISITE*1..3]->(pre:Topic)
        RETURN pre.name as name, pre.status as status
        ORDER BY pre.name
        """
        results = self.query(cypher, {"name": topic_name})
        return [r["name"] for r in results]

    def get_unmastered_prerequisites(self, topic_name: str) -> list[str]:
        cypher = """
        MATCH (t:Topic {name: $name})-[:PREREQUISITE*1..3]->(pre:Topic)
        WHERE coalesce(pre.status, 'grey') <> 'green'
        RETURN pre.name as name, coalesce(pre.status, 'grey') as status
        ORDER BY pre.name
        """
        results = self.query(cypher, {"name": topic_name})
        return [r["name"] for r in results]

    def get_prerequisite_chain_for_feedback(self, topic_name: str) -> list[dict]:
        cypher = """
        MATCH path = (t:Topic {name: $name})-[:PREREQUISITE*]->(pre:Topic)
        RETURN pre.name as name,
               coalesce(pre.status, 'grey') as status,
               length(path) as depth
        ORDER BY depth
        """
        return self.query(cypher, {"name": topic_name})

    # ── Curriculum structure reads ─────────────────

    def get_curriculum_structure(self) -> list[dict]:
        cypher = """
        MATCH (t:Topic)
        WHERE coalesce(t.kg, 'fods') = 'fods'
        OPTIONAL MATCH (t)-[:PREREQUISITE]->(pre:Topic)
        WHERE coalesce(pre.kg, 'fods') = 'fods'
        RETURN t.name as topic,
               coalesce(t.status, 'grey') as status,
               collect(pre.name) as prerequisites
        ORDER BY t.name
        """
        return self.query(cypher)

    def get_topic_techniques(self, topic_name: str) -> list[dict]:
        cypher = """
        MATCH (t:Topic {name: $name})-[]->(tech:Technique)
        WHERE coalesce(t.kg, 'fods') = 'fods'
          AND coalesce(tech.kg, 'fods') = 'fods'
        RETURN tech.name as name,
               coalesce(tech.status, 'grey') as status
        ORDER BY tech.name
        """
        return self.query(cypher, {"name": topic_name})

    def get_related_topics(self, topic_name: str) -> list[str]:
        cypher = """
        MATCH (t:Topic {name: $name})-[r]-(related:Topic)
        RETURN related.name as name
        LIMIT 5
        """
        results = self.query(cypher, {"name": topic_name})
        return [r["name"] for r in results]

    def get_learning_path(self, target_topic: str) -> list[str]:
        cypher = """
        MATCH path = (start:Topic)-[:PREREQUISITE*]->(target:Topic {name: $name})
        WHERE NOT (start)<-[:PREREQUISITE]-()
        RETURN [node in nodes(path) | node.name] as path
        ORDER BY length(path) DESC
        LIMIT 1
        """
        results = self.query(cypher, {"name": target_topic})
        if results and results[0]["path"]:
            return results[0]["path"]
        return [target_topic]

    def get_next_recommended_topic(self) -> str:
        cypher = """
        MATCH (t:Topic)
        WHERE coalesce(t.kg, 'fods') = 'fods'
          AND coalesce(t.status, 'grey') <> 'green'
          AND NOT EXISTS {
              MATCH (t)-[:PREREQUISITE]->(pre:Topic)
              WHERE coalesce(pre.status, 'grey') <> 'green'
          }
        RETURN t.name as name
        ORDER BY t.name
        LIMIT 1
        """
        results = self.query(cypher)
        if results:
            return results[0]["name"]
        return "Python for Data Science"

    def map_concept_to_topic(self, concept_name: str) -> str:
        cypher = """
        MATCH (t:Topic)
        WHERE toLower(t.name) CONTAINS toLower($name)
           OR toLower($name) CONTAINS toLower(t.name)
        RETURN t.name as name
        LIMIT 1
        """
        results = self.query(cypher, {"name": concept_name})
        if results:
            return results[0]["name"]

        cypher2 = """
        MATCH (tech:Technique)
        WHERE coalesce(tech.kg, 'fods') = 'fods'
          AND (toLower(tech.name) CONTAINS toLower($name)
           OR toLower($name) CONTAINS toLower(tech.name))
        RETURN tech.name as name
        LIMIT 1
        """
        results2 = self.query(cypher2, {"name": concept_name})
        if results2:
            return results2[0]["name"]

        return None

    # ── KG visibility ──────────────────────────────

    def get_node_count(self) -> int:
        """FODS KG: total Topic + Technique nodes."""
        result = self.query("""
            MATCH (n)
            WHERE (n:Topic OR n:Technique)
              AND coalesce(n.kg, 'fods') = 'fods'
            RETURN count(n) as count
        """)
        return result[0]["count"] if result else 0

    def get_ts_node_count(self) -> int:
        """Time Series KG: total node count."""
        result = self.query("""
            MATCH (n)
            WHERE n:PipelineStage OR n:Model OR n:EvalMetric OR n:BestPractice
               OR n:AntiPattern OR n:LearningPath OR n:PredictionType
               OR (n:Concept AND coalesce(n.kg, 'timeseries') = 'timeseries')
               OR (n:UseCase AND coalesce(n.kg, 'timeseries') = 'timeseries')
            RETURN count(n) as count
        """)
        return result[0]["count"] if result else 0

    def is_kg_visible(self) -> bool:
        return self.get_node_count() > KG_VISIBLE_THRESHOLD

    def get_mastered_concepts(self, student_id: str = None) -> list[str]:
        results = self.query("""
            MATCH (t:Topic {status: 'green'})
            WHERE coalesce(t.kg, 'fods') = 'fods'
            RETURN t.name as name
        """)
        return [r["name"] for r in results]

    # ── Temporal — learning progression ───────────

    def get_mastery_timeline(self) -> list[dict]:
        return self.query("""
            MATCH (n)
            WHERE (n:Topic OR n:Technique)
              AND coalesce(n.kg, 'fods') = 'fods'
              AND n.status = 'green'
              AND n.mastered_at IS NOT NULL
            RETURN n.name        AS name,
                   labels(n)[0]  AS node_type,
                   n.mastered_at AS mastered_at
            ORDER BY n.mastered_at ASC
        """)

    def sync_mastery_from_letta(self, mastered_concepts: list[str]):
        from datetime import datetime, timezone
        sentinel = "2000-01-01T00:00:00+00:00"
        for name in mastered_concepts:
            self.query("""
                MATCH (n)
                WHERE (n:Topic OR n:Technique)
                  AND n.name = $name
                  AND coalesce(n.kg, 'fods') = 'fods'
                  AND coalesce(n.status, 'grey') <> 'green'
                SET n.status = 'green',
                    n.mastered_at = $ts,
                    n.updated_at  = $ts
            """, {"name": name, "ts": sentinel})

    # ═══════════════════════════════════════════════
    # FRONTEND EXPORT — FODS Curriculum KG
    # ═══════════════════════════════════════════════

    def to_cytoscape_json(self) -> dict:
        """
        FODS Curriculum KG → Cytoscape.js format.
        Topic nodes = large, Technique nodes = small.
        Colored by student mastery status.
        """
        status_colors = {
            "grey":   "#9CA3AF",
            "blue":   "#3B82F6",
            "yellow": "#F59E0B",
            "green":  "#10B981",
            "red":    "#EF4444",
            "orange": "#F97316",
        }

        topics = self.query("""
            MATCH (t:Topic)
            WHERE coalesce(t.kg, 'fods') = 'fods'
            RETURN t.name as name,
                   coalesce(t.status, 'grey') as status,
                   coalesce(t.mastered_at, null) as mastered_at,
                   'topic' as node_type
        """)

        techniques = self.query("""
            MATCH (t:Technique)
            WHERE coalesce(t.kg, 'fods') = 'fods'
            RETURN t.name as name,
                   coalesce(t.status, 'grey') as status,
                   coalesce(t.mastered_at, null) as mastered_at,
                   'technique' as node_type
        """)

        edges_result = self.query("""
            MATCH (a)-[r]->(b)
            WHERE (a:Topic OR a:Technique) AND (b:Topic OR b:Technique)
              AND coalesce(a.kg, 'fods') = 'fods'
              AND coalesce(b.kg, 'fods') = 'fods'
            RETURN a.name as source,
                   b.name as target,
                   type(r) as relationship
        """)

        cytoscape_nodes = []
        seen_ids = set()  # prevent duplicate node IDs

        for node in topics:
            status      = node.get("status", "grey")
            mastered_at = node.get("mastered_at")
            mastered_label = ""
            if mastered_at and mastered_at != "2000-01-01T00:00:00+00:00":
                mastered_label = f" · mastered {mastered_at[:10]}"
            elif mastered_at:
                mastered_label = " · mastered (previous session)"

            node_id = node["name"].lower().replace(" ", "_")
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                cytoscape_nodes.append({
                    "data": {
                        "id":          node_id,
                        "label":       node["name"],
                        "status":      status,
                        "color":       status_colors.get(status, "#9CA3AF"),
                        "node_type":   "topic",
                        "mastered_at": mastered_at or "",
                        "tooltip":     f"{node['name']}{mastered_label}",
                        "difficulty":  "intermediate"
                    }
                })

        for node in techniques:
            status      = node.get("status", "grey")
            mastered_at = node.get("mastered_at")
            mastered_label = ""
            if mastered_at and mastered_at != "2000-01-01T00:00:00+00:00":
                mastered_label = f" · mastered {mastered_at[:10]}"
            elif mastered_at:
                mastered_label = " · mastered (previous session)"

            node_id = node["name"].lower().replace(" ", "_")
            if node_id not in seen_ids:
                seen_ids.add(node_id)
                cytoscape_nodes.append({
                    "data": {
                        "id":          node_id,
                        "label":       node["name"],
                        "status":      status,
                        "color":       status_colors.get(status, "#9CA3AF"),
                        "node_type":   "technique",
                        "mastered_at": mastered_at or "",
                        "tooltip":     f"{node['name']}{mastered_label}",
                        "difficulty":  "beginner"
                    }
                })

        cytoscape_edges = []
        seen_edge_pairs = set()
        for i, edge in enumerate(edges_result):
            src = edge["source"].lower().replace(" ", "_")
            tgt = edge["target"].lower().replace(" ", "_")
            pair = (src, tgt, edge["relationship"])
            if pair in seen_edge_pairs:
                continue
            seen_edge_pairs.add(pair)
            cytoscape_edges.append({
                "data": {
                    "id":           f"e{i}",
                    "source":       edge["source"].lower().replace(" ", "_"),
                    "target":       edge["target"].lower().replace(" ", "_"),
                    "relationship": edge["relationship"],
                    "label":        edge["relationship"].replace("_", " ").lower()
                }
            })

        node_count = len(cytoscape_nodes)
        return {
            "elements":   {"nodes": cytoscape_nodes, "edges": cytoscape_edges},
            "node_count": node_count,
            "visible":    node_count > KG_VISIBLE_THRESHOLD,
            "kg_type":    "fods",
        }

    # ═══════════════════════════════════════════════
    # FRONTEND EXPORT — Time Series KG
    # ═══════════════════════════════════════════════

    def to_cytoscape_json_pipeline(self, view: str = "pipeline") -> dict:
        """
        Time Series KG → Cytoscape.js format.

        view="pipeline"  → 9 PipelineStage nodes + LEADS_TO (clean, default sidebar)
        view="models"    → PipelineStage + Model nodes + USES/LEADS_TO edges
        view="concepts"  → Concept nodes + LEARN_BEFORE / ADDRESSES edges
        view="full"      → All node types (heavy — use in explorer only)
        """
        type_colors = {
            "PipelineStage": "#0284C7",
            "Model":         "#7C3AED",
            "Concept":       "#059669",
            "EvalMetric":    "#D97706",
            "BestPractice":  "#10B981",
            "AntiPattern":   "#EF4444",
            "UseCase":       "#6366F1",
            "LearningPath":  "#EC4899",
            "PredictionType":"#0891B2",
            "Technique":     "#64748B",
        }
        edge_colors = {
            "LEADS_TO":         "#0284C7",
            "USES":             "#7C3AED",
            "SUITABLE_FOR":     "#059669",
            "EVALUATED_BY":     "#D97706",
            "ADDRESSES":        "#10B981",
            "LEARN_BEFORE":     "#6366F1",
            "REQUIRES_CONCEPT": "#EF4444",
            "COMPARED_TO":      "#94A3B8",
            "MUST_PRECEDE":     "#F97316",
            "NEXT_STAGE":       "#0284C7",
            "RELEVANT_TO":      "#64748B",
        }

        if view == "pipeline":
            nodes_raw = self.query("""
                MATCH (n:PipelineStage)
                RETURN n.name as name,
                       coalesce(toString(n.order), '') as order,
                       coalesce(n.description, '') as description,
                       'PipelineStage' as label_type
                ORDER BY n.order
            """)
            edges_raw = self.query("""
                MATCH (a:PipelineStage)-[r:LEADS_TO|NEXT_STAGE]->(b:PipelineStage)
                RETURN a.name as source, b.name as target,
                       type(r) as relationship
            """)

        elif view == "models":
            nodes_raw = self.query("""
                MATCH (n) WHERE n:PipelineStage OR n:Model
                RETURN n.name as name,
                       labels(n)[0] as label_type,
                       coalesce(n.description, n.family, '') as description
            """)
            edges_raw = self.query("""
                MATCH (a)-[r:USES|LEADS_TO|NEXT_STAGE]->(b)
                WHERE (a:PipelineStage OR a:Model)
                  AND (b:PipelineStage OR b:Model)
                RETURN a.name as source, b.name as target,
                       type(r) as relationship
            """)

        elif view == "concepts":
            nodes_raw = self.query("""
                MATCH (n:Concept)
                WHERE coalesce(n.kg, 'timeseries') = 'timeseries'
                RETURN n.name as name,
                       'Concept' as label_type,
                       coalesce(n.description, '') as description
            """)
            edges_raw = self.query("""
                MATCH (a:Concept)-[r:LEARN_BEFORE]->(b:Concept)
                WHERE coalesce(a.kg, 'timeseries') = 'timeseries'
                  AND coalesce(b.kg, 'timeseries') = 'timeseries'
                RETURN a.name as source, b.name as target,
                       'LEARN_BEFORE' as relationship
                UNION
                MATCH (t:Technique)-[r:ADDRESSES]->(c:Concept)
                WHERE coalesce(t.kg, 'timeseries') = 'timeseries'
                  AND coalesce(c.kg, 'timeseries') = 'timeseries'
                RETURN t.name as source, c.name as target,
                       'ADDRESSES' as relationship
            """)

        else:  # full
            nodes_raw = self.query("""
                MATCH (n)
                WHERE n:PipelineStage OR n:Model OR n:EvalMetric OR n:PredictionType
                  OR (n:Concept   AND coalesce(n.kg, 'timeseries') = 'timeseries')
                  OR (n:UseCase   AND coalesce(n.kg, 'timeseries') = 'timeseries')
                RETURN n.name as name,
                       labels(n)[0] as label_type,
                       coalesce(n.description, n.family, '') as description
            """)
            edges_raw = self.query("""
                MATCH (a)-[r]->(b)
                WHERE (a:PipelineStage OR a:Model OR a:EvalMetric OR a:PredictionType
                  OR (a:Concept AND coalesce(a.kg, 'timeseries') = 'timeseries')
                  OR (a:UseCase AND coalesce(a.kg, 'timeseries') = 'timeseries'))
                  AND (b:PipelineStage OR b:Model OR b:EvalMetric OR b:PredictionType
                  OR (b:Concept AND coalesce(b.kg, 'timeseries') = 'timeseries')
                  OR (b:UseCase AND coalesce(b.kg, 'timeseries') = 'timeseries'))
                RETURN a.name as source, b.name as target,
                       type(r) as relationship
                LIMIT 300
            """)

        cytoscape_nodes = []
        seen_ids = set()
        for node in nodes_raw:
            label_type = node.get("label_type", "PipelineStage")
            name       = node.get("name", "")
            desc       = node.get("description", "")
            size       = 28 if label_type == "PipelineStage" else 16
            node_id    = name.lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            if node_id in seen_ids:
                continue
            seen_ids.add(node_id)
            cytoscape_nodes.append({
                "data": {
                    "id":        node_id,
                    "label":     name,
                    "color":     type_colors.get(label_type, "#94A3B8"),
                    "node_type": label_type,
                    "tooltip":   f"[{label_type}] {name}" + (f"\n{desc[:80]}" if desc else ""),
                    "size":      size,
                }
            })

        cytoscape_edges = []
        for i, edge in enumerate(edges_raw):
            rel = edge.get("relationship", "")
            src = edge["source"].lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            tgt = edge["target"].lower().replace(" ", "_").replace("/", "_").replace("(", "").replace(")", "")
            cytoscape_edges.append({
                "data": {
                    "id":           f"pe{i}",
                    "source":       src,
                    "target":       tgt,
                    "relationship": rel,
                    "label":        rel.replace("_", " ").lower(),
                    "color":        edge_colors.get(rel, "#94A3B8"),
                }
            })

        node_count = len(cytoscape_nodes)
        return {
            "elements":   {"nodes": cytoscape_nodes, "edges": cytoscape_edges},
            "node_count": node_count,
            "visible":    node_count > 0,
            "kg_type":    "timeseries",
            "view":       view,
        }
