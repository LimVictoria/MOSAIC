# kg/neo4j_client.py
# Neo4j connection and core operations
# Updated to use Topic and Technique nodes from curriculum KG
# Temporal: mastered_at timestamp written when node reaches green status

from neo4j import GraphDatabase
from config import KG_VISIBLE_THRESHOLD, NEO4J_URI, NEO4J_USER, NEO4J_PASSWORD

class Neo4jClient:
    """
    Core Neo4j client.
    Works with curriculum KG — Topic and Technique nodes.
    Topic nodes   = high-level subjects (e.g. Exploratory Data Analysis)
    Technique nodes = specific skills under each topic (e.g. Pearson Correlation)

    Status values (set per student interaction):
      grey   = not yet reached
      blue   = currently studying
      yellow = being assessed
      green  = mastered
      red    = needs review (failed 3+ times)
      orange = prerequisite gap

    Temporal fields:
      mastered_at = ISO timestamp when node first reached green (never overwritten)
      updated_at  = ISO timestamp of last status change
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
        """
        Update status on a Topic node.
        Called by Solver (blue), Assessment (yellow), Feedback (green/red/orange).
        Writes mastered_at when status becomes green — only set once, never overwritten.
        """
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
        """
        Update status on a Technique node.
        Called by Feedback Agent after detailed assessment.
        Writes mastered_at when status becomes green — only set once, never overwritten.
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MATCH (t:Technique {name: $name})
        SET t.status = $status,
            t.updated_at = $now
        WITH t
        WHERE $status = 'green' AND t.mastered_at IS NULL
        SET t.mastered_at = $now
        RETURN t
        """
        self.query(cypher, {"name": technique_name, "status": status, "now": now})

    def update_node_status(self, name: str, status: str):
        """
        Update status on either Topic or Technique — tries both.
        Used by agents that don't distinguish between node types.
        Writes mastered_at when status becomes green — only set once, never overwritten.
        """
        from datetime import datetime, timezone
        now = datetime.now(timezone.utc).isoformat()
        cypher = """
        MATCH (n)
        WHERE (n:Topic OR n:Technique) AND n.name = $name
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
        """
        Get prerequisite Topics for a given Topic.
        Used by Solver Agent before explaining — checks what student needs first.
        Traverses up to 3 levels deep.
        """
        cypher = """
        MATCH (t:Topic {name: $name})-[:PREREQUISITE*1..3]->(pre:Topic)
        RETURN pre.name as name, pre.status as status
        ORDER BY pre.name
        """
        results = self.query(cypher, {"name": topic_name})
        return [r["name"] for r in results]

    def get_unmastered_prerequisites(self, topic_name: str) -> list[str]:
        """
        Get prerequisites that are NOT yet mastered (status != green).
        Used to enforce prerequisite gating.
        """
        cypher = """
        MATCH (t:Topic {name: $name})-[:PREREQUISITE*1..3]->(pre:Topic)
        WHERE coalesce(pre.status, 'grey') <> 'green'
        RETURN pre.name as name, coalesce(pre.status, 'grey') as status
        ORDER BY pre.name
        """
        results = self.query(cypher, {"name": topic_name})
        return [r["name"] for r in results]

    def get_prerequisite_chain_for_feedback(self, topic_name: str) -> list[dict]:
        """
        Full prerequisite chain with status — used by Feedback Agent
        to trace root cause of mistakes.
        """
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
        """
        Returns full curriculum — all Topics with their prerequisites.
        Passed to LLM system prompt so it knows the learning sequence.
        """
        cypher = """
        MATCH (t:Topic)
        OPTIONAL MATCH (t)-[:PREREQUISITE]->(pre:Topic)
        RETURN t.name as topic,
               coalesce(t.status, 'grey') as status,
               collect(pre.name) as prerequisites
        ORDER BY t.name
        """
        return self.query(cypher)

    def get_topic_techniques(self, topic_name: str) -> list[dict]:
        """
        Get all Techniques under a Topic with their status.
        Used by Assessment Agent to ask technique-specific questions.
        """
        cypher = """
        MATCH (t:Topic {name: $name})-[]->(tech:Technique)
        RETURN tech.name as name,
               coalesce(tech.status, 'grey') as status
        ORDER BY tech.name
        """
        return self.query(cypher, {"name": topic_name})

    def get_related_topics(self, topic_name: str) -> list[str]:
        """
        Get Topics related to a given Topic via any relationship.
        Used by Assessment Agent to generate comprehensive questions.
        """
        cypher = """
        MATCH (t:Topic {name: $name})-[r]-(related:Topic)
        RETURN related.name as name
        LIMIT 5
        """
        results = self.query(cypher, {"name": topic_name})
        return [r["name"] for r in results]

    def get_learning_path(self, target_topic: str) -> list[str]:
        """
        Get ordered learning path to reach a target Topic.
        Returns sequence from root to target.
        """
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
        """
        Returns the next Topic the student should study.
        Finds Topics whose prerequisites are all mastered but
        the topic itself is not yet mastered.
        """
        cypher = """
        MATCH (t:Topic)
        WHERE coalesce(t.status, 'grey') <> 'green'
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
        """
        Maps a free-text concept from student conversation to nearest Topic node.
        Used by KG Builder to avoid creating random nodes.
        Returns matched topic name or None.
        """
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

        # Also check Techniques
        cypher2 = """
        MATCH (tech:Technique)
        WHERE toLower(tech.name) CONTAINS toLower($name)
           OR toLower($name) CONTAINS toLower(tech.name)
        RETURN tech.name as name
        LIMIT 1
        """
        results2 = self.query(cypher2, {"name": concept_name})
        if results2:
            return results2[0]["name"]

        return None

    # ── KG visibility ──────────────────────────────

    def get_node_count(self) -> int:
        """Returns total number of Topic + Technique nodes."""
        result = self.query("""
            MATCH (n) WHERE n:Topic OR n:Technique
            RETURN count(n) as count
        """)
        return result[0]["count"] if result else 0

    def is_kg_visible(self) -> bool:
        return self.get_node_count() > KG_VISIBLE_THRESHOLD

    def get_mastered_concepts(self, student_id: str = None) -> list[str]:
        """
        Returns list of mastered Topic names (status = green).
        student_id kept for API compatibility but status is stored on node directly.
        """
        results = self.query("""
            MATCH (t:Topic {status: 'green'})
            RETURN t.name as name
        """)
        return [r["name"] for r in results]

    # ── Temporal — learning progression ───────────

    def get_mastery_timeline(self) -> list[dict]:
        """
        Returns all mastered nodes ordered by mastered_at timestamp.
        Used to show the student's learning progression over time.
        Returns: [{name, node_type, mastered_at}, ...]
        """
        return self.query("""
            MATCH (n)
            WHERE (n:Topic OR n:Technique)
              AND n.status = 'green'
              AND n.mastered_at IS NOT NULL
            RETURN n.name        AS name,
                   labels(n)[0]  AS node_type,
                   n.mastered_at AS mastered_at
            ORDER BY n.mastered_at ASC
        """)

    def sync_mastery_from_letta(self, mastered_concepts: list[str]):
        """
        Backfill mastery from Letta into Neo4j for concepts mastered
        in previous sessions where mastered_at was not recorded.
        Uses a sentinel timestamp so timeline shows these as 'before current session'.
        Only writes to nodes that are not already green.
        """
        from datetime import datetime, timezone
        sentinel = "2000-01-01T00:00:00+00:00"  # clearly historical
        for name in mastered_concepts:
            self.query("""
                MATCH (n)
                WHERE (n:Topic OR n:Technique) AND n.name = $name
                  AND coalesce(n.status, 'grey') <> 'green'
                SET n.status = 'green',
                    n.mastered_at = $ts,
                    n.updated_at  = $ts
            """, {"name": name, "ts": sentinel})

    # ── Frontend export ────────────────────────────

    def to_cytoscape_json(self) -> dict:
        """
        Convert curriculum KG to Cytoscape.js format for Streamlit frontend.
        Shows Topic nodes as large circles, Technique nodes as smaller ones.
        Includes mastered_at in node data for tooltip display.
        """
        status_colors = {
            "grey":   "#9CA3AF",
            "blue":   "#3B82F6",
            "yellow": "#F59E0B",
            "green":  "#10B981",
            "red":    "#EF4444",
            "orange": "#F97316",
        }

        # Fetch Topic nodes
        topics = self.query("""
            MATCH (t:Topic)
            RETURN t.name as name,
                   coalesce(t.status, 'grey') as status,
                   t.mastered_at as mastered_at,
                   'topic' as node_type
        """)

        # Fetch Technique nodes
        techniques = self.query("""
            MATCH (t:Technique)
            RETURN t.name as name,
                   coalesce(t.status, 'grey') as status,
                   t.mastered_at as mastered_at,
                   'technique' as node_type
        """)

        # Fetch all relationships
        edges_result = self.query("""
            MATCH (a)-[r]->(b)
            WHERE (a:Topic OR a:Technique) AND (b:Topic OR b:Technique)
            RETURN a.name as source,
                   b.name as target,
                   type(r) as relationship
        """)

        cytoscape_nodes = []

        for node in topics:
            status      = node.get("status", "grey")
            mastered_at = node.get("mastered_at")
            # Format mastered_at for tooltip — show date only
            mastered_label = ""
            if mastered_at and mastered_at != "2000-01-01T00:00:00+00:00":
                mastered_label = f" · mastered {mastered_at[:10]}"
            elif mastered_at:
                mastered_label = " · mastered (previous session)"

            cytoscape_nodes.append({
                "data": {
                    "id":          node["name"].lower().replace(" ", "_"),
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

            cytoscape_nodes.append({
                "data": {
                    "id":          node["name"].lower().replace(" ", "_"),
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
        for i, edge in enumerate(edges_result):
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
            "elements": {"nodes": cytoscape_nodes, "edges": cytoscape_edges},
            "node_count": node_count,
            "visible": node_count > KG_VISIBLE_THRESHOLD
        }
