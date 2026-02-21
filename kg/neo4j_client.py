# kg/neo4j_client.py
# Neo4j connection and core operations

from neo4j import GraphDatabase
from config import KG_VISIBLE_THRESHOLD

class Neo4jClient:
    """
    Core Neo4j client.
    Used by KG Builder Agent to write nodes/edges.
    Used by all 3 teaching agents to query relationships.
    Used by API to export Cytoscape JSON for frontend.
    """

    def __init__(self):
        self.driver = GraphDatabase.driver(
            NEO4J_URI,
            auth=(NEO4J_USER, NEO4J_PASSWORD)
        )
        print("Connected to Neo4j")

    def close(self):
        self.driver.close()

    def query(self, cypher: str, params: dict = None) -> list:
        with self.driver.session() as session:
            result = session.run(cypher, params or {})
            return [record.data() for record in result]

    # ── KG Builder writes ──────────────────────────

    def create_concept_node(self, concept: dict):
        """Create or update a concept node. Called by KG Builder Agent."""
        cypher = """
        MERGE (c:Concept {name: $name})
        SET c.description = $description,
            c.difficulty = $difficulty,
            c.topic_area = $topic_area,
            c.status = $status
        RETURN c
        """
        self.query(cypher, {
            "name": concept["name"],
            "description": concept.get("description", ""),
            "difficulty": concept.get("difficulty", "intermediate"),
            "topic_area": concept.get("topic_area", "general"),
            "status": concept.get("status", "grey")
        })

    def create_relationship(self, from_concept: str, to_concept: str, rel_type: str):
        """
        Create a relationship between two concepts.
        rel_type: REQUIRES | BUILDS_ON | PART_OF | USED_IN | RELATED_TO
        """
        cypher = f"""
        MATCH (a:Concept {{name: $from_name}})
        MATCH (b:Concept {{name: $to_name}})
        MERGE (a)-[:{rel_type}]->(b)
        """
        self.query(cypher, {"from_name": from_concept, "to_name": to_concept})

    # ── Teaching agent reads ───────────────────────

    def get_prerequisites(self, concept_name: str) -> list[str]:
        """Used by Solver Agent before explaining — checks what student needs to know first."""
        cypher = """
        MATCH (c:Concept {name: $name})-[:REQUIRES*1..3]->(prereq:Concept)
        RETURN prereq.name as name, prereq.difficulty as difficulty
        ORDER BY prereq.difficulty
        """
        results = self.query(cypher, {"name": concept_name})
        return [r["name"] for r in results]

    def get_related_concepts(self, concept_name: str) -> list[str]:
        """Used by Assessment Agent to generate comprehensive questions."""
        cypher = """
        MATCH (c:Concept {name: $name})-[r]-(related:Concept)
        WHERE type(r) IN ['RELATED_TO', 'BUILDS_ON', 'USED_IN']
        RETURN related.name as name
        LIMIT 5
        """
        results = self.query(cypher, {"name": concept_name})
        return [r["name"] for r in results]

    def get_prerequisite_chain_for_feedback(self, concept_name: str) -> list[dict]:
        """Used by Feedback Agent to trace root cause of mistakes."""
        cypher = """
        MATCH path = (c:Concept {name: $name})-[:REQUIRES*]->(prereq:Concept)
        RETURN prereq.name as name,
               prereq.difficulty as difficulty,
               prereq.status as status,
               length(path) as depth
        ORDER BY depth
        """
        return self.query(cypher, {"name": concept_name})

    def get_learning_path(self, target_concept: str) -> list[str]:
        """Get ordered learning path to reach a target concept."""
        cypher = """
        MATCH path = (start:Concept)-[:REQUIRES*]->(target:Concept {name: $name})
        WHERE NOT (start)-[:REQUIRES]->()
        RETURN [node in nodes(path) | node.name] as path
        ORDER BY length(path) DESC
        LIMIT 1
        """
        results = self.query(cypher, {"name": target_concept})
        if results:
            return results[0]["path"]
        return [target_concept]

    # ── Feedback Agent writes ──────────────────────

    def update_node_status(self, concept_name: str, status: str):
        """
        Update visual status of a node. Called by Feedback Agent after assessment.

        status:
          grey   = not yet reached
          blue   = currently studying
          yellow = being assessed
          green  = mastered
          red    = weak area
          orange = prerequisite gap
        """
        cypher = """
        MATCH (c:Concept {name: $name})
        SET c.status = $status
        RETURN c
        """
        self.query(cypher, {"name": concept_name, "status": status})

    # ── KG visibility ──────────────────────────────

    def get_node_count(self) -> int:
        result = self.query("MATCH (n:Concept) RETURN count(n) as count")
        return result[0]["count"] if result else 0

    def is_kg_visible(self) -> bool:
        """Returns True when KG has more than 1 node — triggers frontend display."""
        return self.get_node_count() > KG_VISIBLE_THRESHOLD

    # ── Frontend export ────────────────────────────

    def to_cytoscape_json(self) -> dict:
        """
        Convert entire Neo4j graph to Cytoscape.js format.
        Called by /api/kg/graph endpoint for Streamlit frontend.
        """
        status_colors = {
            "grey":   "#9CA3AF",
            "blue":   "#3B82F6",
            "yellow": "#F59E0B",
            "green":  "#10B981",
            "red":    "#EF4444",
            "orange": "#F97316",
        }

        nodes_result = self.query("""
            MATCH (n:Concept)
            RETURN n.name as name,
                   n.description as description,
                   n.difficulty as difficulty,
                   n.topic_area as topic_area,
                   n.status as status
        """)

        edges_result = self.query("""
            MATCH (a:Concept)-[r]->(b:Concept)
            RETURN a.name as source,
                   b.name as target,
                   type(r) as relationship
        """)

        cytoscape_nodes = []
        for node in nodes_result:
            status = node.get("status", "grey")
            cytoscape_nodes.append({
                "data": {
                    "id": node["name"].lower().replace(" ", "_"),
                    "label": node["name"],
                    "description": node.get("description", ""),
                    "difficulty": node.get("difficulty", "intermediate"),
                    "topic_area": node.get("topic_area", "general"),
                    "status": status,
                    "color": status_colors.get(status, "#9CA3AF")
                }
            })

        cytoscape_edges = []
        for i, edge in enumerate(edges_result):
            cytoscape_edges.append({
                "data": {
                    "id": f"e{i}",
                    "source": edge["source"].lower().replace(" ", "_"),
                    "target": edge["target"].lower().replace(" ", "_"),
                    "relationship": edge["relationship"],
                    "label": edge["relationship"].replace("_", " ").lower()
                }
            })

        node_count = len(cytoscape_nodes)
        return {
            "elements": {"nodes": cytoscape_nodes, "edges": cytoscape_edges},
            "node_count": node_count,
            "visible": node_count > KG_VISIBLE_THRESHOLD
        }
