def __init__(self):
    import streamlit as st
    
    uri      = st.secrets.get("NEO4J_URI", "")
    user     = st.secrets.get("NEO4J_USER", "neo4j")
    password = st.secrets.get("NEO4J_PASSWORD", "")

    self.driver = GraphDatabase.driver(
        uri,
        auth=(user, password)
    )
    print(f"Neo4j connecting to: {uri}")
