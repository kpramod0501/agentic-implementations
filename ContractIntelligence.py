import os
import asyncio
from typing import TypedDict, List, Optional

import numpy as np
import requests
from fastapi import FastAPI
from pydantic import BaseModel
import serpapi

from langgraph.graph import StateGraph, END

from pymilvus import (
    connections,
    Collection,
    FieldSchema,
    CollectionSchema,
    DataType,
)

from openai import OpenAI


# =========================
# CONFIG (AZURE + OPENAI)
# =========================

MILVUS_HOST = os.getenv("MILVUS_HOST", "your-azure-milvus-host")
MILVUS_PORT = os.getenv("MILVUS_PORT", "19530")

COLLECTION_NAME = "docs_collection"

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "your-key")
client = OpenAI(api_key=OPENAI_API_KEY)

WEB_SEARCH_API = "https://api.example.com/ma-search" # replace with real API


# =========================
# FASTAPI
# =========================

app = FastAPI(title="LangGraph RAG + Web Search (Milvus + Azure)")


# =========================
# REQUEST SCHEMA
# =========================

class SearchRequest(BaseModel):
    query: str


# =========================
# LANGGRAPH STATE
# =========================

class GraphState(TypedDict):
    query: str
    route: Optional[str]
    docs: Optional[List[str]]
    result: Optional[dict]


# =========================
# MILVUS CONNECTION
# =========================

def init_milvus():
    connections.connect(
        alias="default",
        host=MILVUS_HOST,
        port=MILVUS_PORT
    )

    fields = [
        FieldSchema(name="id", dtype=DataType.INT64, is_primary=True, auto_id=True),
        FieldSchema(name="embedding", dtype=DataType.FLOAT_VECTOR, dim=1536),
        FieldSchema(name="text", dtype=DataType.VARCHAR, max_length=2048),
    ]

    schema = CollectionSchema(fields, description="RAG Docs")

    if COLLECTION_NAME not in Collection.list_collections():
        collection = Collection(name=COLLECTION_NAME, schema=schema)
    else:
        collection = Collection(COLLECTION_NAME)

    collection.load()
    return collection


milvus = init_milvus()


# =========================
# EMBEDDINGS (OPENAI)
# =========================

def embed(text: str):
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text
    )
    return np.array(resp.data[0].embedding, dtype=np.float32)


# =========================
# OPTIONAL: INGEST 200 DOCS
# =========================

def bootstrap_ingest():
    """
    Run once if collection is empty.
    """
    sample_docs = [
        f"Document about company strategy {i}" for i in range(200)
    ]

    embeddings = [embed(d) for d in sample_docs]

    milvus.insert([
        list(range(len(sample_docs))),
        embeddings,
        sample_docs
    ])

    milvus.flush()


# =========================
# NODES
# =========================

def orchestration_node(state: GraphState) -> GraphState:
    q = state["query"].lower()

    if any(k in q for k in ["acquisition", "merger", "m&a", "buyout"]):
        state["route"] = "web"
    else:
        state["route"] = "rag"

    return state


def rag_node(state: GraphState) -> GraphState:
    query_vec = embed(state["query"])

    results = milvus.search(
        data=[query_vec],
        anns_field="embedding",
        param={"metric_type": "L2", "params": {"nprobe": 10}},
        limit=5,
        output_fields=["text"]
    )

    state["docs"] = [hit.entity.get("text") for hit in results[0]]
    state["result"] = {"type": "rag", "data": state["docs"]}

    return state


def web_node(state: GraphState) -> GraphState:
    try:
        client = serpapi.Client(api_key="XXXXXXXXX")
        
        results = client.search({
                  "engine": "google",
                  "q": state["query"],
                  "location": "Austin, Texas, United States",
                  "google_domain": "google.com",
                  "hl": "en",
                  "gl": "us"
                })
         r = results["organic_results"]
        #r = requests.get(
        #    WEB_SEARCH_API,
        #    params={"q": state["query"]},
        #    timeout=10
        #)
        state["result"] = {
            "type": "web",
            "data": r.json()
        }
    except Exception as e:
        state["result"] = {
            "type": "web",
            "error": str(e)
        }
    
    # import serpapi



    return state


# =========================
# LANGGRAPH BUILD
# =========================

def build_graph():
    graph = StateGraph(GraphState)

    graph.add_node("orchestrator", orchestration_node)
    graph.add_node("rag", rag_node)
    graph.add_node("web", web_node)

    graph.set_entry_point("orchestrator")

    def route(state):
        return state["route"]

    graph.add_conditional_edges(
        "orchestrator",
        route,
        {
            "rag": "rag",
            "web": "web"
        }
    )

    graph.add_edge("rag", END)
    graph.add_edge("web", END)

    return graph.compile()


workflow = build_graph()


# =========================
# API ENDPOINT
# =========================

@app.post("/search")
async def search(req: SearchRequest):
    state: GraphState = {
        "query": req.query,
        "route": None,
        "docs": None,
        "result": None
    }

    result = await asyncio.to_thread(workflow.invoke, state)

    return result


# =========================
# STARTUP HOOK
# =========================

@app.on_event("startup")
def startup():
    # Optional bootstrap (comment out in prod)
    # bootstrap_ingest()
    pass


# =========================
# RUN
# =========================
"""
Run:
pip install fastapi uvicorn pymilvus openai langgraph requests numpy

uvicorn app:app --reload
"""
