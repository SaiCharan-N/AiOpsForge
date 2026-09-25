"""Day 3 + Day 5: the first LangGraph graph in the project.

This is deliberately NOT the real agent pipeline — it's the smallest possible
proof that (a) LangGraph can route between two nodes based on state, and
(b) a node can call the local Ollama model and get a real response back.
Phase 2 replaces this file's nodes with the actual Planner/Developer/QA
agents; the wiring pattern (StateGraph -> compile -> invoke) stays the same.
"""
from typing import TypedDict

import requests

from app.config import settings


class DemoState(TypedDict):
    topic: str
    node_a_output: str
    node_b_output: str


def node_a(state: DemoState) -> DemoState:
    """Dummy first node — no LLM call, just proves the graph runs at all."""
    topic = state["topic"]
    state["node_a_output"] = f"Received topic: '{topic}'. Handing off to node_b."
    return state


def node_b(state: DemoState) -> DemoState:
    """Second node — calls the local Ollama model over HTTP."""
    prompt = (
        f"In one short sentence, explain why {state['topic']} matters "
        "for software developers."
    )
    response = requests.post(
        f"{settings.ollama_base_url}/api/generate",
        json={"model": settings.ollama_model, "prompt": prompt, "stream": False},
        timeout=120,
    )
    response.raise_for_status()
    state["node_b_output"] = response.json().get("response", "").strip()
    return state


def build_graph():
    from langgraph.graph import StateGraph, END

    graph = StateGraph(DemoState)
    graph.add_node("node_a", node_a)
    graph.add_node("node_b", node_b)
    graph.set_entry_point("node_a")
    graph.add_edge("node_a", "node_b")
    graph.add_edge("node_b", END)
    return graph.compile()


def run_demo(topic: str) -> DemoState:
    app = build_graph()
    return app.invoke({"topic": topic, "node_a_output": "", "node_b_output": ""})
