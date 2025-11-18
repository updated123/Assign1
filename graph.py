from collections import defaultdict
from canonicalize import CanonicalizeTool
from aggregator import AggregatorTool
from typing import Any

State = dict  # your workflow state

def extract_node(state: State):
    out = []
    for r in state["reviews"]:
        # Here, replace with your ExtractTopicsTool logic
        # Example: extract topics from review text
        topics = ["example_topic"]  # dummy placeholder
        out.append({**r, "topics": topics})
    state["extracted"] = out
    return state

def embed_node(state: State):
    # Example placeholder for embedding
    phrases = [t for item in state["extracted"] for t in item["topics"]]
    vectors = [[0.0]] * len(phrases)  # dummy embeddings
    state["phrases"] = phrases
    state["vectors"] = vectors
    return state

def cluster_node(state: State):
    labels = list(range(len(state["phrases"])))  # dummy: one label per phrase
    state["labels"] = labels
    return state

def canonical_node(state: State):
    groups = defaultdict(list)
    for phrase, label in zip(state["phrases"], state["labels"]):
        groups[label].append(phrase)

    canonical_topics = []
    for label, phrases in groups.items():
        can = CanonicalizeTool.invoke({"cluster_phrases": phrases})
        canonical_topics.append({"cluster": label, **can})

    state["canonical"] = canonical_topics
    return state

def aggregate_node(state: State):
    mapped = []
    for item in state["extracted"]:
        review_date = item.get("date", "unknown")
        for topic in state["canonical"]:
            matched = [s for s in topic["synonyms"] if s in item["topics"]]
            if matched:
                mapped.append({"topic": topic["canonical_label"], "date": review_date})
    table = AggregatorTool.invoke({
        "mapped_topics": mapped,
        "start": state.get("start_date"),
        "end": state.get("end_date")
    })
    state["table"] = table
    return state

def report_node(state: State):
    import os
    import pandas as pd

    output_path = state.get("output", "output/report.csv")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)

    df = pd.DataFrame(state["table"]).fillna(0)
    df.to_csv(output_path)
    print(f"Report saved to {output_path}")
    return state

# Graph definition
class StateGraph:
    def __init__(self, state):
        self.state = state
        self.nodes = {}
        self.edges = {}
        self.entry_point = None

    def add_node(self, name, func):
        self.nodes[name] = func

    def set_entry_point(self, name):
        self.entry_point = name

    def add_edge(self, from_node, to_node):
        self.edges[from_node] = to_node

    def compile(self):
        return self

    def invoke(self, state):
        current = self.entry_point
        while current:
            state = self.nodes[current](state)
            current = self.edges.get(current)
        return state

workflow = StateGraph(State({}))
workflow.add_node("extract", extract_node)
workflow.add_node("embed", embed_node)
workflow.add_node("cluster", cluster_node)
workflow.add_node("canonical", canonical_node)
workflow.add_node("aggregate", aggregate_node)
workflow.add_node("report", report_node)

workflow.set_entry_point("extract")
workflow.add_edge("extract", "embed")
workflow.add_edge("embed", "cluster")
workflow.add_edge("cluster", "canonical")
workflow.add_edge("canonical", "aggregate")
workflow.add_edge("aggregate", "report")
workflow.add_edge("report", None)

graph = workflow.compile()
