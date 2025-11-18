

from langchain.tools import StructuredTool
from langchain_openai import ChatOpenAI
from pydantic import BaseModel
import json
from dotenv import load_dotenv
import os
# -------------------------------
# Pydantic Schema for Input
# -------------------------------
class CanonicalPayload(BaseModel):
    cluster_phrases: list[str]

# -------------------------------
# Canonicalization Function
# -------------------------------
load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")
def canonicalize_cluster(phrases):
    """
    Given a list of related topic phrases, generate a canonical topic label,
    synonyms, and topic type. Returns a dictionary safely parsed from JSON.
    """
    # Improved prompt
    prompt = f"""
You are an expert assistant for analyzing app store reviews. 
Given the following cluster of related phrases from reviews:

{phrases}

Your task:
1. Identify the main topic that best summarizes the cluster.
2. List all input phrases as synonyms.
3. Specify the topic type as one of: issue, request, feedback.
4. Return ONLY valid JSON in the following format:

{{
  "canonical_label": "<short descriptive label>",
  "synonyms": [<list of original phrases>],
  "topic_type": "<issue|request|feedback>"
}}

Be concise and accurate. Ensure that different clusters produce distinct canonical labels.
"""

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)


    # Invoke LLM
    resp = llm.invoke(prompt).content
    print("LLM Response:", resp)

    # Safely parse JSON
    try:
        parsed = json.loads(resp)
        # Ensure keys exist
        parsed.setdefault("canonical_label", "unknown")
        parsed.setdefault("synonyms", phrases)
        parsed.setdefault("topic_type", "feedback")
        return parsed
    except json.JSONDecodeError:
        # Fallback if LLM returns invalid JSON
        return {
            "canonical_label": "unknown",
            "synonyms": phrases,
            "topic_type": "feedback"
        }

# -------------------------------
# Structured Tool
# -------------------------------
CanonicalizeTool = StructuredTool.from_function(
    func=lambda cluster_phrases: canonicalize_cluster(cluster_phrases),
    name="CanonicalizeTopic",
    description="Produces canonical label, synonyms, and topic type for a cluster of phrases.",
    args_schema=CanonicalPayload
)
