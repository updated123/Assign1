# tools/embeddings.py

from typing import List
from pydantic import BaseModel
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.tools import StructuredTool


# -------------------------------
# Pydantic input schema
# -------------------------------
class EmbedPayload(BaseModel):
    texts: List[str]   # <-- MUST be "texts" because graph invokes {"texts": ...}


# -------------------------------
# Embedding model
# -------------------------------
hf_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)


# -------------------------------
# Function that performs embeddings
# -------------------------------
def embed_texts(texts: List[str]):
    vectors = hf_model.embed_documents(texts)
    return {"vectors": vectors}


# -------------------------------
# StructuredTool wrapper
# -------------------------------
EmbeddingTool = StructuredTool.from_function(
    func=embed_texts,
    name="EmbeddingTool",
    description="Embeds a list of text phrases using HuggingFace embeddings.",
    args_schema=EmbedPayload
)
