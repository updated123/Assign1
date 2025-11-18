from langchain.tools import StructuredTool
from pydantic import BaseModel
from langchain.embeddings import HuggingFaceEmbeddings

class EmbedPayload(BaseModel):
    texts: list[str]

# Use a free HuggingFace model
hf_model = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

def embed_texts(texts: list[str]):
    vectors = hf_model.embed_documents(texts)
    return vectors

EmbeddingTool = StructuredTool.from_function(
    func=lambda texts: embed_texts(texts),
    name="EmbedTexts",
    description="Embeds list of topic candidate phrases.",
    args_schema=EmbedPayload
)
