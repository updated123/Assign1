from pydantic import BaseModel, Field
from typing import List
from langchain_openai import ChatOpenAI
from langchain.output_parsers import PydanticOutputParser
from langchain_core.tools.structured import StructuredTool
from dotenv import load_dotenv
import os
# -------------------------------
# JSON Schema
# -------------------------------
class TopicOutput(BaseModel):
    topics: List[str] = Field(description="List of extracted topics from review")

parser = PydanticOutputParser(pydantic_object=TopicOutput)


load_dotenv()
api_key = os.getenv("OPENAI_API_KEY")

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0, openai_api_key=api_key)

# -------------------------------
# Function to extract topics
# -------------------------------
def extract_topics(review_text: str) -> dict:
    prompt = f"""
Extract topics from the following user review. 
Output ONLY valid JSON following this schema:

{parser.get_format_instructions()}

Review:
\"\"\"{review_text}\"\"\"
"""
    resp = llm.predict(prompt)
    try:
        return parser.parse(resp).dict()
    except Exception:
        return {"topics": []}

# -------------------------------
# Wrap as a LangChain StructuredTool
# -------------------------------
ExtractTopicsTool = StructuredTool.from_function(
    func=extract_topics,
    name="ExtractTopicsTool",
    description="Extracts topics from a user review and returns JSON."
)
