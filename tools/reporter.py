import pandas as pd
from pydantic import BaseModel
from langchain.tools import StructuredTool

class ReportPayload(BaseModel):
    table: dict
    output_path: str

def generate_report(table, output_path):
    df = pd.DataFrame(table)
    df.to_csv(output_path)
    return {"saved": output_path}

ReportTool = StructuredTool.from_function(
    func=lambda table, output_path: generate_report(table, output_path),
    name="GenerateReport",
    description="Saves a CSV trend table.",
    args_schema=ReportPayload
)
