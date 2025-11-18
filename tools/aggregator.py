import pandas as pd
from pydantic import BaseModel
from langchain.tools import StructuredTool

class AggregationPayload(BaseModel):
    mapped_topics: list[dict]
    start: str
    end: str

def aggregate_counts(mapped, start, end):
    if not mapped:
        return {}

    df = pd.DataFrame(mapped)
    days = pd.date_range(start=start, end=end)
    df['date'] = pd.to_datetime(df['date'], errors='coerce').dt.strftime('%Y-%m-%d')
    df = df.dropna(subset=['date'])  # drop reviews with invalid date

    topics = sorted(df['topic'].unique())
    report = pd.DataFrame(0, index=topics, columns=days.strftime('%Y-%m-%d'))

    for _, r in df.iterrows():
        if r['date'] in report.columns:
            report.loc[r['topic'], r['date']] += 1
        else:
            # Map dates outside window to last column or ignore
            report.iloc[:, -1][r['topic']] += 1

    return report.to_dict()

AggregatorTool = StructuredTool.from_function(
    func=lambda mapped_topics, start, end: aggregate_counts(mapped_topics, start, end),
    name="AggregateTrends",
    description="Aggregates sliding-window topic trends.",
    args_schema=AggregationPayload
)
