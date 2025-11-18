from pydantic import BaseModel
from langchain.tools import StructuredTool
import numpy as np
from sklearn.cluster import AgglomerativeClustering

class ClusterPayload(BaseModel):
    vectors: list[list[float]]

def cluster_vectors(vectors):
    X = np.array(vectors)
    clustering = AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=0.25
    )
    labels = clustering.fit_predict(X)
    return labels.tolist()

ClusteringTool = StructuredTool.from_function(
    func=lambda vectors: cluster_vectors(vectors),
    name="ClusterTopics",
    description="Clusters embedded topic phrases.",
    args_schema=ClusterPayload
)
