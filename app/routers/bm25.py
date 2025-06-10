from fastapi import APIRouter
import joblib
import pandas as pd
import os
import ast

from app.schemas.request import SearchRequest
from app.schemas.response import SearchResponse, SearchResult

router = APIRouter(prefix="/search", tags=["BM25"])

models_dir = os.path.join(os.path.dirname(__file__), "../models")
bm25_model = joblib.load(os.path.join(models_dir, "bm25_model.pkl"))
documents_df = pd.read_csv(os.path.join(models_dir, "bm25_documents_metadata.csv"))

@router.post("/bm25", response_model=SearchResponse)
def search_documents_bm25(request: SearchRequest):
    scores = bm25_model.get_scores(request.query)
    top_indices = sorted(range(len(scores)), key=lambda i: scores[i], reverse=True)[:request.top_k]

    results = []
    for idx in top_indices:
        doc = documents_df.iloc[idx]
        try:
            authors = [a.strip() for a in ast.literal_eval(doc.get("authors", "[]"))]
        except Exception:
            authors = []

        results.append(SearchResult(
            title=doc.get("title", ""),
            abstract=doc.get("abstract", ""),
            url=doc.get("url", ""),
            published=doc.get("published", ""),
            authors=authors
        ))

    return SearchResponse(
        success=True,
        code="SUCCESS",
        message="Data retrieved successfully",
        data=results
    )
