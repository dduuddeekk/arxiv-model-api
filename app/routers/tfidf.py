from fastapi import APIRouter
import joblib
import pandas as pd
import os
import ast
from sklearn.metrics.pairwise import cosine_similarity

from app.schemas.request import SearchRequest
from app.schemas.response import SearchResponse, SearchResult

router = APIRouter(prefix="/search", tags=["TF-IDF"])

models_dir = os.path.join(os.path.dirname(__file__), "../models")
vectorizer = joblib.load(os.path.join(models_dir, "tfidf_vectorizer.pkl"))
tfidf_matrix = joblib.load(os.path.join(models_dir, "tfidf_matrix.pkl"))
documents_df = pd.read_csv(os.path.join(models_dir, "tfidf_journals.csv"))

@router.post("/tfidf", response_model=SearchResponse)
def search_documents_tfidf(request: SearchRequest):
    query_vec = vectorizer.transform([request.query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    top_indices = similarities.argsort()[::-1][:request.top_k]

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
