from fastapi import APIRouter, HTTPException
from app.schemas.request import SearchRequest
from app.schemas.response import SearchResponse, SearchResult
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json
import os
from typing import List
import logging

router = APIRouter(prefix="/search", tags=["FAISS"])

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Path setup
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MODEL_DIR = os.path.join(BASE_DIR, "models")
os.makedirs(MODEL_DIR, exist_ok=True)

# Model files
INDEX_PATH = os.path.join(MODEL_DIR, "my_index.faiss")
METADATA_PATH = os.path.join(MODEL_DIR, "metadata.json")
EMBEDDING_MODEL_PATH = os.path.join(MODEL_DIR, "sentence-transformers", "all-MiniLM-L6-v2")

# Initialize variables that will be loaded
index = None
metadata = []
model = None

def load_resources():
    """Load all required resources (FAISS index, metadata, and embedding model)"""
    global index, metadata, model
    
    try:
        # Load FAISS index
        if os.path.exists(INDEX_PATH):
            index = faiss.read_index(INDEX_PATH)
            logger.info("FAISS index loaded successfully")
        else:
            raise FileNotFoundError(f"FAISS index file not found at {INDEX_PATH}")
        
        # Load metadata
        if os.path.exists(METADATA_PATH):
            with open(METADATA_PATH, "r", encoding="utf-8") as f:
                metadata = json.load(f)
            logger.info(f"Metadata loaded with {len(metadata)} entries")
        else:
            raise FileNotFoundError(f"Metadata file not found at {METADATA_PATH}")
        
        # Load embedding model
        try:
            model = SentenceTransformer(
                EMBEDDING_MODEL_PATH if os.path.exists(EMBEDDING_MODEL_PATH) 
                else "sentence-transformers/all-MiniLM-L6-v2",
                cache_folder=os.path.join(MODEL_DIR, "sentence-transformers")
            )
            logger.info("SentenceTransformer model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load embedding model: {str(e)}")
            raise

    except Exception as e:
        logger.error(f"Error loading resources: {str(e)}")
        raise

# Load resources when module is imported
try:
    load_resources()
except Exception as e:
    logger.error(f"Initialization failed: {str(e)}")
    # You might want to implement a retry mechanism here

@router.post("/faiss", response_model=SearchResponse)
async def search_faiss(request: SearchRequest):
    """
    Perform semantic search using FAISS index
    """
    try:
        # Validate resources are loaded
        if None in [index, model] or not metadata:
            raise HTTPException(
                status_code=503,
                detail="Service unavailable - resources not loaded"
            )

        # Validate top_k is reasonable
        request.top_k = min(max(1, request.top_k), 100)

        # Encode query
        query_embedding = model.encode([request.query])
        query_vector = np.array(query_embedding).astype("float32")

        # Search FAISS index
        distances, indices = index.search(query_vector, request.top_k)

        # Build results
        results: List[SearchResult] = []
        for idx in indices[0]:
            if idx < 0 or idx >= len(metadata):
                continue
            
            item = metadata[idx]
            
            # Fix authors format - handle both string representation and actual lists
            authors = item.get("authors", [])
            if isinstance(authors, str):
                try:
                    # Safely evaluate string representation of list
                    authors = eval(authors) if authors else []
                except:
                    authors = [authors] if authors else []

            results.append(SearchResult(
                title=item.get("title", ""),
                abstract=item.get("abstract", ""),
                url=item.get("url", ""),
                published=item.get("published", ""),
                authors=authors
            ))

        return SearchResponse(
            success=True,
            code="SUCCESS",
            message=f"Found {len(results)} results",
            data=results
        )

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Search error: {str(e)}", exc_info=True)
        raise HTTPException(
            status_code=500,
            detail=f"Search failed: {str(e)}"
        )