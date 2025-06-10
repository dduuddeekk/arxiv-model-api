from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from app.routers import tfidf, bm25

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.include_router(tfidf.router)
app.include_router(bm25.router)
