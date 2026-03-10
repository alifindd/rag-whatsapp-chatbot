from fastapi import FastAPI
from app.routers.chat import router

app = FastAPI(title="RAG Apparel Store Chatbot")
app.include_router(router)
