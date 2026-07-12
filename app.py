import os
import re
import logging
from typing import List, Dict, Optional
from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from dotenv import load_dotenv
from langchain_groq import ChatGroq
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document

# Load environment variables from .env
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("rag-app")

app = FastAPI(title="GigaCorp Customer Support Assistant")


# ----------------------------
# Load Chroma Vector Database
# ----------------------------
embedding = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
)

vector_db = Chroma(
    persist_directory="./chroma_db",
    embedding_function=embedding
)

retriever = vector_db.as_retriever(
    search_kwargs={"k": 4}
)

def normalize_provider(provider: Optional[str]) -> str:
    if provider is None:
        return "groq"
    provider = provider.strip().lower()
    if provider not in {"groq", "openai"}:
        return "groq"
    return provider



def get_user_safe_error_message(error: Exception) -> str:
    return str(error)





def build_fallback_response(question: str, relevant_docs: List[Document]) -> Dict[str, object]:
    sources = []

    for doc in relevant_docs:
        sources.append({
            "content": doc.page_content,
            "line_number": doc.metadata.get("line_number", "-"),
            "section": doc.metadata.get("section", "FAQ"),
            "source": doc.metadata.get("source", "gigacorp_faq.txt")
        })

    if relevant_docs:
        cited_doc = relevant_docs[0]
        answer = (
            "The live Groq model is currently unavailable, so I can’t verify the answer in real time. "
            f"The closest FAQ reference is: '{cited_doc.page_content}' "
            f"({cited_doc.metadata.get('section')}, Line {cited_doc.metadata.get('line_number')})."
        )
    else:
        answer = (
            "The live Groq model is currently unavailable, so I can’t verify that answer in real time. "
            "Please try again in a moment."
        )

    return {
        "answer": answer,
        "sources": sources,
        "fallback": True,
    }


def create_llm(provider: str, api_key: Optional[str]):
    if provider == "groq":
        from langchain_groq import ChatGroq
        return ChatGroq(
            model="llama-3.3-70b-versatile",
            api_key=api_key,
            temperature=0,
        )

    if provider == "openai":
        from langchain_openai import ChatOpenAI
        return ChatOpenAI(
            model="gpt-4o-mini",
            openai_api_key=api_key,
            temperature=0.0,
        )

    raise ValueError("Unsupported provider")

# Pydantic schemas for the API
class ChatMessage(BaseModel):
    role: str  # "user" or "assistant"
    content: str

class ChatRequest(BaseModel):
    message: str
    history: List[ChatMessage]
    

@app.post("/api/chat")
async def chat_endpoint(request: ChatRequest):
    provider = "groq"
    api_key = os.getenv("GROQ_API_KEY")

    relevant_docs = retriever.invoke(request.message)
    sources = []
    for doc in relevant_docs:
        sources.append({
            "content": doc.page_content,
            "line_number": doc.metadata.get("line_number"),
            "section": doc.metadata.get("section")
        })

    try:
        if not api_key:
            raise RuntimeError("No API key provided")
        
        print("=" * 50)
        print("Provider:", provider)
        print("API Key Exists:", api_key is not None)
        print("API Key Prefix:", api_key[:10] if api_key else "None")
        print("=" * 50)

        llm = create_llm(provider, api_key)

        from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
        lc_history = []
        for msg in request.history:
            if msg.role == "user":
                lc_history.append(HumanMessage(content=msg.content))
            elif msg.role == "assistant":
                lc_history.append(AIMessage(content=msg.content))

        context_blocks = []

        for doc in relevant_docs:
            context_blocks.append(
                f"""
        Section: {doc.metadata.get('section','FAQ')}
        Line: {doc.metadata.get('line_number','-')}

        {doc.page_content}
        """
            )

        context_text = "\n\n".join(context_blocks)
        context_text = "\n\n".join(context_blocks) if context_blocks else "No relevant FAQ entries were found."

        system_prompt = (
            "You are a helpful customer support assistant for GigaCorp. "
            "Answer the user's question using ONLY the following retrieved context. "
            "If the context does not contain the answer, say honestly that you cannot "
            "find that information in the GigaCorp knowledge base. "
            "Do not make up facts or use outside knowledge. "
            "Always cite the sources in your answer when referencing them, including the section name and line number, "
            "for example: '(Shipping Policies, Line 4)'. Keep your response clear and direct."
        )

        messages = [SystemMessage(content=system_prompt)]
        messages.extend(lc_history)
        messages.append(HumanMessage(content=f"Question: {request.message}\n\nRetrieved Context:\n{context_text}"))

        print("\n========== Retrieved Documents ==========\n")

        for doc in relevant_docs:
            print(doc.page_content)
            print(doc.metadata)
            print("-----------------------------------")

        response = llm.invoke(messages)
        answer = response.content if hasattr(response, "content") else str(response)

        return {
            "answer": answer,
            "sources": sources,
            "fallback": False,
        }

    except Exception as e:
        logger.error(f"Error handling chat RAG pipeline: {str(e)}", exc_info=True)
        raw_error = get_user_safe_error_message(e)
        message_lower = raw_error.lower()

        if "quota" in message_lower or "rate limit" in message_lower or "429" in message_lower:
            return build_fallback_response(request.message, relevant_docs)

        status_code = 400
        if "invalid" in message_lower or "unauthenticated" in message_lower or "401" in message_lower:
            status_code = 401
        elif "not found" in message_lower or "404" in message_lower:
            status_code = 404
        elif "no api key" in message_lower:
            status_code = 400
        else:
            status_code = 500
        raise HTTPException(
            status_code=status_code,
            detail=raw_error
        )

@app.get("/api/faq")
async def get_faq():
    if not os.path.exists("gigacorp_faq.txt"):
        return {"error": "FAQ file not found."}
    with open("gigacorp_faq.txt", "r", encoding="utf-8") as f:
        content = f.read()
    return {"content": content}

# Serve the web client
@app.get("/")
async def root():
    return FileResponse("static/index.html")

# Serve other static files
app.mount("/static", StaticFiles(directory="static"), name="static")

if __name__ == "__main__":
    import uvicorn
    # Start the app on port 8000
    uvicorn.run(app, host="127.0.0.1", port=8000)
