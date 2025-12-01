import os
import shutil

from typing import List
from fastapi import FastAPI, UploadFile, File, HTTPException
from pydantic import BaseModel
from supabase import create_client, Client
from dotenv import load_dotenv
# LangChain Imports
from langchain_community.document_loaders import PyPDFLoader
from langchain_experimental.text_splitter import SemanticChunker
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_core.documents import Document
load_dotenv()

# Env variables
SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")


app = FastAPI(title="Supabase RAG API")

# Initialize Supabase Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# Initialize Embeddings (Must match the dimension in your SQL table: 384)
embeddings = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")

class QueryRequest(BaseModel):
    query: str
    k: int = 5  # top K chunks


class QueryResponse(BaseModel):
    context: str

@app.post("/upload")
async def upload_document(file: UploadFile = File(...)):
    """
    Upload PDF → Semantic Chunk → Store into Supabase with vector embeddings.
    """
    if not file.filename.lower().endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files allowed")

    temp_path = f"temp_{file.filename}"

    try:
        # Step 1: Save PDF temporarily
        with open(temp_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Step 2: Load PDF
        loader = PyPDFLoader(temp_path)
        docs: List[Document] = loader.load()

        # Step 3: Semantic Chunking
        splitter = SemanticChunker(
            embeddings=embeddings,
            breakpoint_threshold_type="percentile"
        )
        chunks = splitter.split_documents(docs)

        # Step 4: Insert into Supabase
        for chunk in chunks:
            vector = embeddings.embed_query(chunk.page_content)

            supabase.table("documents").insert({
                "content": chunk.page_content,
                "metadata": {"source": file.filename},
                "embedding": vector
            }).execute()

        return {
            "message": "PDF uploaded & embedded successfully",
            "chunks": len(chunks)
        }

    except Exception as e:
        # Log the error for debugging
        print(f"Upload error: {e}")
        raise HTTPException(status_code=500, detail=f"Internal Server Error: {str(e)}")
    finally:
        # Cleanup: Remove the temporary file
        if os.path.exists(temp_path):
            os.remove(temp_path)

@app.post("/query", response_model=QueryResponse)
async def query_db(request: QueryRequest):
    
    try:
        # 1. Generate embedding for the user query
        query_embedding = embeddings.embed_query(request.query)

        # 2. RPC parameters must be a dictionary
        rpc_params = {
            "filter": {},
            "match_count": request.k,
            "match_threshold": 0.0,
            "query_embedding": query_embedding
        }

        # 3. Build RPC request (sync)
        rpc_builder = supabase.rpc("match_documents", rpc_params)

        # 4. Execute RPC (SYNC CALL — correct)
        rpc = rpc_builder.execute()

        if not rpc.data:
            return {"context": "No relevant documents found."}

        # 5. Build context text
        parts = [f"--- CHUNK ---\n{row['content']}" for row in rpc.data]
        context = "\n\n".join(parts)

        return {"context": context}

    except Exception as e:
        print(f"Query error: {e}")
        raise HTTPException(status_code=500, detail=str(e))
    

if __name__ == "__main__":
    import uvicorn
    port = int(os.environ.get("PORT", 8080))  # Default 8080 for local
    uvicorn.run("main:app", host="0.0.0.0", port=port)
