from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import RedirectResponse
from langserve import add_routes
from app.rag_chain import final_chain
from fastapi.middleware.cors import CORSMiddleware
from starlette.staticfiles import StaticFiles
import os
import shutil
import subprocess
from app.rag_chain import final_chain
from sqlalchemy import create_engine, text
from sqlalchemy.exc import SQLAlchemyError
from supabase import create_client, Client
import asyncio


# engine = create_engine(os.getenv('DATABASE_URL'))

app = FastAPI()
url: str = os.environ.get("SUPABASE_URL")
key: str = os.environ.get("SUPABASE_DANGEROUS_KEY")
supabase: Client = create_client(url, key)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "http://localhost:3000"
    ],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
@app.delete("/delete-chat-history")
async def delete_chat_history():
    try:
        data = supabase.table('message_store').delete().neq('id', 0).execute()
        return {"status": "Chat history deleted successfully"}
    except Exception as e:
        return {"status": "Failed to delete chat history", "error": str(e)}

@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

pdf_directory = "./pdf-documents"

app.mount("/rag/static", StaticFiles(directory="./pdf-documents"), name="static")
@app.get("/")
async def redirect_root_to_docs():
    return RedirectResponse("/docs")

@app.post("/upload")
async def upload_files(files: list[UploadFile] = File(...)):
    for file in files:
        try:
            file_path = os.path.join(pdf_directory, file.filename)
            with open(file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)
        except Exception as e:
            raise HTTPException(status_code=500, detail=f"Could not save file: {e}")
    
    return {"message": "Files uploaded successfully", "filenames": [file.filename for file in files]}

@app.post("/load-and-process-pdfs")
async def load_and_process_pdfs():
    try:
        subprocess.run(["python", "./rag-data-loader/rag_load_and_process.py"], check=True)
        return {"message": "PDFs loaded and processed successfully"}
    except subprocess.CalledProcessError as e:
        return {"error": "Failed to execute script"}

# Edit this to add the chain you want to add
add_routes(app, final_chain, path="/rag")

if __name__ == "__main__":
    import uvicorn

    uvicorn.run(app, host="localhost", port=8000)
