from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os
from google import genai
from google.genai.types import GenerateContentConfig
from typing import List, Optional
from pydantic import BaseModel, Field
import ast
import json
from supabase import create_client, Client




app = FastAPI()
app.add_middleware(
    CORSMiddleware,
    allow_origins = [
        "http://localhost:5173",
        "http://127.0.0.1:5173", "http://127.0.0.1:8000"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

load_dotenv()
google_books_key = os.getenv("GOOGLE_BOOKS_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")



client = httpx.AsyncClient()
genai_client = genai.Client(api_key=gemini_api_key)
supabase = create_client(supabase_url, supabase_key)



EXTERNAL_API_URL =  "https://www.googleapis.com/books/v1/volumes"


@app.get("/books/{book_name}")
async def find_book(book_name: str):
    response = await client.get(f"{EXTERNAL_API_URL}?q=intitle:{book_name}&key={google_books_key}")
    return response.json()



class Books_Genre(BaseModel):
    genres: List[str]
    
class Book(BaseModel): # Inherit from BaseModel
    title: str
    author: str
    synopsis: str

class BooksList(BaseModel): # Inherit from BaseModel
    books: List[Book]
    
class DropBookRequest(BaseModel):
    book_id: str
    reason: str = Field(..., min_length=1, max_length=500)
    
class TagsList(BaseModel):
    tags: List[str]
    
class AddBookRequest(BaseModel):
    title: str = Field(..., min_length=1, max_length=200)
    author: str = Field(..., min_length=1, max_length=200)
    synopsis: str = Field(default="", max_length=5000)
    
    

@app.get("/recommend_books/")
async def get_recommendation(prompt: str):
    config = GenerateContentConfig(
        temperature=0.7,
        system_instruction="You are a helpful assistant that provides concise answers. In this case, please recommend the three most appropriate books based on the user's prompt",
        response_mime_type="application/json",
        response_schema=BooksList,
    )
    
    response = genai_client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=[prompt],
        config=config,
    )
    
    print(response)
    return json.loads(response.text)


@app.post("/dropBook")
async def dropBook(payload: DropBookRequest):
    TAGS = [
        "slow_pacing", "dense_prose", "old_english", "boring_characters",
        "confusing_plot", "too_romantic", "too_scary", "too_violent",
        "not_my_genre", "bad_writing", "uninteresting_world", "offensive_content", "edgy"
    ]
    
    config = GenerateContentConfig(
        temperature=0.2,
        system_instruction="Return JSON only. No extra text.",
        response_mime_type="application/json",
        response_schema=TagsList,
    )

    classify_prompt = (
        f"Allowed tags: {TAGS}\n"
        f"User drop reason: {payload.reason}\n"
        "Output format exactly: {\"tags\": [\"tag1\", \"tag2\"]}. "
        "Only use allowed tags. If none fit, return {\"tags\": []}."
    )
    
    llm_res = genai_client.models.generate_content(
        model="gemini-2.5-flash",
        contents=[classify_prompt],
        config=config,
    )
    
    parsed = json.loads(llm_res.text)
    tags = parsed.get("tags", [])
    
    update_res = (
        supabase
        .table("books")
        .update({
            "status": "dropped",
            "drop_reason_text": payload.reason,
            "drop_reason_tags": tags,
        })
        .eq("id", payload.book_id)
        .execute()
    )
    
    return {"ok": True, "tags": tags, "updated": update_res.data} 

def embed_text(text:str) -> float:
    res = genai_client.models.embed_content(
        model="gemini-embedding-001",
        contents=text,
    )
    return res.embeddings[0].values


@app.post("/embeddings/one/{book_id}")
async def embed_one(book_id: str):
    row_res = (
        supabase
        .table("books")
        .select("id, google_id, title, author, synopsis")
        .eq("google_id", book_id)
        .single()
        .execute()
    )
    row = row_res.data
    if not row:
        raise HTTPException(status_code=404, detail="Book not found")

    title = (row.get("title") or "").strip()
    author = (row.get("author") or "").strip()
    synopsis = (row.get("synopsis") or "").strip()

    text_to_embed = "\n".join([
        f"Title: {title}",
        f"Author: {author}",
        f"Synopsis: {synopsis}" if synopsis else ""
    ]).strip()

    vec = embed_text(text_to_embed)
    if not vec:
        raise HTTPException(status_code=500, detail="Embedding failed")

    supabase.table("books").update({"embedding": vec}).eq("google_id", book_id).execute()
    return {"ok": True, "google_id": book_id, "dims": len(vec)}

# @app.get("/supabase")
# async def get_supabase():
#     res = supabase.table("books").select("title").execute()
#     return res.data

# @app.get("/classify")
# async def response(prompt: str):
#     config = GenerateContentConfig(
#         temperature=0.7,
#         system_instruction="You are a helpful assistant that provides concise answers. In this case, please classify the genre of the books given to you. You have [Romance, Thriller, Drama, Horror, Comedy, Philosophy, Memoir, Autobiography, Biography, Educational, Others]. Give the genres of the books fed into you in order",
#         response_mime_type="application/json",
#         response_schema=Books_Genre,
#     )
    
#     response = genai_client.models.generate_content(
#         model='gemini-2.5-flash', 
#         contents=[prompt],
#         config=config,
#     )
#     print(response)
#     return response.text
