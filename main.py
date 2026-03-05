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
import numpy as np

class Book(BaseModel): # Inherit from BaseModel
    title: str
    author: str
    synopsis: str
    
class BooksList3(BaseModel):
    books: List[Book]  

class BooksList20(BaseModel):
    books: List[Book]

class Books_Genre(BaseModel):
    genres: List[str]
    


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
    

def cosine_similarity(a, b):
    a = np.array(a)
    b = np.array(b)
    return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))


def build_rag_context(similar_books):
    dropped = []
    kept = []

    for book in similar_books:
        title = (book.get("title") or "").strip()
        if not title:
            continue

        status = book.get("status")
        reason = (book.get("drop_reason_text") or "").strip()
        tags = book.get("drop_reason_tags")

        if status == "dropped":
            reason_text = reason if reason else "no reason provided"
            tag_text = ""
            if isinstance(tags, list) and len(tags) > 0:
                tag_text = f" (tags: {', '.join(tags)})"

            dropped.append(f"- {title}\n  drop reason: {reason_text}{tag_text}")
        else:
            kept.append(f"- {title}")

    context = (
        "User reading-history signals from their library (most similar to the current prompt):\n\n"
    )

    if kept:
        context += "NOT DROPPED / OK:\n" + "\n".join(kept) + "\n\n"

    if dropped:
        context += "DROPPED / AVOID:\n" + "\n".join(dropped) + "\n\n"

    return context.strip()

def count_embedded_books() -> int:
    # head=True means "don't return rows", just metadata/count
    res = (
        supabase
        .table("books")
        .select("id", count="exact", head=True)
        .not_.is_("embedding", "null")
        .execute()
    )
    return res.count or 0

def master_recommend(prompt):
    return


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

@app.get("/rag")
async def retrieve_similar_books(prompt_embedding, top_k = 5):
    res = (
        supabase
        .table("books")
        .select("title, author, synopsis, embedding, status, drop_reason_text, drop_reason_tags")
        .execute()       
    )
    
    books = res.data
    scored_books = []
    for book in books:
        emd = book.get("embedding")
        if not emd:
            continue
        emd = ast.literal_eval(emd)
        emd = np.array(emd, dtype=float)

            
        score = cosine_similarity(prompt_embedding, emd)
        scored_books.append({
        **book,
        "similarity": score
        })

    scored_books.sort(key=lambda x: x["similarity"], reverse=True)
    return scored_books

@app.get("/test_similarity")
async def test_similarity(prompt: str):

    # embed prompt
    prompt_embedding = embed_text(prompt)

    # get similar books
    books = await retrieve_similar_books(prompt_embedding, top_k=5)

    return books

@app.get("/test_context")
async def test_context(prompt: str):

    prompt_embedding = embed_text(prompt)
    prompt_embedding = np.array(prompt_embedding, dtype=float)

    similar_books = await retrieve_similar_books(prompt_embedding, top_k=5)

    context = build_rag_context(similar_books)

    return {
        "context": context,
        "books_used": similar_books
    }


@app.get("/recommend_rag", response_model=BooksList)
async def recommend_rag(prompt: str, min_history: int = 5):
    embedded_count = count_embedded_books()

    def llm_json(
        prompt_text: str,
        *,
        schema,
        temperature: float = 0.7,
        system_instruction: str = "Return JSON only. No extra text.",
    ):
        config = GenerateContentConfig(
            temperature=temperature,
            system_instruction=system_instruction,
            response_mime_type="application/json",
            response_schema=schema,
        )
        resp = genai_client.models.generate_content(
            model="gemini-2.5-flash",
            contents=[prompt_text],
            config=config,
        )
        return json.loads(resp.text)

    if embedded_count < min_history:
        base_prompt = (
            f"USER PROMPT:\n{prompt}\n\n"
            "TASK:\n"
            "Recommend exactly 3 books as JSON in the schema.\n"
            "Keep synopsis concise.\n"
        )
        return llm_json(
            base_prompt,
            schema=BooksList,
            temperature=0.7,
            system_instruction="Return JSON only. Recommend exactly 3 books that best match the user's prompt. Keep synopsis concise.",
        )

    prompt_embedding = np.array(embed_text(prompt), dtype=float)
    similar_books = await retrieve_similar_books(prompt_embedding, top_k=5)
    context = build_rag_context(similar_books)

    gen_prompt = (
        f"USER PROMPT:\n{prompt}\n\n"
        "TASK:\n"
        "Generate exactly 8 candidate books as JSON in the schema.\n"
        "Keep synopsis concise.\n"
    )
    candidates = llm_json(
        gen_prompt,
        schema=BooksList,
        temperature=0.7,
        system_instruction="Return JSON only. Generate exactly 8 candidate books that match the user's prompt.",
    )

    rerank_prompt = (
        f"USER PROMPT:\n{prompt}\n\n"
        f"USER LIBRARY CONTEXT:\n{context}\n\n"
        "CANDIDATES (choose only from these; do not add new titles):\n"
        f"{json.dumps(candidates, ensure_ascii=False)}\n\n"
        "TASK:\n"
        "Select exactly 3 books FROM THE CANDIDATES that best fit the user.\n"
        "Penalize candidates similar to DROPPED/AVOID signals.\n"
        "Do NOT include any title that appears in USER LIBRARY CONTEXT.\n"
        "Return JSON in the schema with exactly 3 books.\n"
    )
    top3 = llm_json(
        rerank_prompt,
        schema=BooksList,
        temperature=0.3,
        system_instruction="Return JSON only. Output exactly 3 books selected from the provided candidates. Do not add new titles.",
    )

    return top3


@app.get("/books/{book_name}")
async def find_book(book_name: str):
    response = await client.get(f"{EXTERNAL_API_URL}?q=intitle:{book_name}&key={google_books_key}")
    return response.json()

    

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


@app.get("/summarize/{book}")
def summarize_book(book: str) -> str: 
    config = GenerateContentConfig(
        temperature=0.2,
        system_instruction="Return a string",
        # response_mime_type="application/json",
        response_schema=str,
    )
    
    prompt = f"Please give a short but dense summary of what happens in {book} rather than a brief synopsis. Format it similar to a wikipedia plot summary"
    response = genai_client.models.generate_content(
            model='gemini-2.5-flash', 
            contents=[prompt],
            config=config,
        )
    
    return response.text
    
