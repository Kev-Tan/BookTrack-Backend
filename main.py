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
load_dotenv()
google_books_key = os.getenv("GOOGLE_BOOKS_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
supabase_url = os.getenv("SUPABASE_URL")
supabase_key = os.getenv("SUPABASE_SERVICE_ROLE_KEY")


app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5173"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

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


@app.post("/dropBookRequest")
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


@app.get("/supabase")
async def get_supabase():
    res = supabase.table("books").select("title").execute()
    return res.data

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
