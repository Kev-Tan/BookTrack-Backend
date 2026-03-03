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





app = FastAPI()
load_dotenv()
google_books_key = os.getenv("GOOGLE_BOOKS_API_KEY")
gemini_api_key = os.getenv("GEMINI_API_KEY")
app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5173"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

client = httpx.AsyncClient()
genai_client = genai.Client(api_key=gemini_api_key)


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

@app.get("/classify")
async def response(prompt: str):
    config = GenerateContentConfig(
        temperature=0.7,
        system_instruction="You are a helpful assistant that provides concise answers. In this case, please classify the genre of the books given to you. You have [Romance, Thriller, Drama, Horror, Comedy, Philosophy, Memoir, Autobiography, Biography, Educational, Others]. Give the genres of the books fed into you in order",
        response_mime_type="application/json",
        response_schema=Books_Genre,
    )
    
    response = genai_client.models.generate_content(
        model='gemini-2.5-flash', 
        contents=[prompt],
        config=config,
    )
    print(response)
    return response.text
