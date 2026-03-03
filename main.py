from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import httpx
from fastapi.middleware.cors import CORSMiddleware
from dotenv import load_dotenv
import os

load_dotenv()
key = os.getenv("GOOGLE_BOOKS_API_KEY")

app = FastAPI()

app.add_middleware(
    CORSMiddleware,
    allow_origins = ["http://localhost:5173"],
    allow_credentials = True,
    allow_methods=["*"],
    allow_headers=["*"],
)

EXTERNAL_API_URL =  "https://www.googleapis.com/books/v1/volumes"
client = httpx.AsyncClient()


@app.get("/books/{book_name}")
async def find_book(book_name: str):
    response = await client.get(f"{EXTERNAL_API_URL}?q=intitle:{book_name}&key={key}")
    return response.json()
    