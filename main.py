from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
import sqlite3
from fastapi.staticfiles import StaticFiles


# Your existing chatbot code...

import os
os.environ["OPENAI_API_KEY"] = "Your key"

# Embeddings
from langchain.embeddings import OpenAIEmbeddings
embeddings = OpenAIEmbeddings()

from langchain.vectorstores import FAISS
db = FAISS.load_local("faiss_index", embeddings)

#QA CHAIN
#Adding Memory
from langchain.chains import RetrievalQAWithSourcesChain
from langchain.chains.question_answering import load_qa_chain
from langchain.chat_models import ChatOpenAI

llm=ChatOpenAI(temperature=0, model_name='gpt-3.5-turbo')

chain = load_qa_chain(llm, chain_type="stuff")

def get_similar_documents(query):
    docs = db.similarity_search(query)
    return docs

def get_chatbot_response(question, docs):
    response = chain.run(input_documents=docs, question=question)
    return response

# Function to insert the query and chatbot response into the database
def insert_into_database(query, response):
    cursor.execute('INSERT INTO chatbot_history (query, chatbot_response) VALUES (?, ?)', (query, response))
    conn.commit()


# Connect to the SQLite database on Google Drive
database_path = "chatbot_database.db"
conn = sqlite3.connect(database_path)
cursor = conn.cursor()

app = FastAPI()

from fastapi.middleware.cors import CORSMiddleware

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# Mount the static files directory
app.mount("/templates", StaticFiles(directory="templates"), name="templates")

templates = Jinja2Templates(directory="templates")

@app.get("/", response_class=HTMLResponse)
async def read_index(request: Request):
    return templates.TemplateResponse("index.html", {"request": request})

print("Debugging")

from typing import List
from pydantic import BaseModel

class ChatMessage(BaseModel):
    role: str
    content: str

@app.post("/api/chatbot")
async def chat_with_bot(message: ChatMessage):
    query = message.content
    relevant_docs = get_similar_documents(query)
    chatbot_response = get_chatbot_response(query, relevant_docs)
    insert_into_database(query, chatbot_response)
    
    return {"response": chatbot_response}

