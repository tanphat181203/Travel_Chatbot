import os
import sys
from fastapi import FastAPI, HTTPException, Depends, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from datetime import datetime, timezone, timedelta
import psycopg2
from psycopg2 import pool as psycopg2_pool
from jose import JWTError, jwt
import uvicorn
from dotenv import load_dotenv
import time

load_dotenv()

try:
    from src.config import DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, DB_ENDPOINT_ID, GOOGLE_API_KEY
    from src.database import conn_pool
    from src.graph_builder import graph_app
    from src.embedding import embedding_model
    from langchain_core.messages import HumanMessage, AIMessage, BaseMessage
except ImportError as e:
    print(f"Error importing from src: {e}. Using placeholders. API will likely fail at runtime until this is fixed.")
    DB_USER, DB_PASSWORD, DB_HOST, DB_PORT, DB_NAME, DB_ENDPOINT_ID, GOOGLE_API_KEY = [None]*7
    conn_pool = None
    graph_app = None
    embedding_model = None
    class HumanMessage:
        def __init__(self, content):
            self.content = content
    class AIMessage:
        def __init__(self, content):
            self.content = content
    BaseMessage = Dict

JWT_SECRET_KEY = os.getenv("JWT_SECRET")
ALGORITHM = "HS256"

app = FastAPI(
    title="Travel Chatbot API",
    version="1.0.0"
)

reusable_oauth2 = HTTPBearer(
    scheme_name="Bearer"
)

class EmbeddingRequest(BaseModel):
    text: Union[str, List[str]]

class EmbeddingResponse(BaseModel):
    embeddings: List[List[float]]
    model: str
    dimensions: int

class TokenData(BaseModel):
    user_id: Optional[int] = None

async def get_current_user(token: HTTPAuthorizationCredentials = Depends(reusable_oauth2)) -> int:
    credentials_exception = HTTPException(
        status_code=status.HTTP_401_UNAUTHORIZED,
        detail="Could not validate credentials",
        headers={"WWW-Authenticate": "Bearer"},
    )
    try:
        payload = jwt.decode(token.credentials, JWT_SECRET_KEY, algorithms=[ALGORITHM])

        user_id: Optional[int] = payload.get("id")
        if user_id is None:
            user_id = payload.get("userId")

        if user_id is None:
            print(f"JWT payload does not contain 'id' or 'userId' field. Payload: {payload}")
            raise credentials_exception

    except JWTError as e:
        print(f"JWTError: {e}")
        raise credentials_exception
    except Exception as e:
        print(f"An unexpected error occurred during JWT decoding: {e}")
        raise credentials_exception
    return user_id

class ChatMessageInput(BaseModel):
    message: str = Field(..., description="The text message sent by the user to the chatbot.")
    session_id: Optional[str] = Field(None, description="An optional identifier for a specific chat session.")

class ChatResponseOutput(BaseModel):
    user_id: int = Field(..., description="The ID of the user (from JWT token).")
    response: str = Field(..., description="The chatbot's generated textual response.")
    session_id: Optional[str] = Field(None, description="The session identifier, mirrored if provided in input.")
    timestamp: datetime = Field(..., description="UTC timestamp of when the response was generated.")

def get_db_connection():
    if conn_pool is None:
        print("conn_pool is None in get_db_connection. Database module likely not initialized.")
        raise HTTPException(status_code=503, detail="Database connection pool not initialized. Check src.database and .env configuration.")
    try:
        conn = conn_pool.getconn()
        yield conn
    finally:
        if conn:
            conn_pool.putconn(conn)

def fetch_conversation_history(db_conn, user_id: int) -> List[BaseMessage]:
    history: List[BaseMessage] = []
    try:
        with db_conn.cursor(cursor_factory=psycopg2.extras.DictCursor) as cursor:
            cursor.execute(
                "SELECT message, response FROM ChatbotHistory WHERE user_id = %s ORDER BY interaction_time ASC",
                (user_id,)
            )
            records = cursor.fetchall()
            for record in records:
                if record["message"]:
                    history.append(HumanMessage(content=record["message"]))
                if record["response"]:
                    history.append(AIMessage(content=record["response"]))
    except Exception as e:
        print(f"Error fetching conversation history for user_id {user_id}: {e}")
    return history

def save_interaction_to_history(db_conn, user_id: int, user_message: str, chatbot_response: str):
    try:
        with db_conn.cursor() as cursor:
            cursor.execute(
                "INSERT INTO ChatbotHistory (user_id, message, response, interaction_time) VALUES (%s, %s, %s, %s)",
                (user_id, user_message, chatbot_response, datetime.now(timezone.utc))
            )
        db_conn.commit()
    except Exception as e:
        print(f"Error saving interaction to history for user_id {user_id}: {e}")
        db_conn.rollback()

@app.post("/api/chat/", response_model=ChatResponseOutput)
async def chat_endpoint(payload: ChatMessageInput, current_user_id: int = Depends(get_current_user), db_conn = Depends(get_db_connection)):
    if graph_app is None:
        print("graph_app is None in chat_endpoint. Graph_builder module likely not initialized.")
        raise HTTPException(status_code=503, detail="Chatbot graph not initialized. Check src.graph_builder.")

    user_id = current_user_id
    user_message_content = payload.message

    history = fetch_conversation_history(db_conn, user_id)
    
    current_message = HumanMessage(content=user_message_content)
    all_messages = history + [current_message]
    
    inputs = {
        "messages": all_messages,
        "user_query": user_message_content,
        "current_date": None,
        "available_locations": None,
        "extracted_entities": None,
        "search_results": None,
        "final_response": None,
        "error": None,
        "routing_decision": None
    }

    full_response_content = ""
    try:
        result = graph_app.invoke(inputs)
        
        if isinstance(result, dict) and "final_response" in result:
            full_response_content = result["final_response"]
        elif isinstance(result, dict) and "messages" in result and result["messages"]:
            last_message = result["messages"][-1]
            if isinstance(last_message, AIMessage):
                full_response_content = last_message.content
        
        if not full_response_content:
            full_response_content = "Sorry, I could not process your request at this moment."

    except Exception as e:
        print(f"Error during graph invocation for user_id {user_id}: {e}")
        raise HTTPException(status_code=500, detail=f"Error processing message with chatbot: {str(e)}")

    save_interaction_to_history(db_conn, user_id, user_message_content, full_response_content)

    return ChatResponseOutput(
        user_id=user_id,
        response=full_response_content,
        session_id=payload.session_id,
        timestamp=datetime.now(timezone.utc)
    )

@app.get("/api/health")
async def health_check():
    return {"status": "ok", "message": "API is running"}

@app.post("/api/embed", response_model=EmbeddingResponse)
async def get_embedding(request: EmbeddingRequest):
    if embedding_model is None:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail="Embedding service not initialized. Check src.embedding module."
        )

    try:
        embeddings = embedding_model.get_embedding(request.text)

        return {
            "embeddings": embeddings,
            "model": embedding_model.model_name,
            "dimensions": len(embeddings[0])
        }
    except Exception as e:
        raise HTTPException(
            status_code=status.HTTP_500_INTERNAL_SERVER_ERROR,
            detail=f"Error generating embeddings: {str(e)}"
        )

@app.on_event("startup")
async def startup_event():
    try:
        if embedding_model:
            embedding_model.load_model()
    except Exception as e:
        print(f"Failed to load embedding model on startup: {str(e)}")