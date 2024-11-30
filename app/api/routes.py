from fastapi import APIRouter, WebSocket, HTTPException, Depends
from fastapi.security import APIKeyHeader
from typing import Dict, Optional
import uuid
from datetime import datetime
from .websocket import ChatWebSocket
from app.models.message import Message
from app.config.settings import Settings

router = APIRouter()
chat_handler = ChatWebSocket()

# API key security
api_key_header = APIKeyHeader(name="X-API-Key")

async def verify_api_key(api_key: str = Depends(api_key_header)):
    """Verify API key."""
    if api_key != Settings().API_KEY:
        raise HTTPException(
            status_code=403,
            detail="Invalid API key"
        )
    return api_key

@router.websocket("/ws/{client_id}")
async def websocket_endpoint(websocket: WebSocket, client_id: str):
    """WebSocket endpoint for chat communication."""
    await chat_handler.handle_connection(websocket, client_id)

@router.post("/message", response_model=Dict)
async def process_message(
    message: dict,
    api_key: str = Depends(verify_api_key)
):
    """REST endpoint for processing messages (alternative to WebSocket)."""
    try:
        # Create message object
        msg = Message(
            id=str(uuid.uuid4()),
            content=message['content'],
            timestamp=datetime.now().timestamp(),
            sender="user",
            metadata=message.get('metadata', {})
        )
        
        # Process message through therapeutic flow
        result = await chat_handler.flow.process(msg)
        
        return {
            "response": result['response'].dict(),
            "metadata": result['metadata']
        }
        
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error processing message: {str(e)}"
        )

@router.get("/health")
async def health_check():
    """Health check endpoint."""
    return {"status": "healthy"}