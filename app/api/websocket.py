from fastapi import WebSocket, WebSocketDisconnect
from typing import Dict, Set, Optional
import json
import logging
from datetime import datetime
from ..models.message import Message
from ..graphs.therapeutic_flow import TherapeuticFlow
from ..models.state import ConversationState

logger = logging.getLogger(__name__)

class ConnectionManager:
    """Manage WebSocket connections."""
    
    def __init__(self):
        self.active_connections: Dict[str, WebSocket] = {}
        self.session_states: Dict[str, ConversationState] = {}
    
    async def connect(self, websocket: WebSocket, client_id: str):
        """Handle new WebSocket connection."""
        await websocket.accept()
        self.active_connections[client_id] = websocket
        logger.info(f"Client {client_id} connected")
    
    def disconnect(self, client_id: str):
        """Handle WebSocket disconnection."""
        if client_id in self.active_connections:
            del self.active_connections[client_id]
        if client_id in self.session_states:
            del self.session_states[client_id]
        logger.info(f"Client {client_id} disconnected")
    
    async def send_message(self, client_id: str, message: dict):
        """Send message to specific client."""
        if client_id in self.active_connections:
            await self.active_connections[client_id].send_json(message)

class ChatWebSocket:
    """WebSocket handler for chat communication."""
    
    def __init__(self):
        self.manager = ConnectionManager()
        self.flow = TherapeuticFlow()
        
    async def handle_connection(self, websocket: WebSocket, client_id: str):
        """Handle WebSocket connection lifecycle."""
        try:
            await self.manager.connect(websocket, client_id)
            
            # Send welcome message
            welcome_msg = Message(
                id=str(datetime.now().timestamp()),
                content="Hello! I'm here to listen and support you. How are you feeling today?",
                timestamp=datetime.now().timestamp(),
                sender="bot",
                metadata={"message_type": "welcome"}
            )
            await self.manager.send_message(client_id, welcome_msg.dict())
            
            # Handle messages
            try:
                while True:
                    message = await websocket.receive_json()
                    await self.handle_message(client_id, message)
            except WebSocketDisconnect:
                self.manager.disconnect(client_id)
                
        except Exception as e:
            logger.error(f"Error in WebSocket connection: {e}", exc_info=True)
            self.manager.disconnect(client_id)
    
    async def handle_message(self, client_id: str, data: dict):
        """Process incoming message and generate response."""
        try:
            # Create message object
            message = Message(
                id=str(datetime.now().timestamp()),
                content=data['content'],
                timestamp=datetime.now().timestamp(),
                sender="user",
                metadata=data.get('metadata', {})
            )
            
            # Send typing indicator
            await self.manager.send_message(
                client_id,
                {"type": "typing_indicator", "typing": True}
            )
            
            # Process message through therapeutic flow
            current_state = self.manager.session_states.get(client_id)
            result = await self.flow.process(message)
            
            # Update session state
            self.manager.session_states[client_id] = result['state']
            
            # Send response
            await self.manager.send_message(
                client_id,
                {"type": "typing_indicator", "typing": False}
            )
            await self.manager.send_message(client_id, result['response'].dict())
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            # Send error message
            error_msg = Message(
                id=str(datetime.now().timestamp()),
                content="I apologize, but I'm having trouble processing your message. Could you try rephrasing it?",
                timestamp=datetime.now().timestamp(),
                sender="bot",
                metadata={"error": True}
            )
            await self.manager.send_message(client_id, error_msg.dict())