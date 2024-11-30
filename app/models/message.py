from pydantic import BaseModel
from typing import Dict, List, Optional

class Message(BaseModel):
    """Message data model."""
    id: str
    content: str
    timestamp: float
    sender: str
    metadata: Optional[Dict] = None