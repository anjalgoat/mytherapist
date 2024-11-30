from enum import Enum
from pydantic import BaseModel, Field
from typing import List, Dict, Optional
from datetime import datetime
from app.models.message import Message

class EmotionalState(BaseModel):
    """Emotional state assessment model."""
    primary_emotion: str
    intensity: float = Field(ge=0, le=1)
    secondary_emotions: List[str] = []
    valence: float = Field(ge=-1, le=1)
    arousal: float = Field(ge=0, le=1)

class TherapeuticFramework(str, Enum):
    CBT = "cognitive_behavioral"
    DBT = "dialectical_behavioral"
    PERSON_CENTERED = "person_centered"
    MINDFULNESS = "mindfulness"
    SOLUTION_FOCUSED = "solution_focused"

class TherapeuticState(BaseModel):
    """Therapeutic progress and approach tracking."""
    active_framework: TherapeuticFramework
    session_goals: List[str] = []
    progress_markers: Dict[str, float] = {}
    interventions_used: List[str] = []
    
class SafetyStatus(BaseModel):
    """Safety and crisis assessment status."""
    risk_level: float = Field(ge=0, le=1)
    crisis_indicators: List[str] = []
    last_assessment: datetime
    recommended_actions: List[str] = []

class ConversationState(BaseModel):
    """Enhanced conversation state model."""
    messages: List[Message]
    emotional_state: EmotionalState
    therapeutic_state: TherapeuticState
    safety_status: SafetyStatus
    metadata: Dict[str, any] = {}


