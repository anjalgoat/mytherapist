from typing import Dict, Tuple, List
from datetime import datetime
from textblob import TextBlob
from app.models.message import Message
from app.models.state import EmotionalState, SafetyStatus

class AssessmentAgent:
    """Clinical assessment agent for emotional state and safety analysis."""
    
    def __init__(self):
        # Crisis keywords and their weights
        self.crisis_indicators = {
            'suicide': 1.0,
            'kill': 0.8,
            'die': 0.7,
            'hurt': 0.6,
            'harm': 0.6,
            'end': 0.5,
            'worthless': 0.5,
            'hopeless': 0.5
        }
        
        # Simple emotion mapping based on polarity and subjectivity
        self.emotion_map = {
            (-1.0, -0.6): {
                (0.0, 0.4): "detached",
                (0.4, 0.7): "sad",
                (0.7, 1.0): "distressed"
            },
            (-0.6, -0.2): {
                (0.0, 0.4): "tired",
                (0.4, 0.7): "anxious",
                (0.7, 1.0): "frustrated"
            },
            (-0.2, 0.2): {
                (0.0, 0.4): "neutral",
                (0.4, 0.7): "focused",
                (0.7, 1.0): "engaged"
            },
            (0.2, 0.6): {
                (0.0, 0.4): "calm",
                (0.4, 0.7): "pleased",
                (0.7, 1.0): "happy"
            },
            (0.6, 1.0): {
                (0.0, 0.4): "content",
                (0.4, 0.7): "excited",
                (0.7, 1.0): "elated"
            }
        }
    
    async def analyze(self, message: Message, 
                     conversation_history: List[Message]) -> Tuple[EmotionalState, SafetyStatus]:
        """
        Analyze message for emotional content and safety concerns.
        
        Args:
            message: Current message to analyze
            conversation_history: Previous messages for context
            
        Returns:
            Tuple of EmotionalState and SafetyStatus
        """
        # Analyze text using TextBlob
        blob = TextBlob(message.content)
        
        # Get polarity (-1 to 1) and subjectivity (0 to 1)
        polarity = blob.sentiment.polarity
        subjectivity = blob.sentiment.subjectivity
        
        # Map to emotional state
        emotional_state = self._map_to_emotion(polarity, subjectivity)
        
        # Perform safety assessment
        safety_status = await self._assess_safety(message.content, 
                                                emotional_state,
                                                conversation_history)
        
        return emotional_state, safety_status
    
    def _map_to_emotion(self, polarity: float, subjectivity: float) -> EmotionalState:
        """Map TextBlob sentiment to emotional state."""
        # Find the right emotion based on polarity and subjectivity ranges
        primary_emotion = "neutral"
        for (pol_min, pol_max) in self.emotion_map:
            if pol_min <= polarity <= pol_max:
                subj_ranges = self.emotion_map[(pol_min, pol_max)]
                for (subj_min, subj_max) in subj_ranges:
                    if subj_min <= subjectivity <= subj_max:
                        primary_emotion = subj_ranges[(subj_min, subj_max)]
                        break
                break
        
        return EmotionalState(
            primary_emotion=primary_emotion,
            intensity=subjectivity,
            valence=polarity,
            arousal=self._calculate_arousal(polarity, subjectivity),
            secondary_emotions=[]
        )
    
    async def _assess_safety(self, 
                           text: str, 
                           emotional_state: EmotionalState,
                           history: List[Message]) -> SafetyStatus:
        """Assess message for safety concerns and crisis indicators."""
        risk_score = 0.0
        crisis_indicators = []
        
        # Check for crisis keywords
        words = text.lower().split()
        for word in words:
            if word in self.crisis_indicators:
                risk_score = max(risk_score, self.crisis_indicators[word])
                crisis_indicators.append(word)
        
        # Factor in emotional state
        if emotional_state.valence < -0.8 and emotional_state.intensity > 0.7:
            risk_score += 0.2
            
        # Analyze patterns in conversation history
        if history:
            history_risk = self._analyze_history_risk(history)
            risk_score = max(risk_score, history_risk)
            
        return SafetyStatus(
            risk_level=min(1.0, risk_score),
            crisis_indicators=crisis_indicators,
            last_assessment=datetime.now(),
            recommended_actions=self._get_safety_recommendations(risk_score)
        )
        
    def _calculate_arousal(self, polarity: float, subjectivity: float) -> float:
        """Calculate emotional arousal based on sentiment metrics."""
        # Combine absolute polarity and subjectivity for arousal
        # High arousal = strong feelings (positive or negative) + high subjectivity
        return min(1.0, (abs(polarity) + subjectivity) / 2)
    
    def _analyze_history_risk(self, history: List[Message]) -> float:
        """Analyze conversation history for risk patterns."""
        if not history:
            return 0.0
            
        # Look at recent messages for negative patterns
        recent_messages = history[-3:]  # Last 3 messages
        risk_score = 0.0
        
        for msg in recent_messages:
            blob = TextBlob(msg.content)
            # If very negative sentiment in recent messages, increase risk
            if blob.sentiment.polarity < -0.7:
                risk_score += 0.1
                
        return min(1.0, risk_score)
        
    def _get_safety_recommendations(self, risk_score: float) -> List[str]:
        """Get safety recommendations based on risk score."""
        recommendations = []
        
        if risk_score > 0.8:
            recommendations.append("Immediate professional intervention recommended")
            recommendations.append("Provide crisis hotline information")
        elif risk_score > 0.6:
            recommendations.append("Suggest professional consultation")
            recommendations.append("Offer grounding exercises")
        elif risk_score > 0.4:
            recommendations.append("Monitor closely")
            recommendations.append("Provide coping strategies")
            
        return recommendations