from typing import Dict, List
import datetime
from app.models.message import Message
from app.models.state import SafetyStatus

class SafetyAgent:
    """Crisis detection and safety monitoring agent."""
    
    def __init__(self):
        self.crisis_keywords = {
            'suicide': 1.0,
            'kill': 0.9,
            'die': 0.8,
            'hurt': 0.7,
            'end': 0.6
        }
    
    async def evaluate_risk(self, message: Message, history: List[Message] = None) -> SafetyStatus:
        """Evaluate message for crisis indicators and safety concerns."""
        risk_score = 0.0
        crisis_indicators = []
        
        # Check message content
        for word, weight in self.crisis_keywords.items():
            if word in message.content.lower():
                risk_score = max(risk_score, weight)
                crisis_indicators.append(word)
        
        # Check conversation patterns if history provided
        if history:
            pattern_risk = self._evaluate_patterns(history)
            risk_score = max(risk_score, pattern_risk)
        
        return SafetyStatus(
            risk_level=risk_score,
            crisis_indicators=crisis_indicators,
            last_assessment=datetime.datetime.now(),
            recommended_actions=self._get_recommendations(risk_score)
        )
    
    def _evaluate_patterns(self, history: List[Message]) -> float:
        """Evaluate conversation history for concerning patterns."""
        risk_score = 0.0
        # Implement pattern recognition logic
        return risk_score
    
    def _get_recommendations(self, risk_score: float) -> List[str]:
        """Get safety recommendations based on risk level."""
        if risk_score > 0.8:
            return ["Immediate professional help required", "Crisis hotline contact"]
        elif risk_score > 0.6:
            return ["Suggest professional consultation", "Provide support resources"]
        return ["Monitor situation", "Offer support"]