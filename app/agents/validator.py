from typing import Optional
import re
from app.models.message import Message

class ValidatorAgent:
    """Response validation agent."""
    
    def __init__(self):
        self.safety_phrases = [
            "harm yourself",
            "end your life",
            "suicide"
        ]
        
        self.professional_boundaries = [
            "I am not a licensed therapist",
            "This is not medical advice",
            "Please seek professional help"
        ]
    
    async def validate(self, message: Message) -> Optional[str]:
        """Validate therapeutic response for safety and appropriateness."""
        content = message.content.lower()
        
        # Check for safety concerns
        if any(phrase in content for phrase in self.safety_phrases):
            if not self._has_safety_disclaimer(content):
                return "Safety disclaimer required"
        
        # Check professional boundaries
        if message.metadata.get('crisis', False):
            if not self._has_boundary_statement(content):
                return "Professional boundary statement required"
        
        # Check response length
        if len(content.split()) < 10:
            return "Response too short"
        
        return None
    
    def _has_safety_disclaimer(self, content: str) -> bool:
        """Check if content includes appropriate safety disclaimers."""
        safety_patterns = [
            r"crisis.*hotline",
            r"emergency.*services",
            r"professional.*help"
        ]
        return any(re.search(pattern, content) for pattern in safety_patterns)
    
    def _has_boundary_statement(self, content: str) -> bool:
        """Check if content includes professional boundary statements."""
        return any(statement in content for statement in self.professional_boundaries)