# agents/coordinator.py
from typing import Dict, List, Optional, Tuple
from datetime import datetime
import logging
from app.models.message import Message
from app.models.state import (
    ConversationState, 
    TherapeuticFramework, 
    EmotionalState, 
    SafetyStatus,
    TherapeuticState
)
from .assessor import AssessmentAgent
from .therapist import TherapistAgent
from .safety import SafetyAgent

logger = logging.getLogger(__name__)

class CoordinatorAgent:
    """Main therapeutic conversation coordinator agent."""
    
    def __init__(self, config: Dict = None):
        self.assessment_agent = AssessmentAgent()
        self.therapist_agent = TherapistAgent()
        self.safety_agent = SafetyAgent()
        
        self.config = config or {}
        self.crisis_threshold = self.config.get('crisis_threshold', 0.7)
        
        # Define therapeutic framework selection criteria
        self.framework_selection_rules = {
            'anxiety': TherapeuticFramework.CBT,
            'depression': TherapeuticFramework.CBT,
            'emotional_dysregulation': TherapeuticFramework.DBT,
            'trauma': TherapeuticFramework.PERSON_CENTERED,
            'stress': TherapeuticFramework.MINDFULNESS,
            'relationship_issues': TherapeuticFramework.SOLUTION_FOCUSED
        }
        
    async def process_message(
        self, 
        message: Message, 
        state: Optional[ConversationState] = None
    ) -> Tuple[Message, ConversationState]:
        """
        Process incoming message through the therapeutic pipeline.
        
        Args:
            message: Incoming message to process
            state: Current conversation state (optional)
            
        Returns:
            Tuple of (response message, updated state)
        """
        try:
            # Initialize or update conversation state
            current_state = state or await self._initialize_state()
            
            # Update message history
            current_state.messages.append(message)
            if len(current_state.messages) > 10:  # Keep last 10 messages
                current_state.messages.pop(0)
            
            # Perform emotional and safety assessment
            emotional_state, safety_status = await self.assessment_agent.analyze(
                message,
                current_state.messages
            )
            
            # Update state with new assessments
            current_state.emotional_state = emotional_state
            current_state.safety_status = safety_status
            
            # Check for crisis situation
            if safety_status.risk_level >= self.crisis_threshold:
                response, updated_state = await self._handle_crisis(
                    message, 
                    current_state
                )
                return response, updated_state
            
            # Update therapeutic approach if needed
            current_state.therapeutic_state = await self._update_therapeutic_approach(
                current_state
            )
            
            # Generate therapeutic response
            response = await self.therapist_agent.generate_response(
                message,
                current_state
            )
            
            # Update state with response
            current_state.messages.append(response)
            
            return response, current_state
            
        except Exception as e:
            logger.error(f"Error processing message: {e}", exc_info=True)
            return await self._handle_error(current_state), current_state
    
    async def _initialize_state(self) -> ConversationState:
        """Initialize a new conversation state."""
        return ConversationState(
            messages=[],
            emotional_state=EmotionalState(
                primary_emotion="neutral",
                intensity=0.0,
                valence=0.0,
                arousal=0.0,
                secondary_emotions=[]
            ),
            therapeutic_state=TherapeuticState(
                active_framework=TherapeuticFramework.PERSON_CENTERED,
                session_goals=[],
                progress_markers={},
                interventions_used=[]
            ),
            safety_status=SafetyStatus(
                risk_level=0.0,
                crisis_indicators=[],
                last_assessment=datetime.now(),
                recommended_actions=[]
            ),
            metadata={}
        )
    
    async def _handle_crisis(
        self, 
        message: Message, 
        state: ConversationState
    ) -> Tuple[Message, ConversationState]:
        """Handle crisis situations."""
        try:
            # Log crisis detection
            logger.warning(f"Crisis detected. Risk level: {state.safety_status.risk_level}")
            
            # Generate crisis response
            crisis_response = Message(
                content=self._generate_crisis_message(state.safety_status),
                sender="bot",
                timestamp=datetime.now().timestamp(),
                metadata={
                    "crisis": True,
                    "risk_level": state.safety_status.risk_level,
                    "crisis_indicators": state.safety_status.crisis_indicators
                }
            )
            
            # Update state for crisis handling
            state.therapeutic_state.active_framework = TherapeuticFramework.DBT
            state.therapeutic_state.interventions_used.append("crisis_intervention")
            state.messages.append(crisis_response)
            
            return crisis_response, state
            
        except Exception as e:
            logger.error(f"Error handling crisis: {e}", exc_info=True)
            return await self._handle_error(state), state
    
    async def _update_therapeutic_approach(
        self, 
        state: ConversationState
    ) -> TherapeuticState:
        """Update therapeutic approach based on emotional state and progress."""
        current_approach = state.therapeutic_state
        
        # Check if we should change the framework
        new_framework = self._select_framework(
            state.emotional_state,
            state.safety_status
        )
        
        if new_framework != current_approach.active_framework:
            current_approach.active_framework = new_framework
            current_approach.interventions_used = []
            
            # Update session goals based on new framework
            current_approach.session_goals = self._generate_framework_goals(
                new_framework,
                state.emotional_state
            )
        
        return current_approach
    
    def _select_framework(
        self, 
        emotional_state: EmotionalState,
        safety_status: SafetyStatus
    ) -> TherapeuticFramework:
        """Select appropriate therapeutic framework based on user state."""
        # If high risk, prefer DBT for emotion regulation
        if safety_status.risk_level > 0.5:
            return TherapeuticFramework.DBT
            
        # Check primary emotion against framework rules
        if emotional_state.primary_emotion.lower() in self.framework_selection_rules:
            return self.framework_selection_rules[emotional_state.primary_emotion.lower()]
            
        # Default to person-centered approach
        return TherapeuticFramework.PERSON_CENTERED
    
    def _generate_framework_goals(
        self, 
        framework: TherapeuticFramework,
        emotional_state: EmotionalState
    ) -> List[str]:
        """Generate appropriate goals for the therapeutic framework."""
        framework_goals = {
            TherapeuticFramework.CBT: [
                "Identify thought patterns",
                "Challenge cognitive distortions",
                "Develop coping strategies"
            ],
            TherapeuticFramework.DBT: [
                "Improve emotion regulation",
                "Build distress tolerance",
                "Practice mindfulness"
            ],
            TherapeuticFramework.PERSON_CENTERED: [
                "Explore feelings and experiences",
                "Build self-awareness",
                "Develop self-acceptance"
            ],
            TherapeuticFramework.MINDFULNESS: [
                "Present moment awareness",
                "Non-judgmental observation",
                "Emotional awareness"
            ],
            TherapeuticFramework.SOLUTION_FOCUSED: [
                "Identify solutions",
                "Set achievable goals",
                "Build on strengths"
            ]
        }
        
        return framework_goals.get(framework, framework_goals[TherapeuticFramework.PERSON_CENTERED])
    
    def _generate_crisis_message(self, safety_status: SafetyStatus) -> str:
        """Generate appropriate crisis response message."""
        crisis_message = (
            "I notice you're going through a really difficult time right now. "
            "Your safety and well-being are the top priority. "
            "\n\n"
            "Please remember that I'm an AI assistant and not a replacement for "
            "professional help. Here are some immediate steps you can take:\n\n"
        )
        
        # Add recommended actions
        for action in safety_status.recommended_actions:
            crisis_message += f"- {action}\n"
            
        crisis_message += (
            "\nIf you're having thoughts of harming yourself or others, please:\n"
            "1. Call emergency services (911 in the US)\n"
            "2. Contact the National Crisis Hotline: 988\n"
            "3. Reach out to a trusted person or mental health professional\n"
            "\nWould you be willing to tell me if you're safe right now?"
        )
        
        return crisis_message
    
    async def _handle_error(self, state: Optional[ConversationState]) -> Message:
        """Generate safe fallback response for error situations."""
        error_message = (
            "I apologize, but I'm having trouble processing that properly. "
            "Could you rephrase what you're trying to tell me? "
            "I want to make sure I understand and respond appropriately."
        )
        
        return Message(
            content=error_message,
            sender="bot",
            timestamp=datetime.now().timestamp(),
            metadata={"error": True}
        )