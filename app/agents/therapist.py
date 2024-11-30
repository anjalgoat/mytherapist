from typing import Dict, List, Optional
import groq
from datetime import datetime
import logging
import os
from dotenv import load_dotenv
import streamlit as st

from app.models.message import Message
from app.models.state import (
    ConversationState, 
    TherapeuticFramework, 
    EmotionalState, 
    SafetyStatus
)

logger = logging.getLogger(__name__)

class TherapistAgent:
    """Therapeutic response generation agent."""
    
    def __init__(self):
        try:
            # Get API key from Streamlit secrets
            self.api_key = st.secrets["GROQ_API_KEY"]
            self.model_name = st.secrets.get("MODEL_NAME", "mixtral-8x7b-32768")
            
            if not self.api_key:
                raise ValueError("GROQ_API_KEY is missing in Streamlit secrets")
            
            # Initialize Groq client
            self.client = groq.Groq(api_key=self.api_key)
            logger.info("TherapistAgent initialized with Groq client")
        
        except Exception as e:
            logger.error(f"Error initializing TherapistAgent: {e}")
            raise ValueError(f"Failed to initialize TherapistAgent: {str(e)}")

        # Initialize Groq client
        self.client = groq.Groq(api_key=self.api_key)
        logger.info("TherapistAgent initialized with Groq client")
        
        self.framework_prompts = {
            TherapeuticFramework.CBT: self._get_cbt_prompt,
            TherapeuticFramework.DBT: self._get_dbt_prompt,
            TherapeuticFramework.PERSON_CENTERED: self._get_person_centered_prompt,
            TherapeuticFramework.MINDFULNESS: self._get_mindfulness_prompt,
            TherapeuticFramework.SOLUTION_FOCUSED: self._get_solution_focused_prompt
        }
        
    async def generate_response(
        self, 
        message: Message, 
        state: ConversationState
    ) -> Message:
        """Generate therapeutic response based on user input and conversation state."""
        
        # Build the conversation context
        context = self._build_context(message, state)
        
        # Get framework-specific prompt
        framework_prompt = self.framework_prompts[state.therapeutic_state.active_framework](
            state.emotional_state
        )
        
        # Construct the complete prompt
        prompt = self._construct_prompt(context, framework_prompt, state)
        
        try:
            # Generate response using Groq
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=[
                    {"role": "system", "content": prompt},
                    {"role": "user", "content": message.content}
                ],
                temperature=0.7,
                max_tokens=300
            )
            
            # Process and enhance the response
            processed_response = self._process_response(
                completion.choices[0].message.content,
                state
            )
            
            return Message(
                id="response_" + str(datetime.utcnow().timestamp()),
                content=processed_response,
                sender="bot",
                timestamp=datetime.utcnow().timestamp(),
                metadata={
                    "therapeutic_intent": state.therapeutic_state.active_framework.value,
                    "emotional_target": state.emotional_state.primary_emotion
                }
            )
            
        except Exception as e:
            logger.error(f"Error generating response: {e}", exc_info=True)
            return self._generate_fallback_response(state)
    
    def _build_context(self, message: Message, state: ConversationState) -> str:
        """Build context string from conversation state."""
        context_parts = [
            f"User's primary emotion: {state.emotional_state.primary_emotion}",
            f"Emotional intensity: {state.emotional_state.intensity}",
            f"Current therapeutic framework: {state.therapeutic_state.active_framework.value}",
            f"Session goals: {', '.join(state.therapeutic_state.session_goals)}"
        ]
        
        return "\n".join(context_parts)
        
    def _construct_prompt(
        self, 
        context: str, 
        framework_prompt: str, 
        state: ConversationState
    ) -> str:
        """Construct the complete prompt for response generation."""
        base_prompt = """
        You are a professional therapeutic AI assistant. Your responses should be:
        1. Empathetic and understanding
        2. Professional yet warm
        3. Focused on the user's emotional needs
        4. Based on evidence-based therapeutic techniques
        5. Safe and encouraging
        
        Current Context:
        {context}
        
        Therapeutic Framework Guidelines:
        {framework_prompt}
        
        Safety Level: {safety_level}
        
        Generate a response that:
        - Acknowledges the user's emotions
        - Applies appropriate therapeutic techniques
        - Maintains professional boundaries
        - Encourages healthy coping strategies
        """
        
        return base_prompt.format(
            context=context,
            framework_prompt=framework_prompt,
            safety_level=state.safety_status.risk_level
        )
    
    def _get_cbt_prompt(self, emotional_state: EmotionalState) -> str:
        """Get CBT-specific prompt based on emotional state."""
        return """
        Use Cognitive Behavioral Therapy techniques:
        1. Identify cognitive distortions
        2. Challenge negative thought patterns
        3. Encourage behavioral activation
        4. Guide thought recording
        5. Focus on present situations and specific thoughts
        """
        
    def _get_dbt_prompt(self, emotional_state: EmotionalState) -> str:
        """Get DBT-specific prompt based on emotional state."""
        return """
        Use Dialectical Behavior Therapy techniques:
        1. Practice mindfulness
        2. Focus on emotion regulation
        3. Improve distress tolerance
        4. Enhance interpersonal effectiveness
        5. Find balance between acceptance and change
        """
    
    def _get_person_centered_prompt(self, emotional_state: EmotionalState) -> str:
        """Get person-centered therapy prompt based on emotional state."""
        return """
        Use Person-Centered Therapy techniques:
        1. Show unconditional positive regard
        2. Practice empathetic understanding
        3. Maintain genuineness in responses
        4. Reflect feelings and meanings
        5. Support self-discovery and growth
        """
    
    def _get_mindfulness_prompt(self, emotional_state: EmotionalState) -> str:
        """Get mindfulness-based therapy prompt based on emotional state."""
        return """
        Use Mindfulness-Based techniques:
        1. Encourage present-moment awareness
        2. Guide gentle observation of thoughts and feelings
        3. Promote non-judgmental acceptance
        4. Suggest grounding exercises
        5. Support mindful self-compassion
        """
    
    def _get_solution_focused_prompt(self, emotional_state: EmotionalState) -> str:
        """Get solution-focused therapy prompt based on emotional state."""
        return """
        Use Solution-Focused Brief Therapy techniques:
        1. Focus on solutions rather than problems
        2. Look for exceptions to problems
        3. Set concrete, achievable goals
        4. Use scaling questions
        5. Identify and build on existing strengths
        """
    
    def _process_response(self, response: str, state: ConversationState) -> str:
        """Process and enhance the generated response."""
        # Add safety disclaimers if needed
        if state.safety_status.risk_level > 0.6:
            response += "\n\nPlease remember that I'm an AI assistant. If you're in crisis, " \
                       "please contact emergency services or crisis hotline immediately."
        return response
    
    def _generate_fallback_response(self, state: ConversationState) -> Message:
        """Generate a safe fallback response."""
        return Message(
            id="fallback_" + str(datetime.utcnow().timestamp()),
            content="I understand you're going through something important. " \
                   "Could you tell me more about what you're feeling?",
            sender="bot",
            timestamp=datetime.utcnow().timestamp(),
            metadata={"fallback": True}
        )
