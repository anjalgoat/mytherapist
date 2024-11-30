from typing import Dict, Any, Optional, Annotated, Tuple, TypedDict
import datetime
from langgraph.graph import StateGraph, END
from app.agents import CoordinatorAgent, AssessmentAgent, TherapistAgent, ValidatorAgent
from app.models.message import Message
from app.models.state import ConversationState, EmotionalState, SafetyStatus, TherapeuticState

class ConversationContext(TypedDict):
    message: Message
    state: ConversationState
    assessment: Optional[Tuple[EmotionalState, SafetyStatus]]
    response: Optional[Message]
    validated: bool
    error: Optional[str]

class TherapeuticFlow:
    """Main conversation flow orchestrator using LangGraph."""
    
    def __init__(self):
        self.coordinator = CoordinatorAgent()
        self.assessor = AssessmentAgent()
        self.therapist = TherapistAgent()
        self.validator = ValidatorAgent()
        self.current_state: Optional[ConversationState] = None
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the therapeutic conversation flow graph."""
        
        workflow = StateGraph(ConversationContext)
        
        # Define state transitions
        workflow.add_node("assess", self._assess_message)
        workflow.add_node("check_crisis", self._check_crisis)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("validate_response", self._validate_response)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define the flow
        workflow.set_entry_point("assess")
        
        workflow.add_edge("assess", "check_crisis")
        workflow.add_edge("check_crisis", "generate_response")
        workflow.add_edge("check_crisis", "handle_error")  # For crisis situations
        workflow.add_edge("generate_response", "validate_response")
        workflow.add_edge("validate_response", END)
        workflow.add_edge("validate_response", "generate_response")  # For invalid responses
        workflow.add_edge("handle_error", END)
        
        return workflow
        
    async def process(self, message: Message) -> Dict[str, Any]:
        """Process message through the therapeutic flow."""
        try:
            # Initialize conversation context
            context: ConversationContext = {
                "message": message,
                "state": self.current_state or await self.coordinator._initialize_state(),
                "assessment": None,
                "response": None,
                "validated": False,
                "error": None
            }
            
            # Execute the workflow
            final_context = await self.graph.invoke(context)
            
            # Update current state
            self.current_state = final_context["state"]
            
            return {
                "response": final_context["response"],
                "state": final_context["state"],
                "metadata": final_context["response"].metadata if final_context["response"] else {}
            }
            
        except Exception as e:
            # Handle any unexpected errors
            error_response = Message(
                content="I apologize, but I'm having trouble processing your message. Could you try rephrasing it?",
                sender="bot",
                timestamp=datetime.now().timestamp(),
                metadata={"error": str(e)}
            )
            return {"response": error_response, "state": self.current_state, "metadata": {"error": True}}
    
    async def _assess_message(self, context: ConversationContext) -> ConversationContext:
        """Assess incoming message for emotional content and safety."""
        try:
            emotional_state, safety_status = await self.assessor.analyze(
                context["message"],
                context["state"].messages
            )
            context["assessment"] = (emotional_state, safety_status)
            context["state"].emotional_state = emotional_state
            context["state"].safety_status = safety_status
            return context
        except Exception as e:
            context["error"] = f"Assessment error: {str(e)}"
            return context
    
    async def _check_crisis(self, context: ConversationContext) -> str:
        """Check for crisis situations and determine next step."""
        if not context["assessment"]:
            return "handle_error"
            
        _, safety_status = context["assessment"]
        if safety_status.risk_level >= self.coordinator.crisis_threshold:
            context["error"] = "Crisis situation detected"
            return "handle_error"
            
        return "generate_response"
    
    async def _generate_response(self, context: ConversationContext) -> ConversationContext:
        """Generate therapeutic response."""
        try:
            response = await self.therapist.generate_response(
                context["message"],
                context["state"]
            )
            context["response"] = response
            return context
        except Exception as e:
            context["error"] = f"Response generation error: {str(e)}"
            return context
    
    async def _validate_response(self, context: ConversationContext) -> str:
        """Validate generated response."""
        if not context["response"]:
            return "handle_error"
            
        validation_error = await self.validator.validate(context["response"])
        if validation_error:
            context["error"] = validation_error
            return "generate_response"
            
        context["validated"] = True
        return END
    
    async def _handle_error(self, context: ConversationContext) -> ConversationContext:
        """Handle errors and generate appropriate responses."""
        error_message = "I apologize, but I need to ensure your safety and well-being. "
        
        if context["error"] and "Crisis situation" in context["error"]:
            error_message = self.coordinator._generate_crisis_message(context["state"].safety_status)
        elif context["error"]:
            error_message += f"I'm having some trouble: {context['error']}"
            
        context["response"] = Message(
            content=error_message,
            sender="bot",
            timestamp=datetime.now().timestamp(),
            metadata={"error": True, "error_type": context["error"]}
        )
        
        return context