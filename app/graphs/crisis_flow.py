from typing import Dict, Any, List, TypedDict, Optional
from langgraph.graph import StateGraph, END
from datetime import datetime
from app.agents import SafetyAgent
from app.models.message import Message
from app.models.state import SafetyStatus

class CrisisContext(TypedDict):
    message: Message
    history: List[Message]
    safety_status: Optional[SafetyStatus]
    response: Optional[Message]
    requires_escalation: bool
    error: Optional[str]

class CrisisFlow:
    """Crisis intervention flow using LangGraph."""
    
    def __init__(self):
        self.safety_agent = SafetyAgent()
        self.graph = self._build_graph()
        
    def _build_graph(self) -> StateGraph:
        """Build the crisis intervention flow graph."""
        workflow = StateGraph(CrisisContext)
        
        # Define nodes
        workflow.add_node("evaluate_risk", self._evaluate_risk)
        workflow.add_node("check_escalation", self._check_escalation)
        workflow.add_node("generate_response", self._generate_response)
        workflow.add_node("escalate_crisis", self._escalate_crisis)
        workflow.add_node("handle_error", self._handle_error)
        
        # Define flow
        workflow.set_entry_point("evaluate_risk")
        
        workflow.add_edge("evaluate_risk", "check_escalation")
        workflow.add_edge("check_escalation", "generate_response")
        workflow.add_edge("check_escalation", "escalate_crisis")
        workflow.add_edge("generate_response", END)
        workflow.add_edge("escalate_crisis", END)
        workflow.add_edge("handle_error", END)
        
        return workflow
    
    async def handle_crisis(self, message: Message, history: List[Message] = None) -> Dict[str, Any]:
        """Handle crisis situation through the flow."""
        try:
            # Initialize crisis context
            context: CrisisContext = {
                "message": message,
                "history": history or [],
                "safety_status": None,
                "response": None,
                "requires_escalation": False,
                "error": None
            }
            
            # Execute the workflow
            final_context = await self.graph.invoke(context)
            
            return {
                "response": final_context["response"],
                "safety_status": final_context["safety_status"],
                "requires_escalation": final_context["requires_escalation"]
            }
            
        except Exception as e:
            error_response = Message(
                content="I need to ensure your safety. Please contact emergency services if you're in immediate danger.",
                sender="bot",
                timestamp=datetime.now().timestamp(),
                metadata={"error": str(e), "crisis": True}
            )
            return {
                "response": error_response,
                "requires_escalation": True,
                "error": str(e)
            }
    
    async def _evaluate_risk(self, context: CrisisContext) -> CrisisContext:
        """Evaluate risk level of the crisis situation."""
        try:
            context["safety_status"] = await self.safety_agent.evaluate_risk(
                context["message"],
                context["history"]
            )
            return context
        except Exception as e:
            context["error"] = f"Risk evaluation error: {str(e)}"
            return context
    
    async def _check_escalation(self, context: CrisisContext) -> str:
        """Determine if crisis requires escalation."""
        if not context["safety_status"]:
            return "handle_error"
            
        context["requires_escalation"] = context["safety_status"].risk_level > 0.8
        return "escalate_crisis" if context["requires_escalation"] else "generate_response"
    
    async def _generate_response(self, context: CrisisContext) -> CrisisContext:
        """Generate appropriate crisis response."""
        if not context["safety_status"]:
            return context
            
        response_text = (
            "I'm concerned about your safety and well-being. "
            "Let's focus on keeping you safe right now:\n\n"
        )
        
        for action in context["safety_status"].recommended_actions:
            response_text += f"- {action}\n"
            
        response_text += "\nWould you be willing to tell me if you're safe right now?"
        
        context["response"] = Message(
            content=response_text,
            sender="bot",
            timestamp=datetime.now().timestamp(),
            metadata={
                "crisis": True,
                "risk_level": context["safety_status"].risk_level,
                "recommended_actions": context["safety_status"].recommended_actions
            }
        )
        
        return context
    
    async def _escalate_crisis(self, context: CrisisContext) -> CrisisContext:
        """Handle high-risk crisis situations."""
        escalation_text = (
            "🚨 YOUR SAFETY IS MY TOP PRIORITY 🚨\n\n"
            "I need you to know:\n"
            "1. You're not alone\n"
            "2. Help is available right now\n"
            "3. Your life has value\n\n"
            "Please take one of these immediate actions:\n"
            "- Call Emergency Services (911 in the US)\n"
            "- Contact the Crisis Hotline: 988\n"
            "- Go to the nearest emergency room\n"
            "- Call a trusted person who can be with you\n\n"
            "Will you tell me which action you're going to take?"
        )
        
        context["response"] = Message(
            content=escalation_text,
            sender="bot",
            timestamp=datetime.now().timestamp(),
            metadata={
                "crisis": True,
                "escalated": True,
                "risk_level": context["safety_status"].risk_level if context["safety_status"] else 1.0
            }
        )
        
        return context
    
    async def _handle_error(self, context: CrisisContext) -> CrisisContext:
        """Handle errors during crisis management."""
        error_text = (
            "I'm having trouble processing this situation properly, but your safety is paramount. "
            "Please contact emergency services (911) or the crisis hotline (988) immediately if you're in danger."
        )
        
        context["response"] = Message(
            content=error_text,
            sender="bot",
            timestamp=datetime.now().timestamp(),
            metadata={"error": True, "crisis": True}
        )
        
        return context