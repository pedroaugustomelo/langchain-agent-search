from langgraph.graph import END
from utils import AgentState
import logging
from utils.moderation import llamaguard_moderation

def moderation_routing(state: AgentState):
    """Route moderation based on the last message."""

    logging.info(f"🔄 Moderation Routing: allowed={state.allowed}, response_generated={state.response_generated}")

    if state.response_generated or not state.allowed:
        logging.info("✅ Assistant response generated OR content flagged as unsafe. TERMINATING EXECUTION NOW!")
        return END

    return "conversational"


def conversational_routing(state: AgentState):
    """Determine whether to proceed to web search or moderation."""
    if state.search_needed:
        logging.info("🔍 Routing to Web Search due to detected uncertainty.")
        return "web_search"
    
    logging.info("✅ Routing to Moderation for final review.")
    return "moderation"
