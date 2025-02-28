import logging
import os
import json
from utils.AgentState import AgentState
from langchain_community.tools.tavily_search import TavilySearchResults

# Load API key from environment variable
TAVILY_API_KEY = os.getenv("TAVILY_API_KEY")

def web_search_agent(state: AgentState):
    """
    Web search agent function that performs a web search using TavilySearchResults and retrieves relevant snippets.
    
    Args:
        state (AgentState): The current state of the conversation, including search status and messages.
    
    Returns:
        dict: Updated state including search results and next action.
    """
    logging.info("Entering Web Search Agent.")
    
    # If a search has already been attempted, return to conversational agent
    if state.search_attempted:
        logging.info("Web search already performed. Returning to conversational.")
        return {"next": "conversational"}
    
    # Extract user query
    user_query = next((msg["content"] for msg in state.messages if msg["role"] == "user"), None)
    if not user_query:
        logging.error("No valid user query found.")
        return {"next": "conversational"}
    
    logging.info(f"Performing web search for: {user_query}")
    
    # Perform web search
    web_search_tool = TavilySearchResults(max_results=10, api_key=TAVILY_API_KEY)
    search_results = web_search_tool.invoke({"query": user_query})
    
    logging.info(f"Raw Search Results: {search_results}")
    
    # Ensure search results are in the expected format
    if isinstance(search_results, str):
        try:
            search_results = json.loads(search_results)
        except json.JSONDecodeError:
            logging.error("Failed to parse JSON response from web search.")
            return {"next": "conversational"}
    
    if not isinstance(search_results, list):
        logging.error(f"Unexpected search results format: {type(search_results)}")
        return {"next": "conversational"}
    
    # Extract relevant snippets
    snippets = [result.get("content", "") for result in search_results]
    
    logging.info("Web search complete. Returning to conversational agent.")
    
    return {
        "messages": state.messages, 
        "web_snippets": snippets,
        "search_needed": False,
        "search_attempted": True,
        "response_generated": state.response_generated,
        "allowed": state.allowed,
        "next": "conversational"
    }
