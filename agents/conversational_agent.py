import logging
from utils.AgentState import AgentState
from langgraph.graph import END

def conversational_agent(llm, state: AgentState):
    """
    Conversational agent function that processes user queries, determines whether web search is needed,
    and generates responses using a language model (LLM).

    Args:
        llm: The language model used to generate responses.
        state (AgentState): The current state of the conversation, including messages and search status.

    Returns:
        dict: Updated state with messages, response flags, and next action.
    """
    user_query = state.messages[-1]["content"]
    logging.info(f"Processing user query: {user_query}")
    
    updated_messages = state.messages.copy()
    
    # If web snippets are available, use them to enhance the response
    if state.web_snippets:
        logging.info("Web snippets detected. Generating response based on them.")
        
        snippet_prompt = f"""
            Using the provided web snippets: {state.web_snippets}, attempt to answer the user's question: {user_query}.

            If the snippets contain relevant information, craft a well-structured response using them and explicitly list them as sources.

            If the snippets are not relevant or do not sufficiently answer the question, state that you were unable to provide a complete answer despite reviewing them, but still mention them as sources.
        """

        response_with_snippets = llm.invoke(snippet_prompt)
        updated_messages.append({
            "role": "assistant",
            "content": response_with_snippets.content
        })
        
        logging.info(f"Generated response using web snippets: {response_with_snippets.content}")
        
        return {
            "messages": updated_messages,
            "response_generated": True,
            "search_needed": False,
            "search_attempted": False,
            "web_snippets": [],
            "allowed": True,
            "next": "moderation"
        }
    
    # If a response was already generated, skip further processing
    if state.response_generated:
        logging.info("Response already generated. Skipping additional processing.")
        return {
            "messages": updated_messages,
            "response_generated": True,
            "search_needed": False,
            "search_attempted": False,
            "web_snippets": [],
            "allowed": True,
            "next": "moderation"
        }
    
    # If search was not yet attempted, generate an initial response
    if not state.search_attempted:
        llm_response = llm.invoke(f"Answer the user: {user_query}")
        updated_messages.append({
            "role": "assistant",
            "content": llm_response.content
        })
        
        # Evaluate response uncertainty
        uncertainty_prompt = f"""
            You are an evaluator that determines if a given response from a large language model (LLM) failed to answer the user's query.

            User Query: {user_query}
            LLM Response: {llm_response.content}

            - If the response does not provide a meaningful, informative, or relevant answer to the user's query, return "true".
            - If the response is a refusal due to moderation policies, return "true".
            - If the response explicitly states the model's limitations **without offering a useful alternative or explanation**, return "true".
            - If the response correctly answers the user's query or provides a reasonable alternative (such as directing the user to another source), return "false".

            Output Format:
            true or false
        """
        
        logging.info(f"Messages updated: {updated_messages}")
        
        uncertainty_response = llm.invoke(uncertainty_prompt)
        uncertainty = uncertainty_response.content.strip().lower() == "true"
        logging.info(f"Uncertainty detected: {uncertainty}")
        
        # If the response is uncertain and no search has been attempted, trigger a web search
        if uncertainty and not state.web_snippets and not state.search_attempted:
            logging.info("Triggering web search due to uncertainty and lack of snippets.")
            return {
                "messages": updated_messages,
                "response_generated": False,
                "search_needed": True,
                "search_attempted": False,
                "web_snippets": [],
                "allowed": True,
                "next": "web_search"
            }
    
    logging.info("Response finalized. Sending to moderation.")
    return {
        "messages": updated_messages,
        "response_generated": True,
        "search_needed": False,
        "search_attempted": False,
        "web_snippets": [],
        "allowed": True,
        "next": "moderation"
    }
