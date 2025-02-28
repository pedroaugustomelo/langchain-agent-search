import logging
from utils.moderation import llamaguard_moderation
from langgraph.graph import END
from utils.AgentState import AgentState

def moderation_agent(llm, state: AgentState):
    """
    Moderation agent function that evaluates user messages for safety violations using llamaguard_moderation.
    If the content is deemed unsafe, it returns a flagged response; otherwise, it proceeds to the next step.

    Args:
        llm: The language model used for moderation evaluation.
        state (AgentState): The current conversation state containing messages and other tracking information.

    Returns:
        dict: Updated state with moderation results and the next action.
    """
    if not state.messages:
        logging.error("No messages found in state.")
        return {"state": state, "next": END} 

    last_message = state.messages[-1]
    logging.info(f"Moderation Agent Checking: {last_message['content']}")

    # Handle user messages
    if last_message["role"] == "user":
        guardrails_eval = llamaguard_moderation(llm, last_message["content"])
        logging.info(f"Moderation Result: {guardrails_eval}")

        if guardrails_eval.get("allowed") == "UNSAFE":
            logging.info("Content flagged as unsafe.")
            updated_messages = state.messages + [{
                "role": "assistant",
                "content": f"The content is not allowed based on our safety policy: {guardrails_eval.get('flagged_categories')}"
            }]
            return {
                "messages": updated_messages,
                "web_snippets": state.web_snippets,
                "search_needed": state.search_needed,
                "search_attempted": state.search_attempted,
                "response_generated": True,
                "allowed": False,
                "next": END
            }

        logging.info("User input safe. Proceeding to conversational agent.")
        return {"state": state, "next": "conversational"}

    # Handle assistant responses
    if last_message["role"] == "assistant":
        logging.info("Assistant response finalized. Ending process.")
        return {"state": state, "next": END}

    # Handle system messages
    if last_message["role"] == "system":
        logging.info("System message encountered. Continuing process.")
        return {"state": state, "next": "conversational"}

    logging.error("Unexpected message type in moderation. Defaulting to END.")
    return {"state": state, "next": END}
