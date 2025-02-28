from pydantic import BaseModel, Field
from typing import List, Dict

class AgentState(BaseModel):
    messages: List[Dict[str, str]] = Field(default_factory=list)
    web_snippets: List[str] = Field(default_factory=list)
    search_needed: bool = False
    search_attempted: bool = False
    response_generated: bool = False
    allowed: bool = True