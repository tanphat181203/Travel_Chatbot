from typing import List, TypedDict, Optional, Sequence
from langchain_core.messages import BaseMessage
from datetime import date

class GraphState(TypedDict):
    messages: Sequence[BaseMessage]
    user_query: str
    current_date: str
    available_locations: Optional[List[str]]
    extracted_entities: Optional[dict]
    search_results: Optional[List[dict]]
    final_response: Optional[str]
    error: Optional[str]
    routing_decision: Optional[str]