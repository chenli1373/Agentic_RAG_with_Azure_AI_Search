from dataclasses import dataclass
from typing import List, Dict, Any

@dataclass
class ToolContext:
    session_id: str
    history: List[Dict]

@dataclass
class RuntimeResources:
    llm_client: Any
    search_client: Any
    memory_store: Any
    evaluator: Any

@dataclass
class SessionContext:
    session_id: str
    history: List[dict]
    user_id: str = None

@dataclass
class ToolRuntime:
    resources: RuntimeResources
    session: SessionContext