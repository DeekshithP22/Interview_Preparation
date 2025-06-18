# onekey_vr_automation/
# ├── main.py
# ├── agents/
# │   ├── __init__.py
# │   ├── supervisor_agent.py
# │   └── search_summarize_agent.py
# ├── utils/
# │   ├── __init__.py
# │   ├── state_models.py
# │   ├── routing_functions.py
# │   └── search_strategy.py
# ├── prompts/
# │   ├── __init__.py
# │   ├── supervisor_prompts.py
# │   └── search_summarize_prompts.py
# └── tools/
#     ├── __init__.py
#     ├── italy_trusted_sources.py
#     ├── france_trusted_sources.py
#     ├── hospital_sources.py
#     ├── linkedin_professional.py
#     └── untrusted_web_search.py


# #### main.py start 

# """
# OneKey VR Automation - Main Entry Point
# """

# import asyncio
# from langgraph.graph import StateGraph, END, START
# from utils.state_models import AgentState, WorkflowStatus
# from utils.routing_functions import supervisor_routing_decision, search_agent_routing
# from agents.supervisor_agent import SupervisorAgent
# from agents.search_summarize_agent import SearchAndSummarizeAgent


# def create_two_agent_vr_workflow():
#     """
#     Create LangGraph workflow with two agents:
#     1. Supervisor Agent: API calls → Analysis → Decision making
#     2. Search & Summarize Agent: Multi-source search → Summarization
#     """
    
#     # Initialize the two agents
#     supervisor = SupervisorAgent()
#     search_summarize = SearchAndSummarizeAgent()
    
#     # Define workflow graph
#     workflow = StateGraph(AgentState)
    
#     # Supervisor Agent nodes
#     workflow.add_node("execute_vr_api", supervisor.execute_vr_api_call)
#     workflow.add_node("execute_okdb_search", supervisor.execute_okdb_search)
#     workflow.add_node("analyze_comparison", supervisor.analyze_vr_vs_okdb)
#     workflow.add_node("delegate_search", supervisor.delegate_search_task)
#     workflow.add_node("make_dbo_decision", supervisor.make_dbo_decision)
    
#     # Search & Summarize Agent nodes
#     workflow.add_node("select_tools", search_summarize.select_search_tools)
#     workflow.add_node("execute_search", search_summarize.execute_search_tools)
#     workflow.add_node("summarize_results", search_summarize.intelligent_summarization)
    
#     # Workflow edges - Supervisor orchestration
#     workflow.add_edge(START, "execute_vr_api")
    
#     # Supervisor workflow
#     workflow.add_conditional_edges(
#         "execute_vr_api",
#         supervisor_routing_decision,
#         {
#             "execute_okdb_search": "execute_okdb_search",
#             "handle_error": END
#         }
#     )
    
#     workflow.add_conditional_edges(
#         "execute_okdb_search", 
#         supervisor_routing_decision,
#         {
#             "analyze_comparison": "analyze_comparison",
#             "handle_error": END
#         }
#     )
    
#     workflow.add_conditional_edges(
#         "analyze_comparison",
#         supervisor_routing_decision,
#         {
#             "delegate_search": "delegate_search",
#             "handle_error": END
#         }
#     )
    
#     workflow.add_conditional_edges(
#         "delegate_search",
#         supervisor_routing_decision,
#         {
#             "select_tools": "select_tools",  # Hand off to Search & Summarize Agent
#             "make_dbo_decision": "make_dbo_decision"  # Skip search if not needed
#         }
#     )
    
#     # Search & Summarize Agent workflow
#     workflow.add_edge("select_tools", "execute_search")
#     workflow.add_edge("execute_search", "summarize_results")
    
#     # Hand back to Supervisor for final decision
#     workflow.add_conditional_edges(
#         "summarize_results",
#         supervisor_routing_decision,
#         {
#             "make_dbo_decision": "make_dbo_decision"
#         }
#     )
    
#     # Complete workflow
#     workflow.add_edge("make_dbo_decision", END)
    
#     # Compile workflow
#     return workflow.compile()


# async def main():
#     """
#     Example usage of the two-agent VR automation workflow
#     """
#     # Create workflow
#     app = create_two_agent_vr_workflow()
    
#     # Initial state with VR request ID
#     initial_state = {
#         "vr_request_id": "RAR_ITALY_1019000316927770",
#         "vr_api_response": None,
#         "vr_entity_type": None,
#         "okdb_api_response": None,
#         "comparison_analysis": None,
#         "record_status": "",
#         "search_requirements": None,
#         "dbo_action_decision": None,
#         "selected_tools": [],
#         "execution_order": [],
#         "search_results": [],
#         "intelligent_summary": None,
#         "search_confidence": 0.0,
#         "workflow_status": WorkflowStatus.INITIATED,
#         "current_agent": "supervisor",
#         "error_context": None,
#         "messages": []
#     }
    
#     # Execute workflow
#     result = await app.ainvoke(initial_state)
    
#     print("Workflow Status:", result["workflow_status"])
#     print("Record Status:", result["record_status"])
#     print("DBO Action Decision:", result["dbo_action_decision"])
#     print("Search Confidence:", result["search_confidence"])


# if __name__ == "__main__":
#     asyncio.run(main())
    
"""
OneKey VR Automation - Main Entry Point
"""

import asyncio
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage  # ADD THIS IMPORT
from utils.state_models import AgentState, WorkflowStatus
from utils.routing_functions import supervisor_routing_decision, search_agent_routing
from agents.supervisor_agent import SupervisorAgent
from agents.search_summarize_agent import SearchAndSummarizeAgent


# ADD THIS FUNCTION
async def handle_error(state: AgentState) -> AgentState:
    """Global error handler for workflow"""
    error_context = state.get("error_context", {})
    state["messages"].append(
        AIMessage(content=f"Error in {error_context.get('stage', 'unknown')}: {error_context.get('error', 'Unknown error')}")
    )
    state["workflow_status"] = WorkflowStatus.ERROR
    return state


def create_two_agent_vr_workflow():
    """
    Create LangGraph workflow with two agents:
    1. Supervisor Agent: API calls → Analysis → Decision making
    2. Search & Summarize Agent: Multi-source search → Summarization
    """
    
    # Initialize the two agents
    supervisor = SupervisorAgent()
    search_summarize = SearchAndSummarizeAgent()
    
    # Define workflow graph
    workflow = StateGraph(AgentState)
    
    # Supervisor Agent nodes
    workflow.add_node("execute_vr_api", supervisor.execute_vr_api_call)
    workflow.add_node("execute_okdb_search", supervisor.execute_okdb_search)
    workflow.add_node("analyze_comparison", supervisor.analyze_vr_vs_okdb)
    workflow.add_node("delegate_search", supervisor.delegate_search_task)
    workflow.add_node("make_dbo_decision", supervisor.make_dbo_decision)
    
    # Search & Summarize Agent nodes
    workflow.add_node("select_tools", search_summarize.select_search_tools)
    workflow.add_node("execute_search", search_summarize.execute_search_tools)
    workflow.add_node("summarize_results", search_summarize.intelligent_summarization)
    
    # ADD THIS NODE
    workflow.add_node("handle_error", handle_error)
    
    # Workflow edges - Supervisor orchestration
    workflow.add_edge(START, "execute_vr_api")
    
    # Supervisor workflow
    workflow.add_conditional_edges(
        "execute_vr_api",
        supervisor_routing_decision,
        {
            "execute_okdb_search": "execute_okdb_search",
            "handle_error": END
        }
    )
    
    workflow.add_conditional_edges(
        "execute_okdb_search", 
        supervisor_routing_decision,
        {
            "analyze_comparison": "analyze_comparison",
            "handle_error": END
        }
    )
    
    workflow.add_conditional_edges(
        "analyze_comparison",
        supervisor_routing_decision,
        {
            "delegate_search": "delegate_search",
            "handle_error": END
        }
    )
    
    # CHANGE THIS EDGE
    workflow.add_conditional_edges(
        "delegate_search",
        supervisor_routing_decision,
        {
            "select_tools": "select_tools",  # Hand off to Search & Summarize Agent
            "make_dbo_decision": "make_dbo_decision",  # Skip search if not needed
            "handle_error": END  # ADD THIS
        }
    )
    
    # Search & Summarize Agent workflow
    workflow.add_edge("select_tools", "execute_search")
    workflow.add_edge("execute_search", "summarize_results")
    
    # Hand back to Supervisor for final decision
    workflow.add_conditional_edges(
        "summarize_results",
        supervisor_routing_decision,
        {
            "make_dbo_decision": "make_dbo_decision"
        }
    )
    
    # REMOVE THIS LINE
    # workflow.add_edge("make_dbo_decision", END)
    
    # ADD THIS CONDITIONAL EDGE INSTEAD
    workflow.add_conditional_edges(
        "make_dbo_decision",
        lambda x: "complete" if x.get("workflow_status") == WorkflowStatus.DBO_DECISION_READY else "handle_error",
        {
            "complete": END,
            "handle_error": "handle_error"
        }
    )
    
    # ADD THIS EDGE
    workflow.add_edge("handle_error", END)
    
    # Compile workflow
    return workflow.compile()


async def main():
    """
    Example usage of the two-agent VR automation workflow
    """
    # Create workflow
    app = create_two_agent_vr_workflow()
    
    # Initial state with VR request ID
    initial_state = {
        "vr_request_id": "RAR_ITALY_1019000316927770",  # From your example
        "vr_api_response": None,
        "vr_entity_type": None,
        "okdb_api_response": None,
        "comparison_analysis": None,
        "record_status": "",
        "search_requirements": None,
        "dbo_action_decision": None,
        "selected_tools": [],
        "execution_order": [],
        "search_results": [],
        "intelligent_summary": None,
        "search_confidence": 0.0,
        "workflow_status": WorkflowStatus.INITIATED,
        "current_agent": "supervisor",
        "error_context": None,
        "messages": []
    }
    
    # Execute two-agent workflow
    result = await app.ainvoke(initial_state)
    
    print("Workflow Status:", result["workflow_status"])
    print("Record Status:", result["record_status"])
    print("DBO Action Decision:", result["dbo_action_decision"])
    print("Search Confidence:", result["search_confidence"])


if __name__ == "__main__":
    import asyncio
    asyncio.run(main())

### main.py end


### utils/state_models.py 

"""
State Models and Data Definitions
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum
import operator
from langchain_core.messages import BaseMessage


class WorkflowStatus(Enum):
    INITIATED = "initiated"
    VR_RETRIEVED = "vr_retrieved"
    OKDB_SEARCHED = "okdb_searched"
    ANALYSIS_COMPLETED = "analysis_completed"
    SEARCH_DELEGATED = "search_delegated"
    SEARCH_COMPLETED = "search_completed"
    DBO_DECISION_READY = "dbo_decision_ready"
    COMPLETED = "completed"
    ERROR = "error"


class EntityType(Enum):
    ENT_ACTIVITY = "ENT_ACTIVITY"
    ENT_WORKPLACE = "ENT_WORKPLACE"


class AgentState(TypedDict):
    # Input
    vr_request_id: str
    
    # Supervisor Agent state
    vr_api_response: Optional[Dict[str, Any]]
    vr_entity_type: Optional[str]
    okdb_api_response: Optional[Dict[str, Any]]
    comparison_analysis: Optional[Dict[str, Any]]
    record_status: str
    search_requirements: Optional[Dict[str, Any]]
    dbo_action_decision: Optional[Dict[str, Any]]
    
    # Search & Summarize Agent state
    selected_tools: List[str]
    execution_order: List[str]
    search_results: Annotated[List[Dict[str, Any]], operator.add]
    intelligent_summary: Optional[Dict[str, Any]]
    search_confidence: float
    
    # Workflow management
    workflow_status: WorkflowStatus
    current_agent: str
    error_context: Optional[Dict[str, Any]]
    
    # Communication
    messages: Annotated[List[BaseMessage], operator.add]
    
    
    
    ### utils/routing_functions.py
    
    """
Conditional Routing Functions for LangGraph
"""

from utils.state_models import AgentState, WorkflowStatus


# def supervisor_routing_decision(state: AgentState) -> str:
#     """Route workflow based on Supervisor Agent's current status"""
#     workflow_status = state.get("workflow_status")
    
#     if workflow_status == WorkflowStatus.INITIATED:
#         return "execute_vr_api"
#     elif workflow_status == WorkflowStatus.VR_RETRIEVED:
#         return "execute_okdb_search"
#     elif workflow_status == WorkflowStatus.OKDB_SEARCHED:
#         return "analyze_comparison"
#     elif workflow_status == WorkflowStatus.ANALYSIS_COMPLETED:
#         return "delegate_search"
#     elif workflow_status == WorkflowStatus.SEARCH_COMPLETED:
#         return "make_dbo_decision"
#     elif workflow_status == WorkflowStatus.DBO_DECISION_READY:
#         return "complete"
#     else:
#         return "handle_error"

def supervisor_routing_decision(state: AgentState) -> str:
    """Route workflow based on Supervisor Agent's current status"""
    workflow_status = state.get("workflow_status")
    
    # Special handling for analysis completed
    if workflow_status == WorkflowStatus.ANALYSIS_COMPLETED:
        # Check if search is actually needed
        search_requirements = state.get("search_requirements", {})
        if not search_requirements.get("verification_needed", False):
            return "make_dbo_decision"  # Skip search if not needed
        return "delegate_search"
    
    # Standard routing map
    routing_map = {
        WorkflowStatus.INITIATED: "execute_vr_api",
        WorkflowStatus.VR_RETRIEVED: "execute_okdb_search", 
        WorkflowStatus.OKDB_SEARCHED: "analyze_comparison",
        WorkflowStatus.SEARCH_DELEGATED: "select_tools",
        WorkflowStatus.SEARCH_COMPLETED: "make_dbo_decision",
        WorkflowStatus.DBO_DECISION_READY: "complete",
        WorkflowStatus.ERROR: "handle_error"
    }
    
    return routing_map.get(workflow_status, "handle_error")


def search_agent_routing(state: AgentState) -> str:
    """Route Search & Summarize Agent workflow"""
    current_agent = state.get("current_agent")
    
    if current_agent == "search_summarize":
        if not state.get("selected_tools"):
            return "select_tools"
        elif not state.get("search_results"):
            return "execute_search"
        else:
            return "summarize_results"
    else:
        return "supervisor_control"
    

### utils/search_strategy.py

"""
Static Search Strategy Function
"""

from typing import Dict, Any


def determine_okdb_search_strategy(vr_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Static function to determine OK DB search strategy based on entity type
    
    Args:
        vr_data: Parsed VR API response containing entity type and identifiers
    
    Returns:
        Dictionary with search parameters and strategy
    """
    entity_type = vr_data.get("validation", {}).get("entityTypeIco", "")
    individual = vr_data.get("individual", {})
    workplace = vr_data.get("workplace", {})
    address = vr_data.get("address", {})
    
    if entity_type == "ENT_ACTIVITY":
        # Check if individual.originKeyEid is present
        if individual.get("originKeyEid"):
            return {
                "search_method": "originkey_individual",
                "search_params": {
                    "originKeyEid": individual["originKeyEid"],
                    "searchType": "individual_by_originkey"
                },
                "entity_type": "ENT_ACTIVITY"
            }
        else:
            # Fallback to name-based search
            return {
                "search_method": "name_based", 
                "search_params": {
                    "firstName": individual.get("firstName", ""),
                    "lastName": individual.get("lastName", ""),
                    "workplace": workplace.get("usualName", ""),
                    "country": address.get("country", ""),
                    "searchType": "individual_by_name"
                },
                "entity_type": "ENT_ACTIVITY"
            }
    
    elif entity_type == "ENT_WORKPLACE":
        # Check if workplace.originKeyEid is present
        if workplace.get("originKeyEid"):
            return {
                "search_method": "originkey_workplace",
                "search_params": {
                    "originKeyEid": workplace["originKeyEid"],
                    "searchType": "workplace_by_originkey"
                },
                "entity_type": "ENT_WORKPLACE"
            }
        else:
            # Fallback to workplace name-based search
            return {
                "search_method": "workplace_based",
                "search_params": {
                    "workplaceName": workplace.get("usualName", ""),
                    "country": address.get("country", ""),
                    "city": address.get("city", ""),
                    "searchType": "workplace_by_name"
                },
                "entity_type": "ENT_WORKPLACE"
            }
    
    # Default fallback
    return {
        "search_method": "name_based",
        "search_params": {
            "firstName": individual.get("firstName", ""),
            "lastName": individual.get("lastName", ""),
            "searchType": "individual_by_name"
        },
        "entity_type": "UNKNOWN"
    }


### utils/helpers.py

"""
Helper utilities
"""
import json
import re
from typing import Dict, Any

def safe_json_parse(content: str) -> Dict[str, Any]:
    """
    Safely parse JSON from LLM response
    Handles markdown code blocks and malformed JSON
    """
    try:
        # First try direct parsing
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from markdown code blocks
        json_match = re.search(r'```json\s*(.*?)\s*```', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except:
                pass
        
        # Try to find JSON-like content
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
    
    # Return empty dict if all parsing fails
    return {}
    
    
### agents/supervisor_agent.py

"""
Supervisor Agent - API Orchestrator & Decision Maker
"""

import json
from config import Config
from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage
from utils.helpers import safe_json_parse  # ADD THIS
from utils.state_models import AgentState, WorkflowStatus
from utils.search_strategy import determine_okdb_search_strategy
from prompts.supervisor_prompts import (
    VR_VS_OKDB_ANALYSIS_PROMPT,
    DBO_DECISION_PROMPT
)

# Import your existing API classes
# from your_vr_api import VRAPIClient
# from your_okdb_api import OKDBAPIClient


class SupervisorAgent:
    """
    Supervisor Agent: Handles VR API → OK DB API → Analysis → Decision Making
    """
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            temperature=0.1
        )
        
        # Initialize your existing API clients
        # self.vr_api_client = VRAPIClient()
        # self.okdb_api_client = OKDBAPIClient()
    
    async def execute_vr_api_call(self, state: AgentState) -> AgentState:
        """Execute VR API call to get validation request data"""
        try:
            # TODO: Use your VR API client here
            # vr_response = await self.vr_api_client.get_validation_request(state["vr_request_id"])
            
            # Extract entity type from response
            # entity_type = vr_response.get("validation", {}).get("entityTypeIco", "")
            
            # TODO: Update state with VR API response
            # state.update({
            #     "vr_api_response": vr_response,
            #     "vr_entity_type": entity_type,
            #     "workflow_status": WorkflowStatus.VR_RETRIEVED
            # })
            
            state["messages"].append(AIMessage(content="VR API call completed successfully"))
            
        except Exception as e:
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "vr_api_call", "error": str(e)}
            })
        
        return state
    
    async def execute_okdb_search(self, state: AgentState) -> AgentState:
        """Execute OK DB API search using VR data"""
        try:
            # Get search strategy using static function
            vr_data = state["vr_api_response"]
            search_strategy = determine_okdb_search_strategy(vr_data)
            search_params = search_strategy["search_params"]
            
            # TODO: Use your OK DB API client here
            # okdb_response = await self.okdb_api_client.search_records(search_params)
            
            # TODO: Update state with OK DB response
            # state.update({
            #     "okdb_api_response": okdb_response,
            #     "workflow_status": WorkflowStatus.OKDB_SEARCHED
            # })
            
            state["messages"].append(AIMessage(
                content=f"OK DB search completed using {search_strategy['search_method']} method"
            ))
            
        except Exception as e:
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "okdb_search", "error": str(e)}
            })
        
        return state
    
    async def analyze_vr_vs_okdb(self, state: AgentState) -> AgentState:
        """LLM-powered analysis comparing VR data vs OK DB results"""
        try:
            # Execute LLM analysis
            # In analyze_vr_vs_okdb method:
            analysis_response = await self.llm.ainvoke(
                VR_VS_OKDB_ANALYSIS_PROMPT.format(
                    vr_data=state["vr_api_response"],
                    okdb_data=state["okdb_api_response"]
                ).messages  # ADD .messages
            )

            
            # Parse analysis results
            # parsed_analysis = json.loads(analysis_response.content)
            parsed_analysis = safe_json_parse(analysis_response.content)
            
            state.update({
                "comparison_analysis": parsed_analysis,
                "record_status": parsed_analysis["record_status"],
                "search_requirements": parsed_analysis["search_requirements"],
                "workflow_status": WorkflowStatus.ANALYSIS_COMPLETED
            })
            
            state["messages"].append(AIMessage(
                content=f"Analysis complete: {parsed_analysis['analysis_summary']}"
            ))
            
        except Exception as e:
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "analysis", "error": str(e)}
            })
        
        return state
    
    async def delegate_search_task(self, state: AgentState) -> AgentState:
        """Prepare search delegation for Search & Summarize Agent"""
        try:
            search_requirements = state["search_requirements"]
            
            if search_requirements.get("verification_needed", False):
                state.update({
                    "workflow_status": WorkflowStatus.SEARCH_DELEGATED,
                    "current_agent": "search_summarize"
                })
                
                delegation_message = f"Delegating search for: {', '.join(search_requirements['primary_objectives'])}"
                state["messages"].append(AIMessage(content=delegation_message))
            else:
                # No search needed, proceed directly to DBO decision
                state.update({
                    "workflow_status": WorkflowStatus.SEARCH_COMPLETED,
                    "search_confidence": 1.0,
                    "intelligent_summary": {"verification_not_required": True}
                })
                
                state["messages"].append(AIMessage(content="No external verification needed"))
        
        except Exception as e:
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "delegation", "error": str(e)}
            })
        
        return state
    
    async def make_dbo_decision(self, state: AgentState) -> AgentState:
        """LLM makes final DBO action decision based on all available data"""
        try:
            decision_response = await self.llm.ainvoke(
                DBO_DECISION_PROMPT.format(
                    comparison_analysis=state.get("comparison_analysis", {}),
                    search_summary=state.get("intelligent_summary", {}),
                    search_confidence=state.get("search_confidence", 0.0)
                ).messages  # ADD .messages
            )
            
            # Parse decision
            # dbo_decision = json.loads(decision_response.content)
            dbo_decision = safe_json_parse(decision_response.content)
            
            state.update({
                "dbo_action_decision": dbo_decision,
                "workflow_status": WorkflowStatus.DBO_DECISION_READY
            })
            
            state["messages"].append(AIMessage(
                content=f"DBO decision: {dbo_decision['overall_recommendation']}"
            ))
            
        except Exception as e:
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "dbo_decision", "error": str(e)}
            })
        
        return state
    
    
    
    ### agents/search_summarize_agent.py
    
    
    """
Search & Summarize Agent - Multi-source Search & Intelligent Summarization
"""

import json
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage
from utils.state_models import AgentState, WorkflowStatus
from prompts.search_summarize_prompts import (
    TOOL_SELECTION_PROMPT,
    SUMMARIZATION_PROMPT
)

# Import your tool implementations
# from tools.italy_trusted_sources import ItalyTrustedSourcesTool
# from tools.france_trusted_sources import FranceTrustedSourcesTool
# from tools.hospital_sources import HospitalSourcesTool
# from tools.linkedin_professional import LinkedInProfessionalTool
# from tools.untrusted_web_search import UntrustedWebSearchTool


class SearchAndSummarizeAgent:
    """
    Search & Summarize Agent: Multi-source searching + AI summarization
    """
    from utils.helpers import safe_json_parse
    from config import Config

    
    # Update __init__ method:
    def __init__(self):
        self.llm = AzureChatOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            temperature=0.1
        )
            
        # Initialize your existing tool instances
        self.search_tools = {
            "italy_trusted": self._initialize_italy_tool(),
            "france_trusted": self._initialize_france_tool(),
            "hospital_sources": self._initialize_hospital_tool(),
            "linkedin_professional": self._initialize_linkedin_tool(),
            "untrusted_web_search": self._initialize_web_tool()
        }
    
    def _initialize_italy_tool(self):
        """Initialize consolidated Italy tool"""
        # TODO: Import your consolidated Italy tool
        # from tools.italy_trusted_sources import ItalyTrustedSourcesTool
        # return ItalyTrustedSourcesTool()
        pass
    
    def _initialize_france_tool(self):
        """Initialize consolidated France tool"""
        # TODO: Import your consolidated France tool
        # from tools.france_trusted_sources import FranceTrustedSourcesTool
        # return FranceTrustedSourcesTool()
        pass
    
    def _initialize_hospital_tool(self):
        """Initialize hospital sources tool"""
        # TODO: Import your hospital tool
        # from tools.hospital_sources import HospitalSourcesTool
        # return HospitalSourcesTool()
        pass
    
    def _initialize_linkedin_tool(self):
        """Initialize LinkedIn professional search tool"""
        # TODO: Import your LinkedIn tool
        # from tools.linkedin_professional import LinkedInProfessionalTool
        # return LinkedInProfessionalTool()
        pass
    
    def _initialize_web_tool(self):
        """Initialize untrusted web search tool"""
        # TODO: Import your web search tool
        # from tools.untrusted_web_search import UntrustedWebSearchTool
        # return UntrustedWebSearchTool()
        pass
    
    async def select_search_tools(self, state: AgentState) -> AgentState:
        """LLM selects appropriate tools based on search requirements"""
        try:
            # Execute tool selection LLM
            selection_response = await self.llm.ainvoke(
                TOOL_SELECTION_PROMPT.format(
                    search_requirements=state["search_requirements"]
                ).messages  # ADD .messages
)
            
            # Parse selection
            # tool_selection = json.loads(selection_response.content)
            tool_selection = safe_json_parse(selection_response.content)

            
            state.update({
                "selected_tools": tool_selection["selected_tools"],
                "execution_order": tool_selection["execution_order"]
            })
            
            state["messages"].append(AIMessage(
                content=f"Selected tools: {', '.join(tool_selection['selected_tools'])}"
            ))
            
        except Exception as e:
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "tool_selection", "error": str(e)}
            })
        
        return state
    
    async def execute_search_tools(self, state: AgentState) -> AgentState:
        """Execute selected tools in planned order"""
        try:
            execution_order = state["execution_order"]
            search_requirements = state["search_requirements"]
            
            for tool_name in execution_order:
                if tool_name in self.search_tools:
                    tool_instance = self.search_tools[tool_name]
                    
                    # TODO: Execute tool search
                    # search_result = await tool_instance.search(search_requirements)
                    # state["search_results"].append(search_result)
                    
                    state["messages"].append(AIMessage(
                        content=f"Executed search with {tool_name}"
                    ))
            
        except Exception as e:
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "search_execution", "error": str(e)}
            })
        
        return state
    
    async def intelligent_summarization(self, state: AgentState) -> AgentState:
        """LLM creates intelligent summary of all search results"""
        try:
            # Execute summarization LLM
            summary_response = await self.llm.ainvoke(
                SUMMARIZATION_PROMPT.format(
                    search_requirements=state["search_requirements"],
                    search_results=state["search_results"]
                ).messages  # ADD .messages
            )
            
            # Parse summary
            # summary_data = json.loads(summary_response.content)
            summary_data = safe_json_parse(summary_response.content)
            
            state.update({
                "intelligent_summary": summary_data,
                "search_confidence": summary_data["overall_assessment"]["confidence_level"],
                "workflow_status": WorkflowStatus.SEARCH_COMPLETED,
                "current_agent": "supervisor"
            })
            
            state["messages"].append(AIMessage(
                content=f"Search summary complete. Confidence: {summary_data['overall_assessment']['confidence_level']}"
            ))
            
        except Exception as e:
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "summarization", "error": str(e)}
            })
        
        return state
    
    
    
    
    ### prompts/supervisor_prompts.py
    
    
    """
Prompt Templates for Supervisor Agent
"""

from langchain_core.prompts import ChatPromptTemplate


VR_VS_OKDB_ANALYSIS_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are analyzing VR validation requests against OneKey Database records.

Your Analysis Tasks:
1. Determine if VR individual exists in OK DB (new vs existing record)
2. If multiple OK DB results, analyze if they're the same person or different people
3. Compare all attributes between VR and OK DB data
4. Identify any mismatches or discrepancies
5. Determine what needs external verification

Handle These Scenarios:
- No OK DB results → New record
- Single OK DB result → Compare attributes, check for mismatches
- Multiple OK DB results, same person → Individual with multiple workplace associations
- Multiple OK DB results, different people → Need to disambiguate

Key Comparisons:
- Individual: firstName, lastName, originKeyEid
- Workplace: name, location, department
- Status: activity status, business status
- Contact: email, phone, address
- Professional: specialty, qualifications

Respond with JSON:
{{
    "record_status": "new_record/existing_match/existing_mismatch/multiple_matches/ambiguous",
    "analysis_summary": "Brief description of the situation",
    "ok_db_results_count": X,
    "is_same_individual": true/false (if multiple results),
    "primary_match": {{"details of most relevant OK DB record if found"}},
    "detected_mismatches": [
        {{
            "field": "status/workplace/specialty/contact",
            "vr_value": "value from VR",
            "okdb_value": "value from OK DB", 
            "severity": "critical/medium/low",
            "needs_verification": true/false
        }}
    ],
    "workplace_associations": [
        {{"workplace_name": "...", "status": "active/inactive"}}
    ],
    "search_requirements": {{
        "verification_needed": true/false,
        "primary_objectives": ["what needs to be verified"],
        "geographic_region": "country code",
        "confidence_threshold": 0.XX
    }}
}}"""),
    ("human", """
VR API Response: {vr_data}
OK DB API Response: {okdb_data}

Analyze and compare these responses comprehensively.
""")
])

DBO_DECISION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are making final DBO action decisions for OneKey VR validation.

Available Information:
1. VR validation request data
2. OK DB search results and comparison analysis
3. External verification results (if performed)
4. Overall confidence assessments

Decision Categories:
- UPDATE_RECORD: Modify existing OK DB record(s)
- CREATE_RECORD: Create new OK DB record
- NO_ACTION: Current OK DB data is accurate
- MANUAL_REVIEW: Complex case requiring human review
- INVESTIGATE: Need more information

Decision Factors:
- Record status from analysis
- Verification results and confidence
- Risk level of automated decision
- Data completeness

Respond with JSON:
{{
    "dbo_actions": [
        {{
            "action_type": "UPDATE_RECORD/CREATE_RECORD/NO_ACTION/MANUAL_REVIEW/INVESTIGATE",
            "target_record_id": "OK DB record ID if applicable",
            "field_updates": {{"field": "new_value"}},
            "justification": "Clear reasoning for DBO",
            "confidence": 0.XX,
            "manual_review_required": true/false,
            "risk_level": "low/medium/high"
        }}
    ],
    "overall_recommendation": "Primary recommendation for DBO",
    "decision_confidence": 0.XX,
    "processing_summary": "Summary of validation process"
}}"""),
    ("human", """
Comparison Analysis: {comparison_analysis}
Search Summary: {search_summary}
Search Confidence: {search_confidence}

Make final DBO action decision.
""")
])



### prompts/search_and_summarize

"""
Prompt Templates for Search & Summarize Agent
"""

from langchain_core.prompts import ChatPromptTemplate

TOOL_SELECTION_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are selecting search tools for healthcare professional verification.

Available Tools:
1. italy_trusted - Italian medical registries (FNOMCEO, FNOPI, etc.) - Reliability: 0.95
2. france_trusted - French medical directories (Annuaire Santé, etc.) - Reliability: 0.95
3. hospital_sources - Hospital websites and staff directories - Reliability: 0.80
4. linkedin_professional - Professional networking platforms - Reliability: 0.70
5. untrusted_web_search - General web search engines - Reliability: 0.50

Tool Selection Strategy:
- Geographic Priority: Match tools to individual's country/region
- Confidence Requirements: Higher confidence needs → prioritize trusted sources
- Verification Scope: Multiple objectives → include multiple tool types
- Efficiency: Start with most reliable sources, add others if needed

Execution Order Priority:
1. Country-specific trusted medical registries
2. Hospital/institutional sources
3. Professional networks
4. General web search (only if necessary)

Respond with JSON:
{{
    "selected_tools": ["tool1", "tool2", "tool3"],
    "execution_order": ["tool1", "tool2", "tool3"],
    "tool_rationale": {{
        "tool1": "Primary tool for Italian medical professionals",
        "tool2": "Hospital directory for workplace verification",
        "tool3": "Backup source for additional confirmation"
    }},
    "search_strategy": {{
        "primary_search": "italy_trusted for medical registration",
        "secondary_search": "hospital_sources for employment verification",
        "fallback_search": "linkedin_professional if needed"
    }},
    "stopping_criteria": {{
        "confidence_threshold": 0.85,
        "min_sources": 2,
        "stop_on_high_confidence": true
    }},
    "expected_confidence": 0.XX
}}"""),
    ("human", """
Search Requirements: {search_requirements}

Select appropriate tools and execution strategy.
""")
])

SUMMARIZATION_PROMPT = ChatPromptTemplate.from_messages([
   ("system", """You are summarizing search results for healthcare professional verification.

Summarization Tasks:
1. Evidence Synthesis: Combine findings from multiple sources
2. Conflict Resolution: Identify and analyze conflicting information
3. Confidence Assessment: Evaluate reliability of overall findings
4. Gap Analysis: Identify missing information or incomplete verification
5. Risk Assessment: Flag potential issues for manual review

Summary Categories:
- Employment Status: Current workplace and activity status
- Professional Credentials: Medical licenses, certifications, specialties
- Contact Information: Current address, phone, email accuracy
- Workplace Details: Department, role, institutional affiliations

Confidence Calculation:
- Source Reliability: Weight results by source trustworthiness
- Result Consistency: Higher confidence when sources agree
- Completeness: Higher confidence when all objectives verified
- Recency: More recent information weighted higher

Respond with JSON:
{{
   "verification_results": {{
       "primary_objectives_achieved": ["list of verified objectives"],
       "employment_status": {{"status": "confirmed/uncertain", "details": "..."}},
       "professional_credentials": {{"status": "confirmed/uncertain", "details": "..."}}
   }},
   "source_consensus": {{
       "agreement_level": "high/medium/low",
       "conflicting_information": ["any conflicts found"]
   }},
   "overall_assessment": {{
       "primary_finding": "Main conclusion about individual",
       "confidence_level": 0.XX,
       "manual_review_recommended": true/false,
       "manual_review_reasons": ["reasons if recommended"]
   }},
   "summary_narrative": "Human-readable summary for DBO review"
}}"""),
   ("human", """
Search Requirements: {search_requirements}
Search Results: {search_results}

Create intelligent summary of all findings.
""")
])


### utils/init.py

"""
Utils module initialization
"""

from .state_models import AgentState, WorkflowStatus, EntityType
from .routing_functions import supervisor_routing_decision, search_agent_routing
from .search_strategy import determine_okdb_search_strategy
from .helpers import safe_json_parse  # ADD THIS LINE

__all__ = [
    'AgentState',
    'WorkflowStatus',
    'EntityType',
    'supervisor_routing_decision',
    'search_agent_routing',
    'determine_okdb_search_strategy',
    'safe_json_parse'  # ADD THIS LINE
]

### agents/init.py

"""
Agents module initialization
"""

from .supervisor_agent import SupervisorAgent
from .search_summarize_agent import SearchAndSummarizeAgent

__all__ = [
    'SupervisorAgent',
    'SearchAndSummarizeAgent'
]


### prompts/init.py

"""
Prompts module initialization
"""

from .supervisor_prompts import VR_VS_OKDB_ANALYSIS_PROMPT, DBO_DECISION_PROMPT
from .search_summarize_prompts import TOOL_SELECTION_PROMPT, SUMMARIZATION_PROMPT

__all__ = [
    'VR_VS_OKDB_ANALYSIS_PROMPT',
    'DBO_DECISION_PROMPT',
    'TOOL_SELECTION_PROMPT',
    'SUMMARIZATION_PROMPT'
]


### tools/init.py







### config.py

"""
Configuration settings
"""
import os
from dotenv import load_dotenv

load_dotenv()

class Config:
    # Azure OpenAI
    AZURE_OPENAI_API_KEY = os.getenv("AZURE_OPENAI_API_KEY")
    AZURE_OPENAI_ENDPOINT = os.getenv("AZURE_OPENAI_ENDPOINT")
    AZURE_OPENAI_API_VERSION = os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview")
    AZURE_OPENAI_DEPLOYMENT_NAME = os.getenv("AZURE_OPENAI_DEPLOYMENT_NAME", "gpt-4")
    
    # Workflow settings
    MAX_RETRIES = 3
    TIMEOUT_SECONDS = 300
    
    # Tool settings
    TOOL_TIMEOUT = 60  # seconds per tool
    MIN_CONFIDENCE_THRESHOLD = 0.70
    TARGET_CONFIDENCE_THRESHOLD = 0.85
    
    
    
    
### .env
### requirements.txt

# Summary of File Organization
# Core Files:

# main.py - Entry point and workflow creation
# agents/ - Two agent implementations (supervisor & search_summarize)
# utils/ - State models, routing logic, and static search strategy function
# prompts/ - All LLM prompts separated by agent
# tools/ - Your existing tool implementations

# Key Benefits:

# Clean separation of concerns
# Easy to maintain and test
# Clear import structure
# Prompts separated from logic
# Static function isolated in utils
# Your existing tools plug in directly

# This structure keeps the simplified two-agent architecture while organizing it professionally! 