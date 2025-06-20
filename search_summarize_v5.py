### Directory Structure

# search_summarize_agent/
# ├── main.py                     # Standalone workflow entry point
# ├── config.py                   # Configuration
# ├── requirements.txt            # Dependencies
# ├── .env                       # Environment variables
# ├── agents/
# │   ├── __init__.py
# │   └── search_summarize_agent.py
# ├── utils/
# │   ├── __init__.py
# │   ├── state_models.py
# │   ├── routing_functions.py
# │   └── helpers.py
# ├── prompts/
# │   ├── __init__.py
# │   └── search_summarize_prompts.py
# └── tools/
#     ├── __init__.py
#     ├── italy_trusted_sources.py
#     ├── france_trusted_sources.py
#     ├── hospital_sources.py
#     ├── linkedin_professional.py
#     └── untrusted_web_search.py


### main.py

"""
Standalone Search & Summarize Agent - Main Entry Point
"""

import asyncio
import logging
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage
from utils.state_models import SearchAgentState, SearchWorkflowStatus
from utils.routing_functions import (
    search_routing_decision,
    tool_execution_routing,
    summarization_routing
)
from agents.search_summarize_agent import SearchAndSummarizeAgent

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def handle_error(state: SearchAgentState) -> SearchAgentState:
    """Global error handler for search workflow"""
    error_context = state.get("error_context", {})
    error_msg = f"Error in {error_context.get('stage', 'unknown')}: {error_context.get('error', 'Unknown error')}"
    
    logger.error(error_msg)
    state["messages"].append(AIMessage(content=error_msg))
    state["workflow_status"] = SearchWorkflowStatus.ERROR
    
    return state


def create_search_summarize_workflow():
    """
    Create standalone LangGraph workflow for Search & Summarize Agent:
    1. Tool Selection → 2. Search Execution → 3. Intelligent Summarization
    """
    
    # Initialize the search agent
    search_summarize = SearchAndSummarizeAgent()
    
    # Define workflow graph
    workflow = StateGraph(SearchAgentState)
    
    # Search & Summarize Agent nodes
    workflow.add_node("select_tools", search_summarize.select_search_tools)
    workflow.add_node("execute_search", search_summarize.execute_search_tools)
    workflow.add_node("summarize_results", search_summarize.intelligent_summarization)
    
    # Error handler node
    workflow.add_node("handle_error", handle_error)
    
    # WORKFLOW EDGES
    
    # Start with tool selection
    workflow.add_edge(START, "select_tools")
    
    # After tool selection - route to execution or error
    workflow.add_conditional_edges(
        "select_tools",
        search_routing_decision,
        {
            "execute_search": "execute_search",
            "handle_error": "handle_error"
        }
    )
    
    # After search execution - route to summarization or error
    workflow.add_conditional_edges(
        "execute_search",
        tool_execution_routing,
        {
            "summarize_results": "summarize_results",
            "handle_error": "handle_error"
        }
    )
    
    # After summarization - complete or error
    workflow.add_conditional_edges(
        "summarize_results",
        summarization_routing,
        {
            "complete": END,
            "handle_error": "handle_error"
        }
    )
    
    # Error handler to END
    workflow.add_edge("handle_error", END)
    
    # Compile workflow
    return workflow.compile()


async def process_search_request(search_requirements: dict) -> dict:
    """
    Process a search request through the standalone workflow
    
    Args:
        search_requirements: Search requirements dict
        
    Returns:
        Workflow result including intelligent summary
    """
    logger.info(f"Processing search request: {search_requirements.get('primary_objectives', [])}")
    
    # Create workflow
    app = create_search_summarize_workflow()

    # Initial state with search requirements
    initial_state = SearchAgentState(
        search_requirements=search_requirements,
        selected_tools=[],
        execution_order=[],
        search_results=[],
        intelligent_summary=None,
        search_confidence=0.0,
        workflow_status=SearchWorkflowStatus.INITIATED,
        error_context=None,
        messages=[]
    )
    
    # Execute workflow
    result = await app.ainvoke(initial_state)
    
    logger.info(f"Search workflow completed - Status: {result['workflow_status'].value}")
    
    return result


# For testing standalone agent
async def main():
    """
    Example usage - process search requirements
    """
    # Example search requirements
    search_requirements_sample = {
        "verification_needed": True,
        "primary_objectives": [
            "verify_employment_status",
            "confirm_medical_license",
            "validate_workplace_affiliation"
        ],
        "geographic_region": "IT",
        "confidence_threshold": 0.85,
        "entity_details": {
            "firstName": "PAOLO",
            "lastName": "CORVISIERI", 
            "workplaceName": "DISTRETTO SANITARIO FIUMICINO",
            "specialtyCode": "18",
            "country": "IT",
            "city": "VELLETRI"
        },
        "search_context": {
            "entity_type": "ENT_ACTIVITY",
            "verification_type": "medical_professional",
            "urgency": "standard"
        }
    }
    
    result = await process_search_request(search_requirements_sample)
    
    print("\n=== SEARCH WORKFLOW RESULTS ===")
    print(f"Workflow Status: {result['workflow_status'].value}")
    print(f"Selected Tools: {result.get('selected_tools', [])}")
    print(f"Search Confidence: {result.get('search_confidence', 0.0)}")
    
    if result.get("intelligent_summary"):
        summary = result["intelligent_summary"]
        print(f"\nIntelligent Summary:")
        print(f"  Primary Finding: {summary.get('overall_assessment', {}).get('primary_finding', 'N/A')}")
        print(f"  Confidence Level: {summary.get('overall_assessment', {}).get('confidence_level', 0.0)}")
        print(f"  Manual Review: {summary.get('overall_assessment', {}).get('manual_review_recommended', False)}")


if __name__ == "__main__":
    asyncio.run(main())


### config.py

"""
Configuration settings for standalone Search & Summarize Agent
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
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")


### utils/state_models.py

"""
State Models for Standalone Search & Summarize Agent
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum
import operator
from langchain_core.messages import BaseMessage


class SearchWorkflowStatus(Enum):
    INITIATED = "initiated"
    TOOLS_SELECTED = "tools_selected"
    SEARCH_EXECUTING = "search_executing"
    SEARCH_COMPLETED = "search_completed"
    SUMMARIZATION_COMPLETED = "summarization_completed"
    COMPLETED = "completed"
    ERROR = "error"


class SearchAgentState(TypedDict):
    # Input - Search requirements
    search_requirements: Dict[str, Any]  # Primary input for standalone agent
    
    # Search & Summarize Agent state
    selected_tools: List[str]
    execution_order: List[str]
    search_results: Annotated[List[Dict[str, Any]], operator.add]
    intelligent_summary: Optional[Dict[str, Any]]
    search_confidence: float
    
    # Workflow management
    workflow_status: SearchWorkflowStatus
    error_context: Optional[Dict[str, Any]]
    
    # Communication
    messages: Annotated[List[BaseMessage], operator.add]


### utils/routing_functions.py

"""
Routing Functions for Standalone Search & Summarize Agent
"""

from utils.state_models import SearchAgentState, SearchWorkflowStatus
import logging  
logger = logging.getLogger(__name__) 


def search_routing_decision(state: SearchAgentState) -> str:
    """Route workflow based on tool selection status"""
    workflow_status = state.get("workflow_status")
    
    # Check if tools were selected successfully
    if workflow_status == SearchWorkflowStatus.TOOLS_SELECTED:
        selected_tools = state.get("selected_tools", [])
        if selected_tools:
            return "execute_search"
        else:
            logger.warning("No tools selected for search execution")
            return "handle_error"
    
    # Error handling
    if workflow_status == SearchWorkflowStatus.ERROR:
        return "handle_error"
    
    # Default error case
    logger.warning(f"Unexpected workflow status in tool selection: {workflow_status}")
    return "handle_error"


def tool_execution_routing(state: SearchAgentState) -> str:
    """Route workflow based on search execution status"""
    workflow_status = state.get("workflow_status")
    
    # Check if search execution completed
    if workflow_status == SearchWorkflowStatus.SEARCH_COMPLETED:
        return "summarize_results"
    
    # Error handling
    if workflow_status == SearchWorkflowStatus.ERROR:
        return "handle_error"
    
    # Default error case
    logger.warning(f"Unexpected workflow status in search execution: {workflow_status}")
    return "handle_error"


def summarization_routing(state: SearchAgentState) -> str:
    """Route workflow based on summarization status"""
    workflow_status = state.get("workflow_status")
    
    # Check if summarization completed successfully
    if workflow_status == SearchWorkflowStatus.SUMMARIZATION_COMPLETED:
        return "complete"
    
    # Error handling
    if workflow_status == SearchWorkflowStatus.ERROR:
        return "handle_error"
    
    # Default error case
    logger.warning(f"Unexpected workflow status in summarization: {workflow_status}")
    return "handle_error"


### utils/helpers.py

"""
Helper utilities for standalone Search & Summarize Agent
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


def validate_search_requirements(search_requirements: Dict[str, Any]) -> bool:
    """
    Validate that search requirements contain necessary fields
    """
    required_fields = ["verification_needed", "primary_objectives"]
    for field in required_fields:
        if field not in search_requirements:
            return False
    
    # Check if verification is actually needed
    if not search_requirements.get("verification_needed", False):
        return False
    
    # Check if objectives are provided
    objectives = search_requirements.get("primary_objectives", [])
    if not objectives or len(objectives) == 0:
        return False
    
    return True


### agents/search_summarize_agent.py

"""
Standalone Search & Summarize Agent - Multi-source Search & Intelligent Summarization
"""

import logging
from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage
from config import Config
from utils.helpers import safe_json_parse, validate_search_requirements
from utils.state_models import SearchAgentState, SearchWorkflowStatus
from prompts.search_summarize_prompts import (
    TOOL_SELECTION_PROMPT,
    SUMMARIZATION_PROMPT
)

# Configure logging
logger = logging.getLogger(__name__)

# Import your tool implementations
# from tools.italy_trusted_sources import ItalyTrustedSourcesTool
# from tools.france_trusted_sources import FranceTrustedSourcesTool
# from tools.hospital_sources import HospitalSourcesTool
# from tools.linkedin_professional import LinkedInProfessionalTool
# from tools.untrusted_web_search import UntrustedWebSearchTool


class SearchAndSummarizeAgent:
    """
    Standalone Search & Summarize Agent: Multi-source searching + AI summarization
    """
    
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
    
    async def select_search_tools(self, state: SearchAgentState) -> SearchAgentState:
        """LLM selects appropriate tools based on search requirements"""
        logger.info("Starting tool selection for search requirements")
        
        try:
            search_requirements = state.get("search_requirements", {})
            
            # Validate search requirements
            if not validate_search_requirements(search_requirements):
                raise ValueError("Invalid or incomplete search requirements provided")
            
            # Check if search is actually needed
            if not search_requirements.get("verification_needed", False):
                state.update({
                    "selected_tools": [],
                    "execution_order": [],
                    "search_results": [],
                    "intelligent_summary": {"verification_not_required": True},
                    "search_confidence": 1.0,
                    "workflow_status": SearchWorkflowStatus.SUMMARIZATION_COMPLETED
                })
                
                state["messages"].append(AIMessage(
                    content="No search required based on requirements"
                ))
                logger.info("No search required - proceeding to completion")
                return state
            
            # Execute tool selection LLM
            logger.info("Executing LLM for tool selection")
            
            selection_response = await self.llm.ainvoke(
                TOOL_SELECTION_PROMPT.format(
                    search_requirements=search_requirements
                ).messages
            )
            
            # Parse selection
            tool_selection = safe_json_parse(selection_response.content)
            
            # Validate tool selection response
            if not tool_selection.get("selected_tools"):
                raise ValueError("No tools selected by LLM - invalid tool selection response")
            
            # Update state with tool selection
            state.update({
                "selected_tools": tool_selection.get("selected_tools", []),
                "execution_order": tool_selection.get("execution_order", []),
                "workflow_status": SearchWorkflowStatus.TOOLS_SELECTED
            })
            
            selected_tools_list = tool_selection.get("selected_tools", [])
            state["messages"].append(AIMessage(
                content=f"Selected tools: {', '.join(selected_tools_list)}"
            ))
            
            logger.info(f"Tools selected successfully: {selected_tools_list}")
            
        except Exception as e:
            logger.error(f"Tool selection error: {str(e)}")
            state.update({
                "workflow_status": SearchWorkflowStatus.ERROR,
                "error_context": {"stage": "tool_selection", "error": str(e)}
            })
            
            state["messages"].append(AIMessage(
                content=f"Tool selection failed: {str(e)}"
            ))
        
        return state
    
    async def execute_search_tools(self, state: SearchAgentState) -> SearchAgentState:
        """Execute selected tools in planned order"""
        logger.info("Starting search tool execution")
        
        try:
            execution_order = state.get("execution_order", [])
            search_requirements = state.get("search_requirements", {})
            
            if not execution_order:
                raise ValueError("No tools in execution order - cannot proceed with search")
            
            logger.info(f"Executing {len(execution_order)} tools in order")
            
            search_results = []
            successful_executions = 0
            
            for tool_name in execution_order:
                logger.debug(f"Executing tool: {tool_name}")
                
                if tool_name in self.search_tools:
                    tool_instance = self.search_tools[tool_name]
                    
                    try:
                        # TODO: Execute tool search
                        # search_result = await tool_instance.search(search_requirements)
                        # search_results.append(search_result)
                        
                        # Placeholder for tool execution
                        placeholder_result = {
                            "tool_name": tool_name,
                            "execution_status": "success",
                            "results": [],  # TODO: Actual tool results
                            "confidence": 0.8,  # TODO: Actual confidence from tool
                            "execution_time": "2.3s"  # TODO: Actual execution time
                        }
                        search_results.append(placeholder_result)
                        successful_executions += 1
                        
                        state["messages"].append(AIMessage(
                            content=f"Executed search with {tool_name} - Success"
                        ))
                        
                        logger.debug(f"Successfully executed {tool_name}")
                        
                    except Exception as tool_error:
                        logger.warning(f"Tool {tool_name} execution failed: {str(tool_error)}")
                        
                        # Continue with other tools even if one fails
                        error_result = {
                            "tool_name": tool_name,
                            "execution_status": "failed",
                            "error": str(tool_error),
                            "results": [],
                            "confidence": 0.0
                        }
                        search_results.append(error_result)
                        
                        state["messages"].append(AIMessage(
                            content=f"Tool {tool_name} execution failed: {str(tool_error)}"
                        ))
                else:
                    logger.warning(f"Tool {tool_name} not found in available tools")
                    
                    # Add missing tool result
                    missing_result = {
                        "tool_name": tool_name,
                        "execution_status": "not_found",
                        "error": "Tool not available",
                        "results": [],
                        "confidence": 0.0
                    }
                    search_results.append(missing_result)
            
            # Validate that at least one tool executed successfully
            if successful_executions == 0:
                raise ValueError(f"All {len(execution_order)} tools failed to execute successfully")
            
            # Update state with search results
            state.update({
                "search_results": search_results,
                "workflow_status": SearchWorkflowStatus.SEARCH_COMPLETED
            })
            
            state["messages"].append(AIMessage(
                content=f"Search execution completed - {successful_executions}/{len(execution_order)} tools successful"
            ))
            
            logger.info(f"Search tool execution completed - {successful_executions} successful executions")
            
        except Exception as e:
            logger.error(f"Search execution error: {str(e)}")
            state.update({
                "workflow_status": SearchWorkflowStatus.ERROR,
                "error_context": {"stage": "search_execution", "error": str(e)}
            })
            
            state["messages"].append(AIMessage(
                content=f"Search execution failed: {str(e)}"
            ))
        
        return state
    
    async def intelligent_summarization(self, state: SearchAgentState) -> SearchAgentState:
        """LLM creates intelligent summary of all search results"""
        logger.info("Starting intelligent summarization")
        
        try:
            search_results = state.get("search_results", [])
            search_requirements = state.get("search_requirements", {})
            
            if not search_results:
                raise ValueError("No search results available for summarization")
            
            # Execute summarization LLM
            logger.info("Executing LLM for intelligent summarization")
            
            summary_response = await self.llm.ainvoke(
                SUMMARIZATION_PROMPT.format(
                    search_requirements=search_requirements,
                    search_results=search_results
                ).messages
            )
            
            # Parse summary
            summary_data = safe_json_parse(summary_response.content)
            
            # Validate summary response
            if not summary_data:
                raise ValueError("Invalid or empty summary response from LLM")
            
            # Extract confidence level
            overall_assessment = summary_data.get("overall_assessment", {})
            confidence_level = overall_assessment.get("confidence_level", 0.0)
            
            # Validate confidence level
            if not isinstance(confidence_level, (int, float)) or confidence_level < 0 or confidence_level > 1:
                logger.warning(f"Invalid confidence level: {confidence_level}, defaulting to 0.5")
                confidence_level = 0.5
                summary_data["overall_assessment"]["confidence_level"] = confidence_level
            
            # Update state with summary
            state.update({
                "intelligent_summary": summary_data,
                "search_confidence": confidence_level,
                "workflow_status": SearchWorkflowStatus.SUMMARIZATION_COMPLETED
            })
            
            primary_finding = overall_assessment.get("primary_finding", "Summary completed")
            state["messages"].append(AIMessage(
                content=f"Intelligent summary completed. Finding: {primary_finding}, Confidence: {confidence_level}"
            ))
            
            logger.info(f"Intelligent summarization completed with confidence: {confidence_level}")
            
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            state.update({
                "workflow_status": SearchWorkflowStatus.ERROR,
                "error_context": {"stage": "summarization", "error": str(e)}
            })
            
            state["messages"].append(AIMessage(
                content=f"Summarization failed: {str(e)}"
            ))
        
        return state


### prompts/search_summarize_prompts.py

"""
Prompt Templates for Standalone Search & Summarize Agent
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

Consider:
- Primary objectives from search requirements
- Geographic region (IT, FR, etc.)
- Required confidence threshold
- Entity type (ENT_ACTIVITY vs ENT_WORKPLACE)

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

For Each Verification Objective:
- Was it successfully verified?
- What evidence supports the conclusion?
- Are there any conflicts or uncertainties?

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
    "summary_narrative": "Human-readable summary for review"
}}"""),
    ("human", """
Search Requirements: {search_requirements}
Search Results: {search_results}

Create intelligent summary of all findings.
""")
])


### utils/__init__.py

"""
Utils module initialization for standalone Search & Summarize Agent
"""

from .state_models import SearchAgentState, SearchWorkflowStatus
from .routing_functions import (
    search_routing_decision, 
    tool_execution_routing,
    summarization_routing
)
from .helpers import safe_json_parse, validate_search_requirements

__all__ = [
    'SearchAgentState',
    'SearchWorkflowStatus',
    'search_routing_decision',
    'tool_execution_routing',
    'summarization_routing',
    'safe_json_parse',
    'validate_search_requirements'
]


### agents/__init__.py

"""
Agents module initialization for standalone Search & Summarize Agent
"""

from .search_summarize_agent import SearchAndSummarizeAgent

__all__ = [
    'SearchAndSummarizeAgent'
]


### prompts/__init__.py

"""
Prompts module initialization for standalone Search & Summarize Agent
"""

from .search_summarize_prompts import TOOL_SELECTION_PROMPT, SUMMARIZATION_PROMPT

__all__ = [
    'TOOL_SELECTION_PROMPT',
    'SUMMARIZATION_PROMPT'
]


### tools/__init__.py

"""
Tools module initialization
YOUR EXISTING TOOL IMPLEMENTATIONS GO HERE
"""

# TODO: Import your actual tool implementations
# from .italy_trusted_sources import ItalyTrustedSourcesTool
# from .france_trusted_sources import FranceTrustedSourcesTool
# from .hospital_sources import HospitalSourcesTool
# from .linkedin_professional import LinkedInProfessionalTool
# from .untrusted_web_search import UntrustedWebSearchTool

__all__ = [
    # 'ItalyTrustedSourcesTool',
    # 'FranceTrustedSourcesTool',
    # 'HospitalSourcesTool',
    # 'LinkedInProfessionalTool',
    # 'UntrustedWebSearchTool'
]


### .env

# Azure OpenAI Configuration
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4

# Logging
LOG_LEVEL=INFO


### requirements.txt

langgraph>=0.0.26
langchain-core>=0.1.0
langchain-openai>=0.0.5
python-dotenv>=1.0.0
pydantic>=2.0.0
aiohttp>=3.8.0
tenacity>=8.2.0


### Example Tool Placeholder Files

### tools/italy_trusted_sources.py

"""
Placeholder for Italy Trusted Sources Tool
TODO: Implement your actual Italy trusted sources tool
"""

class ItalyTrustedSourcesTool:
    """
    Consolidated Italy tool for medical registries
    TODO: Implement actual tool functionality
    """
    
    def __init__(self):
        # TODO: Initialize your Italy tool
        pass
    
    async def search(self, search_requirements):
        """
        TODO: Implement actual search functionality
        """
        # Placeholder implementation
        return {
            "tool_name": "italy_trusted",
            "status": "placeholder",
            "results": [],
            "confidence": 0.0
        }


### tools/france_trusted_sources.py

"""
Placeholder for France Trusted Sources Tool
TODO: Implement your actual France trusted sources tool
"""

class FranceTrustedSourcesTool:
    """
    Consolidated France tool for medical directories
    TODO: Implement actual tool functionality
    """
    
    def __init__(self):
        # TODO: Initialize your France tool
        pass
    
    async def search(self, search_requirements):
        """
        TODO: Implement actual search functionality
        """
        # Placeholder implementation
        return {
            "tool_name": "france_trusted",
            "status": "placeholder",
            "results": [],
            "confidence": 0.0
        }


### tools/hospital_sources.py

"""
Placeholder for Hospital Sources Tool
TODO: Implement your actual hospital sources tool
"""

class HospitalSourcesTool:
    """
    Hospital websites and staff directories tool
    TODO: Implement actual tool functionality
    """
    
    def __init__(self):
        # TODO: Initialize your hospital tool
        pass
    
    async def search(self, search_requirements):
        """
        TODO: Implement actual search functionality
        """
        # Placeholder implementation
        return {
            "tool_name": "hospital_sources",
            "status": "placeholder",
            "results": [],
            "confidence": 0.0
        }


### tools/linkedin_professional.py

"""
Placeholder for LinkedIn Professional Tool
TODO: Implement your actual LinkedIn professional tool
"""

class LinkedInProfessionalTool:
    """
    Professional networking platforms tool
    TODO: Implement actual tool functionality
    """
    
    def __init__(self):
        # TODO: Initialize your LinkedIn tool
        pass
    
    async def search(self, search_requirements):
        """
        TODO: Implement actual search functionality
        """
        # Placeholder implementation
        return {
            "tool_name": "linkedin_professional",
            "status": "placeholder",
            "results": [],
            "confidence": 0.0
        }


### tools/untrusted_web_search.py

"""
Placeholder for Untrusted Web Search Tool
TODO: Implement your actual web search tool
"""

class UntrustedWebSearchTool:
    """
    General web search engines tool
    TODO: Implement actual tool functionality
    """
    
    def __init__(self):
        # TODO: Initialize your web search tool
        pass
    
    async def search(self, search_requirements):
        """
        TODO: Implement actual search functionality
        """
        # Placeholder implementation
        return {
            "tool_name": "untrusted_web_search",
            "status": "placeholder",
            "results": [],
            "confidence": 0.0
        }


### Testing and Usage Examples

### test_search_agent.py

"""
Test script for standalone Search & Summarize Agent
"""

import asyncio
import json
from main import process_search_request

async def test_basic_search():
    """Test basic search functionality"""
    
    # Test Case 1: Italian Medical Professional
    italian_search = {
        "verification_needed": True,
        "primary_objectives": [
            "verify_employment_status",
            "confirm_medical_license",
            "validate_workplace_affiliation"
        ],
        "geographic_region": "IT",
        "confidence_threshold": 0.85,
        "entity_details": {
            "firstName": "PAOLO",
            "lastName": "CORVISIERI", 
            "workplaceName": "DISTRETTO SANITARIO FIUMICINO",
            "specialtyCode": "18",
            "country": "IT",
            "city": "VELLETRI"
        },
        "search_context": {
            "entity_type": "ENT_ACTIVITY",
            "verification_type": "medical_professional",
            "urgency": "standard"
        }
    }
    
    print("=== Testing Italian Medical Professional ===")
    result = await process_search_request(italian_search)
    print(f"Status: {result['workflow_status'].value}")
    print(f"Tools: {result.get('selected_tools', [])}")
    print(f"Confidence: {result.get('search_confidence', 0.0)}")
    print()
    
    # Test Case 2: French Healthcare Professional
    french_search = {
        "verification_needed": True,
        "primary_objectives": [
            "verify_professional_registration",
            "confirm_current_workplace",
            "validate_specialty_credentials"
        ],
        "geographic_region": "FR",
        "confidence_threshold": 0.80,
        "entity_details": {
            "firstName": "MARIE",
            "lastName": "DUBOIS",
            "workplaceName": "HOPITAL SAINT-ANTOINE",
            "specialtyCode": "22",
            "country": "FR",
            "city": "PARIS"
        },
        "search_context": {
            "entity_type": "ENT_ACTIVITY",
            "verification_type": "medical_professional",
            "urgency": "high"
        }
    }
    
    print("=== Testing French Healthcare Professional ===")
    result = await process_search_request(french_search)
    print(f"Status: {result['workflow_status'].value}")
    print(f"Tools: {result.get('selected_tools', [])}")
    print(f"Confidence: {result.get('search_confidence', 0.0)}")
    print()
    
    # Test Case 3: No Verification Needed
    no_verification = {
        "verification_needed": False,
        "primary_objectives": [],
        "geographic_region": "IT",
        "confidence_threshold": 1.0
    }
    
    print("=== Testing No Verification Needed ===")
    result = await process_search_request(no_verification)
    print(f"Status: {result['workflow_status'].value}")
    print(f"Summary: {result.get('intelligent_summary', {})}")
    print()

async def test_error_cases():
    """Test error handling"""
    
    # Test Case 1: Invalid search requirements
    invalid_search = {
        "verification_needed": True,
        # Missing primary_objectives
        "geographic_region": "IT"
    }
    
    print("=== Testing Invalid Search Requirements ===")
    try:
        result = await process_search_request(invalid_search)
        print(f"Status: {result['workflow_status'].value}")
        print(f"Error: {result.get('error_context', {})}")
    except Exception as e:
        print(f"Exception caught: {str(e)}")
    print()

if __name__ == "__main__":
    asyncio.run(test_basic_search())
    asyncio.run(test_error_cases())


### batch_search_processor.py

"""
Batch processor for multiple search requests
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from main import process_search_request

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def process_search_batch(search_requests: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a batch of search requests
    
    Args:
        search_requests: List of search requirement dictionaries
    
    Returns:
        List of search results
    """
    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        logger.info(f"Processing batch of {len(search_requests)} search requests")
        
        results = []
        for idx, search_request in enumerate(search_requests):
            request_id = search_request.get("request_id", f"request_{idx}")
            logger.info(f"Processing search request {idx + 1}/{len(search_requests)} - ID: {request_id}")
            
            try:
                result = await process_search_request(search_request)
                results.append({
                    "request_id": request_id,
                    "status": "success",
                    "search_result": result
                })
                logger.info(f"Search request {request_id} completed successfully")
                
            except Exception as e:
                logger.error(f"Error processing search request {request_id}: {str(e)}")
                results.append({
                    "request_id": request_id,
                    "status": "error",
                    "error": str(e)
                })
        
        # Save batch results
        output_filename = f"search_batch_results_{batch_id}.json"
        with open(output_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        success_count = len([r for r in results if r["status"] == "success"])
        error_count = len([r for r in results if r["status"] == "error"])
        
        logger.info(f"Batch processing complete. Results saved to: {output_filename}")
        logger.info(f"Summary: {success_count} successful, {error_count} failed")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise

if __name__ == "__main__":
    # Example batch of search requests
    sample_batch = [
        {
            "request_id": "IT_001",
            "verification_needed": True,
            "primary_objectives": ["verify_employment_status", "confirm_medical_license"],
            "geographic_region": "IT",
            "confidence_threshold": 0.85,
            "entity_details": {
                "firstName": "PAOLO",
                "lastName": "CORVISIERI",
                "country": "IT"
            }
        },
        {
            "request_id": "FR_001", 
            "verification_needed": True,
            "primary_objectives": ["verify_professional_registration"],
            "geographic_region": "FR",
            "confidence_threshold": 0.80,
            "entity_details": {
                "firstName": "MARIE",
                "lastName": "DUBOIS",
                "country": "FR"
            }
        }
    ]
    
    asyncio.run(process_search_batch(sample_batch))


### README.md

# Standalone Search & Summarize Agent

This is a standalone implementation of the Search & Summarize Agent that can process search requirements independently using LangGraph workflow management.

## Features

- **Multi-source Search**: Supports 5 different search tools (Italy, France, Hospital, LinkedIn, Web)
- **Intelligent Tool Selection**: LLM-powered tool selection based on search requirements
- **Smart Summarization**: AI-powered synthesis of search results
- **Error Handling**: Comprehensive error handling and recovery
- **Batch Processing**: Support for processing multiple search requests

## Directory Structure

```
search_summarize_agent/
├── main.py                     # Standalone workflow entry point
├── config.py                   # Configuration
├── requirements.txt            # Dependencies
├── .env                       # Environment variables
├── agents/
│   ├── __init__.py
│   └── search_summarize_agent.py
├── utils/
│   ├── __init__.py
│   ├── state_models.py
│   ├── routing_functions.py
│   └── helpers.py
├── prompts/
│   ├── __init__.py
│   └── search_summarize_prompts.py
└── tools/
    ├── __init__.py
    ├── italy_trusted_sources.py
    ├── france_trusted_sources.py
    ├── hospital_sources.py
    ├── linkedin_professional.py
    └── untrusted_web_search.py
```

## Setup

1. Install dependencies:
```bash
pip install -r requirements.txt
```

2. Configure environment variables in `.env`:
```bash
AZURE_OPENAI_API_KEY=your-api-key-here
AZURE_OPENAI_ENDPOINT=https://your-resource.openai.azure.com/
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4
```

3. Implement your actual tools in the `tools/` directory

## Usage

### Single Search Request

```python
import asyncio
from main import process_search_request

search_requirements = {
    "verification_needed": True,
    "primary_objectives": [
        "verify_employment_status",
        "confirm_medical_license"
    ],
    "geographic_region": "IT",
    "confidence_threshold": 0.85,
    "entity_details": {
        "firstName": "PAOLO",
        "lastName": "CORVISIERI",
        "country": "IT"
    }
}

result = await process_search_request(search_requirements)
```

### Batch Processing

```python
from batch_search_processor import process_search_batch

search_requests = [
    # Multiple search requirement dictionaries
]

results = await process_search_batch(search_requests)
```

## Workflow

1. **Tool Selection**: LLM selects appropriate tools based on search requirements
2. **Search Execution**: Selected tools are executed in optimal order
3. **Intelligent Summarization**: Results are synthesized into actionable insights

## TODO Items

- Implement actual tool functionality in `tools/` directory
- Replace placeholder search implementations with real API calls
- Add authentication and rate limiting for external APIs
- Implement caching for repeated searches
- Add more sophisticated confidence scoring algorithms
