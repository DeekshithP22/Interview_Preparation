# standalone_search_state.py
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum
import operator
from langchain_core.messages import BaseMessage

class SearchWorkflowStatus(Enum):
    INITIATED = "initiated"
    TOOLS_SELECTED = "tools_selected"
    SEARCH_EXECUTED = "search_executed"
    RESULTS_SUMMARIZED = "results_summarized"
    COMPLETED = "completed"
    ERROR = "error"

class SearchAgentState(TypedDict):
    # Input requirements
    search_requirements: Dict[str, Any]
    
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

# ================================================================================================

# search_requirements_builder.py
from typing import Dict, Any, List

def build_search_requirements(
    individual_name: str,
    workplace_name: str,
    country: str,
    verification_objectives: List[str] = None,
    confidence_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Build search requirements for standalone testing
    """
    if verification_objectives is None:
        verification_objectives = [
            "verify_current_workplace",
            "verify_employment_status", 
            "verify_professional_credentials"
        ]
    
    return {
        "verification_needed": True,
        "primary_objectives": verification_objectives,
        "individual_details": {
            "name": individual_name,
            "workplace": workplace_name
        },
        "geographic_region": country,
        "confidence_threshold": confidence_threshold,
        "entity_type": "ENT_ACTIVITY",
        "search_context": "standalone_testing"
    }

def create_test_requirements():
    """Create test requirements for PAOLO CORVISIERI"""
    return build_search_requirements(
        individual_name="PAOLO CORVISIERI",
        workplace_name="DISTRETTO SANITARIO FIUMICINO", 
        country="IT",
        verification_objectives=["verify_current_workplace", "verify_employment_status"]
    )

# ================================================================================================

# standalone_search_workflow.py
import asyncio
import logging
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage
from standalone_search_state import SearchAgentState, SearchWorkflowStatus

# Import your existing agent
from agents.search_summarize_agent import SearchAndSummarizeAgent

logger = logging.getLogger(__name__)

async def handle_search_error(state: SearchAgentState) -> SearchAgentState:
    """Error handler for standalone search workflow"""
    error_context = state.get("error_context", {})
    error_msg = f"Search error in {error_context.get('stage', 'unknown')}: {error_context.get('error', 'Unknown error')}"
    
    logger.error(error_msg)
    if "messages" not in state:
        state["messages"] = []
    state["messages"].append(AIMessage(content=error_msg))
    state["workflow_status"] = SearchWorkflowStatus.ERROR
    
    return state

def create_standalone_search_workflow():
    """Create standalone LangGraph workflow for Search & Summarize Agent"""
    
    # Initialize your existing search agent
    search_agent = SearchAndSummarizeAgent()
    
    # Define workflow graph
    workflow = StateGraph(SearchAgentState)
    
    # Add nodes using your existing agent methods (need to create standalone versions)
    workflow.add_node("select_tools", search_agent.select_search_tools_standalone)
    workflow.add_node("execute_search", search_agent.execute_search_tools_standalone) 
    workflow.add_node("summarize_results", search_agent.intelligent_summarization_standalone)
    workflow.add_node("handle_error", handle_search_error)
    
    # Workflow edges
    workflow.add_edge(START, "select_tools")
    workflow.add_edge("select_tools", "execute_search")
    workflow.add_edge("execute_search", "summarize_results")
    
    # Conditional edge from summarize to completion or error
    workflow.add_conditional_edges(
        "summarize_results",
        lambda x: "complete" if x.get("workflow_status") == SearchWorkflowStatus.COMPLETED else "handle_error",
        {
            "complete": END,
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_edge("handle_error", END)
    
    return workflow.compile()

async def run_standalone_search(search_requirements: Dict[str, Any]) -> Dict[str, Any]:
    """Run standalone search workflow"""
    logger.info("Starting standalone search workflow")
    
    # Create workflow
    app = create_standalone_search_workflow()
    
    # Initial state
    initial_state = {
        "search_requirements": search_requirements,
        "selected_tools": [],
        "execution_order": [],
        "search_results": [],
        "intelligent_summary": None,
        "search_confidence": 0.0,
        "workflow_status": SearchWorkflowStatus.INITIATED,
        "error_context": None,
        "messages": []
    }
    
    # Execute workflow
    result = await app.ainvoke(initial_state)
    
    logger.info(f"Standalone search workflow completed - Status: {result['workflow_status'].value}")
    
    return result

# ================================================================================================

# ADDITIONS TO: agents/search_summarize_agent.py
# Add these methods to your existing SearchAndSummarizeAgent class:

class SearchAndSummarizeAgent:
    # ... your existing methods ...
    
    async def select_search_tools_standalone(self, state: SearchAgentState) -> SearchAgentState:
        """Standalone version of tool selection - no VR record dependency"""
        
        try:
            search_requirements = state.get("search_requirements", {})
            
            # Check if verification is needed
            if not search_requirements.get("verification_needed", False):
                state.update({
                    "selected_tools": [],
                    "execution_order": [],
                    "search_results": [],
                    "intelligent_summary": {"verification_not_required": True},
                    "search_confidence": 1.0,
                    "workflow_status": SearchWorkflowStatus.COMPLETED
                })
                
                state["messages"].append(AIMessage(content="No search required"))
                logger.info("No search required for standalone execution")
                return state
            
            # Use your existing tool selection logic
            logger.info("Selecting tools for standalone search")
            
            # TODO: Use your existing TOOL_SELECTION_PROMPT
            selection_response = await self.llm.ainvoke(
                TOOL_SELECTION_PROMPT.format(
                    search_requirements=search_requirements
                ).messages
            )
            
            # TODO: Use your existing safe_json_parse function
            tool_selection = safe_json_parse(selection_response.content)
            
            state.update({
                "selected_tools": tool_selection.get("selected_tools", []),
                "execution_order": tool_selection.get("execution_order", []),
                "workflow_status": SearchWorkflowStatus.TOOLS_SELECTED
            })
            
            state["messages"].append(AIMessage(
                content=f"Selected tools: {', '.join(tool_selection.get('selected_tools', []))}"
            ))
            
            logger.info(f"Tools selected: {tool_selection.get('selected_tools', [])}")
            
        except Exception as e:
            logger.error(f"Tool selection error: {str(e)}")
            state.update({
                "workflow_status": SearchWorkflowStatus.ERROR,
                "error_context": {"stage": "tool_selection", "error": str(e)}
            })
        
        return state
    
    async def execute_search_tools_standalone(self, state: SearchAgentState) -> SearchAgentState:
        """Standalone version of search execution - no VR record dependency"""
        
        try:
            execution_order = state.get("execution_order", [])
            search_requirements = state.get("search_requirements", {})
            
            logger.info(f"Executing {len(execution_order)} tools for standalone search")
            
            # Use your existing search tools
            for tool_name in execution_order:
                if tool_name in self.search_tools:
                    tool_instance = self.search_tools[tool_name]
                    
                    # Execute the actual tool search
                    search_result = await tool_instance.search(search_requirements)
                    state["search_results"].append(search_result)
                    
                    state["messages"].append(AIMessage(
                        content=f"Executed search with {tool_name}"
                    ))
                    
                    logger.debug(f"Executed {tool_name} for standalone search")
            
            state["workflow_status"] = SearchWorkflowStatus.SEARCH_EXECUTED
            logger.info("All tools executed for standalone search")
            
        except Exception as e:
            logger.error(f"Search execution error: {str(e)}")
            state.update({
                "workflow_status": SearchWorkflowStatus.ERROR,
                "error_context": {"stage": "search_execution", "error": str(e)}
            })
        
        return state
    
    async def intelligent_summarization_standalone(self, state: SearchAgentState) -> SearchAgentState:
        """Standalone version of summarization - no VR record dependency"""
        
        try:
            logger.info("Creating summary for standalone search")
            
            # Use your existing SUMMARIZATION_PROMPT
            summary_response = await self.llm.ainvoke(
                SUMMARIZATION_PROMPT.format(
                    search_requirements=state.get("search_requirements", {}),
                    search_results=state.get("search_results", [])
                ).messages
            )
            
            # Use your existing safe_json_parse function
            summary_data = safe_json_parse(summary_response.content)
            
            state.update({
                "intelligent_summary": summary_data,
                "search_confidence": summary_data.get("overall_assessment", {}).get("confidence_level", 0.0),
                "workflow_status": SearchWorkflowStatus.COMPLETED
            })
            
            confidence = summary_data.get("overall_assessment", {}).get("confidence_level", 0.0)
            state["messages"].append(AIMessage(
                content=f"Standalone search summary complete. Confidence: {confidence}"
            ))
            
            logger.info(f"Standalone summary created with confidence: {confidence}")
            
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            state.update({
                "workflow_status": SearchWorkflowStatus.ERROR,
                "error_context": {"stage": "summarization", "error": str(e)}
            })
        
        return state

# ================================================================================================

# test_standalone_search.py
import asyncio
import json
import logging
from datetime import datetime
from search_requirements_builder import create_test_requirements, build_search_requirements
from standalone_search_workflow import run_standalone_search

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_search():
    """Test basic search functionality"""
    print("\n=== BASIC SEARCH TEST ===")
    
    # Create test requirements
    requirements = create_test_requirements()
    
    print("Input Requirements:")
    print(json.dumps(requirements, indent=2))
    
    # Run standalone search
    result = await run_standalone_search(requirements)
    
    print(f"\n=== RESULTS ===")
    print(f"Status: {result['workflow_status'].value}")
    print(f"Tools Selected: {result.get('selected_tools', [])}")
    print(f"Search Confidence: {result.get('search_confidence', 0.0)}")
    
    if result.get("intelligent_summary"):
        summary = result["intelligent_summary"]
        print(f"Primary Finding: {summary.get('overall_assessment', {}).get('primary_finding', 'N/A')}")
    
    return result

async def test_custom_requirements():
    """Test with custom requirements"""
    print("\n=== CUSTOM REQUIREMENTS TEST ===")
    
    requirements = build_search_requirements(
        individual_name="MARCO ROSSI",
        workplace_name="OSPEDALE SAN RAFFAELE",
        country="IT",
        verification_objectives=["verify_credentials", "verify_specialization"],
        confidence_threshold=0.90
    )
    
    print("Custom Requirements:")
    print(json.dumps(requirements, indent=2))
    
    result = await run_standalone_search(requirements)
    
    print(f"\n=== CUSTOM RESULTS ===")
    print(f"Status: {result['workflow_status'].value}")
    print(f"Confidence: {result.get('search_confidence', 0.0)}")
    
    return result

async def main():
    """Run all standalone search tests"""
    print("🔍 STANDALONE SEARCH & SUMMARIZE AGENT TESTING")
    print("=" * 60)
    
    # Test 1: Basic functionality
    basic_result = await test_basic_search()
    
    # Test 2: Custom requirements  
    custom_result = await test_custom_requirements()
    
    # Summary
    print("\n=== TEST SUMMARY ===")
    print(f"Basic Test: {'✅ PASSED' if basic_result['workflow_status'].value == 'completed' else '❌ FAILED'}")
    print(f"Custom Test: {'✅ PASSED' if custom_result['workflow_status'].value == 'completed' else '❌ FAILED'}")
    
    # Save results
    timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
    results_file = f"standalone_search_results_{timestamp}.json"
    
    with open(results_file, 'w') as f:
        json.dump({
            "basic_test": basic_result,
            "custom_test": custom_result,
            "timestamp": timestamp
        }, f, indent=2, default=str)
    
    print(f"Results saved to: {results_file}")

if __name__ == "__main__":
    asyncio.run(main())
