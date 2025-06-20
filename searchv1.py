# standalone_search_agent_fixed.py
import asyncio
import json
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum
import operator
from langchain_core.messages import BaseMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

def safe_json_parse(content: str) -> dict:
    """Safely parse JSON content with fallback"""
    try:
        return json.loads(content)
    except json.JSONDecodeError as e:
        logger.error(f"JSON parse error: {e}")
        # Try to extract JSON from markdown code blocks
        if "```json" in content:
            start = content.find("```json") + 7
            end = content.find("```", start)
            if end != -1:
                try:
                    return json.loads(content[start:end].strip())
                except:
                    pass
        
        # Return default structure based on content
        if "selected_tools" in content.lower():
            return {
                "selected_tools": ["italy_trusted", "general_web_search"],
                "execution_order": ["italy_trusted", "general_web_search"],
                "reasoning": "Default tool selection due to parse error"
            }
        else:
            return {
                "overall_assessment": {
                    "primary_finding": "Analysis completed with parsing limitations",
                    "confidence_level": 0.7,
                    "verification_status": "partial"
                },
                "detailed_findings": {},
                "recommendations": ["Review source JSON format"]
            }

class MockLLM:
    """Mock LLM for testing without actual API calls"""
    
    async def ainvoke(self, messages):
        """Mock LLM response"""
        message_content = str(messages)
        
        if "tool selection" in message_content.lower() or "select" in message_content.lower():
            # Mock tool selection response
            response_content = json.dumps({
                "selected_tools": ["italy_trusted", "general_web_search"],
                "execution_order": ["italy_trusted", "general_web_search"],
                "reasoning": "Selected tools based on Italian geographic region and employment verification needs"
            })
        else:
            # Mock summarization response
            response_content = json.dumps({
                "overall_assessment": {
                    "primary_finding": "Employment verification completed successfully",
                    "confidence_level": 0.85,
                    "verification_status": "confirmed"
                },
                "detailed_findings": {
                    "employment_status": "Active employment confirmed",
                    "workplace_verification": "Workplace details verified",
                    "source_reliability": "High confidence from trusted sources"
                },
                "recommendations": [
                    "Verification process completed successfully",
                    "No additional verification required"
                ]
            })
        
        class MockResponse:
            def __init__(self, content):
                self.content = content
        
        return MockResponse(response_content)

class StandaloneSearchAgent:
    """Standalone Search and Summarize Agent"""
    
    def __init__(self, use_mock_llm: bool = True):
        if use_mock_llm:
            self.llm = MockLLM()
        else:
            # Initialize real LLM here
            # self.llm = AzureChatOpenAI(...)
            raise NotImplementedError("Real LLM initialization needed")
        
        self.search_tools = {
            "italy_trusted": "Mock Italy Trusted Source",
            "general_web_search": "Mock General Web Search",
            "linkedin_search": "Mock LinkedIn Search"
        }
        
        # Define prompts
        self.tool_selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a tool selection expert. Given search requirements, select the most appropriate search tools.
            
            Available tools: italy_trusted, linkedin_search, general_web_search
            
            Return JSON format:
            {
                "selected_tools": ["tool1", "tool2"],
                "execution_order": ["tool1", "tool2"],
                "reasoning": "explanation"
            }"""),
            ("human", "Search requirements: {search_requirements}")
        ])
        
        self.summarization_prompt = ChatPromptTemplate.from_messages([
            ("system", """Analyze search results and create an intelligent summary.
            
            Return JSON format:
            {
                "overall_assessment": {
                    "primary_finding": "summary",
                    "confidence_level": 0.85,
                    "verification_status": "confirmed/partial/unconfirmed"
                },
                "detailed_findings": {},
                "recommendations": []
            }"""),
            ("human", "Requirements: {search_requirements}\nResults: {search_results}")
        ])
    
    async def select_search_tools_standalone(self, state: SearchAgentState) -> SearchAgentState:
        """Standalone tool selection"""
        
        try:
            search_requirements = state.get("search_requirements", {})
            
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
            
            # Execute tool selection LLM
            logger.info("Selecting tools for standalone search")
            
            formatted_prompt = self.tool_selection_prompt.format(
                search_requirements=search_requirements
            )
            
            selection_response = await self.llm.ainvoke(formatted_prompt)
            
            # Parse selection
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
        """Standalone search execution"""
        
        try:
            execution_order = state.get("execution_order", [])
            search_requirements = state.get("search_requirements", {})
            
            logger.info(f"Executing {len(execution_order)} tools for standalone search")
            
            for tool_name in execution_order:
                if tool_name in self.search_tools:
                    # Mock tool execution for testing
                    individual_name = search_requirements.get("individual_details", {}).get("name", "Unknown")
                    workplace = search_requirements.get("individual_details", {}).get("workplace", "Unknown")
                    
                    mock_result = {
                        "tool_name": tool_name,
                        "search_objective": search_requirements.get("primary_objectives", []),
                        "geographic_region": search_requirements.get("geographic_region", ""),
                        "individual_name": individual_name,
                        "workplace": workplace,
                        "results": {
                            "verification_status": "confirmed",
                            "employment_status": "active",
                            "source_reliability": 0.90 if tool_name == "italy_trusted" else 0.80,
                            "details": f"Mock verification completed via {tool_name} for {individual_name} at {workplace}",
                            "additional_info": {
                                "position": "Healthcare Professional",
                                "employment_duration": "2+ years",
                                "verification_date": "2025-06-20"
                            }
                        }
                    }
                    
                    state["search_results"].append(mock_result)
                    
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
        """Standalone summarization"""
        
        try:
            logger.info("Creating summary for standalone search")
            
            # Execute summarization LLM
            formatted_prompt = self.summarization_prompt.format(
                search_requirements=state.get("search_requirements", {}),
                search_results=state.get("search_results", [])
            )
            
            summary_response = await self.llm.ainvoke(formatted_prompt)
            
            # Parse summary
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

def build_search_requirements(
    individual_name: str,
    workplace_name: str,
    country: str,
    verification_objectives: List[str] = None,
    confidence_threshold: float = 0.85
) -> Dict[str, Any]:
    """Build search requirements for standalone testing"""
    
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

async def run_standalone_search_test():
    """Test the standalone search agent"""
    
    print("🔍 Testing Standalone Search Agent")
    print("=" * 50)
    
    # Create test requirements
    requirements = build_search_requirements(
        individual_name="PAOLO CORVISIERI",
        workplace_name="DISTRETTO SANITARIO FIUMICINO",
        country="IT",
        verification_objectives=["verify_current_workplace", "verify_employment_status"]
    )
    
    print("Input Requirements:")
    print(json.dumps(requirements, indent=2))
    
    # Initialize agent
    agent = StandaloneSearchAgent(use_mock_llm=True)
    
    # Initial state
    state = {
        "search_requirements": requirements,
        "selected_tools": [],
        "execution_order": [],
        "search_results": [],
        "intelligent_summary": None,
        "search_confidence": 0.0,
        "workflow_status": SearchWorkflowStatus.INITIATED,
        "error_context": None,
        "messages": []
    }
    
    print("\n--- Step 1: Tool Selection ---")
    state = await agent.select_search_tools_standalone(state)
    print(f"Status: {state['workflow_status'].value}")
    print(f"Selected Tools: {state.get('selected_tools', [])}")
    
    print("\n--- Step 2: Search Execution ---")
    state = await agent.execute_search_tools_standalone(state)
    print(f"Status: {state['workflow_status'].value}")
    print(f"Search Results Count: {len(state.get('search_results', []))}")
    
    print("\n--- Step 3: Summarization ---")
    state = await agent.intelligent_summarization_standalone(state)
    print(f"Final Status: {state['workflow_status'].value}")
    print(f"Confidence: {state.get('search_confidence', 0.0)}")
    
    # Display final summary
    if state.get("intelligent_summary"):
        summary = state["intelligent_summary"]
        print(f"\n=== FINAL SUMMARY ===")
        print(f"Primary Finding: {summary.get('overall_assessment', {}).get('primary_finding', 'N/A')}")
        print(f"Verification Status: {summary.get('overall_assessment', {}).get('verification_status', 'N/A')}")
        print(f"Confidence Level: {summary.get('overall_assessment', {}).get('confidence_level', 0.0)}")
    
    return state

if __name__ == "__main__":
    asyncio.run(run_standalone_search_test())
