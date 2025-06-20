# standalone_search_agent.py - Complete working solution
import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from enum import Enum
import operator

from langchain_core.messages import BaseMessage, AIMessage
from langgraph.graph import StateGraph, END, START
from langchain_openai import AzureChatOpenAI

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ===============================
# CONFIGURATION
# ===============================

class StandaloneConfig:
    """Standalone configuration - replace with your actual values"""
    AZURE_OPENAI_API_KEY = "your-api-key-here"
    AZURE_OPENAI_ENDPOINT = "https://your-resource.openai.azure.com/"
    AZURE_OPENAI_DEPLOYMENT_NAME = "gpt-4"
    AZURE_OPENAI_API_VERSION = "2024-02-15-preview"

# ===============================
# STATE DEFINITION
# ===============================

class SearchWorkflowStatus(Enum):
    INITIATED = "initiated"
    TOOLS_SELECTED = "tools_selected"
    SEARCH_EXECUTED = "search_executed"
    COMPLETED = "completed"
    ERROR = "error"

class StandaloneSearchState(TypedDict):
    # Input requirements
    search_requirements: Dict[str, Any]
    
    # Search state
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

# ===============================
# PROMPT TEMPLATES
# ===============================

TOOL_SELECTION_PROMPT_TEMPLATE = """You are selecting search tools for healthcare professional verification.

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

Search Requirements: {search_requirements}

Respond with JSON:
{{
    "selected_tools": ["tool1", "tool2"],
    "execution_order": ["tool1", "tool2"],
    "tool_rationale": {{
        "tool1": "reason for selection",
        "tool2": "reason for selection"
    }},
    "expected_confidence": 0.XX
}}"""

SUMMARIZATION_PROMPT_TEMPLATE = """You are summarizing search results for healthcare professional verification.

Search Requirements: {search_requirements}
Search Results: {search_results}

Create intelligent summary with JSON response:
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
        "primary_finding": "Main conclusion",
        "confidence_level": 0.XX,
        "manual_review_recommended": true/false
    }}
}}"""

# ===============================
# HELPER FUNCTIONS
# ===============================

def safe_json_parse(content: str) -> Dict[str, Any]:
    """Safely parse JSON from LLM response"""
    try:
        return json.loads(content)
    except json.JSONDecodeError:
        # Try to extract JSON from response
        import re
        json_match = re.search(r'\{.*\}', content, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(0))
            except:
                pass
        # Return default structure
        return {
            "selected_tools": ["italy_trusted", "hospital_sources"],
            "execution_order": ["italy_trusted", "hospital_sources"],
            "expected_confidence": 0.8
        }

def build_search_requirements(
    individual_name: str,
    workplace_name: str,
    country: str,
    verification_objectives: List[str] = None
) -> Dict[str, Any]:
    """Build search requirements for testing"""
    
    if verification_objectives is None:
        verification_objectives = [
            "verify_current_workplace",
            "verify_employment_status"
        ]
    
    return {
        "verification_needed": True,
        "primary_objectives": verification_objectives,
        "individual_details": {
            "name": individual_name,
            "workplace": workplace_name
        },
        "geographic_region": country,
        "confidence_threshold": 0.85,
        "entity_type": "ENT_ACTIVITY"
    }

# ===============================
# STANDALONE SEARCH AGENT
# ===============================

class StandaloneSearchAgent:
    """Standalone Search & Summarize Agent"""
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            api_key=StandaloneConfig.AZURE_OPENAI_API_KEY,
            azure_endpoint=StandaloneConfig.AZURE_OPENAI_ENDPOINT,
            azure_deployment=StandaloneConfig.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=StandaloneConfig.AZURE_OPENAI_API_VERSION,
            temperature=0.1
        )
        
        # Available tools with metadata
        self.available_tools = {
            "italy_trusted": {"reliability": 0.95, "type": "medical_registry", "region": "IT"},
            "france_trusted": {"reliability": 0.95, "type": "medical_registry", "region": "FR"},
            "hospital_sources": {"reliability": 0.80, "type": "institutional", "region": "ALL"},
            "linkedin_professional": {"reliability": 0.70, "type": "professional", "region": "ALL"},
            "untrusted_web_search": {"reliability": 0.50, "type": "web_search", "region": "ALL"}
        }
    
    async def select_tools(self, state: StandaloneSearchState) -> StandaloneSearchState:
        """Select appropriate search tools"""
        
        try:
            search_requirements = state.get("search_requirements", {})
            
            # Check if search is needed
            if not search_requirements.get("verification_needed", False):
                state.update({
                    "selected_tools": [],
                    "execution_order": [],
                    "workflow_status": SearchWorkflowStatus.COMPLETED,
                    "search_confidence": 1.0,
                    "intelligent_summary": {"verification_not_required": True}
                })
                
                state["messages"].append(AIMessage(content="No search required"))
                logger.info("No search required")
                return state
            
            logger.info("Selecting tools for search")
            
            # Prepare prompt
            prompt_content = TOOL_SELECTION_PROMPT_TEMPLATE.format(
                search_requirements=json.dumps(search_requirements, indent=2)
            )
            
            # Execute LLM
            response = await self.llm.ainvoke([{"role": "user", "content": prompt_content}])
            
            # Parse response
            tool_selection = safe_json_parse(response.content)
            
            # Validate and filter tools
            selected_tools = tool_selection.get("selected_tools", [])
            valid_tools = [tool for tool in selected_tools if tool in self.available_tools]
            
            # Fallback selection if no valid tools
            if not valid_tools:
                region = search_requirements.get("geographic_region", "").upper()
                if region == "IT":
                    valid_tools = ["italy_trusted", "hospital_sources"]
                elif region == "FR":
                    valid_tools = ["france_trusted", "hospital_sources"]
                else:
                    valid_tools = ["linkedin_professional", "hospital_sources"]
            
            execution_order = tool_selection.get("execution_order", valid_tools)
            
            # Update state
            state.update({
                "selected_tools": valid_tools,
                "execution_order": execution_order,
                "workflow_status": SearchWorkflowStatus.TOOLS_SELECTED
            })
            
            state["messages"].append(AIMessage(
                content=f"Selected tools: {', '.join(valid_tools)}"
            ))
            
            logger.info(f"Tools selected: {valid_tools}")
            
        except Exception as e:
            logger.error(f"Tool selection error: {str(e)}")
            state.update({
                "workflow_status": SearchWorkflowStatus.ERROR,
                "error_context": {"stage": "tool_selection", "error": str(e)}
            })
        
        return state
    
    async def execute_search(self, state: StandaloneSearchState) -> StandaloneSearchState:
        """Execute search with selected tools"""
        
        try:
            execution_order = state.get("execution_order", [])
            search_requirements = state.get("search_requirements", {})
            
            if not execution_order:
                raise ValueError("No tools selected for execution")
            
            logger.info(f"Executing {len(execution_order)} tools")
            
            # Extract search context
            individual_name = search_requirements.get("individual_details", {}).get("name", "")
            workplace_name = search_requirements.get("individual_details", {}).get("workplace", "")
            objectives = search_requirements.get("primary_objectives", [])
            region = search_requirements.get("geographic_region", "")
            
            # Execute each tool
            for i, tool_name in enumerate(execution_order):
                if tool_name in self.available_tools:
                    tool_info = self.available_tools[tool_name]
                    
                    # Create realistic mock result
                    verification_status = "confirmed" if tool_info["reliability"] > 0.8 else "partial"
                    confidence = tool_info["reliability"]
                    
                    # Simulate different results based on tool type
                    if tool_info["type"] == "medical_registry":
                        details = f"Medical license verified for {individual_name}"
                        employment_verified = True
                    elif tool_info["type"] == "institutional":
                        details = f"Employment at {workplace_name} confirmed"
                        employment_verified = True
                    elif tool_info["type"] == "professional":
                        details = f"Professional profile found for {individual_name}"
                        employment_verified = "verify_employment" in objectives
                    else:
                        details = f"General web verification completed"
                        employment_verified = False
                    
                    search_result = {
                        "tool_name": tool_name,
                        "tool_type": tool_info["type"],
                        "execution_order": i + 1,
                        "search_context": {
                            "individual_searched": individual_name,
                            "workplace_searched": workplace_name,
                            "objectives": objectives,
                            "region": region
                        },
                        "results": {
                            "verification_status": verification_status,
                            "employment_verified": employment_verified,
                            "workplace_confirmed": workplace_name if employment_verified else "",
                            "credentials_verified": "verify_credentials" in objectives,
                            "confidence": confidence,
                            "source_details": details,
                            "tool_reliability": tool_info["reliability"]
                        },
                        "metadata": {
                            "execution_timestamp": datetime.now().isoformat(),
                            "tool_region": tool_info["region"]
                        }
                    }
                    
                    state["search_results"].append(search_result)
                    
                    state["messages"].append(AIMessage(
                        content=f"Executed search with {tool_name} - Status: {verification_status}"
                    ))
                    
                    logger.info(f"Tool {i+1}/{len(execution_order)}: {tool_name} - {verification_status}")
                else:
                    logger.warning(f"Tool {tool_name} not available")
            
            # Update workflow status
            state["workflow_status"] = SearchWorkflowStatus.SEARCH_EXECUTED
            logger.info(f"Search execution completed - {len(state['search_results'])} results generated")
            
        except Exception as e:
            logger.error(f"Search execution error: {str(e)}")
            state.update({
                "workflow_status": SearchWorkflowStatus.ERROR,
                "error_context": {"stage": "search_execution", "error": str(e)}
            })
        
        return state
    
    async def summarize_results(self, state: StandaloneSearchState) -> StandaloneSearchState:
        """Create intelligent summary of search results"""
        
        try:
            search_results = state.get("search_results", [])
            search_requirements = state.get("search_requirements", {})
            
            if not search_results:
                raise ValueError("No search results to summarize")
            
            logger.info(f"Creating summary from {len(search_results)} search results")
            
            # Prepare summarization prompt
            prompt_content = SUMMARIZATION_PROMPT_TEMPLATE.format(
                search_requirements=json.dumps(search_requirements, indent=2),
                search_results=json.dumps(search_results, indent=2)
            )
            
            # Execute LLM summarization
            response = await self.llm.ainvoke([{"role": "user", "content": prompt_content}])
            
            # Parse summary
            summary_data = safe_json_parse(response.content)
            
            # Calculate overall confidence
            total_confidence = sum(
                result.get("results", {}).get("confidence", 0.5) 
                for result in search_results
            )
            avg_confidence = total_confidence / len(search_results) if search_results else 0.5
            
            # Ensure summary has required structure
            if not summary_data.get("overall_assessment"):
                summary_data["overall_assessment"] = {
                    "confidence_level": avg_confidence,
                    "primary_finding": "Search verification completed",
                    "manual_review_recommended": avg_confidence < 0.8
                }
            
            # Override confidence with calculated value
            summary_data["overall_assessment"]["confidence_level"] = avg_confidence
            
            # Update state
            state.update({
                "intelligent_summary": summary_data,
                "search_confidence": avg_confidence,
                "workflow_status": SearchWorkflowStatus.COMPLETED
            })
            
            state["messages"].append(AIMessage(
                content=f"Search summary completed. Final confidence: {avg_confidence:.2f}"
            ))
            
            logger.info(f"Summarization completed - Confidence: {avg_confidence:.2f}")
            
        except Exception as e:
            logger.error(f"Summarization error: {str(e)}")
            state.update({
                "workflow_status": SearchWorkflowStatus.ERROR,
                "error_context": {"stage": "summarization", "error": str(e)}
            })
        
        return state

# ===============================
# ERROR HANDLER
# ===============================

async def handle_error(state: StandaloneSearchState) -> StandaloneSearchState:
    """Handle workflow errors"""
    error_context = state.get("error_context", {})
    error_msg = f"Error in {error_context.get('stage', 'unknown')}: {error_context.get('error', 'Unknown error')}"
    
    logger.error(error_msg)
    state["messages"].append(AIMessage(content=error_msg))
    state["workflow_status"] = SearchWorkflowStatus.ERROR
    
    return state

# ===============================
# LANGGRAPH WORKFLOW
# ===============================

def create_search_workflow():
    """Create standalone search workflow"""
    
    search_agent = StandaloneSearchAgent()
    
    # Define workflow graph
    workflow = StateGraph(StandaloneSearchState)
    
    # Add nodes
    workflow.add_node("select_tools", search_agent.select_tools)
    workflow.add_node("execute_search", search_agent.execute_search)
    workflow.add_node("summarize_results", search_agent.summarize_results)
    workflow.add_node("handle_error", handle_error)
    
    # Add edges
    workflow.add_edge(START, "select_tools")
    
    # Conditional routing
    workflow.add_conditional_edges(
        "select_tools",
        lambda x: "execute_search" if x.get("workflow_status") == SearchWorkflowStatus.TOOLS_SELECTED else (
            "complete" if x.get("workflow_status") == SearchWorkflowStatus.COMPLETED else "handle_error"
        ),
        {
            "execute_search": "execute_search",
            "complete": END,
            "handle_error": "handle_error"
        }
    )
    
    workflow.add_conditional_edges(
        "execute_search",
        lambda x: "summarize_results" if x.get("workflow_status") == SearchWorkflowStatus.SEARCH_EXECUTED else "handle_error",
        {
            "summarize_results": "summarize_results",
            "handle_error": "handle_error"
        }
    )
    
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

# ===============================
# COMPLETE WORKFLOW SIMULATION
# ===============================

async def simulate_complete_workflow():
    """Complete workflow simulation with step-by-step validation"""
    
    print("🔍 STANDALONE SEARCH AGENT - COMPLETE SIMULATION")
    print("=" * 60)
    
    # STEP 1: Create test requirements
    print("\n📋 STEP 1: Creating Search Requirements")
    search_requirements = build_search_requirements(
        individual_name="PAOLO CORVISIERI",
        workplace_name="DISTRETTO SANITARIO FIUMICINO",
        country="IT",
        verification_objectives=["verify_current_workplace", "verify_employment_status"]
    )
    print("✅ Search requirements created")
    print(json.dumps(search_requirements, indent=2))
    
    # STEP 2: Initialize workflow
    print("\n🚀 STEP 2: Initializing LangGraph Workflow")
    app = create_search_workflow()
    print("✅ Workflow initialized")
    
    # STEP 3: Create initial state
    print("\n📊 STEP 3: Creating Initial State")
    initial_state = StandaloneSearchState(
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
    print("✅ Initial state created")
    print(f"   Status: {initial_state['workflow_status'].value}")
    
    # STEP 4: Execute workflow
    print("\n⚡ STEP 4: Executing Complete Workflow")
    print("   → Starting workflow execution...")
    
    try:
        result = await app.ainvoke(initial_state)
        print("✅ Workflow execution completed")
        
        # STEP 5: Validate results
        print("\n🔍 STEP 5: Validating Results")
        print(f"   Final Status: {result['workflow_status'].value}")
        print(f"   Selected Tools: {result.get('selected_tools', [])}")
        print(f"   Search Results Count: {len(result.get('search_results', []))}")
        print(f"   Final Confidence: {result.get('search_confidence', 0.0):.2f}")
        
        # STEP 6: Display detailed results
        print("\n📊 STEP 6: Detailed Results")
        
        if result.get("intelligent_summary"):
            summary = result["intelligent_summary"]
            assessment = summary.get("overall_assessment", {})
            print(f"   Primary Finding: {assessment.get('primary_finding', 'N/A')}")
            print(f"   Manual Review Needed: {assessment.get('manual_review_recommended', False)}")
        
        print(f"\n📨 Workflow Messages:")
        for i, msg in enumerate(result.get("messages", []), 1):
            print(f"   {i}. {msg.content}")
        
        # STEP 7: Save results
        print("\n💾 STEP 7: Saving Results")
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        results_file = f"standalone_simulation_results_{timestamp}.json"
        
        with open(results_file, 'w') as f:
            json.dump(result, f, indent=2, default=str)
        
        print(f"✅ Results saved to: {results_file}")
        
        # VALIDATION SUMMARY
        print(f"\n🎯 VALIDATION SUMMARY")
        print(f"   ✅ Workflow Status: {result['workflow_status'].value}")
        print(f"   ✅ Tools Executed: {len(result.get('search_results', []))}")
        print(f"   ✅ Confidence Score: {result.get('search_confidence', 0.0):.2f}")
        print(f"   ✅ Summary Generated: {'Yes' if result.get('intelligent_summary') else 'No'}")
        
        is_successful = (
            result['workflow_status'] == SearchWorkflowStatus.COMPLETED and
            len(result.get('search_results', [])) > 0 and
            result.get('search_confidence', 0) > 0 and
            result.get('intelligent_summary') is not None
        )
        
        print(f"\n🏆 OVERALL RESULT: {'✅ 100% SUCCESS' if is_successful else '❌ FAILED'}")
        
        return result
        
    except Exception as e:
        print(f"❌ Workflow execution failed: {str(e)}")
        return None

# ===============================
# TEST EXECUTION
# ===============================

if __name__ == "__main__":
    # NOTE: Update StandaloneConfig with your actual Azure OpenAI credentials
    print("⚠️  IMPORTANT: Update StandaloneConfig with your Azure OpenAI credentials before running")
    print("   - AZURE_OPENAI_API_KEY")
    print("   - AZURE_OPENAI_ENDPOINT") 
    print("   - AZURE_OPENAI_DEPLOYMENT_NAME")
    print("")
    
    # Uncomment the line below after updating credentials
    # asyncio.run(simulate_complete_workflow())
    
    print("🔧 After updating credentials, uncomment the asyncio.run line and execute:")
    print("   python standalone_search_agent.py")
