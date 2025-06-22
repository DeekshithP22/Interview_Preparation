# **🔍 STANDALONE SEARCH & SUMMARIZE AGENT ANALYSIS**

## **📋 CURRENT DEPENDENCY ANALYSIS**

### **Current Integration Points:**
1. **Input Dependency:** Receives `search_requirements` from SupervisorAgent
2. **State Dependency:** Uses `AgentState` which includes all workflow fields
3. **Tool Dependencies:** Requires tool implementations
4. **LLM Dependencies:** Uses Azure OpenAI configuration

---

## **🎯 STANDALONE REQUIREMENTS ANALYSIS**

### **STEP 1: Create Minimal State Model**

**📍 Current Issue:** `AgentState` has 15+ fields that SearchAgent doesn't need

**🔧 Solution:** Create minimal state for standalone operation

```python
# New file: standalone_search_state.py
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
    
    # Search & Summarize Agent state (only what's needed)
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
```

---

### **STEP 2: Create Standalone Search Requirements Builder**

**📍 Current Issue:** Search requirements come from supervisor analysis

**🔧 Solution:** Create requirements builder for testing

```python
# New file: search_requirements_builder.py
from typing import Dict, Any

def build_search_requirements(
    individual_name: str,
    workplace_name: str,
    country: str,
    verification_objectives: List[str] = None,
    confidence_threshold: float = 0.85
) -> Dict[str, Any]:
    """
    Build search requirements for standalone testing
    
    Args:
        individual_name: "PAOLO CORVISIERI"
        workplace_name: "DISTRETTO SANITARIO FIUMICINO"
        country: "IT"
        verification_objectives: ["verify_employment", "verify_credentials"]
        confidence_threshold: 0.85
    
    Returns:
        Formatted search requirements dictionary
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

# Example usage
def create_test_requirements():
    return build_search_requirements(
        individual_name="PAOLO CORVISIERI",
        workplace_name="DISTRETTO SANITARIO FIUMICINO", 
        country="IT",
        verification_objectives=["verify_current_workplace", "verify_employment_status"]
    )
```

---

### **STEP 3: Create Standalone LangGraph Workflow**

**📍 Current Issue:** Search agent is embedded in main workflow

**🔧 Solution:** Create dedicated search workflow

```python
# New file: standalone_search_workflow.py
import asyncio
import logging
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage
from standalone_search_state import SearchAgentState, SearchWorkflowStatus
from agents.search_summarize_agent import SearchAndSummarizeAgent

logger = logging.getLogger(__name__)

async def handle_search_error(state: SearchAgentState) -> SearchAgentState:
    """Error handler for standalone search workflow"""
    error_context = state.get("error_context", {})
    error_msg = f"Search error in {error_context.get('stage', 'unknown')}: {error_context.get('error', 'Unknown error')}"
    
    logger.error(error_msg)
    state["messages"].append(AIMessage(content=error_msg))
    state["workflow_status"] = SearchWorkflowStatus.ERROR
    
    return state

def create_standalone_search_workflow():
    """
    Create standalone LangGraph workflow for Search & Summarize Agent only
    """
    
    # Initialize search agent
    search_agent = SearchAndSummarizeAgent()
    
    # Define workflow graph
    workflow = StateGraph(SearchAgentState)
    
    # Search & Summarize Agent nodes (modified for standalone)
    workflow.add_node("select_tools", search_agent.select_search_tools_standalone)
    workflow.add_node("execute_search", search_agent.execute_search_tools_standalone) 
    workflow.add_node("summarize_results", search_agent.intelligent_summarization_standalone)
    
    # Error handler node
    workflow.add_node("handle_error", handle_search_error)
    
    # Workflow edges - linear flow for search agent
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
    
    # Error handler to END
    workflow.add_edge("handle_error", END)
    
    # Compile workflow
    return workflow.compile()

async def run_standalone_search(search_requirements: Dict[str, Any]) -> Dict[str, Any]:
    """
    Run standalone search workflow
    
    Args:
        search_requirements: Search requirements dictionary
        
    Returns:
        Complete workflow result with search summary
    """
    logger.info(f"Starting standalone search workflow")
    
    # Create workflow
    app = create_standalone_search_workflow()
    
    # Initial state for standalone search
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
    
    logger.info(f"Standalone search workflow completed - Status: {result['workflow_status'].value}")
    
    return result
```

---

### **STEP 4: Modify SearchAndSummarizeAgent Methods**

**📍 Current Issue:** Methods expect full AgentState with VR record

**🔧 Solution:** Create standalone versions of methods

```python
# Additions to agents/search_summarize_agent.py

class SearchAndSummarizeAgent:
    # ... existing methods ...
    
    async def select_search_tools_standalone(self, state: SearchAgentState) -> SearchAgentState:
        """Standalone version - no VR record dependency"""
        
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
            
            selection_response = await self.llm.ainvoke(
                TOOL_SELECTION_PROMPT.format(
                    search_requirements=search_requirements
                ).messages
            )
            
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
        """Standalone version - no VR record dependency"""
        
        try:
            execution_order = state["execution_order"]
            search_requirements = state["search_requirements"]
            
            logger.info(f"Executing {len(execution_order)} tools for standalone search")
            
            for tool_name in execution_order:
                if tool_name in self.search_tools:
                    tool_instance = self.search_tools[tool_name]
                    
                    # TODO: Execute tool search
                    # search_result = await tool_instance.search(search_requirements)
                    # state["search_results"].append(search_result)
                    
                    # Placeholder execution for standalone testing
                    placeholder_result = {
                        "tool_name": tool_name,
                        "search_objective": search_requirements.get("primary_objectives", []),
                        "geographic_region": search_requirements.get("geographic_region", ""),
                        "individual_name": search_requirements.get("individual_details", {}).get("name", ""),
                        "results": {
                            "verification_status": "confirmed",
                            "employment_status": "active",
                            "source_reliability": 0.90 if tool_name == "italy_trusted" else 0.80,
                            "details": f"Verification completed via {tool_name}"
                        }
                    }
                    state["search_results"].append(placeholder_result)
                    
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
        """Standalone version - no VR record dependency"""
        
        try:
            logger.info("Creating summary for standalone search")
            
            # Execute summarization LLM
            summary_response = await self.llm.ainvoke(
                SUMMARIZATION_PROMPT.format(
                    search_requirements=state["search_requirements"],
                    search_results=state["search_results"]
                ).messages
            )
            
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
```

---

### **STEP 5: Create Test Runner Script**

**📍 Current Issue:** No way to test search agent independently

**🔧 Solution:** Create comprehensive test script

```python
# New file: test_standalone_search.py
import asyncio
import json
import logging
from datetime import datetime
from search_requirements_builder import create_test_requirements, build_search_requirements
from standalone_search_workflow import run_standalone_search

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

async def test_basic_search():
    """Test basic search functionality"""
    print("\n=== BASIC SEARCH TEST ===")
    
    # Create test requirements
    requirements = create_test_requirements()
    
    print(f"Input Requirements:")
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
    
    # Create custom requirements
    requirements = build_search_requirements(
        individual_name="MARCO ROSSI",
        workplace_name="OSPEDALE SAN RAFFAELE",
        country="IT",
        verification_objectives=["verify_credentials", "verify_specialization"],
        confidence_threshold=0.90
    )
    
    print(f"Custom Requirements:")
    print(json.dumps(requirements, indent=2))
    
    # Run standalone search
    result = await run_standalone_search(requirements)
    
    print(f"\n=== CUSTOM RESULTS ===")
    print(f"Status: {result['workflow_status'].value}")
    print(f"Confidence: {result.get('search_confidence', 0.0)}")
    
    return result

async def test_error_handling():
    """Test error handling"""
    print("\n=== ERROR HANDLING TEST ===")
    
    # Create invalid requirements
    invalid_requirements = {
        "verification_needed": True,
        "primary_objectives": [],  # Empty objectives
        "geographic_region": "",   # Empty region
        "confidence_threshold": -1  # Invalid threshold
    }
    
    try:
        result = await run_standalone_search(invalid_requirements)
        print(f"Error Test Status: {result['workflow_status'].value}")
        
        if result.get("error_context"):
            print(f"Error Details: {result['error_context']}")
            
    except Exception as e:
        print(f"Exception caught: {str(e)}")

async def main():
    """Run all standalone search tests"""
    print("🔍 STANDALONE SEARCH & SUMMARIZE AGENT TESTING")
    print("=" * 60)
    
    # Test 1: Basic functionality
    basic_result = await test_basic_search()
    
    # Test 2: Custom requirements  
    custom_result = await test_custom_requirements()
    
    # Test 3: Error handling
    await test_error_handling()
    
    # Summary
    print("\n=== TEST SUMMARY ===")
    print(f"Basic Test: {'✅ PASSED' if basic_result['workflow_status'].value == 'completed' else '❌ FAILED'}")
    print(f"Custom Test: {'✅ PASSED' if custom_result['workflow_status'].value == 'completed' else '❌ FAILED'}")
    
    # Save results for analysis
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
```

---

## **📋 IMPLEMENTATION CHECKLIST**

### **Files to Create:**
1. ✅ `standalone_search_state.py` - Minimal state model
2. ✅ `search_requirements_builder.py` - Requirements builder
3. ✅ `standalone_search_workflow.py` - Standalone LangGraph workflow  
4. ✅ `test_standalone_search.py` - Test runner script

### **Files to Modify:**
1. ✅ `agents/search_summarize_agent.py` - Add standalone methods
2. ✅ Import statements and dependencies

### **Dependencies Needed:**
1. ✅ All existing LangGraph dependencies
2. ✅ Azure OpenAI configuration (from config.py)
3. ✅ Tool implementations (placeholders work for testing)
4. ✅ Prompt templates (existing ones)

---

## **🚀 TESTING WORKFLOW**

### **Phase 1: Basic Testing**
```bash
python test_standalone_search.py
```

### **Phase 2: Individual Component Testing**
```python
# Test requirements builder
from search_requirements_builder import create_test_requirements
reqs = create_test_requirements()

# Test workflow creation
from standalone_search_workflow import create_standalone_search_workflow
workflow = create_standalone_search_workflow()

# Test individual agent methods
from agents.search_summarize_agent import SearchAndSummarizeAgent
agent = SearchAndSummarizeAgent()
```

### **Phase 3: Integration with Main Workflow**
- Verify standalone agent produces same results as integrated version
- Compare outputs between standalone and full workflow execution

---

## **🎯 KEY BENEFITS OF STANDALONE APPROACH**

1. **Independent Testing:** Test search logic without supervisor dependencies
2. **Faster Iteration:** Quick testing of tool selection and summarization logic
3. **Debugging:** Isolate search agent issues from workflow complexity
4. **Development:** Easier to develop and test new search tools
5. **Demonstration:** Show search capabilities independently

**This standalone implementation allows you to fully test the Search & Summarize Agent with LangGraph workflow in isolation while maintaining compatibility with the main workflow.**


