### Directory Structure

# onekey_vr_automation/
# ├── batch_processor.py          # Batch processing entry point
# ├── main.py                     # Workflow definition
# ├── config.py                   # Configuration
# ├── requirements.txt            # Dependencies
# ├── .env                       # Environment variables
# ├── agents/
# │   ├── __init__.py
# │   ├── supervisor_agent.py
# │   └── search_summarize_agent.py
# ├── utils/
# │   ├── __init__.py
# │   ├── state_models.py
# │   ├── routing_functions.py
# │   ├── search_strategy.py
# │   └── helpers.py
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



### batch_processor.py

# """
# Batch Processor for VR Records
# Handles date-based VR API calls and sequential processing
# """

# import asyncio
# import json
# import logging
# from datetime import datetime
# from typing import List, Dict, Any
# from main import process_single_vr_record

# # Configure logging
# logging.basicConfig(level=logging.INFO)
# logger = logging.getLogger(__name__)

# # Import your existing VR API and data class
# # from vr_api import get_vr_records_for_date_range, VRDataClass


# async def process_vr_batch(date_range: Dict[str, str]) -> List[Dict[str, Any]]:
#     """
#     Process a batch of VR records for a given date range
    
#     Args:
#         date_range: {"start_date": "2025-06-01", "end_date": "2025-06-02"}
    
#     Returns:
#         List of workflow results for each VR record
#     """
#     try:
#         # Step 1: Call VR API with date range
#         logger.info(f"Fetching VR records for date range: {date_range}")
#         # raw_vr_records = await get_vr_records_for_date_range(
#         #     date_range["start_date"], 
#         #     date_range["end_date"]
#         # )
        
#         # Step 2: Process with VR data class
#         # processed_vr_records = [VRDataClass(record).to_dict() for record in raw_vr_records]
#         # logger.info(f"Processing {len(processed_vr_records)} VR records")
        
#         # Step 3: Sequential processing through workflow
#         results = []
#         # for idx, vr_record in enumerate(processed_vr_records):
#         #     logger.info(f"Processing record {idx + 1}/{len(processed_vr_records)}")
#         #     
#         #     try:
#         #         result = await process_single_vr_record(vr_record)
#         #         results.append({
#         #             "vr_validation_id": vr_record["id"],
#         #             "status": "success",
#         #             "workflow_result": result
#         #         })
#         #     except Exception as e:
#         #         logger.error(f"Error processing VR ID {vr_record['validation']['id']}: {str(e)}")
#         #         results.append({
#         #             "vr_validation_id": vr_record["validation"]["id"],
#         #             "status": "error",
#         #             "error": str(e)
#         #         })
        
#         # Step 4: Save results
#         output_filename = f"vr_batch_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
#         with open(output_filename, 'w') as f:
#             json.dump(results, f, indent=2)
        
#         logger.info(f"Batch processing complete. Results saved to {output_filename}")
#         return results
        
#     except Exception as e:
#         logger.error(f"Batch processing error: {str(e)}")
#         raise


# async def process_daily_vr_records():
#     """
#     Process today's VR records
#     """
#     today = datetime.now().strftime("%Y-%m-%d")
#     date_range = {
#         "start_date": today,
#         "end_date": today
#     }
    
#     return await process_vr_batch(date_range)


# if __name__ == "__main__":
#     # Example: Process specific date range
#     date_range = {
#         "start_date": "2025-06-02",
#         "end_date": "2025-06-02"
#     }
    
#     asyncio.run(process_vr_batch(date_range))




"""
Simple File-Based Batch Processor for VR Records
Handles date-based VR API calls and sequential processing
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import List, Dict, Any
from main import process_single_vr_record

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Import your existing VR API and data class
# from vr_api import get_vr_records_for_date_range
# from vr_data_class import VRDataClass


async def process_vr_batch(date_range: Dict[str, str]) -> List[Dict[str, Any]]:
    """
    Process a batch of VR records with file-based pipeline
    
    Args:
        date_range: {"start_date": "2025-06-01", "end_date": "2025-06-02"}
    
    Returns:
        List of workflow results for each VR record
    """
    batch_id = f"{date_range['start_date']}_{date_range['end_date']}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    
    try:
        # Step 1: Call VR API and store raw data
        logger.info(f"Fetching VR records for date range: {date_range}")
        
        # TODO: Replace with your actual VR API call
        # raw_vr_records = await get_vr_records_for_date_range(
        #     date_range["start_date"], 
        #     date_range["end_date"]
        # )
        raw_vr_records = []  # Placeholder
        
        # Save raw data to JSON file
        raw_data_filename = f"raw_vr_data_{batch_id}.json"
        with open(raw_data_filename, 'w', encoding='utf-8') as f:
            json.dump(raw_vr_records, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Raw VR data saved to: {raw_data_filename} ({len(raw_vr_records)} records)")
        
        # Step 2: Process raw data with VR data class and store processed data
        logger.info(f"Processing {len(raw_vr_records)} raw VR records")
        
        processed_vr_records = []
        for raw_record in raw_vr_records:
            # TODO: Replace with your actual VRDataClass
            # processed_record = VRDataClass(raw_record).to_dict()
            processed_record = raw_record  # Placeholder
            processed_vr_records.append(processed_record)
        
        # Save processed data to JSON file
        processed_data_filename = f"processed_vr_data_{batch_id}.json"
        with open(processed_data_filename, 'w', encoding='utf-8') as f:
            json.dump(processed_vr_records, f, indent=2, ensure_ascii=False)
        
        logger.info(f"Processed VR data saved to: {processed_data_filename} ({len(processed_vr_records)} records)")
        
        # Step 3: Sequential processing through workflow
        logger.info(f"Starting workflow processing for {len(processed_vr_records)} records")
        
        results = []
        for idx, vr_record in enumerate(processed_vr_records):
            vr_id = vr_record.get("validation.id", f"unknown_{idx}")
            logger.info(f"Processing record {idx + 1}/{len(processed_vr_records)} - VR ID: {vr_id}")
            
            try:
                result = await process_single_vr_record(vr_record)
                results.append({
                    "vr_validation_id": vr_id,
                    "status": "success",
                    "workflow_result": result
                })
                logger.info(f"Workflow completed for VR ID: {vr_id}")
                
            except Exception as e:
                logger.error(f"Error processing VR ID {vr_id}: {str(e)}")
                results.append({
                    "vr_validation_id": vr_id,
                    "status": "error",
                    "error": str(e)
                })
        
        # Step 4: Save workflow results
        results_filename = f"vr_workflow_results_{batch_id}.json"
        with open(results_filename, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        success_count = len([r for r in results if r["status"] == "success"])
        error_count = len([r for r in results if r["status"] == "error"])
        
        logger.info(f"Batch processing complete. Results saved to: {results_filename}")
        logger.info(f"Summary: {success_count} successful, {error_count} failed")
        
        return results
        
    except Exception as e:
        logger.error(f"Batch processing error: {str(e)}")
        raise


async def process_daily_vr_records():
    """
    Process today's VR records
    """
    today = datetime.now().strftime("%Y-%m-%d")
    date_range = {
        "start_date": today,
        "end_date": today
    }
    
    return await process_vr_batch(date_range)


if __name__ == "__main__":
    # Example: Process specific date range
    date_range = {
        "start_date": "2025-06-02",
        "end_date": "2025-06-02"
    }
    
    asyncio.run(process_vr_batch(date_range))
    
    
    

#### main.py


"""
OneKey VR Automation - Main Entry Point
"""

import asyncio
import logging
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import AIMessage
from utils.state_models import AgentState, WorkflowStatus
from utils.routing_functions import (
    supervisor_routing_decision, 
    post_delegation_routing,
    search_completion_routing,
    determine_next_step_after_okdb_search
)
from agents.supervisor_agent import SupervisorAgent
from agents.search_summarize_agent import SearchAndSummarizeAgent

# Configure logging
logger = logging.getLogger(__name__)


async def handle_error(state: AgentState) -> AgentState:
    """Global error handler for workflow"""
    error_context = state.get("error_context", {})
    error_msg = f"Error in {error_context.get('stage', 'unknown')}: {error_context.get('error', 'Unknown error')}"
    
    logger.error(error_msg)
    state["messages"].append(AIMessage(content=error_msg))
    state["workflow_status"] = WorkflowStatus.ERROR
    
    return state


# def create_agent_vr_workflow():
#     """
#     Create LangGraph workflow with two agents:
#     1. Supervisor Agent: OK DB Search → Analysis → Decision making
#     2. Search & Summarize Agent: Multi-source search → Summarization
#     """
    
#     # Initialize the two agents
#     supervisor = SupervisorAgent()
#     search_summarize = SearchAndSummarizeAgent()
    
#     # Define workflow graph
#     workflow = StateGraph(AgentState)
    
#     # Supervisor Agent nodes
#     workflow.add_node("execute_okdb_search", supervisor.execute_okdb_search)
#     workflow.add_node("analyze_comparison", supervisor.analyze_vr_vs_okdb)
#     workflow.add_node("delegate_search", supervisor.delegate_search_task)
#     workflow.add_node("make_dbo_decision", supervisor.make_dbo_decision)
    
#     # Search & Summarize Agent nodes
#     workflow.add_node("select_tools", search_summarize.select_search_tools)
#     workflow.add_node("execute_search", search_summarize.execute_search_tools)
#     workflow.add_node("summarize_results", search_summarize.intelligent_summarization)
    
#     # Error handler node
#     workflow.add_node("handle_error", handle_error)
    
#     workflow.add_node("compare_okdb_results", supervisor.compare_okdb_results)
    
#     # Workflow edges - Start directly with OK DB search
#     workflow.add_edge(START, "execute_okdb_search")
    
#     # Supervisor workflow
#     # workflow.add_conditional_edges(
#     #     "execute_okdb_search", 
#     #     supervisor_routing_decision,
#     #     {
#     #         "analyze_comparison": "analyze_comparison",
#     #         "handle_error": "handle_error"
#     #     }
#     # )
    
#     workflow.add_conditional_edges(
#         "execute_okdb_search",
#         determine_next_step_after_search,  # NEW ROUTING FUNCTION
#         {
#             "compare_okdb_results": "compare_okdb_results",
#             "analyze_comparison": "analyze_comparison"
#         }
#     )
    
#     workflow.add_edge("compare_okdb_results", "analyze_comparison")
    
#     workflow.add_conditional_edges(
#         "analyze_comparison",
#         supervisor_routing_decision,
#         {
#             "delegate_search": "delegate_search",
#             "make_dbo_decision": "make_dbo_decision",  # Skip search if not needed
#             "handle_error": "handle_error"
#         }
#     )
    
#     # CORRECTED: Use custom routing after delegation
#     workflow.add_conditional_edges(
#         "delegate_search",
#         post_delegation_routing,  # Custom routing function
#         {
#             "select_tools": "select_tools",
#             "make_dbo_decision": "make_dbo_decision"
#         }
#     )
    
#     # Search & Summarize Agent workflow
#     workflow.add_edge("select_tools", "execute_search")
#     workflow.add_edge("execute_search", "summarize_results")
    
#     # Hand back to Supervisor for final decision
#     workflow.add_conditional_edges(
#         "summarize_results",
#         search_completion_routing,  # Custom routing for search completion
#         {
#             "make_dbo_decision": "make_dbo_decision",
#             "handle_error": "handle_error"
#         }
#     )
    
#     # Final decision edge
#     workflow.add_conditional_edges(
#         "make_dbo_decision",
#         lambda x: "complete" if x.get("workflow_status") == WorkflowStatus.DBO_DECISION_READY else "handle_error",
#         {
#             "complete": END,
#             "handle_error": "handle_error"
#         }
#     )
    
#     # Error handler to END
#     workflow.add_edge("handle_error", END)
    
#     # Compile workflow
#     return workflow.compile()


def create_agent_vr_workflow():
    """
    Create LangGraph workflow with enhanced dual search support:
    1. Supervisor Agent: OK DB Search → OK DB Comparison → VR Analysis → Decision making
    2. Search & Summarize Agent: Multi-source search → Summarization
    """
    
    # Initialize the two agents
    supervisor = SupervisorAgent()
    search_summarize = SearchAndSummarizeAgent()
    
    # Define workflow graph
    workflow = StateGraph(AgentState)
    
    # Supervisor Agent nodes
    workflow.add_node("execute_okdb_search", supervisor.execute_okdb_search)
    workflow.add_node("compare_okdb_results", supervisor.compare_okdb_results)  # NEW NODE
    workflow.add_node("analyze_comparison", supervisor.analyze_vr_vs_okdb)
    workflow.add_node("delegate_search", supervisor.delegate_search_task)
    workflow.add_node("make_dbo_decision", supervisor.make_dbo_decision)
    
    # Search & Summarize Agent nodes
    workflow.add_node("select_tools", search_summarize.select_search_tools)
    workflow.add_node("execute_search", search_summarize.execute_search_tools)
    workflow.add_node("summarize_results", search_summarize.intelligent_summarization)
    
    # Error handler node
    workflow.add_node("handle_error", handle_error)
    
    # WORKFLOW EDGES
    
    # Start with OK DB search
    workflow.add_edge(START, "execute_okdb_search")
    
    # After OK DB search - determine if comparison is needed
    workflow.add_conditional_edges(
        "execute_okdb_search",
        determine_next_step_after_okdb_search,  # NEW ROUTING FUNCTION
        {
            "compare_okdb_results": "compare_okdb_results",  # Dual search path
            "analyze_comparison": "analyze_comparison",       # Single search path
            "handle_error": "handle_error"
        }
    )
    
    # After OK DB comparison - proceed to VR analysis
    workflow.add_conditional_edges(
        "compare_okdb_results",
        supervisor_routing_decision,
        {
            "analyze_comparison": "analyze_comparison",
            "handle_error": "handle_error"
        }
    )
    
    # After VR analysis - determine if external search is needed
    workflow.add_conditional_edges(
        "analyze_comparison",
        supervisor_routing_decision,
        {
            "delegate_search": "delegate_search",
            "make_dbo_decision": "make_dbo_decision",  # Skip search if not needed
            "handle_error": "handle_error"
        }
    )
    
    # After search delegation - route to search agent or skip
    workflow.add_conditional_edges(
        "delegate_search",
        post_delegation_routing,  # Existing routing function
        {
            "select_tools": "select_tools",
            "make_dbo_decision": "make_dbo_decision"
        }
    )
    
    # Search & Summarize Agent workflow (unchanged)
    workflow.add_edge("select_tools", "execute_search")
    workflow.add_edge("execute_search", "summarize_results")
    
    # Hand back to Supervisor for final decision
    workflow.add_conditional_edges(
        "summarize_results",
        search_completion_routing,  # Existing routing function
        {
            "make_dbo_decision": "make_dbo_decision",
            "handle_error": "handle_error"
        }
    )
    
    # Final decision edge
    workflow.add_conditional_edges(
        "make_dbo_decision",
        lambda x: "complete" if x.get("workflow_status") == WorkflowStatus.DBO_DECISION_READY else "handle_error",
        {
            "complete": END,
            "handle_error": "handle_error"
        }
    )
    
    # Error handler to END
    workflow.add_edge("handle_error", END)
    
    # Compile workflow
    return workflow.compile()



async def process_single_vr_record(vr_record: dict) -> dict:
    """
    Process a single VR record through the workflow
    
    Args:
        vr_record: Processed VR record from VR data class
        
    Returns:
        Workflow result including DBO decision
    """
    logger.info(f"Processing VR ID: {vr_record.get('id')}")
    
    # Create workflow
    app = create_agent_vr_workflow()

    # Initial state with VR record
    initial_state = AgentState(
        vr_record=vr_record,
        vr_entity_type=vr_record.get("validation.entityTypeIco"),
        okdb_primary_results=None,   
        okdb_secondary_results=None,  
        okdb_comparison_analysis=None, 
        search_strategy=None,          
        okdb_api_response=None,
        comparison_analysis=None,
        record_status="",
        search_requirements=None,
        dbo_action_decision=None,
        selected_tools=[],
        execution_order=[],
        search_results=[],
        intelligent_summary=None,
        search_confidence=0.0,
        workflow_status=WorkflowStatus.INITIATED,
        current_agent="supervisor",
        error_context=None,
        messages=[]
    )
    
    # Execute workflow
    result = await app.ainvoke(initial_state)
    
    logger.info(f"Completed VR ID: {vr_record.get('validation', {}).get('id')} - Status: {result['workflow_status'].value}")
    
    return result


# For testing single record
async def main():
    """
    Example usage - process single VR record
    """
    # Example VR record (already processed by VR data class)
    vr_sample_1 = {
        "refAreaEid": "RAR_ITALY",
        "id": 1019000131693245,
        "customerId": 2436,
        "externalId": "47064409",
        "customerRequestEid": "VVAIT797937-NA-NA-CGD",
        "vrTypeCode": "VMR",
        "countryCode": "IT",
        "entityTypeIco": "ENT_ACTIVITY",
        "integrationDate": "2025-06-02T18:06:29Z",
        "requestDate": "2025-05-30T12:00:00Z",
        "requestComment": "Claudia Gambini kol426 Presso l'ospedale di Velletri di Via Orti Ginnetti fa due giorni di visite diabetologiche e un giorno di diabete gestazionale-DCR000266334",
        "statusIco": "VAS_NOT_PROCESSED",
        "isForced": False,
        "businessStatusCode": "C",
        "statusDate": "2025-06-02T18:06:29Z",
        "requesterLastName": "dcr_integration_user",
        "slaRemainingNumDays": 1,
        "slaDate": "2025-06-03",
        "slaNumDays": 5,
        "individualWithoutActivityAccepted": False,
        # Individual fields
        "individualOriginKeyEid": "WITO43596247",
        "firstName": "PAOLO",
        "lastName": "CORVISIERI",
        "specialityCode1": "18",
        "specialityCode2": "29",
        # Workplace fields
        "workplaceOriginKeyEid": "WITO62731655",
        "workplaceUsualName": "DISTRETTO SANITARIO FIUMICINO",
        # Address fields
        "country": "IT",
        "city": "VELLETRI",
        "postalCity": "VELLETRI"
    }
    
    result = await process_single_vr_record(vr_sample_1)
    
    print("\n=== WORKFLOW RESULTS ===")
    print(f"Workflow Status: {result['workflow_status'].value}")
    print(f"Record Status: {result.get('record_status', 'N/A')}")
    
    if result.get("dbo_action_decision"):
        print(f"\nDBO Decision:")
        print(f"  Action: {result['dbo_action_decision'].get('overall_recommendation')}")
        print(f"  Confidence: {result['dbo_action_decision'].get('decision_confidence')}")


if __name__ == "__main__":
    asyncio.run(main())






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
    
    # Logging
    LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO")
    
    
    
    
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
    OKDB_SEARCHED = "okdb_searched"
    OKDB_COMPARISON_COMPLETED = "okdb_comparison_completed"  # NEW
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
    # Input - VR record
    vr_record: Dict[str, Any]  # Complete processed VR record
    
    # Supervisor Agent state
    vr_entity_type: Optional[str]
    okdb_api_response: Optional[Dict[str, Any]]
    okdb_primary_results: Optional[Dict[str, Any]]    # originKeyEid search results
    okdb_secondary_results: Optional[List[Dict]]       # name-based search results
    okdb_comparison_analysis: Optional[Dict[str, Any]] # OK DB vs OK DB comparison
    search_strategy: Optional[Dict[str, Any]]          # Store search strategy
    comparison_analysis: Optional[Dict[str, Any]]
    record_status: str  # new_record, existing_match, multiple_matches, etc.
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
import logging  
logger = logging.getLogger(__name__) 


# def supervisor_routing_decision(state: AgentState) -> str:
#     """Route workflow based on Supervisor Agent's current status"""
#     workflow_status = state.get("workflow_status")
    
#     # Special handling for analysis completed
#     if workflow_status == WorkflowStatus.ANALYSIS_COMPLETED:
#         # Check if search is actually needed
#         search_requirements = state.get("search_requirements", {})
#         if not search_requirements.get("verification_needed", False):
#             return "make_dbo_decision"  # Skip search if not needed
#         return "delegate_search"
    
#     # Standard routing map - CORRECTED: Removed SEARCH_DELEGATED
#     routing_map = {
#         WorkflowStatus.INITIATED: "execute_okdb_search",
#         WorkflowStatus.OKDB_SEARCHED: "determine_next_step_after_search",  # NEW LOGIC
#         WorkflowStatus.OKDB_COMPARISON_COMPLETED: "analyze_comparison",
#         WorkflowStatus.ANALYSIS_COMPLETED: "delegate_search",
#         WorkflowStatus.SEARCH_COMPLETED: "make_dbo_decision",
#         WorkflowStatus.DBO_DECISION_READY: "complete",
#         WorkflowStatus.ERROR: "handle_error"
#     }
    
#     return routing_map.get(workflow_status, "handle_error")


def supervisor_routing_decision(state: AgentState) -> str:
    """Route workflow based on Supervisor Agent's current status"""
    workflow_status = state.get("workflow_status")
    
    # ADD THIS NEW CASE:
    if workflow_status == WorkflowStatus.OKDB_COMPARISON_COMPLETED:
        return "analyze_comparison"
    
    # Special handling for analysis completed
    if workflow_status == WorkflowStatus.ANALYSIS_COMPLETED:
        # Check if search is actually needed
        search_requirements = state.get("search_requirements", {})
        if not search_requirements.get("verification_needed", False):
            return "make_dbo_decision"  # Skip search if not needed
        return "delegate_search"
    
    # Standard routing map - KEEP EXISTING
    routing_map = {
        WorkflowStatus.INITIATED: "execute_okdb_search",
        WorkflowStatus.OKDB_SEARCHED: "analyze_comparison",
        WorkflowStatus.SEARCH_COMPLETED: "make_dbo_decision",
        WorkflowStatus.DBO_DECISION_READY: "complete",
        WorkflowStatus.ERROR: "handle_error"
    }
    
    return routing_map.get(workflow_status, "handle_error")


def determine_next_step_after_okdb_search(state: AgentState) -> str:
    """
    Route after OK DB search based on search strategy type
    """
    try:
        workflow_status = state.get("workflow_status")
        
        # Check for errors first
        if workflow_status == WorkflowStatus.ERROR:
            return "handle_error"
        
        # Check if OK DB search was successful
        if workflow_status != WorkflowStatus.OKDB_SEARCHED:
            logger.warning(f"Unexpected workflow status after OK DB search: {workflow_status}")
            return "handle_error"
        
        # Determine routing based on search strategy
        search_strategy = state.get("search_strategy", {})
        requires_comparison = search_strategy.get("requires_okdb_comparison", False)
        
        if requires_comparison:
            # Dual search requires OK DB comparison
            logger.debug("Routing to OK DB comparison for dual search results")
            return "compare_okdb_results"
        else:
            # Single search can proceed directly to VR analysis
            logger.debug("Routing to VR analysis for single search")
            return "analyze_comparison"
            
    except Exception as e:
        logger.error(f"Error in post-OK DB search routing: {str(e)}")
        return "handle_error"

def post_delegation_routing(state: AgentState) -> str:
    """Route after search delegation based on search requirements"""
    search_requirements = state.get("search_requirements", {})
    
    if search_requirements.get("verification_needed", False):
        return "select_tools"
    else:
        return "make_dbo_decision"


def search_completion_routing(state: AgentState) -> str:
    """Route after search completion"""
    if state.get("workflow_status") == WorkflowStatus.ERROR:
        return "handle_error"
    return "make_dbo_decision"


def search_agent_routing(state: AgentState) -> str:
    """Route Search & Summarize Agent workflow - kept for potential future use"""
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
        vr_data: FLAT VR record 
    """
    entity_type = vr_data.get("entityTypeIco", "")  # ✅ Direct access
    
    if entity_type == "ENT_ACTIVITY":
        # Check if individualOriginKeyEid is present
        if vr_data.get("individualOriginKeyEid"):
            return {
                "search_method": "originkey_individual",
                "search_params": {
                    "originKeyEid": vr_data["individualOriginKeyEid"],
                    "searchType": "individual_by_originkey"
                },
                "entity_type": "ENT_ACTIVITY"
            }
        else:
            # Fallback to name-based search
            return {
                "search_method": "name_based", 
                "search_params": {
                    "firstName": vr_data.get("firstName", ""),
                    "lastName": vr_data.get("lastName", ""),
                    "workplace": vr_data.get("workplaceUsualName", ""),
                    "country": vr_data.get("country", ""),
                    "searchType": "individual_by_name"
                },
                "entity_type": "ENT_ACTIVITY"
            }
    
    elif entity_type == "ENT_WORKPLACE":
        # Check if workplaceOriginKeyEid is present
        if vr_data.get("workplaceOriginKeyEid"):
            return {
                "search_method": "originkey_workplace",
                "search_params": {
                    "originKeyEid": vr_data["workplaceOriginKeyEid"],
                    "searchType": "workplace_by_originkey"
                },
                "entity_type": "ENT_WORKPLACE"
            }
        else:
            # Fallback to workplace name-based search
            return {
                "search_method": "workplace_based",
                "search_params": {
                    "workplaceName": vr_data.get("workplaceUsualName", ""),
                    "country": vr_data.get("country", ""),
                    "city": vr_data.get("city", ""),
                    "searchType": "workplace_by_name"
                },
                "entity_type": "ENT_WORKPLACE"
            }
    
    # Default fallback
    return {
        "search_method": "name_based",
        "search_params": {
            "firstName": vr_data.get("firstName", ""),
            "lastName": vr_data.get("lastName", ""),
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

import logging
from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage
from config import Config
from utils.helpers import safe_json_parse
from utils.state_models import AgentState, WorkflowStatus
from utils.search_strategy import determine_okdb_search_strategy
from prompts.supervisor_prompts import (
    VR_VS_OKDB_ANALYSIS_PROMPT,
    DBO_DECISION_PROMPT,
    OKDB_COMPARISON_PROMPT 
)

# Configure logging
logger = logging.getLogger(__name__)

# Import your existing OK DB API class
# from your_okdb_api import OKDBAPIClient


class SupervisorAgent:
    """
    Supervisor Agent: Handles OK DB API → Analysis → Decision Making
    Note: VR record is passed in, no VR API call needed
    """
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            api_key=Config.AZURE_OPENAI_API_KEY,
            azure_endpoint=Config.AZURE_OPENAI_ENDPOINT,
            azure_deployment=Config.AZURE_OPENAI_DEPLOYMENT_NAME,
            api_version=Config.AZURE_OPENAI_API_VERSION,
            temperature=0.1
        )
        
        # Initialize OK DB API client only
        # self.okdb_api_client = OKDBAPIClient()
    
    # async def execute_okdb_search(self, state: AgentState) -> AgentState:
    #     """Execute OK DB API search using VR record data"""
    #     vr_id = state["vr_record"].get("id")
    #     logger.info(f"Starting OK DB search for VR ID: {vr_id}")
        
    #     try:
    #         # Get search strategy using static function
    #         vr_record = state["vr_record"]
    #         search_strategy = determine_okdb_search_strategy(vr_record)
    #         search_params = search_strategy["search_params"]
            
    #         logger.debug(f"Search strategy: {search_strategy['search_method']}")
            
    #         # TODO: Use your OK DB API client here
    #         # okdb_response = await self.okdb_api_client.search_records(search_params)
            
    #         # TODO: Update state with OK DB response
    #         # state.update({
    #         #     "okdb_api_response": okdb_response,
    #         #     "workflow_status": WorkflowStatus.OKDB_SEARCHED
    #         # })
            
    #         state["messages"].append(AIMessage(
    #             content=f"OK DB search completed using {search_strategy['search_method']} method"
    #         ))
            
    #         logger.info(f"OK DB search completed for VR ID: {vr_id}")
            
    #     except Exception as e:
    #         logger.error(f"OK DB search error for VR ID {vr_id}: {str(e)}")
    #         state.update({
    #             "workflow_status": WorkflowStatus.ERROR,
    #             "error_context": {"stage": "okdb_search", "error": str(e)}
    #         })
        
    #     return state
    
    
    async def execute_okdb_search(self, state: AgentState) -> AgentState:
        """Execute OK DB API search using VR record data - supports dual search strategy"""
        vr_id = state["vr_record"].get("validation.id")
        logger.info(f"Starting OK DB search for VR ID: {vr_id}")
        
        try:
            # Get search strategy using static function
            vr_record = state["vr_record"]
            search_strategy = determine_okdb_search_strategy(vr_record)
            state["search_strategy"] = search_strategy
            
            search_method = search_strategy.get("search_method", "")
            logger.debug(f"VR ID {vr_id}: Search strategy determined - {search_method}")
            
            if search_method == "dual_search":
                logger.info(f"VR ID {vr_id}: Executing dual search strategy")
                
                # Execute primary search (originKeyEid)
                primary_search_config = search_strategy["primary_search"]
                primary_params = primary_search_config["params"]
                logger.debug(f"VR ID {vr_id}: Starting primary search - {primary_search_config['method']}")
                
                primary_results = None
                try:
                    # TODO: Replace with your actual OK DB API client call
                    # primary_results = await self.okdb_api_client.search_by_origin_key(primary_params)
                    
                    # Placeholder for primary search
                    primary_results = {
                        "search_type": "primary_originkey",
                        "origin_key": primary_params.get("originKeyEid"),
                        "method": primary_search_config["method"],
                        "results": []  # TODO: Actual API response
                    }
                    logger.info(f"VR ID {vr_id}: Primary search completed successfully")
                    
                except Exception as e:
                    logger.error(f"VR ID {vr_id}: Primary search failed - {str(e)}")
                    # Continue with secondary search even if primary fails
                    primary_results = None
                
                # Execute secondary search (name-based)
                secondary_search_config = search_strategy["secondary_search"]
                secondary_params = secondary_search_config["params"]
                logger.debug(f"VR ID {vr_id}: Starting secondary search - {secondary_search_config['method']}")
                
                secondary_results = []
                try:
                    # TODO: Replace with your actual OK DB API client call
                    # secondary_results = await self.okdb_api_client.search_by_criteria(secondary_params)
                    
                    # Placeholder for secondary search
                    secondary_results = [{
                        "search_type": "secondary_name_based",
                        "method": secondary_search_config["method"],
                        "search_criteria": secondary_params,
                        "results": []  # TODO: Actual API response - list of candidates
                    }]
                    logger.info(f"VR ID {vr_id}: Secondary search completed successfully")
                    
                except Exception as e:
                    logger.error(f"VR ID {vr_id}: Secondary search failed - {str(e)}")
                    secondary_results = []
                
                # Validate that we have at least one successful search result
                if primary_results is None and not secondary_results:
                    raise Exception("Both primary and secondary searches failed - no OK DB results available")
                
                # Update state with dual search results
                state.update({
                    "okdb_primary_results": primary_results,
                    "okdb_secondary_results": secondary_results,
                    "workflow_status": WorkflowStatus.OKDB_SEARCHED
                })
                
                state["messages"].append(AIMessage(
                    content=f"Dual OK DB search completed - Primary: {'Success' if primary_results else 'Failed'}, Secondary: {'Success' if secondary_results else 'Failed'}"
                ))
                
                logger.info(f"VR ID {vr_id}: Dual search strategy completed")
                
            else:
                # Single search strategy (existing logic)
                logger.info(f"VR ID {vr_id}: Executing single search strategy - {search_method}")
                
                search_params = search_strategy.get("search_params", {})
                if not search_params:
                    raise Exception("Search parameters are missing from search strategy")
                
                logger.debug(f"VR ID {vr_id}: Single search params - {search_params.get('searchType', 'unknown')}")
                
                try:
                    # TODO: Use your OK DB API client here
                    # single_results = await self.okdb_api_client.search_records(search_params)
                    
                    # Placeholder for single search
                    single_results = {
                        "search_type": "single_search",
                        "search_criteria": search_params,
                        "method": search_method,
                        "results": []  # TODO: Actual API response
                    }
                    
                    if single_results is None:
                        logger.warning(f"VR ID {vr_id}: Single search returned no results")
                        single_results = {}
                    
                    # Update state with single search results
                    state.update({
                        "okdb_api_response": single_results,
                        "workflow_status": WorkflowStatus.OKDB_SEARCHED
                    })
                    
                    state["messages"].append(AIMessage(
                        content=f"Single OK DB search completed using {search_method} method"
                    ))
                    
                    logger.info(f"VR ID {vr_id}: Single search strategy completed")
                    
                except Exception as e:
                    logger.error(f"VR ID {vr_id}: Single search execution failed - {str(e)}")
                    raise
            
            logger.info(f"OK DB search completed successfully for VR ID: {vr_id}")
            
        except Exception as e:
            logger.error(f"OK DB search error for VR ID {vr_id}: {str(e)}")
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {
                    "stage": "okdb_search", 
                    "error": str(e),
                    "vr_id": vr_id,
                    "search_method": search_strategy.get("search_method", "unknown") if 'search_strategy' in locals() else "unknown"
                }
            })
            
            state["messages"].append(AIMessage(
                content=f"OK DB search failed: {str(e)}"
            ))
        
        return state
    
    
    
    async def compare_okdb_results(self, state: AgentState) -> AgentState:
        """
        Compare OK DB originKeyEid results vs name-based results
        Determine if they represent the same person/workplace
        """
        vr_id = state["vr_record"].get("validation.id")
        logger.info(f"Starting OK DB results comparison for VR ID: {vr_id}")
        
        try:
            # Validate that we have dual search results to compare
            primary_results = state.get("okdb_primary_results")
            secondary_results = state.get("okdb_secondary_results", [])
            search_strategy = state.get("search_strategy", {})
            entity_type = search_strategy.get("entity_type", "UNKNOWN")
            
            logger.debug(f"VR ID {vr_id}: Comparing OK DB results for entity type: {entity_type}")
            
            # Validate input data
            if primary_results is None and not secondary_results:
                raise ValueError("No OK DB results available for comparison - both primary and secondary results are missing")
            
            if primary_results is None:
                logger.warning(f"VR ID {vr_id}: Primary search results missing, using best secondary result")
                
                # Select best secondary result when primary is unavailable
                final_okdb_result = {}
                if secondary_results:
                    first_search = secondary_results[0] if secondary_results else {}
                    results_list = first_search.get("results", [])
                    if results_list:
                        final_okdb_result = results_list[0]  # Take first candidate
                        logger.debug(f"VR ID {vr_id}: Selected best secondary result from {len(results_list)} candidates")
                    else:
                        final_okdb_result = first_search
                
                comparison_analysis = {
                    "same_entity": False,
                    "confidence_level": 0.5,
                    "comparison_summary": "Primary search failed - using best secondary result",
                    "primary_entity": None,
                    "best_secondary_match": final_okdb_result,
                    "recommended_result": final_okdb_result,
                    "multiple_candidates_found": len(secondary_results) > 1,
                    "requires_manual_review": True,
                    "comparison_method": "fallback_to_secondary"
                }
                
            elif not secondary_results:
                logger.warning(f"VR ID {vr_id}: Secondary search results missing, using primary result")
                
                # Use primary result when secondary is unavailable
                final_okdb_result = primary_results
                comparison_analysis = {
                    "same_entity": True,
                    "confidence_level": 0.9,
                    "comparison_summary": "Secondary search failed - using primary (originKeyEid) result",
                    "primary_entity": primary_results,
                    "best_secondary_match": None,
                    "recommended_result": primary_results,
                    "multiple_candidates_found": False,
                    "requires_manual_review": False,
                    "comparison_method": "primary_only"
                }
                
            else:
                # Both results available - perform LLM-powered comparison
                logger.info(f"VR ID {vr_id}: Performing LLM comparison between primary and secondary results")
                
                try:
                    # Execute LLM comparison
                    comparison_response = await self.llm.ainvoke(
                        OKDB_COMPARISON_PROMPT.format(
                            primary_results=primary_results,
                            secondary_results=secondary_results,
                            entity_type=entity_type,
                            vr_id=vr_id
                        ).messages
                    )
                    
                    # Parse comparison results
                    comparison_analysis = safe_json_parse(comparison_response.content)
                    
                    # Validate comparison analysis structure
                    required_fields = ["same_entity", "confidence_level", "recommended_result"]
                    missing_fields = [field for field in required_fields if field not in comparison_analysis]
                    if missing_fields:
                        logger.warning(f"VR ID {vr_id}: LLM comparison missing fields: {missing_fields}")
                        # Set default values for missing fields
                        comparison_analysis.setdefault("same_entity", False)
                        comparison_analysis.setdefault("confidence_level", 0.5)
                        comparison_analysis.setdefault("requires_manual_review", True)
                    
                    # Determine final result based on comparison
                    if comparison_analysis.get("same_entity", False):
                        # Use primary (originKeyEid) result as authoritative when entities match
                        final_okdb_result = primary_results
                        logger.info(f"VR ID {vr_id}: Entities match - using primary (originKeyEid) result")
                    else:
                        # Use recommended result from comparison or best secondary match
                        recommended = comparison_analysis.get("recommended_result")
                        if recommended:
                            final_okdb_result = recommended
                            logger.info(f"VR ID {vr_id}: Entities differ - using LLM recommended result")
                        else:
                            # Fallback to primary result if no recommendation
                            final_okdb_result = primary_results
                            logger.warning(f"VR ID {vr_id}: No recommendation from LLM - falling back to primary result")
                    
                    comparison_analysis["comparison_method"] = "llm_powered"
                    
                    confidence = comparison_analysis.get("confidence_level", 0.5)
                    same_entity = comparison_analysis.get("same_entity", False)
                    logger.info(f"VR ID {vr_id}: LLM comparison completed - Same entity: {same_entity}, Confidence: {confidence}")
                    
                except Exception as e:
                    logger.error(f"VR ID {vr_id}: LLM comparison failed - {str(e)}")
                    
                    # Fallback to rule-based comparison when LLM fails
                    logger.info(f"VR ID {vr_id}: Performing rule-based comparison as LLM fallback")
                    
                    # Simple rule-based logic - conservative approach
                    final_okdb_result = primary_results  # Default to primary when LLM fails
                    
                    comparison_analysis = {
                        "same_entity": False,  # Conservative approach
                        "confidence_level": 0.6,
                        "comparison_summary": "Rule-based comparison performed (LLM fallback)",
                        "primary_entity": primary_results,
                        "best_secondary_match": secondary_results[0] if secondary_results else None,
                        "recommended_result": primary_results,
                        "multiple_candidates_found": len(secondary_results) > 1,
                        "requires_manual_review": True,
                        "comparison_method": "rule_based_fallback",
                        "llm_error": str(e)
                    }
                    
                    logger.info(f"VR ID {vr_id}: Rule-based comparison completed - defaulting to primary result")
            
            # Validate final result
            if final_okdb_result is None:
                raise ValueError("Failed to determine final OK DB result from comparison")
            
            # Update state with comparison results
            state.update({
                "okdb_comparison_analysis": comparison_analysis,
                "okdb_api_response": final_okdb_result,  # Set final result for VR comparison
                "workflow_status": WorkflowStatus.OKDB_COMPARISON_COMPLETED
            })
            
            # Add informative message
            comparison_summary = comparison_analysis.get("comparison_summary", "Comparison completed")
            requires_review = comparison_analysis.get("requires_manual_review", False)
            
            state["messages"].append(AIMessage(
                content=f"OK DB comparison completed: {comparison_summary}" + 
                    (f" - Manual review recommended" if requires_review else "")
            ))
            
            logger.info(f"OK DB results comparison completed for VR ID: {vr_id}")
            
        except Exception as e:
            logger.error(f"OK DB comparison error for VR ID {vr_id}: {str(e)}")
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {
                    "stage": "okdb_comparison", 
                    "error": str(e),
                    "vr_id": vr_id,
                    "entity_type": state.get("search_strategy", {}).get("entity_type", "unknown")
                }
            })
            
            state["messages"].append(AIMessage(
                content=f"OK DB comparison failed: {str(e)}"
            ))
        
        return state
        
    
    
    async def analyze_vr_vs_okdb(self, state: AgentState) -> AgentState:
        """LLM-powered analysis comparing VR data vs OK DB results"""
        vr_id = state["vr_record"].get("validation.id")
        logger.info(f"Analyzing VR vs OK DB for VR ID: {vr_id}")
        
        try:
            # Execute LLM analysis
            analysis_response = await self.llm.ainvoke(
                VR_VS_OKDB_ANALYSIS_PROMPT.format(
                    vr_data=state["vr_record"],
                    okdb_data=state["okdb_api_response"]
                ).messages
            )
            
            # Parse analysis results
            parsed_analysis = safe_json_parse(analysis_response.content)
            
            state.update({
                "comparison_analysis": parsed_analysis,
                "record_status": parsed_analysis.get("record_status", ""),
                "search_requirements": parsed_analysis.get("search_requirements", {}),
                "workflow_status": WorkflowStatus.ANALYSIS_COMPLETED
            })
            
            state["messages"].append(AIMessage(
                content=f"Analysis complete: {parsed_analysis.get('analysis_summary', 'Analysis completed')}"
            ))
            
            logger.info(f"Analysis completed for VR ID {vr_id}: {parsed_analysis.get('record_status')}")
            
        except Exception as e:
            logger.error(f"Analysis error for VR ID {vr_id}: {str(e)}")
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "analysis", "error": str(e)}
            })
        
        return state
    
    async def delegate_search_task(self, state: AgentState) -> AgentState:
        """Prepare search delegation for Search & Summarize Agent"""
        vr_id = state["vr_record"].get("validation.id")
        
        try:
            search_requirements = state["search_requirements"]
            
            if search_requirements.get("verification_needed", False):
                state.update({
                    "workflow_status": WorkflowStatus.SEARCH_DELEGATED,
                    "current_agent": "search_summarize"
                })
                
                objectives = search_requirements.get("primary_objectives", [])
                delegation_message = f"Delegating search for: {', '.join(objectives)}"
                state["messages"].append(AIMessage(content=delegation_message))
                
                logger.info(f"Search delegated for VR ID {vr_id}")
            else:
                # No search needed, proceed directly to DBO decision
                state.update({
                    "workflow_status": WorkflowStatus.SEARCH_COMPLETED,
                    "search_confidence": 1.0,
                    "intelligent_summary": {"verification_not_required": True}
                })
                
                state["messages"].append(AIMessage(content="No external verification needed"))
                logger.info(f"No search needed for VR ID {vr_id}, proceeding to DBO decision")
        
        except Exception as e:
            logger.error(f"Delegation error for VR ID {vr_id}: {str(e)}")
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "delegation", "error": str(e)}
            })
        
        return state
    
    async def make_dbo_decision(self, state: AgentState) -> AgentState:
        """LLM makes final DBO action decision based on all available data"""
        vr_id = state["vr_record"].get("validation.id")
        logger.info(f"Making DBO decision for VR ID: {vr_id}")
        
        try:
            # Enhanced to include raw VR and OK DB data
            decision_response = await self.llm.ainvoke(
                DBO_DECISION_PROMPT.format(
                    vr_record=state["vr_record"],
                    okdb_data=state.get("okdb_api_response", {}),
                    comparison_analysis=state.get("comparison_analysis", {}),
                    search_summary=state.get("intelligent_summary", {}),
                    search_confidence=state.get("search_confidence", 0.0)
                ).messages
            )
            
            # Parse decision
            dbo_decision = safe_json_parse(decision_response.content)
            
            state.update({
                "dbo_action_decision": dbo_decision,
                "workflow_status": WorkflowStatus.DBO_DECISION_READY
            })
            
            state["messages"].append(AIMessage(
                content=f"DBO decision: {dbo_decision.get('overall_recommendation', 'Decision made')}"
            ))
            
            logger.info(f"DBO decision made for VR ID {vr_id}: {dbo_decision.get('overall_recommendation')}")
            
        except Exception as e:
            logger.error(f"DBO decision error for VR ID {vr_id}: {str(e)}")
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "dbo_decision", "error": str(e)}
            })
        
        return state
    
    
    
    
    

### agents/search_summarize_agent.py



"""
Search & Summarize Agent - Multi-source Search & Intelligent Summarization
"""

import logging
from typing import Dict, Any
from langchain_openai import AzureChatOpenAI
from langchain_core.messages import AIMessage
from config import Config
from utils.helpers import safe_json_parse
from utils.state_models import AgentState, WorkflowStatus
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
    Search & Summarize Agent: Multi-source searching + AI summarization
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
    
    async def select_search_tools(self, state: AgentState) -> AgentState:
        """LLM selects appropriate tools based on search requirements"""
        vr_id = state["vr_record"].get("validation.id")
        
        try:
            search_requirements = state.get("search_requirements", {})
            
            # CORRECTED: Handle case where no search is needed
            if not search_requirements.get("verification_needed", False):
                state.update({
                    "selected_tools": [],
                    "execution_order": [],
                    "search_results": [],
                    "intelligent_summary": {"verification_not_required": True},
                    "search_confidence": 1.0,
                    "workflow_status": WorkflowStatus.SEARCH_COMPLETED,
                    "current_agent": "supervisor"
                })
                
                state["messages"].append(AIMessage(
                    content="No search required - proceeding to DBO decision"
                ))
                logger.info(f"No search required for VR ID {vr_id}")
                return state
            
            # Execute tool selection LLM
            logger.info(f"Selecting tools for VR ID {vr_id}")
            
            selection_response = await self.llm.ainvoke(
                TOOL_SELECTION_PROMPT.format(
                    search_requirements=search_requirements
                ).messages
            )
            
            # Parse selection
            tool_selection = safe_json_parse(selection_response.content)
            
            state.update({
                "selected_tools": tool_selection.get("selected_tools", []),
                "execution_order": tool_selection.get("execution_order", [])
            })
            
            state["messages"].append(AIMessage(
                content=f"Selected tools: {', '.join(tool_selection.get('selected_tools', []))}"
            ))
            
            logger.info(f"Tools selected for VR ID {vr_id}: {tool_selection.get('selected_tools', [])}")
            
        except Exception as e:
            logger.error(f"Tool selection error for VR ID {vr_id}: {str(e)}")
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "tool_selection", "error": str(e)}
            })
        
        return state
    
    async def execute_search_tools(self, state: AgentState) -> AgentState:
        """Execute selected tools in planned order"""
        vr_id = state["vr_record"].get("validation.id")
        
        try:
            execution_order = state["execution_order"]
            search_requirements = state["search_requirements"]
            
            logger.info(f"Executing {len(execution_order)} tools for VR ID {vr_id}")
            
            for tool_name in execution_order:
                if tool_name in self.search_tools:
                    tool_instance = self.search_tools[tool_name]
                    
                    # TODO: Execute tool search
                    # search_result = await tool_instance.search(search_requirements)
                    # state["search_results"].append(search_result)
                    
                    state["messages"].append(AIMessage(
                        content=f"Executed search with {tool_name}"
                    ))
                    
                    logger.debug(f"Executed {tool_name} for VR ID {vr_id}")
            
            logger.info(f"All tools executed for VR ID {vr_id}")
            
        except Exception as e:
            logger.error(f"Search execution error for VR ID {vr_id}: {str(e)}")
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "search_execution", "error": str(e)}
            })
        
        return state
    
    async def intelligent_summarization(self, state: AgentState) -> AgentState:
        """LLM creates intelligent summary of all search results"""
        vr_id = state["vr_record"].get("validation.id")
        logger.info(f"Creating summary for VR ID {vr_id}")
        
        try:
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
                "workflow_status": WorkflowStatus.SEARCH_COMPLETED,
                "current_agent": "supervisor"
            })
            
            confidence = summary_data.get("overall_assessment", {}).get("confidence_level", 0.0)
            state["messages"].append(AIMessage(
                content=f"Search summary complete. Confidence: {confidence}"
            ))
            
            logger.info(f"Summary created for VR ID {vr_id} with confidence: {confidence}")
            
        except Exception as e:
            logger.error(f"Summarization error for VR ID {vr_id}: {str(e)}")
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
     You are analyzing VR validation requests against OneKey Database records.

    IMPORTANT: VR data is provided as a FLAT JSON structure where:
    - Individual fields: firstName, lastName, individualOriginKeyEid, specialityCode1, specialityCode2
    - Workplace fields: workplaceUsualName, workplaceOriginKeyEid
    - Address fields: country, city, postalCity
    - Validation fields: id, entityTypeIco, businessStatusCode, statusIco

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
    "record_status": "new_record/existing_match/existing_mismatch/multiple_matches/multiple_workplace_associations/ambiguous",
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
        1. Raw VR validation request data
        2. Raw OK DB search results
        3. Comparison analysis between VR and OK DB
        4. External verification results (if performed)
        5. Overall confidence assessments

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
        - Multiple workplace scenarios
        - Entity type (ENT_ACTIVITY vs ENT_WORKPLACE)

        For Each Decision Provide:
        - Specific field updates required
        - Justification with evidence
        - Confidence level (0-1)
        - Risk assessment
        - Manual review flag if needed

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
        VR Record (FLAT JSON): {vr_record}
        OK DB Data: {okdb_data}
        Comparison Analysis: {comparison_analysis}
        Search Summary: {search_summary}
        Search Confidence: {search_confidence}

        Make final DBO action decision considering all data.
        """)
        ])

OKDB_COMPARISON_PROMPT = ChatPromptTemplate.from_messages([
    ("system", """You are comparing OK DB search results to determine if they represent the same entity.

Primary Results: Results from originKeyEid search (most authoritative)
Secondary Results: Results from name/workplace search (may contain multiple candidates)

Your Tasks:
1. Compare primary result with secondary results
2. Determine if they represent the same person/workplace
3. If secondary results have multiple candidates, identify the best match
4. Assess confidence level

For Individuals (ENT_ACTIVITY):
- Compare: firstName, lastName, workplace, specialties, address
- Same person if: name matches + workplace context aligns

For Workplaces (ENT_WORKPLACE):
- Compare: workplace name, address, type, activity codes
- Same workplace if: name similarity + address proximity

Respond with JSON:
{{
    "same_entity": true/false,
    "confidence_level": 0.XX,
    "comparison_summary": "Brief explanation",
    "primary_entity": {{"key details from originKeyEid result"}},
    "best_secondary_match": {{"best match from name search if different"}},
    "recommended_result": {{"which result to use for VR comparison"}},
    "multiple_candidates_found": true/false,
    "requires_manual_review": true/false
}}"""),
    ("human", """
Primary Results (originKeyEid search): {primary_results}
Secondary Results (name-based search): {secondary_results}
Entity Type: {entity_type}

Compare these OK DB results and determine if they represent the same entity.
""")
])



### prompts/search_summarize_prompts.py

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
from .routing_functions import (
    supervisor_routing_decision, 
    search_agent_routing,
    post_delegation_routing,
    search_completion_routing
)
from .search_strategy import determine_okdb_search_strategy
from .helpers import safe_json_parse

__all__ = [
    'AgentState',
    'WorkflowStatus',
    'EntityType',
    'supervisor_routing_decision',
    'search_agent_routing',
    'post_delegation_routing',
    'search_completion_routing',
    'determine_okdb_search_strategy',
    'safe_json_parse'
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

from .supervisor_prompts import VR_VS_OKDB_ANALYSIS_PROMPT, DBO_DECISION_PROMPT, OKDB_COMPARISON_PROMPT
from .search_summarize_prompts import TOOL_SELECTION_PROMPT, SUMMARIZATION_PROMPT

__all__ = [
    'VR_VS_OKDB_ANALYSIS_PROMPT',
    'DBO_DECISION_PROMPT',
    'TOOL_SELECTION_PROMPT',
    'SUMMARIZATION_PROMPT'
]




### tools/init.py

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

# API Endpoints (if needed)
VR_API_ENDPOINT=https://your-vr-api.com
OKDB_API_ENDPOINT=https://your-okdb-api.com




### requirements.txt

langgraph>=0.0.26
langchain-core>=0.1.0
langchain-openai>=0.0.5
python-dotenv>=1.0.0
pydantic>=2.0.0
aiohttp>=3.8.0
tenacity>=8.2.0



