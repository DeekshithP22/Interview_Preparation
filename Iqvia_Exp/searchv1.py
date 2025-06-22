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



































































### test_search_agent.py

"""
Test script for standalone Search & Summarize Agent
Updated to include VR data for summarization
"""

import asyncio
import json
from main import process_search_request

async def test_basic_search():
    """Test basic search functionality with VR data"""
    
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
    
    # VR data for Italian case
    italian_vr_data = {
        "validation.refAreaEid": "RAR_ITALY",
        "validation.id": 1019001316927770,
        "validation.customerId": 7433,
        "validation.externalId": "47064408",
        "validation.customerRequestEid": "1-ALD8GCL/1-ALD8GE2",
        "validation.vrTypeCode": "VMR",
        "validation.countryCode": "IT",
        "validation.entityTypeIco": "ENT_ACTIVITY",
        "validation.integrationDate": "2025-06-02T07:16:43Z",
        "validation.requestDate": "2025-06-02T07:16:43Z",
        "validation.requestComment": "Automatically Created",
        "validation.statusIco": "VAS_NOT_PROCESSED",
        "validation.isForced": False,
        "validation.businessStatusCode": "C",
        "validation.statusDate": "2025-06-02T07:16:43Z",
        "validation.requesterId": "97433",
        "validation.requesterLastName": "fathimathsireen.ma@iqvia.com",
        "validation.slaRemainingNumDays": 4,
        "validation.slaDate": "2025-06-12",
        "validation.slaNumDays": 8,
        "validation.slaWorkInstructionsFileUrl": "nullVMR_WORK_INST_01",
        "validation.slaWorkInstructionsSummaryCode": "WORK_INSTRUCTION_DEFAULT",
        "validation.individualWithoutActivityAccepted": False,
        "individual.firstName": "Marcello",
        "individual.lastName": "Marchetti",
        "workplace.usualName": "Fondazione IRCCS Istituto Neurologico Carlo Besta",
        "address.country": "IT",
        "address.city": "Milano",
        "address.postalCity": "Milano",
        "matchingCandidatesKeys": [
            "WIT10546253201",
            "WIT10546253202"
        ]
    }
    
    print("=== Testing Italian Medical Professional ===")
    result = await process_search_request(italian_search, italian_vr_data)
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
    
    # VR data for French case
    french_vr_data = {
        "validation.refAreaEid": "RAR_FRANCE",
        "validation.id": 1019001316927771,
        "validation.customerId": 7434,
        "validation.externalId": "47064409",
        "validation.customerRequestEid": "1-FR_HOSPITAL/1-FR_DOC",
        "validation.vrTypeCode": "VMR",
        "validation.countryCode": "FR",
        "validation.entityTypeIco": "ENT_ACTIVITY",
        "validation.integrationDate": "2025-06-02T08:20:15Z",
        "validation.requestDate": "2025-06-02T08:20:15Z",
        "validation.requestComment": "French medical professional verification",
        "validation.statusIco": "VAS_NOT_PROCESSED",
        "validation.isForced": False,
        "validation.businessStatusCode": "C",
        "validation.statusDate": "2025-06-02T08:20:15Z",
        "validation.requesterId": "97434",
        "validation.requesterLastName": "marie.verification@iqvia.com",
        "validation.slaRemainingNumDays": 5,
        "validation.slaDate": "2025-06-15",
        "validation.slaNumDays": 10,
        "validation.individualWithoutActivityAccepted": False,
        "individual.firstName": "Marie",
        "individual.lastName": "Dubois",
        "workplace.usualName": "Hopital Saint-Antoine",
        "address.country": "FR",
        "address.city": "Paris",
        "address.postalCity": "Paris",
        "matchingCandidatesKeys": [
            "WITFR10546253301",
            "WITFR10546253302"
        ]
    }
    
    print("=== Testing French Healthcare Professional ===")
    result = await process_search_request(french_search, french_vr_data)
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
    result = await process_search_request(no_verification, None)
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
        result = await process_search_request(invalid_search, None)
        print(f"Status: {result['workflow_status'].value}")
        print(f"Error: {result.get('error_context', {})}")
    except Exception as e:
        print(f"Exception caught: {str(e)}")
    print()

async def test_with_vr_comparison():
    """Test VR data comparison in summarization"""
    
    # Test with VR data that has potential mismatches
    search_requirements = {
        "verification_needed": True,
        "primary_objectives": [
            "verify_employment_status",
            "confirm_medical_license",
            "validate_contact_information"
        ],
        "geographic_region": "IT",
        "confidence_threshold": 0.85,
        "entity_details": {
            "firstName": "Marcello",
            "lastName": "Marchetti",
            "workplaceName": "Fondazione IRCCS Istituto Neurologico Carlo Besta",
            "country": "IT",
            "city": "Milano"
        }
    }
    
    # VR data with specific details for comparison
    vr_data_for_comparison = {
        "validation.refAreaEid": "RAR_ITALY",
        "validation.id": 1019001316927770,
        "validation.entityTypeIco": "ENT_ACTIVITY",
        "validation.statusIco": "VAS_NOT_PROCESSED",
        "individual.firstName": "Marcello",
        "individual.lastName": "Marchetti",
        "workplace.usualName": "Fondazione IRCCS Istituto Neurologico Carlo Besta",
        "address.country": "IT",
        "address.city": "Milano",
        "address.postalCity": "Milano"
    }
    
    print("=== Testing VR Data Comparison in Summarization ===")
    result = await process_search_request(search_requirements, vr_data_for_comparison)
    print(f"Status: {result['workflow_status'].value}")
    
    if result.get("intelligent_summary"):
        summary = result["intelligent_summary"]
        print(f"Summary includes VR comparison: {summary}")
    print()

if __name__ == "__main__":
    asyncio.run(test_basic_search())
    asyncio.run(test_error_cases())
    asyncio.run(test_with_vr_comparison())


### batch_search_processor.py

"""
Batch processor for multiple search requests
Updated to handle VR data along with search requirements
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

async def process_search_batch(search_requests_with_vr: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """
    Process a batch of search requests with VR data
    
    Args:
        search_requests_with_vr: List of dictionaries containing both search requirements and VR data
    
    Returns:
        List of search results
    """
    batch_id = datetime.now().strftime('%Y%m%d_%H%M%S')
    
    try:
        logger.info(f"Processing batch of {len(search_requests_with_vr)} search requests with VR data")
        
        results = []
        for idx, request_data in enumerate(search_requests_with_vr):
            request_id = request_data.get("request_id", f"request_{idx}")
            search_requirements = request_data.get("search_requirements", {})
            vr_data = request_data.get("vr_data", {})
            
            logger.info(f"Processing search request {idx + 1}/{len(search_requests_with_vr)} - ID: {request_id}")
            
            try:
                result = await process_search_request(search_requirements, vr_data)
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
                    "error": str(e),
                    "vr_id": vr_data.get("validation.id", "unknown")
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

async def process_vr_batch_from_file(vr_records_file: str) -> List[Dict[str, Any]]:
    """
    Process VR records from a JSON file
    
    Args:
        vr_records_file: Path to JSON file containing VR records
    
    Returns:
        List of search results
    """
    try:
        # Load VR records from file
        with open(vr_records_file, 'r', encoding='utf-8') as f:
            vr_records = json.load(f)
        
        logger.info(f"Loaded {len(vr_records)} VR records from {vr_records_file}")
        
        # Convert VR records to search requests
        search_requests_with_vr = []
        for idx, vr_record in enumerate(vr_records):
            # Generate search requirements based on VR record
            search_requirements = generate_search_requirements_from_vr(vr_record)
            
            search_requests_with_vr.append({
                "request_id": f"VR_{vr_record.get('validation.id', idx)}",
                "search_requirements": search_requirements,
                "vr_data": vr_record
            })
        
        # Process the batch
        return await process_search_batch(search_requests_with_vr)
        
    except Exception as e:
        logger.error(f"Error processing VR batch from file: {str(e)}")
        raise

def generate_search_requirements_from_vr(vr_record: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate search requirements from VR record
    
    Args:
        vr_record: VR record data
    
    Returns:
        Search requirements dictionary
    """
    entity_type = vr_record.get("validation.entityTypeIco", "")
    country_code = vr_record.get("validation.countryCode", "")
    
    # Base search requirements
    search_requirements = {
        "verification_needed": True,
        "geographic_region": country_code,
        "confidence_threshold": 0.85,
        "search_context": {
            "entity_type": entity_type,
            "verification_type": "medical_professional" if entity_type == "ENT_ACTIVITY" else "workplace",
            "urgency": "standard"
        }
    }
    
    # Set objectives based on entity type
    if entity_type == "ENT_ACTIVITY":
        search_requirements["primary_objectives"] = [
            "verify_employment_status",
            "confirm_medical_license",
            "validate_workplace_affiliation",
            "verify_professional_credentials"
        ]
        
        # Add individual details
        search_requirements["entity_details"] = {
            "firstName": vr_record.get("individual.firstName", ""),
            "lastName": vr_record.get("individual.lastName", ""),
            "workplaceName": vr_record.get("workplace.usualName", ""),
            "country": vr_record.get("address.country", ""),
            "city": vr_record.get("address.city", "")
        }
        
    elif entity_type == "ENT_WORKPLACE":
        search_requirements["primary_objectives"] = [
            "verify_workplace_existence",
            "confirm_workplace_status",
            "validate_workplace_registration",
            "verify_contact_information"
        ]
        
        # Add workplace details
        search_requirements["entity_details"] = {
            "workplaceName": vr_record.get("workplace.usualName", ""),
            "country": vr_record.get("address.country", ""),
            "city": vr_record.get("address.city", ""),
            "postalCity": vr_record.get("address.postalCity", "")
        }
    
    return search_requirements

if __name__ == "__main__":
    # Example batch of search requests with VR data
    sample_batch = [
        {
            "request_id": "IT_001",
            "search_requirements": {
                "verification_needed": True,
                "primary_objectives": [
                    "verify_employment_status", 
                    "confirm_medical_license",
                    "validate_workplace_affiliation"
                ],
                "geographic_region": "IT",
                "confidence_threshold": 0.85,
                "entity_details": {
                    "firstName": "Marcello",
                    "lastName": "Marchetti",
                    "workplaceName": "Fondazione IRCCS Istituto Neurologico Carlo Besta",
                    "country": "IT",
                    "city": "Milano"
                },
                "search_context": {
                    "entity_type": "ENT_ACTIVITY",
                    "verification_type": "medical_professional",
                    "urgency": "standard"
                }
            },
            "vr_data": {
                "validation.refAreaEid": "RAR_ITALY",
                "validation.id": 1019001316927770,
                "validation.customerId": 7433,
                "validation.externalId": "47064408",
                "validation.entityTypeIco": "ENT_ACTIVITY",
                "validation.countryCode": "IT",
                "individual.firstName": "Marcello",
                "individual.lastName": "Marchetti",
                "workplace.usualName": "Fondazione IRCCS Istituto Neurologico Carlo Besta",
                "address.country": "IT",
                "address.city": "Milano",
                "address.postalCity": "Milano"
            }
        },
        {
            "request_id": "FR_001", 
            "search_requirements": {
                "verification_needed": True,
                "primary_objectives": [
                    "verify_professional_registration",
                    "confirm_current_workplace",
                    "validate_specialty_credentials"
                ],
                "geographic_region": "FR",
                "confidence_threshold": 0.80,
                "entity_details": {
                    "firstName": "Marie",
                    "lastName": "Dubois",
                    "workplaceName": "Hopital Saint-Antoine",
                    "country": "FR",
                    "city": "Paris"
                },
                "search_context": {
                    "entity_type": "ENT_ACTIVITY",
                    "verification_type": "medical_professional",
                    "urgency": "high"
                }
            },
            "vr_data": {
                "validation.refAreaEid": "RAR_FRANCE",
                "validation.id": 1019001316927771,
                "validation.customerId": 7434,
                "validation.externalId": "47064409",
                "validation.entityTypeIco": "ENT_ACTIVITY",
                "validation.countryCode": "FR",
                "individual.firstName": "Marie",
                "individual.lastName": "Dubois",
                "workplace.usualName": "Hopital Saint-Antoine",
                "address.country": "FR",
                "address.city": "Paris",
                "address.postalCity": "Paris"
            }
        }
    ]
    
    asyncio.run(process_search_batch(sample_batch))
