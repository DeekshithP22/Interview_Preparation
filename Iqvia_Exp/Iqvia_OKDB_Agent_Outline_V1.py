"""
OneKey VR Automation - Simplified Two-Agent Architecture
Supervisor Agent: VR API → OK DB API → Analysis → Decision Making → DBO Actions
Search & Summarize Agent: Multi-source searching → Intelligent summarization
"""

from typing import Dict, List, Any, Optional, TypedDict, Annotated
from langgraph.graph import StateGraph, END, START
from langchain_core.messages import BaseMessage, HumanMessage, AIMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.prompts import ChatPromptTemplate
import operator
from enum import Enum

# Import your existing API classes and tools
# from your_vr_api import VRAPIClient, VRDataClass
# from your_okdb_api import OKDBAPIClient, OKDBDataClass
# from tools.italy_trusted_sources import ItalyTrustedSourcesTool
# from tools.france_trusted_sources import FranceTrustedSourcesTool
# from tools.hospital_sources import HospitalSourcesTool
# from tools.linkedin_professional import LinkedInProfessionalTool
# from tools.untrusted_web_search import UntrustedWebSearchTool

# =============================================================================
# STATE DEFINITIONS
# =============================================================================

class WorkflowStatus(Enum):
    INITIATED = "initiated"                 # Workflow started
    VR_RETRIEVED = "vr_retrieved"          # VR API call completed
    OKDB_SEARCHED = "okdb_searched"        # OK DB API search completed
    ANALYSIS_COMPLETED = "analysis_completed"  # Supervisor analysis done
    SEARCH_DELEGATED = "search_delegated"  # Search task assigned to Search&Summarize Agent
    SEARCH_COMPLETED = "search_completed"  # Search&Summarize Agent finished
    DBO_DECISION_READY = "dbo_decision_ready"  # Final DBO action determined
    COMPLETED = "completed"                # Workflow finished
    ERROR = "error"                       # Error state

class EntityType(Enum):
    ENT_ACTIVITY = "ENT_ACTIVITY"
    ENT_WORKPLACE = "ENT_WORKPLACE"

class AgentState(TypedDict):
    # Input
    vr_request_id: str                     # VR request identifier
    
    # Supervisor Agent state - API calls and analysis
    vr_api_response: Optional[Dict[str, Any]]      # Raw VR API response
    vr_entity_type: Optional[str]                  # ENT_ACTIVITY or ENT_WORKPLACE
    okdb_api_response: Optional[Dict[str, Any]]    # Raw OK DB API response
    comparison_analysis: Optional[Dict[str, Any]]   # Supervisor's VR vs OK DB analysis
    record_status: str                             # "existing", "new", "mismatch", "unclear"
    search_requirements: Optional[Dict[str, Any]]   # What needs to be searched/verified
    dbo_action_decision: Optional[Dict[str, Any]]   # Final DBO action from Supervisor
    
    # Search & Summarize Agent state
    selected_tools: List[str]                      # Tools chosen for search
    execution_order: List[str]                     # Order of tool execution
    search_results: Annotated[List[Dict[str, Any]], operator.add]  # Results from all tools
    intelligent_summary: Optional[Dict[str, Any]]  # AI-generated summary
    search_confidence: float                       # Overall search confidence
    
    # Workflow management
    workflow_status: WorkflowStatus
    current_agent: str                            # "supervisor" or "search_summarize"
    error_context: Optional[Dict[str, Any]]        # Error details if any
    
    # Communication between agents
    messages: Annotated[List[BaseMessage], operator.add]

# =============================================================================
# STATIC HELPER FUNCTION FOR OK DB SEARCH STRATEGY
# =============================================================================

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

# =============================================================================
# SUPERVISOR AGENT - API ORCHESTRATOR & DECISION MAKER
# =============================================================================

class SupervisorAgent:
    """
    Supervisor Agent: Handles VR API → OK DB API → Analysis → Decision Making
    
    Responsibilities:
    1. Execute VR API call and parse response
    2. Execute OK DB API search using VR data
    3. Compare VR vs OK DB responses (existing/new/mismatch analysis)
    4. Decide what verification is needed
    5. Delegate search task to Search&Summarize Agent
    6. Make final DBO action decisions
    """
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4",
            api_version="2024-02-15-preview",
            temperature=0.1
        )
        
        # Initialize your existing API clients
        # self.vr_api_client = VRAPIClient()
        # self.okdb_api_client = OKDBAPIClient()
    
    async def execute_vr_api_call(self, state: AgentState) -> AgentState:
        """
        Execute VR API call to get validation request data
        
        Uses your existing VR API client and data classes
        Handles authentication, request formatting, response parsing
        """
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
        """
        Execute OK DB API search using VR data
        
        Uses the static search strategy function to determine search parameters
        """
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
        """
        LLM-powered analysis comparing VR data vs OK DB results
        
        Handles all comparison scenarios:
        - Single/multiple OK DB results
        - Same person with multiple workplaces
        - Different people with similar names
        - Existing vs new records
        - Data mismatches
        """
        # Create comprehensive analysis prompt
        analysis_prompt = ChatPromptTemplate.from_messages([
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
        
        try:
            # Execute LLM analysis
            analysis_response = await self.llm.ainvoke(
                analysis_prompt.format_messages(
                    vr_data=state["vr_api_response"],
                    okdb_data=state["okdb_api_response"]
                )
            )
            
            # Parse analysis results
            import json
            parsed_analysis = json.loads(analysis_response.content)
            
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
        """
        Prepare search delegation for Search & Summarize Agent
        
        Formats search requirements and sets up delegation
        """
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
        """
        LLM makes final DBO action decision based on all available data
        
        Considers:
        - Original VR vs OK DB analysis
        - Search & Summarize Agent results (if verification was done)
        - Confidence levels and risk assessment
        """
        decision_prompt = ChatPromptTemplate.from_messages([
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
        
        try:
            # Execute decision-making LLM
            decision_response = await self.llm.ainvoke(
                decision_prompt.format_messages(
                    comparison_analysis=state.get("comparison_analysis", {}),
                    search_summary=state.get("intelligent_summary", {}),
                    search_confidence=state.get("search_confidence", 0.0)
                )
            )
            
            # Parse decision
            import json
            dbo_decision = json.loads(decision_response.content)
            
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

# =============================================================================
# SEARCH & SUMMARIZE AGENT - MULTI-SOURCE SEARCH & INTELLIGENT SUMMARIZATION
# =============================================================================

class SearchAndSummarizeAgent:
    """
    Search & Summarize Agent: Multi-source searching + AI summarization
    
    Responsibilities:
    1. Receive search requirements from Supervisor
    2. Select appropriate tools for verification
    3. Execute searches across multiple sources
    4. Intelligently summarize all search results
    5. Assess confidence and flag manual review needs
    """
    
    def __init__(self):
        self.llm = AzureChatOpenAI(
            azure_deployment="gpt-4",
            api_version="2024-02-15-preview",
            temperature=0.1
        )
        
        # Initialize your existing tool instances - consolidated approach
        self.search_tools = {
            "italy_trusted": self._initialize_italy_tool(),
            "france_trusted": self._initialize_france_tool(),
            "hospital_sources": self._initialize_hospital_tool(),
            "linkedin_professional": self._initialize_linkedin_tool(),
            "untrusted_web_search": self._initialize_web_tool()
        }
    
    def _initialize_italy_tool(self):
        """Initialize consolidated Italy tool (wraps your 12+ Italian site automations)"""
        # TODO: Import your consolidated Italy tool
        # from tools.italy_trusted_sources import ItalyTrustedSourcesTool
        # return ItalyTrustedSourcesTool()
        pass
    
    def _initialize_france_tool(self):
        """Initialize consolidated France tool (wraps your 9+ French site automations)"""
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
        """
        LLM selects appropriate tools based on search requirements from Supervisor
        
        Considers geographic region, confidence needs, verification objectives
        """
        tool_selection_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are selecting search tools for healthcare professional verification.

Available Tools:
1. italy_trusted - Italian medical registries (FNOMCEO, FNOPI, etc.) - Reliability: 0.95
2. france_trusted - French medical directories (Annuaire Santé, etc.) - Reliability: 0.95
3. hospital_sources - Hospital websites and staff directories - Reliability: 0.80
4. linkedin_professional - Professional networking platforms - Reliability: 0.70
5. untrusted_web_search - General web search engines - Reliability: 0.50

Tool Selection Strategy:
- Geographic Priority: Match tools to individual's country/region
- Confidence Requirements: Higher confidence needs → include more trusted sources
- Verification Scope: Broad verification → include multiple tool types
- Resource Efficiency: Stop early if high confidence achieved

Execution Order Priority:
1. Trusted medical registries (country-specific)
2. Institutional sources (hospitals, organizations)  
3. Professional networks (LinkedIn)
4. General web search (if needed)

Respond with JSON:
{{
    "selected_tools": ["tool1", "tool2", "tool3"],
    "execution_order": ["tool1", "tool2", "tool3"],
    "tool_rationale": {{
        "tool1": "reason for selection",
        "tool2": "reason for selection"
    }},
    "stopping_strategy": "When to stop searching",
    "confidence_target": 0.XX
}}"""),
            ("human", """
Search Requirements: {search_requirements}

Select tools and execution strategy.
""")
        ])
        
        try:
            # Execute tool selection LLM
            selection_response = await self.llm.ainvoke(
                tool_selection_prompt.format_messages(
                    search_requirements=state["search_requirements"]
                )
            )
            
            # Parse selection
            import json
            tool_selection = json.loads(selection_response.content)
            
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
        """
        Execute selected tools in planned order with intelligent stopping
        
        Each tool searches for the same individual across different sources
        Builds evidence and confidence progressively
        """
        try:
            execution_order = state["execution_order"]
            search_requirements = state["search_requirements"]
            
            for tool_name in execution_order:
                if tool_name in self.search_tools:
                    tool_instance = self.search_tools[tool_name]
                    
                    # TODO: Execute tool search
                    # search_result = await tool_instance.search(search_requirements)
                    
                    # Add to results
                    # state["search_results"].append(search_result)
                    
                    # Check stopping criteria with LLM
                    # should_continue = await self._evaluate_stopping_criteria(state)
                    # if not should_continue:
                    #     break
                    
                    state["messages"].append(AIMessage(
                        content=f"Executed search with {tool_name}"
                    ))
            
        except Exception as e:
            state.update({
                "workflow_status": WorkflowStatus.ERROR,
                "error_context": {"stage": "search_execution", "error": str(e)}
            })
        
        return state
    
    async def _evaluate_stopping_criteria(self, state: AgentState) -> bool:
        """
        LLM evaluates whether to continue searching or stop
        
        Considers current confidence, remaining tools, diminishing returns
        """
        # TODO: Implement LLM-based stopping decision
        # For now, simple logic - continue if less than target confidence
        current_results = state.get("search_results", [])
        if len(current_results) >= 2:  # At least 2 sources
            return False  # Stop searching
        return True  # Continue
    
    async def intelligent_summarization(self, state: AgentState) -> AgentState:
        """
        LLM creates intelligent summary of all search results
        
        Synthesizes findings, identifies conflicts, assesses overall confidence
        """
        summarization_prompt = ChatPromptTemplate.from_messages([
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
        
        try:
            # Execute summarization LLM
            summary_response = await self.llm.ainvoke(
                summarization_prompt.format_messages(
                    search_requirements=state["search_requirements"],
                    search_results=state["search_results"]
                )
            )
            
            # Parse summary
            import json
            summary_data = json.loads(summary_response.content)
            
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

# =============================================================================
# CONDITIONAL ROUTING FUNCTIONS
# =============================================================================

def supervisor_routing_decision(state: AgentState) -> str:
    """Route workflow based on Supervisor Agent's current status"""
    workflow_status = state.get("workflow_status")
    
    if workflow_status == WorkflowStatus.INITIATED:
        return "execute_vr_api"
    elif workflow_status == WorkflowStatus.VR_RETRIEVED:
        return "execute_okdb_search"
    elif workflow_status == WorkflowStatus.OKDB_SEARCHED:
        return "analyze_comparison"
    elif workflow_status == WorkflowStatus.ANALYSIS_COMPLETED:
        return "delegate_search"
    elif workflow_status == WorkflowStatus.SEARCH_COMPLETED:
        return "make_dbo_decision"
    elif workflow_status == WorkflowStatus.DBO_DECISION_READY:
        return "complete"
    else:
        return "handle_error"

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

# =============================================================================
# LANGGRAPH WORKFLOW DEFINITION
# =============================================================================

def create_agent_workflow():
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
    
    workflow.add_conditional_edges(
        "delegate_search",
        supervisor_routing_decision,
        {
            "select_tools": "select_tools",  # Hand off to Search & Summarize Agent
            "make_dbo_decision": "make_dbo_decision"  # Skip search if not needed
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
    
    # Complete workflow
    workflow.add_edge("make_dbo_decision", END)
    
    # Compile workflow
    return workflow.compile()

# =============================================================================
# USAGE EXAMPLE
# =============================================================================

async def main():
    """
    Example usage of the two-agent VR automation workflow
    """
    # Create workflow
    app = create_agent_workflow()
    
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
    
    
    
    
    

# =============================================================================
# DOCUMENTATION
# =============================================================================

"""
ONEKEY VR AUTOMATION - TWO-AGENT ARCHITECTURE

OVERVIEW:
This system automates the VR (Validation Request) processing workflow using two specialized agents:
1. Supervisor Agent - Handles API orchestration and decision making
2. Search & Summarize Agent - Performs multi-source verification

WORKFLOW FLOW:
1. START → Supervisor: Execute VR API call
2. Supervisor: Execute OK DB API search (using static strategy function)
3. Supervisor: Analyze VR vs OK DB data (handles all comparison scenarios)
4. Supervisor: Generate search requirements (if verification needed)
5. Supervisor: Delegate to Search & Summarize Agent
6. Search & Summarize: Select appropriate tools
7. Search & Summarize: Execute searches
8. Search & Summarize: Create intelligent summary
9. Return to Supervisor: Make DBO decision
10. END

KEY FEATURES:
- Entity Type Handling: ENT_ACTIVITY vs ENT_WORKPLACE logic
- Smart Search Strategy: Uses originKeyEid when available, falls back to name-based search
- Multiple Results Handling: Supervisor's LLM handles all scenarios (same person multiple workplaces, different people, etc.)
- Simple Architecture: Only one static function for search strategy, all complex logic handled by LLMs
- Tool Consolidation: 21 website automations wrapped into 5 logical tools

INTEGRATION POINTS:
1. VR API Client - Your existing implementation
2. OK DB API Client - Your existing implementation  
3. Search Tools:
   - italy_trusted - Wraps 12+ Italian site automations
   - france_trusted - Wraps 9+ French site automations
   - hospital_sources - Hospital directory searches
   - linkedin_professional - Professional network searches
   - untrusted_web_search - General web searches

STATE MANAGEMENT:
The AgentState contains all necessary information flowing through the workflow:
- API responses (VR and OK DB)
- Analysis results and record status
- Search requirements and results
- Final DBO decisions

ERROR HANDLING:
- Graceful error handling at each stage
- Workflow status tracking for debugging
- Error context preservation for troubleshooting

EXAMPLE USAGE:
```python
# Initialize workflow
app = create_two_agent_vr_workflow()

# Execute with VR request ID
initial_state = {
    "vr_request_id": "RAR_ITALY_1019000316927770",
    # ... other initial state values
}

result = await app.ainvoke(initial_state)

# Access results
dbo_actions = result["dbo_action_decision"]["dbo_actions"]
```

SUPERVISOR AGENT RESPONSIBILITIES:
1. Execute VR API call
2. Determine OK DB search strategy (static function)
3. Execute OK DB API search
4. Analyze all data comprehensively (single LLM call handles all scenarios)
5. Generate search requirements if needed
6. Make final DBO decisions

SEARCH & SUMMARIZE AGENT RESPONSIBILITIES:
1. Select appropriate search tools based on geography and requirements
2. Execute searches in optimal order
3. Evaluate stopping criteria
4. Create intelligent summary of all findings
5. Calculate overall confidence

DBO ACTION TYPES:
- UPDATE_RECORD: Modify existing OK DB record(s)
- CREATE_RECORD: Create new OK DB record
- NO_ACTION: Current data is accurate
- MANUAL_REVIEW: Complex case needs human review
- INVESTIGATE: More information needed

CONFIDENCE LEVELS:
- italy_trusted: 0.95 (Official registries)
- france_trusted: 0.95 (Official registries)
- hospital_sources: 0.80 (Hospital directories)
- linkedin_professional: 0.70 (Professional networks)
- untrusted_web_search: 0.50 (General web)

BENEFITS:
- Simple, clean architecture
- Minimal code complexity
- Leverages LLM intelligence for complex decisions
- Easy integration with existing systems
- Scalable and maintainable
"""