"""
OneKey VR Automation - Two-Agent Architecture LangGraph Framework
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

# Import existing API classes and tools
# from vr_api import VRAPIClient, VRDataClass
# from okdb_api import OKDBAPIClient, OKDBDataClass
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

class AgentState(TypedDict):
    # Input
    vr_request_id: str                     # VR request identifier
    
    # Supervisor Agent state - API calls and analysis
    vr_api_response: Optional[Dict[str, Any]]      # Raw VR API response
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
        # TODO: Use your VR API client here
        # vr_response = await self.vr_api_client.get_validation_request(state["vr_request_id"])
        # parsed_vr_data = VRDataClass(vr_response)
        
        # TODO: Update state with VR API response
        # state.update({
        #     "vr_api_response": vr_response,
        #     "workflow_status": WorkflowStatus.VR_RETRIEVED
        # })
        
        pass
    
    async def execute_okdb_search(self, state: AgentState) -> AgentState:
        """
        Execute OK DB API search using VR data
        
        Uses your existing OK DB API client and data classes
        Searches for existing records matching VR individual data
        """
        # TODO: Extract search parameters from VR response
        # vr_data = state["vr_api_response"]
        # search_params = self._extract_search_params_from_vr(vr_data)
        
        # TODO: Use your OK DB API client here
        # okdb_response = await self.okdb_api_client.search_records(search_params)
        # parsed_okdb_data = OKDBDataClass(okdb_response)
        
        # TODO: Update state with OK DB response
        # state.update({
        #     "okdb_api_response": okdb_response,
        #     "workflow_status": WorkflowStatus.OKDB_SEARCHED
        # })
        
        pass
    
    async def analyze_vr_vs_okdb(self, state: AgentState) -> AgentState:
        """
        LLM-powered analysis comparing VR data vs OK DB results
        
        Determines:
        - Is this an existing record or new record?
        - Are there mismatches in status, workplace, specialty?
        - What verification is needed?
        - What confidence level is required?
        """
        # TODO: Create analysis prompt for LLM
        analysis_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are analyzing VR validation requests against existing OneKey Database records.

Your Analysis Tasks:
1. Record Status Classification:
   - "existing_match": Individual found with consistent data
   - "existing_mismatch": Individual found but data conflicts detected  
   - "new_record": Individual not found in OK DB
   - "ambiguous": Multiple potential matches, unclear which is correct

2. Mismatch Detection:
   - Status inconsistencies (VR active vs OK DB inactive)
   - Workplace changes or discrepancies
   - Specialty assignment conflicts
   - Contact information differences

3. Verification Requirements:
   - What specific facts need external verification?
   - Which sources would be most reliable?
   - What confidence level is needed for decision?

4. Geographic and Context Analysis:
   - Country/region for tool selection
   - Institution type for search strategy
   - Specialty field for targeted verification

Respond with JSON:
{{
    "record_status": "existing_match/existing_mismatch/new_record/ambiguous",
    "individual_disambiguation": {{
        "confidence": 0.XX,
        "primary_match_id": "OK DB ID if found",
        "matching_factors": ["factors that confirm match"],
        "conflicting_factors": ["factors that suggest mismatch"]
    }},
    "detected_mismatches": [
        {{
            "field": "status/workplace/specialty/contact",
            "vr_value": "value from VR",
            "okdb_value": "value from OK DB", 
            "severity": "high/medium/low",
            "verification_needed": "specific verification requirement"
        }}
    ],
    "search_requirements": {{
        "primary_objectives": ["list of verification goals"],
        "geographic_region": "country code",
        "confidence_threshold": 0.XX,
        "specialized_searches": ["specific search needs"]
    }},
    "analysis_summary": "Human-readable summary for DBO context"
}}"""),
            ("human", """
VR API Response: {vr_data}
OK DB API Response: {okdb_data}

Analyze and compare these responses.
""")
        ])
        
        # TODO: Execute LLM analysis
        # analysis_response = await self.llm.ainvoke(
        #     analysis_prompt.format(
        #         vr_data=state["vr_api_response"],
        #         okdb_data=state["okdb_api_response"]
        #     )
        # )
        
        # TODO: Parse analysis results and update state
        # parsed_analysis = parse_json(analysis_response.content)
        # state.update({
        #     "comparison_analysis": parsed_analysis,
        #     "record_status": parsed_analysis["record_status"],
        #     "search_requirements": parsed_analysis["search_requirements"],
        #     "workflow_status": WorkflowStatus.ANALYSIS_COMPLETED
        # })
        
        pass
    
    async def delegate_search_task(self, state: AgentState) -> AgentState:
        """
        Prepare search delegation for Search & Summarize Agent
        
        Formats search requirements and sets up delegation
        """
        # TODO: Prepare delegation message
        # search_requirements = state["search_requirements"]
        # delegation_message = f"Search delegation: {search_requirements['primary_objectives']}"
        
        # TODO: Update workflow status
        # state.update({
        #     "workflow_status": WorkflowStatus.SEARCH_DELEGATED,
        #     "current_agent": "search_summarize"
        # })
        
        # state["messages"].append(AIMessage(content=delegation_message))
        
        pass
    
    async def make_dbo_decision(self, state: AgentState) -> AgentState:
        """
        LLM makes final DBO action decision based on all available data
        
        Considers:
        - Original VR vs OK DB analysis
        - Search & Summarize Agent results
        - Confidence levels and risk assessment
        """
        # TODO: Create decision-making prompt
        decision_prompt = ChatPromptTemplate.from_messages([
            ("system", """You are making final DBO action decisions for OneKey VR validation.

Available Information:
1. Original VR validation request
2. Existing OK DB records (if any)
3. External verification results from trusted sources
4. Confidence assessments from multiple sources

Your Decision Categories:
- "UPDATE_RECORD": Modify existing OK DB record with new information
- "CREATE_RECORD": Create new OK DB record for new individual
- "NO_ACTION": Current OK DB data is accurate, no changes needed
- "MANUAL_REVIEW": Complex case requiring human DBO review
- "INVESTIGATE": Conflicting information needs further investigation

For each decision, provide:
- Specific action steps for DBO
- Supporting evidence and reasoning
- Risk assessment and confidence level
- Manual review flags if needed

Respond with JSON:
{{
    "dbo_actions": [
        {{
            "action_type": "UPDATE_RECORD/CREATE_RECORD/NO_ACTION/MANUAL_REVIEW/INVESTIGATE",
            "target_record_id": "OK DB record ID if applicable",
            "field_updates": {{"field": "new_value"}},
            "justification": "Clear reasoning for DBO",
            "confidence": 0.XX,
            "supporting_evidence": ["evidence sources"],
            "manual_review_required": true/false,
            "risk_level": "low/medium/high"
        }}
    ],
    "overall_recommendation": "Primary recommendation for DBO",
    "decision_confidence": 0.XX,
    "processing_summary": "Summary of entire validation process"
}}"""),
            ("human", """
Original Analysis: {comparison_analysis}
Search Results: {search_results}
Search Summary: {intelligent_summary}
Search Confidence: {search_confidence}

Make final DBO action decision.
""")
        ])
        
        # TODO: Execute decision-making LLM
        # decision_response = await self.llm.ainvoke(
        #     decision_prompt.format(
        #         comparison_analysis=state["comparison_analysis"],
        #         search_results=state["search_results"],
        #         intelligent_summary=state["intelligent_summary"],
        #         search_confidence=state["search_confidence"]
        #     )
        # )
        
        # TODO: Parse decision and update state
        # dbo_decision = parse_json(decision_response.content)
        # state.update({
        #     "dbo_action_decision": dbo_decision,
        #     "workflow_status": WorkflowStatus.DBO_DECISION_READY
        # })
        
        pass
    
    def _extract_search_params_from_vr(self, vr_data: Dict[str, Any]) -> Dict[str, Any]:
        """
        Extract search parameters from VR response for OK DB API call
        
        Maps VR data fields to OK DB search parameters
        """
        # TODO: Implement VR to OK DB search parameter mapping
        # return {
        #     "firstName": vr_data["individual"]["firstName"],
        #     "lastName": vr_data["individual"]["lastName"], 
        #     "workplace": vr_data["workplace"]["usualName"],
        #     "country": vr_data["address"]["country"]
        # }
        pass

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
Geographic Region: {geographic_region}
Required Confidence: {confidence_threshold}

Select tools and execution strategy.
""")
        ])
        
        # TODO: Execute tool selection LLM
        # selection_response = await self.llm.ainvoke(
        #     tool_selection_prompt.format(
        #         search_requirements=state["search_requirements"],
        #         geographic_region=state["search_requirements"]["geographic_region"],
        #         confidence_threshold=state["search_requirements"]["confidence_threshold"]
        #     )
        # )
        
        # TODO: Parse selection and update state
        # tool_selection = parse_json(selection_response.content)
        # state.update({
        #     "selected_tools": tool_selection["selected_tools"],
        #     "execution_order": tool_selection["execution_order"]
        # })
        
        pass
    
    async def execute_search_tools(self, state: AgentState) -> AgentState:
        """
        Execute selected tools in planned order with intelligent stopping
        
        Each tool searches for the same individual across different sources
        Builds evidence and confidence progressively
        """
        # TODO: Execute each tool in sequence
        # execution_order = state["execution_order"]
        # search_requirements = state["search_requirements"]
        
        # for tool_name in execution_order:
        #     if tool_name in self.search_tools:
        #         tool_instance = self.search_tools[tool_name]
        #         
        #         # Execute tool search
        #         search_result = await tool_instance.search(search_requirements)
        #         
        #         # Add to results
        #         state["search_results"].append(search_result)
        #         
        #         # Check stopping criteria with LLM
        #         should_continue = await self._evaluate_stopping_criteria(state)
        #         if not should_continue:
        #             break
        
        pass
    
    async def _evaluate_stopping_criteria(self, state: AgentState) -> bool:
        """
        LLM evaluates whether to continue searching or stop
        
        Considers current confidence, remaining tools, diminishing returns
        """
        # TODO: Implement LLM-based stopping decision
        # stopping_prompt = ChatPromptTemplate.from_messages([...])
        # stopping_response = await self.llm.ainvoke(stopping_prompt.format(...))
        # decision = parse_json(stopping_response.content)
        # return decision["should_continue"]
        pass
    
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
        "employment_status": {{
            "status": "confirmed/uncertain/conflicted",
            "evidence": ["supporting evidence"],
            "confidence": 0.XX
        }},
        "professional_credentials": {{
            "status": "confirmed/uncertain/conflicted", 
            "evidence": ["supporting evidence"],
            "confidence": 0.XX
        }}
    }},
    "source_analysis": {{
        "sources_consulted": ["source1", "source2"],
        "highest_confidence_source": "source_name",
        "conflicting_sources": ["sources with conflicts"]
    }},
    "overall_assessment": {{
        "primary_finding": "Main conclusion about individual",
        "confidence_level": 0.XX,
        "verification_completeness": "percentage of objectives met",
        "manual_review_recommended": true/false,
        "manual_review_reasons": ["reasons if recommended"]
    }},
    "summary_narrative": "Human-readable summary for DBO review"
}}"""),
            ("human", """
Search Requirements: {search_requirements}
Search Results: {search_results}
Tools Used: {selected_tools}

Create intelligent summary of all findings.
""")
        ])
        
        # TODO: Execute summarization LLM
        # summary_response = await self.llm.ainvoke(
        #     summarization_prompt.format(
        #         search_requirements=state["search_requirements"],
        #         search_results=state["search_results"],
        #         selected_tools=state["selected_tools"]
        #     )
        # )
        
        # TODO: Parse summary and update state
        # summary_data = parse_json(summary_response.content)
        # state.update({
        #     "intelligent_summary": summary_data,
        #     "search_confidence": summary_data["overall_assessment"]["confidence_level"],
        #     "workflow_status": WorkflowStatus.SEARCH_COMPLETED,
        #     "current_agent": "supervisor"
        # })
        
        pass

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
            "select_tools": "select_tools"  # Hand off to Search & Summarize Agent
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
    app = create_two_agent_vr_workflow()
    
    # Initial state with VR request ID
    initial_state = {
        "vr_request_id": "RAR_ITALY_1019000316927770",  # From your example
        "vr_api_response": None,
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