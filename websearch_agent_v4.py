"""
Web Search Agent using Tavily and OpenAI
This module provides a search function to find healthcare professionals
using structured input and targeted search queries.
"""

import os
import json
import logging
import re
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from openai import AzureOpenAI
from tavily import TavilyClient
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


# Simplified Pydantic model only for LLM response
class LLMSummaryResponse(BaseModel):
    """Model for LLM to provide comprehensive summary from search results"""
    llm_final_answer: str = Field(description="Comprehensive summary analyzing all search results with main search, linkedin, and workplace summaries")


# Function to extract structured data from raw text using OpenAI
def extract_llm_response(raw_text: str) -> LLMSummaryResponse:
    """
    Uses OpenAI to extract structured LLM response data from raw text.
    
    Args:
        raw_text (str): Unstructured scraped text from a website.
    
    Returns:
        LLMSummaryResponse: A populated Pydantic model with extracted data.
    """
    # Set up OpenAI token provider for Azure OpenAI authentication
    token_provider = get_bearer_token_provider(
        ClientSecretCredential(
            os.environ["AZURE_OPENAI_TENANT_ID"],
            os.environ["AZURE_OPENAI_PRINCIPAL_ID"], 
            os.environ["AZURE_OPENAI_PRINCIPAL_SECRET"],
        ),
        "api://825a47b7-8e55-49b5-99c5-d7ecf65bd64d/.default",
    )
    
    # Initialize OpenAI client
    client = AzureOpenAI(
        api_version=os.environ["AZURE_OPENAI_API_VERSION"],
        azure_endpoint=os.environ["AZURE_OPENAI_ENDPOINT_URL"],
        azure_ad_token_provider=token_provider,
    )
    
    schema_json = json.dumps(LLMSummaryResponse.model_json_schema(), indent=2)
    
    prompt = f"""
    You are a data extraction assistant. Given the following raw text from a medical or professional registry website, extract the relevant fields and return a JSON object that matches this Pydantic model:

    {schema_json}

    If any field is missing or not found, set it to null. Here's the raw text:

    \"\"\"{raw_text}\"\"\"
"""
    
    completion = client.chat.completions.create(
        model=os.environ["AZURE_OPENAI_CHATGPT_DEPLOYMENT"],
        messages=[{"role": "user", "content": prompt}],
        temperature=0.2,
    )
    
    response_text = completion.choices[0].message.content.strip()
    
    # Remove Markdown-style code block markers if present
    if response_text.startswith("```json"):
        response_text = re.sub(r'^```json\s*|\s*```$', "", response_text.strip(), flags=re.DOTALL)
    
    try:
        data = json.loads(response_text)
        return LLMSummaryResponse(**data)
    except Exception as e:
        print("Error parsing response:", e)
        print("Raw LLM output:", response_text)
        raise


class WebSearchAgent:
    """Agent for searching healthcare professionals using Tavily API"""
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        """
        Initialize the search agent
        
        Args:
            tavily_api_key: Tavily API key (defaults to env variable)
        """
        # Initialize Tavily client
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("Tavily API key not provided. Set TAVILY_API_KEY environment variable.")
        
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        
        # Country code mapping
        self.country_mapping = {
            "IT": "Italy",
            "US": "United States", 
            "UK": "United Kingdom",
            "DE": "Germany",
            "FR": "France",
            "ES": "Spain",
            "IN": "India",
            "CA": "Canada",
            "AU": "Australia"
        }
        
        logger.info("WebSearchAgent initialized successfully")
    
    def _get_country_name(self, geographic_region: str) -> str:
        """Convert geographic region code to country name"""
        return self.country_mapping.get(geographic_region.upper(), geographic_region)
    
    def _construct_main_search_query(self, doctor_info: Dict[str, Any]) -> str:
        """Construct comprehensive main search query"""
        first_name = doctor_info.get("firstName", "")
        last_name = doctor_info.get("lastName", "") 
        workplace = doctor_info.get("workplaceName", "")
        address = doctor_info.get("address", "")
        geographic_region = doctor_info.get("geographic_region", "")
        
        # Comprehensive search targeting official records, profiles, and contact info
        main_query = f'I need complete details, specialization about the medical professional, {first_name} {last_name} working at{workplace} {address}'

        logger.info(f"Main search query: {main_query}")
        return main_query
    
    def _construct_linkedin_query(self, doctor_info: Dict[str, Any]) -> str:
        """Construct highly targeted LinkedIn profile search"""
        first_name = doctor_info.get("firstName", "")
        last_name = doctor_info.get("lastName", "")
        workplace = doctor_info.get("workplaceName", "")
        address = doctor_info.get("address", "")
        
        # Target LinkedIn profiles specifically - using exact name matching and workplace
        linkedin_query = f'site:linkedin.com/in/{first_name} {last_name} {workplace} {address}'
        
        logger.info(f"LinkedIn search query: {linkedin_query}")
        return linkedin_query
    
    def _construct_workplace_query(self, doctor_info: Dict[str, Any]) -> str:
        """Construct workplace-specific search to find doctor in institutional websites"""
        first_name = doctor_info.get("firstName", "")
        last_name = doctor_info.get("lastName", "")
        workplace = doctor_info.get("workplaceName", "")
        address = doctor_info.get("address", "")
        
        # Target institutional websites - medical centers, hospitals, university websites
        # Use terms that appear in staff directories, faculty pages, doctor profiles
        workplace_query = f'{first_name} {last_name} {workplace} {address} staff OR faculty OR directory OR team OR doctors'
        
        logger.info(f"Workplace search query: {workplace_query}")
        return workplace_query
    
    def _search_tavily_main(self, doctor_info: Dict[str, Any], max_results: int = 10) -> Dict[str, Any]:
        """
        Perform main Tavily search with structured doctor information
        """
        try:
            main_query = self._construct_main_search_query(doctor_info)
            country = self._get_country_name(doctor_info.get("geographic_region", ""))
            
            logger.info(f"Performing main search: {main_query}")
            
            response = self.tavily_client.search(
                query=main_query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=True,
                country=country,
                time_range="year",
                chunks_per_source=2,
            )
                        # Filter results to ensure relevance
            response = self._filter_relevant_results(
                response,
                "workplace", 
                f"{doctor_info.get('firstName', '')} {doctor_info.get('lastName', '')}"
            )
            
            logger.info(f"Main search completed. Found {len(response.get('results', []))} results")
            return response
            
        except Exception as e:
            logger.error(f"Error in main Tavily search: {str(e)}")
            return {"results": [], "answer": ""}
        
    def _search_linkedin_profile(self, doctor_info: Dict[str, Any]) -> Dict[str, Any]:
        """Search specifically for LinkedIn profile using structured info"""
        try:
            linkedin_query = self._construct_linkedin_query(doctor_info)
            country = self._get_country_name(doctor_info.get("geographic_region", ""))
            
            logger.info(f"Performing LinkedIn search: {linkedin_query}")
            
            response = self.tavily_client.search(
                query=linkedin_query,
                search_depth="advanced",
                max_results=3,
                include_answer=True,
                include_raw_content=True,
                country=country,
                chunks_per_source=3,
                time_range="year",
                include_domains=["linkedin.com"]
            )
            
            # Filter results to ensure relevance
            response = self._filter_relevant_results(
                response, 
                "linkedin", 
                f"{doctor_info.get('firstName', '')} {doctor_info.get('lastName', '')}"
            )
            
            logger.info(f"LinkedIn search completed. Found {len(response.get('results', []))} results")
            return response
            
        except Exception as e:
            logger.error(f"Error in LinkedIn search: {str(e)}")
            return {"results": [], "answer": ""}

    def _search_workplace_website(self, doctor_info: Dict[str, Any]) -> Dict[str, Any]:
        """Search for doctor's information on workplace/institutional websites"""
        try:
            workplace_query = self._construct_workplace_query(doctor_info)
            country = self._get_country_name(doctor_info.get("geographic_region", ""))
            
            logger.info(f"Performing workplace search: {workplace_query}")
            
            response = self.tavily_client.search(
                query=workplace_query,
                search_depth="advanced",
                max_results=5,
                include_answer=True,
                include_raw_content=True,
                country=country,
                time_range="year",
                chunks_per_source=2,
            )
            
            # Filter results to ensure relevance
            response = self._filter_relevant_results(
                response,
                "workplace", 
                f"{doctor_info.get('firstName', '')} {doctor_info.get('lastName', '')}"
            )
            
            logger.info(f"Workplace search completed. Found {len(response.get('results', []))} results")
            return response
            
        except Exception as e:
            logger.error(f"Error in workplace website search: {str(e)}") 
            return {"results": [], "answer": ""}
        
    
    def _extract_urls_from_results(self, search_results: Dict[str, Any]) -> List[str]:
        """Extract all URLs from search results and remove duplicates"""
        urls = []
        
        try:
            # Extract from main results
            main_results = search_results.get("main_search", {})
            if "results" in main_results:
                for result in main_results["results"]:
                    if "url" in result and result["url"]:
                        urls.append(result["url"])
            
            # Extract from LinkedIn results
            linkedin_results = search_results.get("linkedin_search", {})
            if "results" in linkedin_results:
                for result in linkedin_results["results"]:
                    if "url" in result and result["url"]:
                        urls.append(result["url"])
            
            # Extract from workplace results
            workplace_results = search_results.get("workplace_search", {})
            if "results" in workplace_results:
                for result in workplace_results["results"]:
                    if "url" in result and result["url"]:
                        urls.append(result["url"])
            
            # Remove duplicates while preserving order
            seen = set()
            unique_urls = []
            for url in urls:
                if url not in seen:
                    seen.add(url)
                    unique_urls.append(url)
            
            return unique_urls
            
        except Exception as e:
            logger.error(f"Error extracting URLs: {str(e)}")
            return []
    
    
    def _format_search_results_for_llm(self, search_results: Dict[str, Any]) -> str:
        """Format search results for LLM processing"""
        formatted_results = []
        
        try:
            # Add main search results
            main_results = search_results.get("main_search", {})
            formatted_results.append("=== MAIN SEARCH RESULTS ===")
            
            if main_results.get("results"):
                for i, result in enumerate(main_results["results"], 1):
                    formatted_results.append(f"Result {i}:")
                    formatted_results.append(f"Title: {result.get('title', 'No title')}")
                    formatted_results.append(f"URL: {result.get('url', 'No URL')}")
                    formatted_results.append(f"Content: {result.get('content', 'No content')[:1000]}...")
                    
                    if result.get("raw_content"):
                        formatted_results.append(f"Additional Details: {result['raw_content'][:1000]}...")
                    
                    formatted_results.append(f"Score: {result.get('score', 'N/A')}")
                    formatted_results.append("-" * 50)
            else:
                formatted_results.append("No main search results found.")
            
            # Add LinkedIn results if available
            linkedin_results = search_results.get("linkedin_search", {})
            if linkedin_results.get("results"):
                formatted_results.append("\n=== LINKEDIN PROFILE RESULTS ===")
                for i, result in enumerate(linkedin_results["results"], 1):
                    formatted_results.append(f"LinkedIn Result {i}:")
                    formatted_results.append(f"Title: {result.get('title', 'No title')}")
                    formatted_results.append(f"URL: {result.get('url', 'No URL')}")
                    formatted_results.append(f"Content: {result.get('content', 'No content')[:500]}...")
                    formatted_results.append("-" * 30)
            else:
                formatted_results.append("\n=== LINKEDIN PROFILE RESULTS ===")
                formatted_results.append("No relevant LinkedIn profiles found.")
            
            # Add workplace website results if available  
            workplace_results = search_results.get("workplace_search", {})
            if workplace_results.get("results"):
                formatted_results.append("\n=== WORKPLACE WEBSITE RESULTS ===")
                for i, result in enumerate(workplace_results["results"], 1):
                    formatted_results.append(f"Workplace Result {i}:")
                    formatted_results.append(f"Title: {result.get('title', 'No title')}")
                    formatted_results.append(f"URL: {result.get('url', 'No URL')}")
                    formatted_results.append(f"Content: {result.get('content', 'No content')[:500]}...")
                    formatted_results.append("-" * 30)
            else:
                formatted_results.append("\n=== WORKPLACE WEBSITE RESULTS ===")
                formatted_results.append("No workplace directory results found.")
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error formatting search results: {str(e)}")
            return str(search_results)
    
    
    def search(self, doctor_info: Dict[str, Any]) -> Dict[str, Any]:
        """
        Search for healthcare professionals based on structured input
        
        Args:
            doctor_info: Dictionary containing structured doctor information
                        Expected keys: firstName, lastName, workplaceName, address, geographic_region
            
        Returns:
            Dictionary with search_results, tavily_answer, and llm_answer
            
        Raises:
            Exception: If search or processing fails
        """
        try:
            first_name = doctor_info.get("firstName", "")
            last_name = doctor_info.get("lastName", "")
            workplace = doctor_info.get("workplaceName", "")
            
            logger.info(f"Starting search for: Dr {first_name} {last_name} at {workplace}")
            
            # Initialize response structure
            response = {
                "search_results": {
                    "main_search": {"results": []},
                    "linkedin_search": {"results": []},
                    "workplace_search": {"results": []}
                },
                "tavily_answer": "",
                "llm_answer": ""
            }
            
            # Perform main search
            main_results = self._search_tavily_main(doctor_info)
            
            # Extract and format main search results
            response["search_results"]["main_search"]["results"] = [
                {
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0)
                } for result in main_results.get("results", [])
            ]
            
            # Extract Tavily answer from main search
            response["tavily_answer"] = main_results.get("answer", "")
            
            # Perform LinkedIn search
            linkedin_results = self._search_linkedin_profile(doctor_info)
            
            response["search_results"]["linkedin_search"]["results"] = [
                {
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0)
                } for result in linkedin_results.get("results", [])
            ]
            
            # Perform workplace search
            workplace_results = self._search_workplace_website(doctor_info)
            
            response["search_results"]["workplace_search"]["results"] = [
                {
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0)
                } for result in workplace_results.get("results", [])
            ]
            
            # Format search results for LLM
            formatted_results_for_llm = self._format_search_results_for_llm(response["search_results"])
            
            # Get LLM analysis using OpenAI
            try:
                logger.info("Processing results with OpenAI...")
                
                # Create comprehensive prompt for OpenAI
                prompt = f"""You are a helpful AI assistant specialized in analyzing healthcare professional search results.
Based on the search results provided, provide a comprehensive summary that includes:

1. **Main Search Summary:** Analysis of official records, academic documents, and institutional information
2. **LinkedIn Profile Summary:** Professional profile information and career details  
3. **Workplace Search Summary:** Additional workplace directories and institutional connections

Guidelines:
- Provide clear, factual summaries for each section
- Highlight key qualifications, positions, and affiliations
- Note any gaps in contact information or details
- Keep each section concise but informative
- If a search type has no relevant results, mention that briefly

Geographic Region: {self._get_country_name(doctor_info.get("geographic_region", ""))}
Query for: Dr {first_name} {last_name} at {workplace}

Search Results:
{formatted_results_for_llm}"""
                
                llm_response = extract_llm_response(prompt)
                response["llm_answer"] = llm_response.llm_final_answer
                
            except Exception as llm_error:
                logger.error(f"Error in LLM processing: {str(llm_error)}")
                response["llm_answer"] = f"Error processing search results with AI analysis: {str(llm_error)}"
            
            logger.info(f"Search completed successfully for Dr {first_name} {last_name}")
            return response
            
        except Exception as e:
            logger.error(f"Error in search operation: {str(e)}")
            # Return error response
            return {
                "search_results": {
                    "main_search": {"results": []},
                    "linkedin_search": {"results": []},
                    "workplace_search": {"results": []}
                },
                "tavily_answer": "",
                "llm_answer": f"Search operation failed: {str(e)}"
            }
            
    def _filter_relevant_results(self, results: Dict[str, Any], search_type: str, professional_name: str) -> Dict[str, Any]:
        """Filter only relevant results based on search type"""
        if not results.get("results"):
            return results
        
        filtered_results = []
        name_parts = professional_name.lower().split()
        
        for result in results["results"]:
            content = (result.get("content", "") + result.get("title", "")).lower()
            
            # Check if the professional's name appears in the content
            if any(name_part in content for name_part in name_parts):
                filtered_results.append(result)
                if len(filtered_results) >= 5:  # Keep top 3 relevant results
                    break
        
        results["results"] = filtered_results
        return results


# Convenience function for direct import and use
def search_healthcare_professionals(
    doctor_info: Dict[str, Any],
    tavily_api_key: Optional[str] = None
) -> Dict[str, Any]:
    """
    Convenience function to search for healthcare professionals
    
    Args:
        doctor_info: Structured dictionary with doctor information
        tavily_api_key: Optional Tavily API key
        
    Returns:
        Dictionary with search results, tavily answer, and llm analysis
    """
    agent = WebSearchAgent(tavily_api_key)
    return agent.search(doctor_info)