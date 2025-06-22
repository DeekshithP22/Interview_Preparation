"""
Healthcare Professional Search Agent using Tavily and LangChain
This module provides a search function to find healthcare professionals
in a specific country using Tavily API and structured output.
"""

import os
import json
import logging
from typing import List, Dict, Any, Optional
from pydantic import BaseModel, Field
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
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


# Pydantic models for structured output
class ContactInfo(BaseModel):
    """Contact information for a healthcare professional"""
    phone: Optional[str] = Field(default=None, description="Phone number")
    email: Optional[str] = Field(default=None, description="Email address")
    website: Optional[str] = Field(default=None, description="Website URL")


class Address(BaseModel):
    """Address information for a healthcare professional"""
    street: Optional[str] = Field(default=None, description="Street address")
    city: Optional[str] = Field(default=None, description="City")
    state: Optional[str] = Field(default=None, description="State or province")
    postal_code: Optional[str] = Field(default=None, description="Postal/ZIP code")
    country: str = Field(description="Country")
    full_address: Optional[str] = Field(default=None, description="Complete formatted address")


class HealthcareProfessional(BaseModel):
    """Structured information about a healthcare professional"""
    name: str = Field(description="Full name of the healthcare professional")
    specialty: str = Field(description="Medical specialty or field of expertise")
    qualifications: Optional[List[str]] = Field(
        default=None, 
        description="List of qualifications, degrees, certifications"
    )
    address: Address = Field(description="Practice address")
    contact: ContactInfo = Field(description="Contact information")
    hospital_affiliations: Optional[List[str]] = Field(
        default=None,
        description="List of affiliated hospitals or medical centers"
    )


class SearchResponse(BaseModel):
    """Response model for the search operation"""
    query: str = Field(description="The original search query")
    country: str = Field(description="Country searched in")
    professionals: List[HealthcareProfessional] = Field(
        description="List of healthcare professionals found"
    )
    total_results: int = Field(description="Total number of results found")
    sources_used: List[str] = Field(default_factory=list, description="List of source URLs used")
    search_results: List[Dict[str, Any]] = Field(default_factory=list, description="Individual search results from each source")



class LLMExtractionResponse(BaseModel):
    """Model for LLM to extract healthcare professionals from search results"""
    professionals: List[HealthcareProfessional] = Field(
        description="List of healthcare professionals found in the search results"
    )
    total_results: int = Field(description="Total number of healthcare professionals found")


class HealthcareSearchAgent:
    """Agent for searching healthcare professionals using Tavily API"""
    
    def __init__(self, tavily_api_key: Optional[str] = None, google_api_key: Optional[str] = None):
        """
        Initialize the search agent
        
        Args:
            tavily_api_key: Tavily API key (defaults to env variable)
            google_api_key: Google API key for Gemini (defaults to env variable)
        """
        # Initialize Tavily client
        self.tavily_api_key = tavily_api_key or os.getenv("TAVILY_API_KEY")
        if not self.tavily_api_key:
            raise ValueError("Tavily API key not provided. Set TAVILY_API_KEY environment variable.")
        
        self.tavily_client = TavilyClient(api_key=self.tavily_api_key)
        
        # Initialize LLM
        self.google_api_key = google_api_key or os.getenv("GEMINI_API_KEY")
        if not self.google_api_key:
            raise ValueError("Google API key not provided. Set GEMINI_API_KEY environment variable.")
        
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-1.5-flash",
            api_key=self.google_api_key,
            max_output_tokens=8192,
            temperature=0.1
        ).with_structured_output(LLMExtractionResponse)
        
        # Create prompt template
        self.prompt_template = ChatPromptTemplate.from_messages([
            ("system", """You are a helpful AI assistant specialized in finding healthcare professionals.
            Based on the search results provided, extract and structure information about healthcare professionals.
            
            Guidelines:
            1. Extract detailed information about each healthcare professional mentioned
            2. Parse addresses, contact information, and specialties carefully
            3. Include all relevant qualifications and affiliations
            4. If information is not available for a field, leave it as null
            5. Ensure all information is accurate based on the search results
            6. Structure the response according to the required format
            
            Country context: {country}
            
            Search Results:
            {search_results}
            """),
            ("human", "Query: {query}")
        ])
        
        logger.info("HealthcareSearchAgent initialized successfully")
    
    def _search_tavily(self, query: str, country: str, max_results: int = 10) -> Dict[str, Any]:
        """
        Perform Tavily search with country-specific query
        
        Args:
            query: Search query
            country: Country to search in
            max_results: Maximum number of results
            
        Returns:
            Dictionary containing search results
        """
        try:
            # Enhance query with country information
            enhanced_query = f"Need details about the doctor, {query}, contact details as well if there is any"
            
            logger.info(f"Searching Tavily for: {enhanced_query}")
            
            response = self.tavily_client.search(
                query=enhanced_query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=True,
                country=country,
                # include_domains=["healthcare.gov", "healthgrades.com", "doctor.com", 
                #                "zocdoc.com", "vitals.com", "webmd.com", "mayo.edu", 
                #                "hospital-websites.com", ".gov", ".edu", ".org"]
            )
            
            logger.info(f"Tavily search completed. Found {len(response.get('results', []))} results")
            return response
            
        except Exception as e:
            logger.error(f"Error in Tavily search: {str(e)}")
            raise
        
        
    def _search_linkedin_profile(self, professional_name: str, workplace: str, specialty: str, location: str, country:str) -> Dict[str, Any]:
        """Search for LinkedIn profile of a healthcare professional"""
        try:
            linkedin_query = f'site:linkedin.com/in/ "{professional_name}" "{workplace}" {specialty} {location}'
            
            logger.info(f"Searching LinkedIn for: {linkedin_query}")
            
            response = self.tavily_client.search(
                query=linkedin_query,
                search_depth="advanced",
                max_results=2,
                include_answer=True,
                include_raw_content=True,
                country =country
            )
            
            response = self._filter_relevant_results(response, "linkedin", professional_name)


            return {
                "search_type": "linkedin",
                "query": linkedin_query,
                "results": response
            }
        except Exception as e:
            logger.error(f"Error in LinkedIn search: {str(e)}")
            return {"search_type": "linkedin", "query": "", "results": {}}

    def _search_workplace_website(self, professional_name: str, workplace_name: str, specialty: str, location: str) -> Dict[str, Any]:
        """Search for doctor's information on workplace website"""
        try:
            workplace_query = f'"{professional_name}" "{workplace_name}" {location} {specialty} profile OR directory OR staff'
            
            logger.info(f"Searching for doctor on workplace website: {workplace_query}")
            
            response = self.tavily_client.search(
                query=workplace_query,
                search_depth="advanced",
                max_results=3,
                include_answer=True,
                include_raw_content=True,
                country= country
            )
            
            return {
                "search_type": "workplace",
                "query": workplace_query,
                "results": response
            }
        except Exception as e:
            logger.error(f"Error in workplace website search: {str(e)}")
            return {"search_type": "workplace", "query": "", "results": {}}
        
    
    def _extract_urls_from_results(self, search_results: Dict[str, Any]) -> List[str]:
        """Extract all URLs from search results"""
        urls = []
        
        try:
            # Extract from main results
            main_results = search_results.get("main_results", search_results)
            if "results" in main_results:
                for result in main_results["results"]:
                    if "url" in result and result["url"]:
                        urls.append(result["url"])
            
            # Extract from LinkedIn results
            if "linkedin_results" in search_results and "results" in search_results["linkedin_results"]:
                for result in search_results["linkedin_results"]["results"]:
                    if "url" in result and result["url"]:
                        urls.append(result["url"])
            
            # Extract from workplace results
            if "workplace_results" in search_results and "results" in search_results["workplace_results"]:
                for result in search_results["workplace_results"]["results"]:
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
    
    
    
    def _format_search_results(self, search_results: Dict[str, Any]) -> str:
        """Format search results for LLM processing"""
        formatted_results = []
        
        try:
            # Handle main results
            main_results = search_results.get("main_results", search_results)
            
            # Add main search results
            formatted_results.append("=== MAIN SEARCH RESULTS ===")
            if main_results.get("answer"):
                formatted_results.append(f"AI Summary: {main_results['answer']}")
                formatted_results.append("-" * 50)
            
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
            
            # Add LinkedIn results if available
            if "linkedin_results" in search_results and search_results["linkedin_results"].get("results"):
                formatted_results.append("\n=== LINKEDIN PROFILE RESULTS ===")
                for i, result in enumerate(search_results["linkedin_results"]["results"], 1):
                    formatted_results.append(f"LinkedIn Result {i}:")
                    formatted_results.append(f"URL: {result.get('url', 'No URL')}")
                    formatted_results.append(f"Content: {result.get('content', 'No content')[:500]}...")
                    formatted_results.append("-" * 30)
            
            # Add workplace website results if available  
            if "workplace_results" in search_results and search_results["workplace_results"].get("results"):
                formatted_results.append("\n=== WORKPLACE WEBSITE RESULTS ===")
                for i, result in enumerate(search_results["workplace_results"]["results"], 1):
                    formatted_results.append(f"Workplace Result {i}:")
                    formatted_results.append(f"URL: {result.get('url', 'No URL')}")
                    formatted_results.append(f"Content: {result.get('content', 'No content')[:500]}...")
                    formatted_results.append("-" * 30)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error formatting search results: {str(e)}")
            return str(search_results)
    
    
    
    def search(self, requirements: str, country: str = "italy") -> SearchResponse:
        """
        Search for healthcare professionals based on requirements and country
        
        Args:
            requirements: Search requirements/query (e.g., "cardiologist specialized in heart surgery")
            country: Country to search in (default: "United States")
            
        Returns:
            SearchResponse object with structured results
            
        Raises:
            Exception: If search or processing fails
        """
        try:
            logger.info(f"Starting search for: {requirements} in {country}")
            
            # Parse requirements
            parts = [part.strip() for part in requirements.split(',')]
            professional_name = parts[0] if len(parts) > 0 else requirements
            specialty = parts[1] if len(parts) > 1 else "healthcare professional"
            workplace = parts[2] if len(parts) > 2 else ""
            
            # Clean name for search
            name_for_search = professional_name.replace("Dr.", "").replace("Dr", "").strip()
            
            # List to store all search results
            all_search_results_list = []
            
            # Perform main Tavily search
            search_results = self._search_tavily(requirements, country)
            all_search_results_list.append({
                "search_type": "main",
                "query": f"{requirements}",
                "results": search_results
            })
            
            # Perform LinkedIn search
            if name_for_search:
                linkedin_results = self._search_linkedin_profile(
                    name_for_search,
                    workplace,
                    specialty,
                    country
                )
                all_search_results_list.append(linkedin_results)
            
            # Perform workplace website search
            if workplace and name_for_search:
                workplace_results = self._search_workplace_website(
                    name_for_search,
                    workplace,
                    specialty,
                    country
                )
                all_search_results_list.append(workplace_results)
            
            # Combine all results for processing
            all_search_results = {
                "main_results": search_results,
                "linkedin_results": linkedin_results.get("results", {}) if name_for_search else {},
                "workplace_results": workplace_results.get("results", {}) if workplace and name_for_search else {}
            }
            
            # Extract URLs from all results
            source_urls = self._extract_urls_from_results(all_search_results)
            logger.info(f"Extracted {len(source_urls)} unique URLs from search results")
            
            # Format search results for LLM
            formatted_results = self._format_search_results(all_search_results)
            
            # Create prompt and get structured response from LLM
            messages = self.prompt_template.format_messages(
                country=country,
                search_results=formatted_results,
                query=requirements
            )
            
            logger.info("Processing results with LLM...")
            llm_response = self.llm.invoke(messages)
            
            # Create the final SearchResponse object
            final_response = SearchResponse(
                query=requirements,
                country=country,
                professionals=llm_response.professionals,
                total_results=llm_response.total_results,
                sources_used=source_urls,
                search_results=all_search_results_list  # Add the list of all search results
            )
            
            logger.info(f"Search completed successfully. Found {len(final_response.professionals)} professionals")
            return final_response
            
        except Exception as e:
            logger.error(f"Error in search operation: {str(e)}")
            # Return error response
            return SearchResponse(
                query=requirements,
                country=country,
                professionals=[],
                total_results=0,
                sources_used=[],
                search_results=[]
            )
            
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
                    if len(filtered_results) >= 2:  # Keep only top 2 relevant results
                        break
            
            results["results"] = filtered_results
            return results

# Convenience function for direct import and use
def search_healthcare_professionals(
    requirements: str, 
    country: str = "United States",
    tavily_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None
) -> SearchResponse:
    """
    Convenience function to search for healthcare professionals
    
    Args:
        requirements: Search requirements/query
        country: Country to search in
        tavily_api_key: Optional Tavily API key
        google_api_key: Optional Google API key
        
    Returns:
        SearchResponse object with structured results
    """
    agent = HealthcareSearchAgent(tavily_api_key, google_api_key)
    return agent.search(requirements, country)





# Example usage
if __name__ == "__main__":
    # Example searches
    queries = [
        ("Dr Likhitha P, Paediatrician, AIIMS Delhi", "India")
    ]
    
    # Initialize agent
    agent = HealthcareSearchAgent()
    
    for requirements, country in queries:
        print(f"\n{'='*80}")
        print(f"Searching for: {requirements} in {country}")
        print('='*80)
        
        try:
            result = agent.search(requirements, country)
            struct_result = (json.dumps(result.model_dump(), indent=2))
            
            # print(f"\nFound {result.total_results} professionals:")
            # for i, prof in enumerate(result.professionals, 1):
            #     print(f"\n{i}. {prof.name}")
            #     print(f"   Specialty: {prof.specialty}")
            #     print(f"   Location: {prof.address.city}, {prof.address.country}")
            #     if prof.contact.phone:
            #         print(f"   Phone: {prof.contact.phone}")
            #     if prof.hospital_affiliations:
            #         print(f"   Affiliations: {', '.join(prof.hospital_affiliations)}")
            
            print(struct_result)
            
            print(f"\nSources used: {len(result.sources_used)}")
            for url in result.sources_used:  # Show first 3 URLs
                print(f"  - {url}")
            
        except Exception as e:
            print(f"Error: {str(e)}")