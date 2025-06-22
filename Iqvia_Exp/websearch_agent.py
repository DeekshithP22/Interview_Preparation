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


class LLMExtractionResponse(BaseModel):
    """Model for LLM to extract healthcare professionals from search results"""
    professionals: List[HealthcareProfessional] = Field(
        description="List of healthcare professionals found in the search results"
    )
    total_results: int = Field(description="Total number of healthcare professionals found")


class WebSearchAgent:
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
        
        logger.info("WebSearchAgent initialized successfully")
    
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
            enhanced_query = f"{query} in {country}"
            
            logger.info(f"Searching Tavily for: {enhanced_query}")
            
            response = self.tavily_client.search(
                query=enhanced_query,
                search_depth="advanced",
                max_results=max_results,
                include_answer=True,
                include_raw_content=True,
                include_domains=["healthcare.gov", "healthgrades.com", "doctor.com", 
                               "zocdoc.com", "vitals.com", "webmd.com", "mayo.edu", 
                               "hospital-websites.com", ".gov", ".edu", ".org"]
            )
            
            logger.info(f"Tavily search completed. Found {len(response.get('results', []))} results")
            return response
            
        except Exception as e:
            logger.error(f"Error in Tavily search: {str(e)}")
            raise
    
    def _extract_urls_from_results(self, search_results: Dict[str, Any]) -> List[str]:
        """Extract all URLs from search results"""
        urls = []
        
        try:
            # Extract URLs from results
            if "results" in search_results:
                for result in search_results["results"]:
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
            # Add Tavily's AI answer if available
            if search_results.get("answer"):
                formatted_results.append(f"AI Summary: {search_results['answer']}")
                formatted_results.append("-" * 50)
            
            # Add individual search results
            if search_results.get("results"):
                for i, result in enumerate(search_results["results"], 1):
                    formatted_results.append(f"Result {i}:")
                    formatted_results.append(f"Title: {result.get('title', 'No title')}")
                    formatted_results.append(f"URL: {result.get('url', 'No URL')}")
                    formatted_results.append(f"Content: {result.get('content', 'No content')[:1000]}...")
                    
                    # Add raw content if available
                    if result.get("raw_content"):
                        formatted_results.append(f"Additional Details: {result['raw_content'][:1000]}...")
                    
                    formatted_results.append(f"Score: {result.get('score', 'N/A')}")
                    formatted_results.append("-" * 50)
            
            return "\n".join(formatted_results)
            
        except Exception as e:
            logger.error(f"Error formatting search results: {str(e)}")
            return str(search_results)
    
    def search(self, requirements: str, country: str = "United States") -> SearchResponse:
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
            
            # Perform Tavily search
            search_results = self._search_tavily(requirements, country)
            
            # Extract URLs from results
            source_urls = self._extract_urls_from_results(search_results)
            logger.info(f"Extracted {len(source_urls)} unique URLs from search results")
            
            # Format search results for LLM
            formatted_results = self._format_search_results(search_results)
            
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
                sources_used=source_urls
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
                sources_used=[]
            )


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
    agent = WebSearchAgent(tavily_api_key, google_api_key)
    return agent.search(requirements, country)


# Example usage
if __name__ == "__main__":
    # Example searches
    queries = [
        ("Dr Likhitha P, Paediatrician, AIIMS Delhi", "India")
    ]
    
    # Initialize agent
    agent = WebSearchAgent()
    
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