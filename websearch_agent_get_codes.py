"""
Enhanced Web Search Agent for IQVIA Healthcare Professional Validation
Complete working code with 2-step workplace strategy, exact validation, and clean output - ASYNC VERSION
"""

import json
import logging
import os
import re
import asyncio
from typing import List, Dict, Any, Optional
import yaml

from tavily import TavilyClient
from dotenv import load_dotenv
from urllib.parse import urlparse

from azure.identity import ClientSecretCredential, get_bearer_token_provider
from langchain_openai import AzureChatOpenAI

# from app.config import Config
from app.my_agent.prompts.search_summarize_prompts import WEBSEARCH_PROMPT, WEBSEARCH_PROMPT_ANALYSIS
from app.my_agent.utils.models import ValidationResult

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)


def load_specialty_mapping(region: str = "IT") -> Dict[str, str]:
    """Load specialty code mapping from JSON file based on region"""
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Path to config directory (websearch_config is in same folder as this script)
    config_dir = os.path.join(current_dir, "websearch_config", "specialty_mappings")
    
    # Map region to JSON file names
    json_files = {
        "IT": "GET_CODES_ITALY_SP.json",
        "FR": "GET_CODES_FRANCE_SP.json",
        "US": "GET_CODES_US_SP.json",
        "UK": "GET_CODES_UK_SP.json",
    }
    
    filename = json_files.get(region, json_files["IT"])
    json_file_path = os.path.join(config_dir, filename)
    
    print(f" Looking for specialty mapping at: {json_file_path}")
    
    try:
        with open(json_file_path, "r", encoding="utf-8") as file:
            data = json.load(file)
            
        # Extract specialty mapping from the JSON structure
        specialty_mapping = {}
        
        if "response" in data and isinstance(data["response"], dict) and "response" in data["response"]:
            for specialty in data["response"]["results"]:
                eid = specialty.get("eid", "")
                long_localized_label = specialty.get("longLocalizedLabel", "")
                
                if eid and long_localized_label:
                    specialty_mapping[eid] = long_localized_label
        
        print(f" Loaded {len(specialty_mapping)} specialties for region {region}")
        return specialty_mapping
        
    except FileNotFoundError:
        print(f" Specialty mapping file {json_file_path} not found. Using empty mapping.")
        return {}
    except json.JSONDecodeError:
        print(f" Invalid JSON in specialty mapping file {json_file_path}")
        return {}
    except Exception as e:
        print(f" Error loading specialty mapping: {str(e)}")
        return {}

# ASYNC WRAPPER CLASSES
class AsyncTavilyWrapper:
    """Async wrapper for Tavily search operations"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.tavily_client = TavilyClient(api_key=api_key)
        
    async def search_async(self, **kwargs):
        """Async wrapper for Tavily search"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.tavily_client.search(**kwargs))

class AsyncLLMWrapper:
    """Async wrapper for LLM operations"""
    
    def __init__(self, llm):
        self.llm = llm
        
    async def invoke_async(self, prompt):
        """Async wrapper for LLM invoke"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, lambda: self.llm.invoke(prompt))

# class GeographicConfig:
class GeographicConfig:
    """Manages geographic configuration from YAML file"""
    
    _config = None
    
    @classmethod
    def get_config(cls) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        if cls._config is None:
            cls._config = cls._load_config()
        return cls._config
    

    @classmethod
    def _load_config(cls) -> Dict[str, Any]:
        """Load configuration from YAML file"""
        try:
            # Get the directory where this script is located
            current_dir = os.path.dirname(os.path.abspath(__file__))
            
            # Path to config file (websearch_config is in same folder as this script)
            config_file = os.path.join(current_dir, "websearch_config", "geographic_config.yaml")
            
            logger.info(f" Looking for geographic config at: {config_file}")
            
            with open(config_file, "r", encoding="utf-8") as file:
                config = yaml.safe_load(file)

            logger.info(f" Loaded geographic configuration from {config_file}")
            return config
            
        except FileNotFoundError:
            logger.warning(f" Geographic config file not found. Using default config.")
            return cls._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f" Error parsing YAML config: {e}. Using default config.")
            return cls._get_default_config()
        except Exception as e:
            logger.error(f" Error loading geographic config: {e}. Using default config.")
            return cls._get_default_config()
    
    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """Fallback default configuration"""
        return {
            "IT": {
                "region_name": "Italy",
                "professional_terms": ["medico", "dottore", "specialista"],
                "professional_domains": ["ospedale", "clinica"],
                "medical_institutions": [".it"],
            }
        }
        
        
        
class TavilyPayloadBuilder:
    """Dynamic Tavily API payload builder with intelligent configuration"""
    
    def __init__(self):
        self.geo_config = GeographicConfig.get_config()
        
    def build_hcp_main_payload(
        self, doctor_info: Dict[str, Any], specialty_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build optimized payload for HCP main search - NATURAL LANGUAGE APPROACH"""
        region = doctor_info.get("geographic_region", "")
        config = self.geo_config.get(region, self.geo_config["IT"])
        
        first_name = doctor_info.get("firstName", "")
        last_name = doctor_info.get("lastName", "")
        workplace = doctor_info.get("workplaceName", "")
        address = doctor_info.get("address", "")
        
        # NATURAL LANGUAGE QUERY
        if specialty_name:
            query = (
                f"Find information about a healthcare professional named {first_name} {last_name}, "
                f"who specializes in {specialty_name}, currently working at {workplace}, located at {address}."
            )
        else:
            query = f"Find information about a healthcare professional named {first_name} {last_name},working at {workplace}, {address}"
        
        if specialty_name:
            query = (
                f"Find information about a healthcare professional named {first_name} {last_name}, "
                f"who specializes in {specialty_name}, currently working at {workplace}, located at {address}."
            )
        else:
            query = (
                f"Find information about a healthcare professional named {first_name} {last_name}, "
                f"currently working at {workplace}, located at {address}."
            )
            
            
        # Include professional terms and domains
        include_domains = config["professional_domains"].copy()
        include_domains.extend(config["institution_domains"])
        
        return {
            "query": query,
            "search_depth": "advanced",
            "max_results": 10,
            "chunks_per_source": 2,
            "include_answer": True,
            "country": config["country"],
            # "include_raw_content": True,
            # "time_range": "year",
            # "include_domains": include_domains,
            # "exclude_domains": config.get("exclude_domains", []),
        }
    
    def build_linkedin_payload(
        self, doctor_info: Dict[str, Any], specialty_name: Optional[str] = None
    ) -> Dict[str, Any]:
        """Build optimized payload for LinkedIn search - TARGETED LINKEDIN SEARCH"""
        region = doctor_info.get("geographic_region", "")
        config = self.geo_config.get(region, self.geo_config["IT"])
        
        first_name = doctor_info.get("firstName", "")
        last_name = doctor_info.get("lastName", "")
        workplace = doctor_info.get("workplaceName", "")
        
        # TARGETED LINKEDIN QUERY
        if specialty_name:
            query = (
                f"LinkedIn profile of healthcare professional {first_name} {last_name}, "
                f"specialized in {specialty_name}, working at {workplace}"
            )
        else:
            query = f"LinkedIn profile of Healthcare Professional {first_name} {last_name} {workplace}"
        
        if specialty_name:
            query = (
                f"LinkedIn profile of healthcare professional {first_name} {last_name}, "
                f"specialized in {specialty_name}, working at {workplace}"
            )
        else:
            query = f"LinkedIn profile of healthcare professional {first_name} {last_name}, " + f"working at {workplace}"
        
        return {
            "query": query,
            "search_depth": "advanced",
            "max_results": 10,
            "chunks_per_source": 2,
            "include_answer": True,
            "country": config["country"],
            "include_domains": ["linkedin.com/in"],
            "exclude_domains": ["linkedin.com/jobs", "linkedin.com/learning"],
            # "include_raw_content": True,
            # "time_range": "year",
        }
    
    def build_workplace_payload(self, doctor_info: Dict[str, Any]) -> Dict[str, Any]:
        """Build optimized payload for workplace search - SMART SITE OPERATOR STRATEGY"""
        region = doctor_info.get("geographic_region", "")
        config = self.geo_config.get(region, self.geo_config["IT"])
        
        first_name = doctor_info.get("firstName", "")
        last_name = doctor_info.get("lastName", "")
        workplace = doctor_info.get("workplaceName", "")
        
        query = f"{first_name} {last_name} {workplace} site:.org OR site:.edu OR site:.gov"
        
        if region == "IT":
            query = f"{first_name} {last_name} {workplace} site:.it OR site:.org"
        elif region == "UK":
            query = f"{first_name} {last_name} {workplace} site:.ac.uk OR site:.nhs.uk OR site:.gov.uk"
        elif region == "DE":
            query = f"{first_name} {last_name} {workplace} site:.de OR site:.org"
        elif region == "FR":
            query = f"{first_name} {last_name} {workplace} site:.fr OR site:.org"
        
        return {
            "query": query,
            "search_depth": "advanced",
            "max_results": 5,
            "chunks_per_source": 2,
            "include_answer": True,
            "country": config["country"],
            # "include_domains": config.get("institution_domains", []),
            # "exclude_domains": config.get("exclude_domains", []),
            # "include_raw_content": True,
            # "time_range": "year",
        }
    
    def build_workplace_validation_payload(self, workplace_info: Dict[str, Any]) -> Dict[str, Any]:
        """STEP 1: Find official website first - OFFICIAL SITE FINDER"""
        region = workplace_info.get("geographic_region", "")
        config = self.geo_config.get(region, self.geo_config["US"])
        
        workplace = workplace_info.get("workplaceName", "")
        
        # STEP 1: Find official website first
        query = f"{workplace} official site"
        
        return {
            "query": query,
            "search_depth": "advanced",
            "max_results": 5,
            "chunks_per_source": 2,
            "include_answer": True,
            "country": config["country"],
            "exclude_domains": config["exclude_domains"] + [".com/reviews"],
            # "include_raw_content": True,
            # "time_range": "year",
            # "include_domains": config.get("institution_domains", [] + config.get("professional_domains", [])),
        }
    
    def build_targeted_workplace_payload(self, doctor_info: Dict[str, Any], official_domain: str) -> Dict[str, Any]:
        """STEP 2: Search person on specific official workplace domain"""
        region = doctor_info.get("geographic_region", "")
        config = self.geo_config.get(region, self.geo_config["US"])
        
        first_name = doctor_info.get("firstName", "")
        last_name = doctor_info.get("lastName", "")
        
        # STEP 2: Search person on specific official workplace domain found in STEP 1
        query = f"{first_name} {last_name} site:{official_domain}"
        
        return {
            "query": query,
            "search_depth": "advanced",
            "max_results": 5,
            "chunks_per_source": 2,
            "include_answer": True,
            "country": config["country"],
        }


class ResultValidator:
    """Advanced result validation with EXACT matching"""
    
    def __init__(self, geographic_config: Dict[str, Any], async_llm: Optional[Any] = None):
        self.geo_config = geographic_config
    
    def calculate_exact_name_match(self, content: str, first_name: str, last_name: str) -> bool:
        """EXACT name matching - name must be present exactly in content"""
        content_lower = content.lower()
        first_lower = first_name.lower().strip()
        last_lower = last_name.lower().strip()
        
        # Check for exact full name
        full_name = f"{first_lower} {last_lower}"
        if full_name in content_lower:
            return True
        
        # Check for exact individual names (both must be present)
        first_present = first_lower in content_lower
        last_present = last_lower in content_lower
        
        # Both names must be present exactly
        return first_present and last_present
    
    def validate_exact_city_match(self, content: str, address: str) -> bool:
        """EXACT city matching - city must be present exactly in content"""
        if not address:
            return True  # No city to validate
        
        content_lower = content.lower()
        address_lower = address.lower().strip()
        
        # Direct exact city match
        return address_lower in content_lower
    
    def validate_exact_workplace_match(self, content: str, workplace: str) -> bool:
        """EXACT workplace matching - workplace name must be present exactly in content"""
        if not workplace:
            return True  # No workplace to validate
        
        content_lower = content.lower()
        workplace_lower = workplace.lower().strip()
        
        # Direct exact workplace match
        if workplace_lower in content_lower:
            return True
        
        # Check for key terms from workplace name (at least 70% of significant terms must match exactly)
        workplace_terms = [term for term in workplace_lower.split() if len(term) >= 4]
        if len(workplace_terms) >= 2:
            exact_matches = sum(1 for term in workplace_terms if term in content_lower)
            return exact_matches >= (len(workplace_terms) * 0.7)  # 70% of terms must match exactly
        
        return False
    
    def validate_geographic_match(self, url: str, content: str, region: str) -> bool:
        """Validate geographic relevance"""
        config = self.geo_config.get(region, {})
        
        # Domain-based validation
        domain = urlparse(url).netloc.lower()
        
        # Check professional domains
        professional_domains = config.get("professional_domains", [])
        if any(prof_domain in domain for prof_domain in professional_domains):
            return True
        
        # Check institution domains
        institution_domains = config.get("institution_domains", [])
        if any(domain.endswith(inst_domain) for inst_domain in institution_domains):
            return True
        
        # Content-based geographic validation
        content_lower = content.lower()
        region_terms = {
            "IT": ["italy", "italia", "italian", "milano", "milan", "rome", "roma"],
            "US": ["united states", "usa", "american", "america"],
            "UK": ["united kingdom", "uk", "british", "england", "london"],
        }
        
        terms = region_terms.get(region, [])
        return any(term in content_lower for term in terms)
    
    def validate_professional_context(self, content: str, region: str, specialty: Optional[str] = None) -> bool:
        """Validate professional medical context"""
        config = self.geo_config.get(region, {})
        content_lower = content.lower()
        
        # Check professional terms
        professional_terms = config.get("professional_terms", [])
        professional_match = any(term.lower() in content_lower for term in professional_terms)
        
        # Medical context terms
        medical_terms = [
            "medical",
            "medicine",
            "health",
            "hospital",
            "clinic",
            "patient",
            "treatment",
            "surgery",
            "research",
            "clinical",
            "oncologist",
            "radiation",
            "radiotherapy",
            "neurological",
            "neurosurgery",
            "radioterapia",
            "neurologico",
        ]
        medical_match = any(term in content_lower for term in medical_terms)
        
        return professional_match or medical_match
    
    def _fallback_medical_check(self, content: str) -> bool:
        """Fallback keyword-based medical validation"""
        content_lower = content.lower()
        basic_medical_terms = [
            "medical",
            "medicine",
            "doctor",
            "physician",
            "hospital",
            "clinic",
            "medico",
            "dottore",
            "medicina",
            "ospedale",
            "clinica",
            "radiation",
            "oncologist",
            "neurosurgeon",
            "chirurgo",
            "specialista",
        ]
        result = any(term in content_lower for term in basic_medical_terms)
        logger.info(f"Fallback medical validation: {result}")
        return result
        
    def validate_url(self, result: Dict[str, Any], search_input: Dict[str, Any]) -> ValidationResult:
            """EXACT validation with differentiated logic for HCP vs Workplace"""
            url = result.get("url", "")
            title = result.get("title", "")
            content = result.get("content", "")
            
            # Combine all content for analysis
            full_content = f"{title} {content}"
            
            # Get search parameters
            entity_type = search_input.get("entity_type", "")
            first_name = search_input.get("firstName", "")
            last_name = search_input.get("lastName", "")
            workplace = search_input.get("workplaceName", "")
            region = search_input.get("geographic_region", "")
            address = search_input.get("address", "")
            
            validation_reasons = []
            
            # ENHANCED LOGIC: Differentiate between HCP search and Workplace search
            if entity_type == "ent_activity" or (first_name and last_name):
                # HCP validation - EXACT name and city, WORKPLACE CONTEXT REQUIRED
                exact_name_match = self.calculate_exact_name_match(full_content, first_name, last_name)
                
                if not exact_name_match:
                    # If exact name doesn't match, REJECT immediately
                    return ValidationResult(
                        url=url,
                        is_valid=False,
                        confidence_score=0.0,
                        validation_reasons=[f"EXACT name match FAILED for {first_name} {last_name} - REJECTED"],
                        geographic_match=False,
                        name_match=False,
                        workplace_match=False,
                    )
                
                validation_reasons.append(f"EXACT name match confirmed: {first_name} {last_name}")
                
                # EXACT city matching - REQUIRED if city provided
                exact_city_match = self.validate_exact_city_match(full_content, address)
                
                if address and not exact_city_match:
                    # If city is provided but doesn't match exactly, REJECT
                    return ValidationResult(
                        url=url,
                        is_valid=False,
                        confidence_score=0.2,
                        validation_reasons=[f"EXACT city match FAILED for {address} - REJECTED"],
                        geographic_match=False,
                        name_match=True,
                        workplace_match=False,
                    )
                
                if exact_city_match and address:
                    validation_reasons.append(f"EXACT city match confirmed: {address}")
                
                # FOR HCP SEARCH: Workplace context is REQUIRED (person should be associated with workplace)
                workplace_match = self.validate_exact_workplace_match(full_content, workplace)
                if workplace_match:
                    validation_reasons.append("Workplace context confirmed: Person associated with workplace")
                else:
                    # For HCP search, we need SOME workplace context (more flexible)
                    workplace_keywords = workplace.lower().split() if workplace else []
                    workplace_context = any(
                        keyword in full_content.lower() for keyword in workplace_keywords if len(keyword) >= 4
                    )
                    if workplace_context:
                        validation_reasons.append("Workplace context partially confirmed")
                        workplace_match = True
                    else:
                        validation_reasons.append("Workplace context missing (acceptable if strong professional context)")
                
                # Professional context - REQUIRED for HCP
                professional_context = self.validate_professional_context(full_content, region)
                
                if not professional_context:
                    return ValidationResult(
                        url=url,
                        is_valid=False,
                        confidence_score=0.3,
                        validation_reasons=["Professional medical context MISSING - REJECTED"],
                        geographic_match=False,
                        name_match=True,
                        workplace_match=workplace_match,
                    )
                
                validation_reasons.append("Professional medical context confirmed")
                
                # HCP validation criteria: Name + City + Professional context (workplace optional but preferred)
                is_valid = (
                    exact_name_match  # EXACT name match REQUIRED
                    and (not address or exact_city_match)  # EXACT city match REQUIRED (if city provided)
                    and professional_context  # Professional context REQUIRED
                )
                
                confidence_score = 1.0 if (is_valid and workplace_match) else 0.8  # Higher confidence with workplace match
                
            elif entity_type == "ent_workplace":
                # WORKPLACE VALIDATION - DIFFERENT LOGIC: Looking for workplace itself, not individuals
                exact_workplace_match = self.validate_exact_workplace_match(full_content, workplace)
                
                if not exact_workplace_match:
                    # If exact workplace doesn't match, REJECT immediately
                    return ValidationResult(
                        url=url,
                        is_valid=False,
                        confidence_score=0.0,
                        validation_reasons=[f"EXACT workplace match FAILED for {workplace} - REJECTED"],
                        geographic_match=False,
                        name_match=False,
                        workplace_match=False,
                    )
                
                validation_reasons.append(f"EXACT workplace match confirmed: {workplace}")
                
                # EXACT city matching for workplace - REQUIRED
                exact_city_match = self.validate_exact_city_match(full_content, address)
                
                if address and not exact_city_match:
                    # If city is provided but doesn't match exactly, REJECT
                    return ValidationResult(
                        url=url,
                        is_valid=False,
                        confidence_score=0.2,
                        validation_reasons=[f"EXACT city match FAILED for {address} - REJECTED"],
                        geographic_match=False,
                        name_match=False,
                        workplace_match=True,
                    )
                
                if exact_city_match and address:
                    validation_reasons.append(f"EXACT city match confirmed: {address}")
                
                # WORKPLACE validation: Only workplace + city required (no individual names)
                is_valid = exact_workplace_match and (  # EXACT workplace match REQUIRED
                    not address or exact_city_match
                )  # EXACT city match REQUIRED (if city provided)
                
                confidence_score = 1.0  # Full confidence if exact workplace and city match
                exact_name_match = False  # Not applicable for workplace validation
                workplace_match = exact_workplace_match
                
            else:
                # FALLBACK: Default to HCP validation
                return self.validate_url(result, {**search_input, "entity_type": "ent_activity"})
            
            return ValidationResult(
                url=url,
                is_valid=is_valid,
                confidence_score=confidence_score,
                validation_reasons=validation_reasons,
                geographic_match=True,  # Set to True if validation passes
                name_match=exact_name_match,
                workplace_match=workplace_match,
            )



class WebSearchAgent:
    """Enhanced Web Search Agent with 2-step workplace strategy and exact validation - ASYNC VERSION"""
    
    def __init__(self, tavily_api_key: Optional[str] = None):
        """Initialize the web search agent"""
        token_provider = get_bearer_token_provider(
            ClientSecretCredential(
                tenant_id=os.getenv("AZURE_TENANT_ID"),
                client_id=os.getenv("AZURE_OPENAI_PRINCIPAL_ID"),
                client_secret=os.getenv("AZURE_OPENAI_PRINCIPAL_SECRET"),
            ),
            "api://825a47b7-8e55-49b5-99c5-d7ecf65bd64d/.default",
        )
        
        self.llm = AzureChatOpenAI(
            azure_ad_token_provider=token_provider,
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_URL"),
            azure_deployment=os.getenv("AZURE_OPENAI_CHATGPT_DEPLOYMENT", "gpt-4"),
            api_version=os.getenv("AZURE_OPENAI_API_VERSION", "2024-02-15-preview"),
            temperature=0.1,
        )
        
        # ASYNC WRAPPERS
        self.tavily_client = AsyncTavilyWrapper("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")
        
        # ASYNC LLM WRAPPER
        self.async_llm: Any = AsyncLLMWrapper(self.llm)
        
        self.payload_builder = TavilyPayloadBuilder()
        self.geo_config = GeographicConfig.get_config()
        self.validator = ResultValidator(self.geo_config)
        self.specialty_mapping: Dict[str, str] = {}
        logger.info("Enhanced WebSearchAgent initialized successfully with GPT4 - ASYNC VERSION")
    
    def get_specialty_name(self, specialty_code: str, region: str = "IT") -> Optional[str]:
        """Get specialty name from code using region-specific mapping"""
        region_mapping = load_specialty_mapping(region)
        return region_mapping.get(specialty_code)
    
    def _extract_official_domain(
        self, workplace_results: List[Dict[str, Any]], workplace_name: str = ""
    ) -> Optional[str]:
        """Extract official domain from workplace search results - DYNAMIC VERSION"""
        # Create workplace keywords for domain matching
        workplace_keywords = []
        if workplace_name:
            # Extract significant words from workplace name
            words = workplace_name.lower().split()
            workplace_keywords = [
                word
                for word in words
                if len(word) >= 4
                and word
                not in [
                    "foundation",
                    "fondazione",
                    "institute",
                    "istituto",
                    "hospital",
                    "ospedale",
                    "university",
                    "universita",
                    "center",
                    "centre",
                    "medical",
                    "clinic",
                    "clinica",
                ]
            ]
        
        for result in workplace_results:
            url = result.get("url", "")
            content = result.get("content", "")
            title = result.get("title", "")
            
            # Combine content for analysis
            full_text = f"{title} {content}".lower()
            
            if url:
                domain = urlparse(url).netloc
                # Remove www. prefix if present
                if domain.startswith("www."):
                    domain = domain[4:]
                
                # Check if it's likely an official domain
                official_indicators = [
                    ".org",
                    ".edu",
                    ".gov",
                    ".it",
                    ".uk",
                    ".de",
                    ".fr",
                    ".com",
                ]
                if any(domain.endswith(indicator) for indicator in official_indicators):
                    # Check if this domain contains workplace keywords
                    if workplace_keywords:
                        domain_matches_workplace = any(keyword in domain for keyword in workplace_keywords)
                        if domain_matches_workplace:
                            logger.info(f"Found direct domain match: {domain}")
                            return domain
                
                # Generic institutional domain indicators
                institutional_indicators = [
                    "hospital",
                    "clinic",
                    "medical",
                    "university",
                    "institute",
                    "ospedale",
                    "clinica",
                    "universita",
                    "istituto",
                    "fondazione",
                ]
                if any(indicator in domain for indicator in institutional_indicators):
                    logger.info(f"Found institutional domain: {domain}")
                    return domain
        
        # ENHANCED: Look for referenced domains in content DYNAMICALLY
        
        # Dynamic patterns based on workplace keywords
        website_patterns = [
            r"(?:www\.)?([a-zA-Z0-9-]+\.(?:it|org|edu|gov|uk|de|fr|com))",  # General websites
            r"(?:web|sito|website)[:.\s]*(?:www\.)?([a-zA-Z0-9-]+\.(?:it|org|edu|gov|uk|de|fr|com))",  # Web indicators
        ]
        
        # Add workplace-specific patterns
        if workplace_keywords:
            for keyword in workplace_keywords:
                # Look for domains containing workplace keywords
                workplace_patterns = [
                    rf"(?:www\.)?([a-zA-Z0-9-]*{re.escape(keyword)}[a-zA-Z0-9-]*\.(?:it|org|edu|gov|uk|de|fr|com))",
                    rf"(?:www\.)?(?:[a-zA-Z0-9-]+\.)?({re.escape(keyword)}\.(?:it|org|edu|gov|uk|de|fr|com))",
                ]
                website_patterns.extend(workplace_patterns)
        
        # Find all potential domains in content
        found_domains = set()
        for pattern in website_patterns:
            matches = re.findall(pattern, full_text, re.IGNORECASE)
            for match in matches:
                if match and len(match) > 5:  # Valid domain length
                    found_domains.add(match.lower())
        
        # Score domains based on workplace relevance
        best_domain = None
        best_score = 0
        
        for domain in found_domains:
            score = 0
            
            # Score based on workplace keyword presence
            if workplace_keywords:
                for keyword in workplace_keywords:
                    if keyword in domain:
                        score += 10
            
            # Score based on institutional indicators
            institutional_indicators = [
                "hospital",
                "clinic",
                "medical",
                "university",
                "institute",
                "ospedale",
                "clinica",
                "universita",
                "istituto",
                "fondazione",
            ]
            for indicator in institutional_indicators:
                if indicator in domain:
                    score += 5
            
            # Prefer certain TLDs
            preferred_tlds = [".org", ".edu", ".gov"]
            if any(domain.endswith(tld) for tld in preferred_tlds):
                score += 3
            
            # Avoid generic domains
            generic_indicators = [
                "facebook",
                "linkedin",
                "twitter",
                "youtube",
                "google",
            ]
            if any(generic in domain for generic in generic_indicators):
                score -= 20
            
            if score > best_score:
                best_score = score
                best_domain = domain
        
        if best_domain and best_score > 0:
            logger.info(f"Found best matching domain: {best_domain} (score: {best_score})")
            return best_domain
        
        return None
    
    def _deduplicate_results(self, clean_response: Dict[str, Any]) -> Dict[str, Any]:
        """Remove 100% exact URL duplicates across all search results"""
        seen_urls = set()
        
        # Check what keys exist in search_results
        search_results = clean_response.get("search_results", {})
        
        # Process online search results (if exists) - RENAMED FROM main_search
        if "online_search" in search_results:
            deduplicated_online = []
            for result in search_results["online_search"]["results"]:
                url = result.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    deduplicated_online.append(result)
            clean_response["search_results"]["online_search"]["results"] = deduplicated_online
        
        # Process LinkedIn search results (if exists)
        if "linkedin_search" in search_results:
            deduplicated_linkedin = []
            for result in search_results["linkedin_search"]["results"]:
                url = result.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    deduplicated_linkedin.append(result)
            clean_response["search_results"]["linkedin_search"]["results"] = deduplicated_linkedin
        
        # Process workplace search results (if exists)
        if "workplace_search" in search_results:
            deduplicated_workplace = []
            for result in search_results["workplace_search"]["results"]:
                url = result.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    deduplicated_workplace.append(result)
            clean_response["search_results"]["workplace_search"]["results"] = deduplicated_workplace
        
        # Process workplace validation results (if exists)
        if "workplace_validation" in search_results:
            deduplicated_workplace_validation = []
            for result in search_results["workplace_validation"]["results"]:
                url = result.get("url", "")
                if url not in seen_urls:
                    seen_urls.add(url)
                    deduplicated_workplace_validation.append(result)
            clean_response["search_results"]["workplace_validation"]["results"] = deduplicated_workplace_validation
        
        return clean_response
    
    async def _validate_results_async(
        self, results: List[Dict[str, Any]], search_input: Dict[str, Any]
    ) -> List[Dict[str, Any]]:
        """Async validation of search results - STRICT FILTERING"""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, self._validate_results, results, search_input)
    
    def _validate_results(self, results: List[Dict[str, Any]], search_input: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Validate search results - STRICT FILTERING"""
        validated_results = []
        
        for result in results:
            validation = self.validator.validate_url(result, search_input)
            
            validated_result = {
                "url": result.get("url", ""),
                "title": result.get("title", ""),
                "content": result.get("content", ""),  # RAW CONTENT FROM TAVILY - NO LLM MODIFICATION
                "score": result.get("score", 0.0),
                "validation": {
                    "is_valid": validation.is_valid,
                    "confidence_score": validation.confidence_score,
                    "validation_reasons": validation.validation_reasons,
                    "geographic_match": validation.geographic_match,
                    "name_match": validation.name_match,
                    "workplace_match": validation.workplace_match,
                },
            }
            
            # STRICT FILTERING - Only include VALIDATED results
            if validation.is_valid:  # REMOVED confidence_score >= 0.5 fallback
                validated_results.append(validated_result)
        
        # Sort by confidence score
        validated_results.sort(key=lambda x: x["validation"]["confidence_score"], reverse=True)
        
        return validated_results
    
    
    
    async def search_hcp(self, doctor_info: Dict[str, Any]) -> Dict[str, Any]:
        """Search for Healthcare Professional - TRUE 2-STEP WORKPLACE STRATEGY - ASYNC"""
        logger.info(f"Starting ASYNC HCP search for: {doctor_info.get('firstName')} {doctor_info.get('lastName')}")
        
        
        specialty_code = doctor_info.get("specialtyCode")
        region = doctor_info.get("geographic_region", "IT")
        specialty_name = None
        if specialty_code:
            specialty_name = self.get_specialty_name(specialty_code, region)
            logger.info(f"Specialty code {specialty_code} mapped to: {specialty_name} for region {region}")
        
        # Initialize response structure
        clean_response: Dict[str, Any] = {
            "search_results": {
                "online_search": {"results": []},
                "linkedin_search": {"results": []},
                "workplace_search": {"results": []},
            },
            "tavily_answer": "",
            "llm_answer": "",
        }
        
        try:
            # Add entity_type for validation
            search_input = {**doctor_info, "entity_type": "ent_activity"}
            
            # BUILD ALL PAYLOADS FIRST
            main_payload = self.payload_builder.build_hcp_main_payload(doctor_info, specialty_name)
            linkedin_payload = self.payload_builder.build_linkedin_payload(doctor_info, specialty_name)
            workplace_validation_payload = self.payload_builder.build_workplace_validation_payload(doctor_info)
            
            # CONCURRENT EXECUTION: ALL THREE SEARCHES SIMULTANEOUSLY
            logger.info("Running main, LinkedIn, and workplace validation searches concurrently...")
            main_task = self.tavily_client.search_async(**main_payload)
            linkedin_task = self.tavily_client.search_async(**linkedin_payload)
            workplace_task = self.tavily_client.search_async(**workplace_validation_payload)
            
            # Wait for all three to complete
            online_results, linkedin_results, workplace_site_results = await asyncio.gather(
                main_task, linkedin_task, workplace_task
            )
            
            print(f"Main search results: {online_results}")
            print(f"LinkedIn search results: {linkedin_results}")
            print(f"Workplace search results: {workplace_site_results}")
            clean_response["tavily_answer"] = online_results.get("answer", "") or ""
            
            # CONCURRENT VALIDATION - but workplace needs different entity_type
            logger.info("Running validation concurrently...")
            validated_main_task = self._validate_results_async(online_results.get("results", []), search_input)
            validated_linkedin_task = self._validate_results_async(linkedin_results.get("results", []), search_input)
            
            # STEP 3: TRUE 2-STEP WORKPLACE STRATEGY
            logger.info("Step 3b: Extracting official domain from workplace results...")
            
            # Step 3b: Extract official domain from results
            official_domain = self._extract_official_domain(
                workplace_site_results.get("results", []),
                doctor_info.get("workplaceName", ""),
            )
            
            if official_domain:
                logger.info(f"Step 3c: Found official domain: {official_domain}")
                # Step 3c: Search for person on that specific domain using the new method
                targeted_payload = self.payload_builder.build_targeted_workplace_payload(doctor_info, official_domain)
                targeted_workplace_results = await self.tavily_client.search_async(**targeted_payload)
                # Use HCP validation for targeted search (looking for person)
                validated_workplace = await self._validate_results_async(
                    targeted_workplace_results.get("results", []), search_input
                )
            else:
                logger.info("Step 3c: No official domain found, using fallback search")
                # Fallback: Use original workplace search
                workplace_payload = self.payload_builder.build_workplace_payload(doctor_info)
                workplace_results = await self.tavily_client.search_async(**workplace_payload)
                # Use HCP validation for fallback search (looking for person)
                validated_workplace = await self._validate_results_async(
                    workplace_results.get("results", []), search_input
                )
            
            # Wait for main and LinkedIn validation to complete
            validated_main, validated_linkedin = await asyncio.gather(validated_main_task, validated_linkedin_task)
            
            # Extract validated URLs and content
            for result in validated_main:
                validation = result.get("validation", {})
                if validation.get("is_valid", False):
                    clean_response["search_results"]["online_search"]["results"].append(
                        {
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "content": result.get("content", ""),
                            "score": result.get("score", 0.0),
                        }
                    )
            
            for result in validated_linkedin:
                validation = result.get("validation", {})
                if validation.get("is_valid", False):
                    clean_response["search_results"]["linkedin_search"]["results"].append(
                        {
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),
                            "content": result.get("content", ""),
                            "score": result.get("score", 0.0),
                        }
                    )
            
            # if official_domain:
            #     # Filter results to only include official workplace domain URLs
            #     for result in validated_workplace:
            #         validation = result.get("validation", {})
            #         if validation.get("is_valid", False):
            #             # Extract domain from result URL
            #             result_url = result.get("url", "")
            #             if result_url:
            #                 result_domain = urlparse(result_url).netloc
            #                 # Remove www. prefix if present
            #                 if result_domain.startswith("www."):
            #                     result_domain = result_domain[4:]
            #                 # Only add if result is from official workplace domain
            #                 if result_domain == official_domain or result_domain.endswith(f".{official_domain}"):
            #                     clean_response["search_results"]["workplace_search"]["results"].append(
            #                         {
            #                             "url": result.get("url", ""),
            #                             "title": result.get("title", ""),
            #                             "content": result.get("content", ""),
            #                             "score": result.get("score", 0.0),
            #                         }
            #                     )
            #                     logger.info(f"Added official workplace URL: {result_url}")
            #                 else:
            #                     logger.info(f"Rejected non-official URL: {result_url} (not from {official_domain})")
            # else:
            #     # If no official domain found, don't add any workplace results
            #     logger.warning("No official domain found - no workplace results added for safety")
            
            
            if official_domain:
                # Filter results to only include official workplace domain URLs
                hcp_found_on_official_domain = False
                
                for result in validated_workplace:
                    validation = result.get("validation", {})
                    if validation.get("is_valid", False):
                        # Extract domain from result URL
                        result_url = result.get("url", "")
                        if result_url:
                            result_domain = urlparse(result_url).netloc
                            # Remove www. prefix if present
                            if result_domain.startswith("www."):
                                result_domain = result_domain[4:]
                            # Only add if result is from official workplace domain
                            if result_domain == official_domain or result_domain.endswith(f".{official_domain}"):
                                clean_response["search_results"]["workplace_search"]["results"].append(
                                    {
                                        "url": result.get("url", ""),
                                        "title": result.get("title", ""),
                                        "content": result.get("content", ""),
                                        "score": result.get("score", 0.0),
                                    }
                                )
                                hcp_found_on_official_domain = True
                                logger.info(f"Added official workplace URL: {result_url}")
                            else:
                                logger.info(f"Rejected non-official URL: {result_url} (not from {official_domain})")
                
                # NEW: If official domain found but no HCP data, add the official domain info
                if not hcp_found_on_official_domain:
                    logger.info(f"Official domain {official_domain} found but no HCP data - adding domain info")
                    clean_response["search_results"]["workplace_search"]["results"].append(
                        {
                            "url": f"https://{official_domain}",
                            "title": f"Official Website - {doctor_info.get('workplaceName', '')}",
                            "content": f"Found official domain {official_domain} for {doctor_info.get('workplaceName', '')}, but no specific information about {doctor_info.get('firstName', '')} {doctor_info.get('lastName', '')} was found on this official website.",
                            "score": 0.5,
                        }
                    )
            else:
                # If no official domain found, don't add any workplace results
                logger.warning("No official domain found - no workplace results added for safety")           
            
            # Generate LLM analysis asynchronously
            clean_response["llm_answer"] = await self._generate_llm_analysis_async(
                clean_response, doctor_info, specialty_name
            )
            
            logger.info(f"ASYNC HCP search completed. Tavily answer: {len(clean_response['tavily_answer'])} chars")
            
        except Exception as e:
            logger.error(f"Error in ASYNC HCP search: {str(e)}")
            clean_response["llm_answer"] = f"Search operation failed: {str(e)}"
        
        return clean_response
    
    async def search_workplace(self, workplace_info: Dict[str, Any]) -> Dict[str, Any]:
        """Search for workplace validation - Clean Output - ASYNC"""
        logger.info(f"Starting ASYNC workplace search for: {workplace_info.get('workplaceName')}")
        
        # Initialize response structure
        clean_response: Dict[str, Any] = {
            "search_results": {"workplace_validation": {"results": []}},
            "tavily_answer": "",
            "llm_answer": "",
        }
        
        try:
            # Add entity_type for validation
            search_input = {**workplace_info, "entity_type": "ent_workplace"}
            
            # Workplace validation search
            workplace_payload = self.payload_builder.build_workplace_validation_payload(workplace_info)
            workplace_results = await self.tavily_client.search_async(**workplace_payload)
            print(f"Workplace validation results: {workplace_results}")
            clean_response["tavily_answer"] = workplace_results.get("answer", "") or ""
            
            # Validate workplace results
            validated_workplace = await self._validate_results_async(workplace_results.get("results", []), search_input)
            
            for result in validated_workplace:
                validation = result.get("validation", {})
                if validation.get("is_valid", False):  # ONLY validated results
                    clean_response["search_results"]["workplace_validation"]["results"].append(
                        {
                            "url": result.get("url", ""),
                            "title": result.get("title", ""),  # RAW TAVILY TITLE
                            "content": result.get("content", ""),  # RAW TAVILY CONTENT - NO MODIFICATION
                            "score": result.get("score", 0.0),  # RAW TAVILY SCORE
                        }
                    )
            
            # Generate LLM analysis asynchronously
            clean_response["llm_answer"] = await self._generate_workplace_llm_analysis_async(
                clean_response, workplace_info
            )
            
            logger.info(
                f"ASYNC workplace search completed. Tavily answer: {len(clean_response['tavily_answer'])} chars"
            )
            
        except Exception as e:
            logger.error(f"Error in ASYNC workplace search: {str(e)}")
            clean_response["llm_answer"] = f"Workplace search operation failed: {str(e)}"
        
        return clean_response
    
    async def _generate_llm_analysis_async(
        self,
        clean_response: Dict[str, Any],
        doctor_info: Dict[str, Any],
        specialty_name: Optional[str] = None,
    ) -> str:
        """Generate LLM analysis for HCP - ASYNC"""
        try:
            online_results = clean_response["search_results"]["online_search"]["results"]
            linkedin_results = clean_response["search_results"]["linkedin_search"]["results"]
            workplace_results = clean_response["search_results"]["workplace_search"]["results"]
            
            search_summary = f"""
            MAIN SEARCH RESULTS: {len(online_results)} validated results found
            LINKEDIN SEARCH RESULTS: {len(linkedin_results)} validated results found
            WORKPLACE SEARCH RESULTS: {len(workplace_results)} validated results found

            MAIN SEARCH CONTENT:
            """
                        
            if online_results:
                for i, result in enumerate(online_results, 1):
                    search_summary += f"Result {i}: {result['content']}...\n"
            else:
                search_summary += "No validated main search results found.\n"
            
            search_summary += "\nLINKEDIN SEARCH CONTENT:\n"
            
            if linkedin_results:
                for i, result in enumerate(linkedin_results, 1):
                    search_summary += f"LinkedIn Result {i}: {result['content']}...\n"
            else:
                search_summary += "No validated LinkedIn results found.\n"
            
            search_summary += "\nWORKPLACE SEARCH CONTENT:\n"
            
            if workplace_results:
                for i, result in enumerate(workplace_results, 1):
                    search_summary += f"Workplace Result {i}: {result['content']}...\n"
            else:
                search_summary += "No validated workplace results found.\n"
            
            doctor_firstname = doctor_info.get("firstName", "")
            doctor_lastname = doctor_info.get("lastName", "")
            doctor_workplace = doctor_info.get("workplaceName", "")
            if specialty_name is None:
                specialty_name = "Not specified"
            
            prompt = WEBSEARCH_PROMPT.format(
                doctor_firstname=doctor_firstname,
                doctor_lastname=doctor_lastname,
                doctor_workplace=doctor_workplace,
                specialty_name=specialty_name,
                search_summary=search_summary,
            )
            
            response_llm = await self.async_llm.invoke_async(prompt)
            
            if isinstance(response_llm, dict) and "llm_answer" in response_llm:
                return response_llm["llm_answer"].strip()
            else:
                return str(response_llm)
        
        except Exception as e:
            logger.error(f"Error in ASYNC LLM analysis: {str(e)}")
            return f"""Main Search Summary:\n\nLLM analysis failed: {str(e)}\n\nLinkedIn Profile Summary:\n\nUnavailable due to processing error.\n\nWorkplace Validation Summary:\n\nUnavailable due to processing error.\n"""



    async def _generate_workplace_llm_analysis_async(
        self, clean_response: Dict[str, Any], workplace_info: Dict[str, Any]
    ) -> str:
        """Generate LLM analysis for workplace - ASYNC"""
        try:
            workplace_results = clean_response["search_results"]["workplace_validation"]["results"]

            search_summary = f"""
                WORKPLACE VALIDATION RESULTS: {len(workplace_results)} validated results found

                WORKPLACE VALIDATION CONTENT:
                """

            if workplace_results:
                for i, result in enumerate(workplace_results, 1):
                    search_summary += f"Result {i}: {result['content']}...\n"
            else:
                search_summary += "No validated workplace results found.\n"

            workplace_name = workplace_info.get("workplaceName", "")
            workplace_address = workplace_info.get("address", "")

            
            # prompt = f"""You are an expert institutional validation analyst for IQVIA. Based on the workplace validation results, provide a comprehensive analysis in this exact format:

            #         # \"\"\"\"Workplace Validation Summary:

            #         # [Analyze workplace validation results - institutional legitimacy, contact information, official status, geographic verification for {workplace_name}]

            #         # \"\"\"\"

            #         # Workplace Target: {workplace_name} in {workplace_address}

            #         # {search_summary}"""

            prompt = WEBSEARCH_PROMPT_ANALYSIS.format(
                workplace_name=workplace_name,
                workplace_address=workplace_address,
                search_summary=search_summary,
            )

            response_llm = await self.async_llm.invoke_async(prompt)

            if hasattr(response_llm, "llm_final_answer"):
                return response_llm.llm_final_answer
            else:
                return str(response_llm)

        except Exception as e:
            logger.error(f"Error in ASYNC workplace LLM analysis: {str(e)}")
            return f"Workplace Validation Summary:\n\nLLM analysis failed: {str(e)}\n"

    async def search(self, search_input: Dict[str, Any]) -> Dict[str, Any]:
        """Main search function that routes to appropriate search type - ASYNC"""
        entity_type = search_input.get("entity_type", "")

        if entity_type == "ent_activity":
            search_type = "hcp"
        elif entity_type == "ent_workplace":
            search_type = "workplace"
        elif "firstName" in search_input and "lastName" in search_input:
            search_type = "hcp"
        else:
            search_type = "workplace"

        logger.info(f"Detected search type: {search_type}")

        if search_type == "hcp":
            return await self.search_hcp(search_input)
        else:
            return await self.search_workplace(search_input)