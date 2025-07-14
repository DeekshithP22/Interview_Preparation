Perfect! I can see the JSON structure. Let me give you step-by-step instructions to integrate this specialty mapping:

## 🔧 **Step-by-Step Integration:**

### **STEP 1: Update the JSON File Loading Function**

**Find this function:**
```python
def load_specialty_mapping(json_file_path: str = "specialty_mapping.json") -> Dict[str, str]:
```

**Replace it with:**
```python
def load_specialty_mapping(region: str = "IT") -> Dict[str, str]:
    """Load specialty code mapping from JSON file based on region"""
    # Map region to JSON file names
    json_files = {
        "IT": "GET_CODES_ITALY_SP.json",
        "FR": "GET_CODES_FRANCE_SP.json",
        "US": "GET_CODES_US_SP.json",  # Add if you have US
        "UK": "GET_CODES_UK_SP.json"   # Add if you have UK
    }
    
    json_file_path = json_files.get(region, json_files["IT"])  # Default to Italy
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Extract specialty mapping from the JSON structure
        specialty_mapping = {}
        
        if "response" in data and "results" in data["response"]:
            for specialty in data["response"]["results"]:
                eid = specialty.get("eid", "")
                long_localized_label = specialty.get("longLocalizedLabel", "")
                
                if eid and long_localized_label:
                    specialty_mapping[eid] = long_localized_label
        
        logger.info(f"Loaded {len(specialty_mapping)} specialties for region {region}")
        return specialty_mapping
        
    except FileNotFoundError:
        logger.warning(f"Specialty mapping file {json_file_path} not found. Using empty mapping.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in specialty mapping file {json_file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading specialty mapping: {str(e)}")
        return {}
```

### **STEP 2: Update the EnhancedWebSearchAgent __init__ Method**

**Find this line in __init__:**
```python
self.specialty_mapping = load_specialty_mapping(specialty_mapping_file)
```

**Replace with:**
```python
# Initialize with empty mapping - will be loaded dynamically per search
self.specialty_mapping = {}
```

### **STEP 3: Update the get_specialty_name Method**

**Find this method:**
```python
def get_specialty_name(self, specialty_code: str) -> Optional[str]:
    """Get specialty name from code using mapping"""
    return self.specialty_mapping.get(specialty_code)
```

**Replace with:**
```python
def get_specialty_name(self, specialty_code: str, region: str = "IT") -> Optional[str]:
    """Get specialty name from code using region-specific mapping"""
    # Load region-specific mapping
    region_mapping = load_specialty_mapping(region)
    return region_mapping.get(specialty_code)
```

### **STEP 4: Update the search_hcp Method**

**Find these lines:**
```python
# Get specialty name if code provided
specialty_code = doctor_info.get("specialtyCode")
specialty_name = None
if specialty_code:
    specialty_name = self.get_specialty_name(specialty_code)
    logger.info(f"Specialty code {specialty_code} mapped to: {specialty_name}")
```

**Replace with:**
```python
# Get specialty name if code provided
specialty_code = doctor_info.get("specialtyCode")
region = doctor_info.get("geographic_region", "IT")
specialty_name = None
if specialty_code:
    specialty_name = self.get_specialty_name(specialty_code, region)
    logger.info(f"Specialty code {specialty_code} mapped to: {specialty_name} for region {region}")
```

### **STEP 5: Update EnhancedWebSearchAgent Constructor Parameters**

**Find the __init__ method signature:**
```python
def __init__(self, tavily_api_key: Optional[str] = None, google_api_key: Optional[str] = None, 
             specialty_mapping_file: str = "specialty_mapping.json"):
```

**Replace with:**
```python
def __init__(self, tavily_api_key: Optional[str] = None, google_api_key: Optional[str] = None):
```

**Remove this line from __init__:**
```python
self.specialty_mapping = load_specialty_mapping(specialty_mapping_file)
```

### **STEP 6: Update the Convenience Function**

**Find this function:**
```python
async def search_healthcare_professionals_async(
    search_input: Dict[str, Any],
    tavily_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None,
    specialty_mapping_file: str = "specialty_mapping.json"
) -> Dict[str, Any]:
```

**Replace with:**
```python
async def search_healthcare_professionals_async(
    search_input: Dict[str, Any],
    tavily_api_key: Optional[str] = None,
    google_api_key: Optional[str] = None
) -> Dict[str, Any]:
```

**And update the function body:**
```python
agent = EnhancedWebSearchAgent(tavily_api_key, google_api_key)
return await agent.search(search_input)
```

### **STEP 7: Test the Integration**

**Example input:**
```python
hcp_search_input = {
    "entity_type": "ent_activity",
    "geographic_region": "IT",  # Will load GET_CODES_ITALY_SP.json
    "firstName": "Marcello",
    "lastName": "Marchetti",
    "specialtyCode": "01",  # Maps to eid in Italian JSON
    # ...
}
```

## 🎯 **How It Works:**

1. **Input**: `specialtyCode: "01"`, `geographic_region: "IT"`
2. **Loads**: `GET_CODES_ITALY_SP.json`
3. **Finds**: `eid: "01"` → `longLocalizedLabel: "Medic. generica"`
4. **Uses**: "Medic. generica" in search queries

**This ensures region-specific specialty mapping with proper JSON structure handling!** 🚀


























You're absolutely right! Let me give you the complete 7-step instructions with the Pydantic model and OpenAI compatibility:

## 🔧 **Complete Step-by-Step Instructions:**

### **STEP 1: Add Pydantic Model**

**Location**: At the top of the file, add this model after the existing models (`LLMSummaryResponse`, `ValidationResult`):

```python
class MedicalValidationResponse(BaseModel):
    """Model for LLM medical validation response"""
    is_medical_professional: bool = Field(description="True if content describes a healthcare professional, False otherwise")
    specialty_detected: Optional[str] = Field(description="Medical specialty detected in content, if any")
    confidence: float = Field(description="Confidence score between 0.0 and 1.0")
    reasoning: str = Field(description="Brief explanation of the validation decision")
```

### **STEP 2: Add LLM Validation Method**

**Location**: In the `ResultValidator` class, add this method **after** the `validate_professional_context` method:

```python
async def llm_validate_medical_context(self, content: str, specialty_name: str = None, llm_wrapper=None) -> bool:
    """LLM-based medical professional context validation with structured output"""
    if not llm_wrapper:
        return self._fallback_medical_check(content)
    
    try:
        # Prepare specialty context
        specialty_context = f" Expected specialty: {specialty_name}." if specialty_name else ""
        
        prompt = f"""Analyze this content to determine if it describes a healthcare professional.

Content: "{content[:800]}"
{specialty_context}

Consider these indicators of healthcare professionals:
- Medical degrees, certifications, specializations
- Medical job titles (doctor, physician, surgeon, oncologist, etc.)
- Employment at medical institutions (hospitals, clinics, medical centers)
- Medical research, publications, clinical work
- Medical specialties (cardiology, neurology, dermatology, radiation oncology, etc.)

Respond in this exact JSON format:
{{
    "is_medical_professional": true/false,
    "specialty_detected": "detected specialty or null",
    "confidence": 0.0-1.0,
    "reasoning": "brief explanation"
}}"""

        # Get response from your OpenAI LLM
        response = await llm_wrapper.invoke_async(prompt)
        
        # Parse response text to extract JSON
        if hasattr(response, 'llm_final_answer'):
            response_text = response.llm_final_answer
        else:
            response_text = str(response)
        
        # Try to parse JSON response
        import json
        try:
            response_json = json.loads(response_text)
            
            is_medical = response_json.get('is_medical_professional', False)
            confidence = response_json.get('confidence', 0.5)
            specialty_detected = response_json.get('specialty_detected', '')
            reasoning = response_json.get('reasoning', '')
            
            logger.info(f"LLM Medical Validation - Result: {is_medical}, Confidence: {confidence:.2f}, "
                       f"Specialty: {specialty_detected}, Reasoning: {reasoning[:100]}")
            
            # Use confidence threshold for decision
            return is_medical and confidence >= 0.7
            
        except json.JSONDecodeError:
            logger.warning(f"Could not parse LLM JSON response: {response_text[:100]}, using fallback")
            return self._fallback_medical_check(content)
            
    except Exception as e:
        logger.error(f"LLM medical validation failed: {str(e)}")
        return self._fallback_medical_check(content)

def _fallback_medical_check(self, content: str) -> bool:
    """Fallback keyword-based medical validation"""
    content_lower = content.lower()
    basic_medical_terms = [
        "medical", "medicine", "doctor", "physician", "hospital", "clinic",
        "medico", "dottore", "medicina", "ospedale", "clinica", "radiation",
        "oncologist", "neurosurgeon", "chirurgo", "specialista"
    ]
    result = any(term in content_lower for term in basic_medical_terms)
    logger.info(f"Fallback medical validation: {result}")
    return result
```

### **STEP 3: Update ResultValidator Constructor**

**Location**: In the `ResultValidator` class `__init__` method

**Find this:**
```python
def __init__(self, geographic_config: Dict[str, Any]):
    self.geo_config = geographic_config
```

**Replace with:**
```python
def __init__(self, geographic_config: Dict[str, Any], llm_wrapper=None):
    self.geo_config = geographic_config
    self.llm_wrapper = llm_wrapper
```

### **STEP 4: Update validate_url Method Signature**

**Location**: In the `ResultValidator` class

**Find this:**
```python
def validate_url(self, result: Dict[str, Any], search_input: Dict[str, Any]) -> ValidationResult:
```

**Replace with:**
```python
async def validate_url(self, result: Dict[str, Any], search_input: Dict[str, Any]) -> ValidationResult:
```

### **STEP 5: Update Professional Context Call in validate_url**

**Location**: In the `validate_url` method in `ResultValidator` class

**Find this line:**
```python
professional_context = self.validate_professional_context(full_content, region)
```

**Replace with:**
```python
# Get specialty name for LLM context
specialty_code = search_input.get("specialtyCode")
region = search_input.get("geographic_region", "IT")
specialty_name = None
if specialty_code:
    region_mapping = load_specialty_mapping(region)
    specialty_name = region_mapping.get(specialty_code)

# Use LLM-based validation for professional context
professional_context = await self.llm_validate_medical_context(full_content, specialty_name, self.llm_wrapper)
```

### **STEP 6: Update _validate_results Method**

**Location**: In the `EnhancedWebSearchAgent` class

**Find this method:**
```python
def _validate_results(self, results: List[Dict[str, Any]], search_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate search results - STRICT FILTERING"""
    validated_results = []
    
    for result in results:
        validation = self.validator.validate_url(result, search_input)
        # ... rest of method
```

**Replace with:**
```python
async def _validate_results(self, results: List[Dict[str, Any]], search_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate search results - STRICT FILTERING"""
    validated_results = []
    
    for result in results:
        validation = await self.validator.validate_url(result, search_input)
        # ... rest of method stays exactly the same
```

### **STEP 7: Update EnhancedWebSearchAgent Constructor**

**Location**: In the `EnhancedWebSearchAgent` class `__init__` method

**Find this line:**
```python
self.validator = ResultValidator(self.geo_config)
```

**Replace with:**
```python
self.validator = ResultValidator(self.geo_config, self.async_llm)
```

### **STEP 8: Update _validate_results_async Method**

**Location**: In the `EnhancedWebSearchAgent` class

**Find this method:**
```python
async def _validate_results_async(self, results: List[Dict[str, Any]], search_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Async validation of search results - STRICT FILTERING"""
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, self._validate_results, results, search_input)
```

**Replace with:**
```python
async def _validate_results_async(self, results: List[Dict[str, Any]], search_input: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Async validation of search results - STRICT FILTERING"""
    return await self._validate_results(results, search_input)
```

## ✅ **Complete List - 8 Steps Total:**

1. Add Pydantic model
2. Add LLM validation method  
3. Update ResultValidator constructor
4. Make validate_url async
5. Update professional context call
6. Make _validate_results async
7. Update EnhancedWebSearchAgent constructor
8. Update _validate_results_async method

**Follow these 8 steps in order after integrating get_codes and you'll have intelligent LLM-based medical validation!** 🚀















#!/usr/bin/env python3
"""
Simple test script for Enhanced Web Search Agent
"""

import asyncio
from websearch_agent_get_code import EnhancedWebSearchAgent

async def test_agent():
    """Simple test of the search agent"""
    
    # Create agent
    agent = EnhancedWebSearchAgent()
    
    # Test HCP search
    hcp_input = {
        "entity_type": "ent_activity",
        "firstName": "Marcello",
        "lastName": "Marchetti", 
        "workplaceName": "Fondazione IRCCS Istituto Neurologico Carlo Besta",
        "address": "Milano",
        "specialtyCode": "01",
        "geographic_region": "IT"
    }
    
    print("Testing HCP search...")
    result = await agent.search(hcp_input)
    
    # Print results
    print(f"Online results: {len(result['search_results']['online_search']['results'])}")
    print(f"LinkedIn results: {len(result['search_results']['linkedin_search']['results'])}")
    print(f"Workplace results: {len(result['search_results']['workplace_search']['results'])}")
    
    print("\nFirst online result:")
    if result['search_results']['online_search']['results']:
        first_result = result['search_results']['online_search']['results'][0]
        print(f"URL: {first_result['url']}")
        print(f"Title: {first_result['title']}")
    
    print(f"\nTavily answer: {result['tavily_answer']}")

if __name__ == "__main__":





















Perfect! Let's organize everything properly. Here's the complete step-by-step reorganization:

## 📁 **STEP 1: Create the Directory Structure**

**Navigate to your current location and create these folders:**

```
your_current_directory/
├── websearch_agent_get_code.py  (your existing file)
├── config/                      👈 CREATE THIS FOLDER
│   ├── geographic_config.yaml   👈 CREATE THIS FILE
│   └── specialty_mappings/      👈 CREATE THIS FOLDER
│       ├── GET_CODES_ITALY_SP.json    👈 MOVE/CREATE
│       ├── GET_CODES_FRANCE_SP.json   👈 MOVE/CREATE
│       ├── GET_CODES_US_SP.json       👈 CREATE (optional)
│       └── GET_CODES_UK_SP.json       👈 CREATE (optional)
```

## 📝 **STEP 2: Create geographic_config.yaml**

**Create `config/geographic_config.yaml` with this content:**

```yaml
# Geographic Configuration for Healthcare Professional Search

IT:
  region_name: "Italy"
  language_code: "it"
  professional_terms:
    - "medico"
    - "dottore"
    - "specialista"
    - "chirurgo" 
    - "oncologo"
    - "neurologo"
    - "radiologo"
    - "cardiologo"
    - "dermatologo"
    - "pediatra"
    - "ginecologo"
  professional_domains:
    - "ospedale"
    - "clinica"
    - "università"
    - "irccs"
    - "asl"
    - "ao"
  medical_institutions:
    - ".it"
    - "ospedale"
    - "clinica"
    - "università"

FR:
  region_name: "France"
  language_code: "fr"
  professional_terms:
    - "médecin"
    - "docteur"
    - "spécialiste"
    - "chirurgien"
    - "oncologue"
    - "neurologue"
    - "radiologue"
    - "cardiologue"
    - "dermatologue"
    - "pédiatre"
  professional_domains:
    - "hôpital"
    - "clinique"
    - "université"
    - "chu"
    - "chru"
  medical_institutions:
    - ".fr"
    - "hopital"
    - "clinique"
    - "université"

US:
  region_name: "United States"
  language_code: "en"
  professional_terms:
    - "doctor"
    - "physician"
    - "specialist"
    - "surgeon"
    - "oncologist"
    - "neurologist"
    - "radiologist"
    - "cardiologist"
    - "dermatologist"
    - "pediatrician"
  professional_domains:
    - "hospital"
    - "clinic"
    - "university"
    - "medical"
    - "health"
  medical_institutions:
    - ".edu"
    - ".org"
    - "hospital"
    - "clinic"
    - "university"

UK:
  region_name: "United Kingdom" 
  language_code: "en"
  professional_terms:
    - "doctor"
    - "physician"
    - "consultant"
    - "surgeon"
    - "oncologist"
    - "neurologist"
    - "radiologist"
    - "cardiologist"
    - "dermatologist"
    - "paediatrician"
  professional_domains:
    - "hospital"
    - "clinic"
    - "university"
    - "nhs"
  medical_institutions:
    - ".uk"
    - ".ac.uk"
    - "hospital"
    - "clinic"
    - "university"
```

## 🔧 **STEP 3: Update websearch_agent_get_code.py - Replace GeographicConfig Class**

**Find the `GeographicConfig` class in your file and replace it entirely with:**

```python
import yaml
import os
from typing import Dict, Any

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
            config_file = os.path.join(current_dir, "config", "geographic_config.yaml")
            
            with open(config_file, 'r', encoding='utf-8') as file:
                config = yaml.safe_load(file)
            
            logger.info(f"Loaded geographic configuration from {config_file}")
            return config
            
        except FileNotFoundError:
            logger.warning(f"Geographic config file not found at {config_file}. Using default config.")
            return cls._get_default_config()
        except yaml.YAMLError as e:
            logger.error(f"Error parsing YAML config: {e}. Using default config.")
            return cls._get_default_config()
        except Exception as e:
            logger.error(f"Error loading geographic config: {e}. Using default config.")
            return cls._get_default_config()
    
    @classmethod
    def _get_default_config(cls) -> Dict[str, Any]:
        """Fallback default configuration"""
        return {
            "IT": {
                "region_name": "Italy",
                "professional_terms": ["medico", "dottore", "specialista"],
                "professional_domains": ["ospedale", "clinica"],
                "medical_institutions": [".it"]
            }
        }
```

## 🔧 **STEP 4: Update load_specialty_mapping Function**

**Find the `load_specialty_mapping` function and replace it with:**

```python
def load_specialty_mapping(region: str = "IT") -> Dict[str, str]:
    """Load specialty code mapping from JSON file based on region"""
    
    # Get the directory where this script is located
    current_dir = os.path.dirname(os.path.abspath(__file__))
    config_dir = os.path.join(current_dir, "config", "specialty_mappings")
    
    # Map region to JSON file names
    json_files = {
        "IT": "GET_CODES_ITALY_SP.json",
        "FR": "GET_CODES_FRANCE_SP.json",
        "US": "GET_CODES_US_SP.json", 
        "UK": "GET_CODES_UK_SP.json"
    }
    
    filename = json_files.get(region, json_files["IT"])  # Default to Italy
    json_file_path = os.path.join(config_dir, filename)
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Extract specialty mapping from the JSON structure
        specialty_mapping = {}
        
        if "response" in data and "results" in data["response"]:
            for specialty in data["response"]["results"]:
                eid = specialty.get("eid", "")
                long_localized_label = specialty.get("longLocalizedLabel", "")
                
                if eid and long_localized_label:
                    specialty_mapping[eid] = long_localized_label
        
        logger.info(f"Loaded {len(specialty_mapping)} specialties for region {region} from {json_file_path}")
        return specialty_mapping
        
    except FileNotFoundError:
        logger.warning(f"Specialty mapping file {json_file_path} not found. Using empty mapping.")
        return {}
    except json.JSONDecodeError:
        logger.error(f"Invalid JSON in specialty mapping file {json_file_path}")
        return {}
    except Exception as e:
        logger.error(f"Error loading specialty mapping: {str(e)}")
        return {}
```

## 📦 **STEP 5: Add Required Imports**

**At the top of your websearch_agent_get_code.py file, make sure you have:**

```python
import os
import yaml
import json
```

**If you don't have PyYAML installed, install it:**
```bash
pip install PyYAML
```

## 📄 **STEP 6: Move Your JSON Files**

**Move your existing JSON files to the new location:**
- Move `GET_CODES_ITALY_SP.json` to `config/specialty_mappings/`
- Move `GET_CODES_FRANCE_SP.json` to `config/specialty_mappings/`

## ✅ **STEP 7: Test the New Structure**

**Run your test to make sure everything works:**
```bash
python simple_test.py
```

## 🎯 **Benefits of This Organization:**

1. **Clean Separation** - Code vs Configuration
2. **Easy Updates** - Just update YAML/JSON files
3. **Version Control** - Track config changes separately
4. **Multi-Environment** - Different configs for dev/prod
5. **Maintainable** - Non-technical people can update configs

**This structure is production-ready and follows best practices!** 🚀
    asyncio.run(test_agent())
















    Perfect! I can see your directory structure. You have the config files in the right place under `tools/websearch_config/`. Let me fix the path resolution.

## 🔧 **The Issue:**

Your files are in:
```
app/backend/app/my_agent/tools/websearch_config/
├── specialty_mappings/
│   ├── GET_CODES_FRANCE_SP.json
│   └── GET_CODES_ITALY_SP.json
└── geographic_config.yaml
```

But you're running from `app/backend/`, so the relative paths need to be adjusted.

## 🔧 **STEP 1: Update load_specialty_mapping Function**

**In your `websearch_agent_get_code.py`, find and replace the `load_specialty_mapping` function:**

```python
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
        "UK": "GET_CODES_UK_SP.json"
    }
    
    filename = json_files.get(region, json_files["IT"])
    json_file_path = os.path.join(config_dir, filename)
    
    print(f"🔍 Looking for specialty mapping at: {json_file_path}")
    
    try:
        with open(json_file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
            
        # Extract specialty mapping from the JSON structure
        specialty_mapping = {}
        
        if "response" in data and "results" in data["response"]:
            for specialty in data["response"]["results"]:
                eid = specialty.get("eid", "")
                long_localized_label = specialty.get("longLocalizedLabel", "")
                
                if eid and long_localized_label:
                    specialty_mapping[eid] = long_localized_label
        
        print(f"✅ Loaded {len(specialty_mapping)} specialties for region {region}")
        return specialty_mapping
        
    except FileNotFoundError:
        print(f"❌ Specialty mapping file {json_file_path} not found. Using empty mapping.")
        return {}
    except json.JSONDecodeError:
        print(f"❌ Invalid JSON in specialty mapping file {json_file_path}")
        return {}
    except Exception as e:
        print(f"❌ Error loading specialty mapping: {str(e)}")
        return {}
```

## 🔧 **STEP 2: Update GeographicConfig._load_config Method**

**In the `GeographicConfig` class, update the `_load_config` method:**

```python
@classmethod
def _load_config(cls) -> Dict[str, Any]:
    """Load configuration from YAML file"""
    try:
        # Get the directory where this script is located
        current_dir = os.path.dirname(os.path.abspath(__file__))
        
        # Path to config file (websearch_config is in same folder as this script)
        config_file = os.path.join(current_dir, "websearch_config", "geographic_config.yaml")
        
        print(f"🔍 Looking for geographic config at: {config_file}")
        
        with open(config_file, 'r', encoding='utf-8') as file:
            config = yaml.safe_load(file)
        
        print(f"✅ Loaded geographic configuration from {config_file}")
        return config
        
    except FileNotFoundError:
        print(f"❌ Geographic config file not found. Using default config.")
        return cls._get_default_config()
    except yaml.YAMLError as e:
        print(f"❌ Error parsing YAML config: {e}. Using default config.")
        return cls._get_default_config()
    except Exception as e:
        print(f"❌ Error loading geographic config: {e}. Using default config.")
        return cls._get_default_config()
```

## 🎯 **What This Does:**

- Uses `os.path.dirname(os.path.abspath(__file__))` to get the directory where `websearch_agent_get_code.py` is located
- Looks for `websearch_config` folder in the same directory as the script
- Finds your files: `tools/websearch_config/specialty_mappings/` and `tools/websearch_config/geographic_config.yaml`

**This will work regardless of where you run the script from (app/backend)!** 🚀

Now test it again and the file not found errors should be resolved!
