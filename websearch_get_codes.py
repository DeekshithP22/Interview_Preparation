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
