Looking at your blob storage requirements and workflow, let me provide a comprehensive step-by-step guide for storage integration:

## Complete Storage Integration Guide

### Storage Requirements Mapped to Blob Paths:
1. **VR Record** → `agent-output/vr-data/{vrId}/{YYYYMMDD}/vr-data_{vrId}_{YYYYMMDDHHMMSS}.json`
2. **OK DB Results** → Store with VR data in same file
3. **VR vs OK DB Comparison** → `agent-output/comparison/{vrId}/{YYYYMMDD}/comparison_{vrId}_{YYYYMMDDHHMMSS}.json`
4. **Search Results** → `agent-output/search-results/{vrId}/{YYYYMMDD}/tool_{toolName}_{vrId}_{YYYYMMDDHHMMSS}.json`
5. **AI Summary** → `agent-output/ai-summary/{vrId}/{YYYYMMDD}/summary_{vrId}_{YYYYMMDDHHMMSS}.json`
6. **Failed Records** → `agent-output/failed-records/{YYYYMMDD}/failed_{vrId}_{YYYYMMDDHHMMSS}.json`
7. **Daily VR Batch** → `agent-output/vr-api-batches/{YYYYMMDD}/vr_api_batch_{start_date}_{end_date}_{YYYYMMDDHHMMSS}.json`

## Step-by-Step Implementation

### Step 1: Update State Model
**File**: `my_agent/utils/state_models.py`

Add storage tracking to AgentState:
```python
class AgentState(TypedDict):
    # ... existing fields ...
    
    # Storage tracking (ADD THESE)
    run_id: str  # Format: {vrId}_{YYYYMMDDHHMMSS}
    batch_id: Optional[str]  # From batch processor
    storage_paths: Dict[str, str]  # Track all storage paths
    storage_status: Dict[str, bool]  # Track upload success
```

### Step 2: Create Storage Manager
**File**: `my_agent/utils/storage_integration.py` (NEW FILE)

```python
import json
import logging
from datetime import datetime
from typing import Dict, Any, Optional, List
from azure.storage.blob import BlobServiceClient
import os

logger = logging.getLogger(__name__)

class AgentStorageManager:
    def __init__(self):
        connection_string = os.getenv("AZURE_STORAGE_CONNECTION_STRING")
        self.blob_service_client = BlobServiceClient.from_connection_string(connection_string)
        self.container_name = "agent-output"
        
    def generate_blob_path(self, vr_id: str, file_type: str, tool_name: Optional[str] = None) -> str:
        """Generate blob storage path following the structure"""
        date_folder = datetime.now().strftime("%Y%m%d")
        timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
        
        if file_type == "vr_data":
            filename = f"vr-data_{vr_id}_{timestamp}.json"
            return f"agent-output/vr-data/{vr_id}/{date_folder}/{filename}"
            
        elif file_type == "comparison":
            filename = f"comparison_{vr_id}_{timestamp}.json"
            return f"agent-output/comparison/{vr_id}/{date_folder}/{filename}"
            
        elif file_type == "search_result" and tool_name:
            filename = f"tool_{tool_name}_{vr_id}_{timestamp}.json"
            return f"agent-output/search-results/{vr_id}/{date_folder}/{filename}"
            
        elif file_type == "ai_summary":
            filename = f"summary_{vr_id}_{timestamp}.json"
            return f"agent-output/ai-summary/{vr_id}/{date_folder}/{filename}"
            
        elif file_type == "failed_record":
            filename = f"failed_{vr_id}_{timestamp}.json"
            return f"agent-output/failed-records/{date_folder}/{filename}"
            
        elif file_type == "vr_api_batch":
            filename = f"vr_api_batch_{timestamp}.json"
            return f"agent-output/vr-api-batches/{date_folder}/{filename}"
    
    async def store_json_data(self, data: Dict[str, Any], vr_id: str, file_type: str, 
                            tool_name: Optional[str] = None) -> Optional[str]:
        """Store JSON data to blob storage"""
        try:
            blob_path = self.generate_blob_path(vr_id, file_type, tool_name)
            json_content = json.dumps(data, indent=2, ensure_ascii=False)
            
            blob_client = self.blob_service_client.get_blob_client(
                container=self.container_name, 
                blob=blob_path
            )
            blob_client.upload_blob(json_content, overwrite=True)
            
            logger.info(f"Stored {file_type} for VR ID {vr_id} at {blob_path}")
            return blob_path
            
        except Exception as e:
            logger.error(f"Failed to store {file_type} for VR ID {vr_id}: {str(e)}")
            return None
```

### Step 3: Update Supervisor Agent Storage Points

**File**: `my_agent/agents/supervisor_agent.py`

Add storage manager and update methods:

```python
from my_agent.utils.storage_integration import AgentStorageManager

class SupervisorAgent:
    def __init__(self):
        # ... existing init ...
        self.storage_manager = AgentStorageManager()
    
    async def execute_okdb_search(self, state: AgentState) -> AgentState:
        # ... existing search logic ...
        
        # STORAGE POINT 1 & 2: Store VR data + OK DB results together
        if state.get("workflow_status") == WorkflowStatus.OKDB_SEARCHED:
            vr_id = str(state["vr_record"].get("id"))
            
            # Prepare combined data
            vr_okdb_data = {
                "run_id": state.get("run_id"),
                "batch_id": state.get("batch_id"),
                "timestamp": datetime.now().isoformat(),
                "vr_record": state["vr_record"],
                "search_strategy": state.get("search_strategy"),
                "okdb_results": {
                    "primary_results": state.get("okdb_primary_results"),
                    "secondary_results": state.get("okdb_secondary_results"),
                    "single_result": state.get("okdb_api_response") if not state.get("okdb_primary_results") else None
                },
                "record_status": state.get("record_status", "unknown")
            }
            
            # Store VR + OK DB data
            storage_path = await self.storage_manager.store_json_data(
                data=vr_okdb_data,
                vr_id=vr_id,
                file_type="vr_data"
            )
            
            if storage_path:
                state["storage_paths"]["vr_okdb_data"] = storage_path
                state["storage_status"]["vr_okdb_data"] = True
        
        return state
    
    async def analyze_vr_vs_okdb(self, state: AgentState) -> AgentState:
        # ... existing analysis logic ...
        
        # STORAGE POINT 3: Store VR vs OK DB comparison (with fuzzy logic output)
        if state.get("comparison_analysis") and state.get("workflow_status") == WorkflowStatus.ANALYSIS_COMPLETED:
            vr_id = str(state["vr_record"].get("id"))
            
            # Assuming you have fuzzy logic output
            comparison_data = {
                "run_id": state.get("run_id"),
                "batch_id": state.get("batch_id"),
                "timestamp": datetime.now().isoformat(),
                "vr_id": vr_id,
                "comparison_analysis": state["comparison_analysis"],
                "fuzzy_logic_output": state.get("fuzzy_logic_output", {}),  # Your fuzzy logic result
                "record_status": state.get("record_status"),
                "search_requirements": state.get("search_requirements"),
                "okdb_comparison_analysis": state.get("okdb_comparison_analysis")
            }
            
            storage_path = await self.storage_manager.store_json_data(
                data=comparison_data,
                vr_id=vr_id,
                file_type="comparison"
            )
            
            if storage_path:
                state["storage_paths"]["comparison"] = storage_path
                state["storage_status"]["comparison"] = True
        
        return state
    
    async def make_dbo_decision(self, state: AgentState) -> AgentState:
        # ... existing decision logic ...
        
        # STORAGE POINT 5: Store final AI summary
        if state.get("dbo_action_decision"):
            vr_id = str(state["vr_record"].get("id"))
            
            ai_summary_data = {
                "run_id": state.get("run_id"),
                "batch_id": state.get("batch_id"),
                "timestamp": datetime.now().isoformat(),
                "vr_id": vr_id,
                "dbo_decision": state["dbo_action_decision"],
                "workflow_status": state["workflow_status"].value,
                "search_confidence": state.get("search_confidence", 0.0),
                "processing_summary": {
                    "record_status": state.get("record_status"),
                    "verification_performed": state.get("search_requirements", {}).get("verification_needed", False),
                    "tools_used": state.get("selected_tools", [])
                }
            }
            
            storage_path = await self.storage_manager.store_json_data(
                data=ai_summary_data,
                vr_id=vr_id,
                file_type="ai_summary"
            )
            
            if storage_path:
                state["storage_paths"]["ai_summary"] = storage_path
                state["storage_status"]["ai_summary"] = True
        
        return state
```

### Step 4: Update Search & Summarize Agent Storage

**File**: `my_agent/agents/search_summarize_agent.py`

```python
from my_agent.utils.storage_integration import AgentStorageManager

class SearchAndSummarizeAgent:
    def __init__(self):
        # ... existing init ...
        self.storage_manager = AgentStorageManager()
    
    async def execute_search_tools(self, state: AgentState) -> AgentState:
        vr_id = str(state["vr_record"].get("id"))
        
        try:
            # ... existing search logic ...
            
            # STORAGE POINT 4: Store each tool's results
            for tool_name in execution_order:
                if tool_name in self.search_tools:
                    # Execute tool
                    search_result = await self.search_tools[tool_name].search(...)
                    
                    if search_result:
                        # Store individual tool result
                        tool_data = {
                            "run_id": state.get("run_id"),
                            "batch_id": state.get("batch_id"),
                            "timestamp": datetime.now().isoformat(),
                            "vr_id": vr_id,
                            "tool_name": tool_name,
                            "search_result": search_result,
                            "search_requirements": state.get("search_requirements")
                        }
                        
                        storage_path = await self.storage_manager.store_json_data(
                            data=tool_data,
                            vr_id=vr_id,
                            file_type="search_result",
                            tool_name=tool_name
                        )
                        
                        if storage_path:
                            state["storage_paths"][f"search_{tool_name}"] = storage_path
            
        except Exception as e:
            # ... error handling ...
            pass
        
        return state
    
    async def intelligent_summarization(self, state: AgentState) -> AgentState:
        # ... existing summarization logic ...
        
        # Store search summary (separate from individual tool results)
        if state.get("intelligent_summary"):
            vr_id = str(state["vr_record"].get("id"))
            
            search_summary_data = {
                "run_id": state.get("run_id"),
                "batch_id": state.get("batch_id"),
                "timestamp": datetime.now().isoformat(),
                "vr_id": vr_id,
                "intelligent_summary": state["intelligent_summary"],
                "search_confidence": state.get("search_confidence"),
                "tools_executed": state.get("selected_tools", []),
                "total_results": len(state.get("search_results", []))
            }
            
            # Store as a search result with special name
            storage_path = await self.storage_manager.store_json_data(
                data=search_summary_data,
                vr_id=vr_id,
                file_type="search_result",
                tool_name="search_summary"
            )
            
            if storage_path:
                state["storage_paths"]["search_summary"] = storage_path
        
        return state
```

### Step 5: Update Main Workflow

**File**: `my_agent/main.py`

```python
async def process_single_vr_record(vr_record: dict, batch_id: Optional[str] = None) -> dict:
    vr_id = vr_record.get('id', 'unknown')
    logger.info(f"Processing VR ID: {vr_id}")
    
    # Create workflow
    app = create_agent_vr_workflow()
    
    # Generate run_id
    timestamp = datetime.now().strftime("%Y%m%d%H%M%S")
    run_id = f"{vr_id}_{timestamp}"

    # Initial state with storage tracking
    initial_state = AgentState(
        # ... existing fields ...
        
        # Storage tracking
        run_id=run_id,
        batch_id=batch_id,
        storage_paths={},
        storage_status={}
    )
    
    # Execute workflow
    result = await app.ainvoke(initial_state)
    
    # Log storage summary
    logger.info(f"Storage summary for VR ID {vr_id}:")
    for storage_type, path in result.get("storage_paths", {}).items():
        logger.info(f"  - {storage_type}: {path}")
    
    return result
```

### Step 6: Update Batch Processor for Failed Records

**File**: `my_agent/batch_processor.py`

```python
async def process_vr_batch(date_range: Dict[str, str]) -> List[Dict[str, Any]]:
    # ... existing code ...
    storage_manager = AgentStorageManager()
    
    # STORAGE POINT 7: Store raw VR API batch
    if raw_vr_records:
        batch_data = {
            "batch_id": batch_id,
            "date_range": date_range,
            "extraction_timestamp": datetime.now().isoformat(),
            "total_records": len(raw_vr_records),
            "records": raw_vr_records
        }
        
        batch_path = await storage_manager.store_json_data(
            data=batch_data,
            vr_id="batch",  # Special ID for batch
            file_type="vr_api_batch"
        )
        logger.info(f"Stored VR API batch at: {batch_path}")
    
    # Process records
    for vr_record in processed_vr_records:
        try:
            result = await process_single_vr_record(vr_record, batch_id=batch_id)
            # ... success handling ...
            
        except Exception as e:
            # STORAGE POINT 6: Store failed records
            vr_id = str(vr_record.get("id", "unknown"))
            
            failed_record_data = {
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat(),
                "vr_record": vr_record,
                "error": str(e),
                "error_type": type(e).__name__,
                "workflow_stage": "unknown"  # Could track where it failed
            }
            
            failed_path = await storage_manager.store_json_data(
                data=failed_record_data,
                vr_id=vr_id,
                file_type="failed_record"
            )
            
            logger.error(f"Stored failed record at: {failed_path}")
```

### Step 7: Add Error Handler Storage

**File**: `my_agent/main.py`

Update the error handler to store failed records:

```python
async def handle_error(state: AgentState) -> AgentState:
    """Global error handler for workflow"""
    error_context = state.get("error_context", {})
    vr_id = str(state["vr_record"].get("id", "unknown"))
    
    # Store failed record
    storage_manager = AgentStorageManager()
    
    failed_data = {
        "run_id": state.get("run_id"),
        "batch_id": state.get("batch_id"),
        "timestamp": datetime.now().isoformat(),
        "vr_record": state["vr_record"],
        "error_context": error_context,
        "workflow_status": state.get("workflow_status", "").value if hasattr(state.get("workflow_status", ""), "value") else str(state.get("workflow_status", "")),
        "last_successful_stage": state.get("current_agent", "unknown"),
        "partial_results": {
            "okdb_searched": bool(state.get("okdb_api_response")),
            "comparison_done": bool(state.get("comparison_analysis")),
            "search_done": bool(state.get("search_results"))
        }
    }
    
    await storage_manager.store_json_data(
        data=failed_data,
        vr_id=vr_id,
        file_type="failed_record"
    )
    
    # ... rest of error handling ...
    
    return state
```

## Summary of Storage Points:

1. ✅ **VR + OK DB Results** - Stored together after OK DB search
2. ✅ **VR vs OK DB Comparison** - Stored after analysis with fuzzy logic output
3. ✅ **Search Results** - Each tool's results stored individually
4. ✅ **Search Summary** - Stored as special search result
5. ✅ **Final AI Summary** - Stored after DBO decision
6. ✅ **Failed Records** - Stored in error handler and batch processor
7. ✅ **Daily VR Batch** - Stored at beginning of batch processing

## Benefits for UI Team:

- **Consistent paths**: `{file_type}/{vrId}/{date}/{filename}`
- **Easy filtering**: Can query by date, VR ID, or file type
- **Complete audit trail**: Every step is stored
- **Failed record recovery**: Easy to find and reprocess failures
- **Batch tracking**: Can see all records from a daily run

This implementation ensures complete storage coverage with minimal performance impact!







































    def __init__(self):
        account_url = os.getenv("AZURE_BLOB_STORAGE_ACCOUNT_URL")
        container_name = os.getenv("AZURE_BLOB_STORAGE_CONTAINER")
        account_key = os.getenv("AZURE_BLOB_STORAGE_ACCOUNT_KEY")

        self.container_name = container_name

        # Use account key if available (typically local dev), otherwise managed identity
        if account_key:
            account_name = account_url.split("//")[1].split(".")[0]
            credential = StorageSharedKeyCredential(account_name, account_key)
        else:
            credential = DefaultAzureCredential()

        self.blob_service_client = BlobServiceClient(
            account_url=account_url,
            credential=credential
        )

        self.container_client = self.blob_service_client.get_container_client(container_name)
