# ## Task 1: Complete Corrected Batch Processing File

# **File**: `my_agent/batch_processor.py`

# ```python
# """
# Production Batch Processor for VR Records
# Designed for Azure Function deployment with comprehensive error handling and blob storage
# """

# import asyncio
# import json
# import logging
# import sys
# import traceback
# from datetime import datetime, timedelta
# from typing import List, Dict, Any, Optional, Tuple
# from dataclasses import dataclass
# import os

# # Import your workflow
# from my_agent.main import process_single_vr_record
# from my_agent.utils.storage_integration import AgentStorageManager
# from my_agent.utils.state_models import WorkflowStatus

# # Configure logging
# logging.basicConfig(
#     level=logging.INFO,
#     format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
#     handlers=[logging.StreamHandler(sys.stdout)]
# )
# logger = logging.getLogger(__name__)

# # TODO: Import your actual VR API client
# # from your_vr_api import VRAPIClient


# @dataclass
# class BatchMetrics:
#     """Track batch processing metrics"""
#     batch_id: str
#     start_time: datetime
#     date_range: Dict[str, str]
#     total_raw_records: int = 0
#     valid_records: int = 0
#     invalid_records: int = 0
#     processed_successfully: int = 0
#     failed_processing: int = 0
#     skipped_duplicates: int = 0
#     api_errors: int = 0
#     validation_errors: List[Dict] = None
#     processing_errors: List[Dict] = None
    
#     def __post_init__(self):
#         if self.validation_errors is None:
#             self.validation_errors = []
#         if self.processing_errors is None:
#             self.processing_errors = []
    
#     def to_dict(self) -> Dict:
#         return {
#             "batch_id": self.batch_id,
#             "start_time": self.start_time.isoformat(),
#             "end_time": datetime.now().isoformat(),
#             "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
#             "date_range": self.date_range,
#             "metrics": {
#                 "total_raw_records": self.total_raw_records,
#                 "valid_records": self.valid_records,
#                 "invalid_records": self.invalid_records,
#                 "processed_successfully": self.processed_successfully,
#                 "failed_processing": self.failed_processing,
#                 "skipped_duplicates": self.skipped_duplicates,
#                 "api_errors": self.api_errors,
#                 "success_rate": (self.processed_successfully / self.valid_records * 100) if self.valid_records > 0 else 0
#             },
#             "validation_errors": self.validation_errors[:100],  # Limit to 100 for storage
#             "processing_errors": self.processing_errors[:100]   # Limit to 100 for storage
#         }


# class VRBatchProcessor:
#     """Production-ready batch processor with comprehensive error handling"""
    
#     def __init__(self):
#         self.storage_manager = AgentStorageManager()
#         # TODO: Initialize your VR API client
#         # self.vr_api_client = VRAPIClient()
#         self.processed_vr_ids = set()  # Track processed IDs to avoid duplicates
        
#     def validate_vr_record(self, record: Dict[str, Any], index: int) -> Tuple[bool, Optional[str]]:
#         """
#         Minimal validation - only check what's absolutely required for workflow
#         Returns: (is_valid, error_message)
#         """
#         try:
#             # Critical: Must have an ID for tracking and storage
#             vr_id = record.get("id")
#             if not vr_id:
#                 return False, "Missing VR ID - required for processing"
            
#             # Critical: Must have entity type for routing decisions
#             entity_type = record.get("entityTypeIco")
#             if not entity_type:
#                 return False, "Missing entityTypeIco - required for workflow routing"
            
#             # Validate entity type values
#             valid_entity_types = ["ENT_ACTIVITY", "ENT_WORKPLACE"]
#             if entity_type not in valid_entity_types:
#                 return False, f"Invalid entityTypeIco: {entity_type}"
            
#             # Check for duplicate processing
#             if str(vr_id) in self.processed_vr_ids:
#                 return False, f"Duplicate VR ID: {vr_id}"
            
#             # All other fields are optional - workflow handles missing data gracefully
#             return True, None
            
#         except Exception as e:
#             return False, f"Validation error: {str(e)}"
    
#     async def fetch_vr_records(self, date_range: Dict[str, str]) -> Tuple[List[Dict], Optional[str]]:
#         """
#         Fetch VR records from API - returns already processed data
#         Returns: (records, error_message)
#         """
#         try:
#             logger.info(f"Fetching VR records for date range: {date_range}")
            
#             # TODO: Replace with your actual VR API call
#             # The API returns data that's already processed by VR data class
#             # processed_records = await self.vr_api_client.get_records_by_date_range(
#             #     start_date=date_range["start_date"],
#             #     end_date=date_range["end_date"]
#             # )
            
#             # Mock implementation - replace with actual
#             processed_records = []
            
#             if not isinstance(processed_records, list):
#                 return [], "VR API returned invalid response format"
            
#             logger.info(f"Fetched {len(processed_records)} processed records from VR API")
#             return processed_records, None
            
#         except asyncio.TimeoutError:
#             error_msg = "VR API timeout error"
#             logger.error(error_msg)
#             return [], error_msg
            
#         except ConnectionError as e:
#             error_msg = f"VR API connection error: {str(e)}"
#             logger.error(error_msg)
#             return [], error_msg
            
#         except Exception as e:
#             error_msg = f"VR API error: {str(e)}"
#             logger.error(error_msg)
#             return [], error_msg
    
#     async def store_batch_logs(self, batch_id: str, log_type: str, log_data: Dict):
#         """Store batch processing logs to blob storage"""
#         try:
#             log_entry = {
#                 "batch_id": batch_id,
#                 "timestamp": datetime.now().isoformat(),
#                 "log_type": log_type,
#                 "data": log_data
#             }
            
#             date_folder = datetime.now().strftime("%Y%m%d")
#             filename = f"batch_log_{log_type}_{batch_id}_{datetime.now().strftime('%H%M%S')}.json"
#             blob_path = f"agent-output/batch-logs/{date_folder}/{filename}"
            
#             blob_client = self.storage_manager.blob_service_client.get_blob_client(
#                 container=self.storage_manager.container_name,
#                 blob=blob_path
#             )
#             blob_client.upload_blob(
#                 json.dumps(log_entry, indent=2),
#                 overwrite=True
#             )
            
#         except Exception as e:
#             logger.error(f"Failed to store batch log: {str(e)}")
    
#     async def process_batch(self, date_range: Dict[str, str]) -> Dict[str, Any]:
#         """
#         Main batch processing logic with comprehensive error handling
#         """
#         # Initialize batch
#         batch_start_time = datetime.now()
#         batch_timestamp = batch_start_time.strftime('%Y%m%d_%H%M%S')
#         batch_id = f"batch_{date_range['start_date']}_{batch_timestamp}"
        
#         # Initialize metrics
#         metrics = BatchMetrics(
#             batch_id=batch_id,
#             start_time=batch_start_time,
#             date_range=date_range
#         )
        
#         # Store batch start log
#         await self.store_batch_logs(batch_id, "start", {
#             "date_range": date_range,
#             "environment": os.getenv("ENVIRONMENT", "production")
#         })
        
#         try:
#             # Step 1: Fetch VR records (already processed by data class)
#             processed_records, api_error = await self.fetch_vr_records(date_range)
            
#             if api_error:
#                 metrics.api_errors += 1
#                 await self.store_batch_logs(batch_id, "api_error", {"error": api_error})
                
#                 # If no records at all, exit early
#                 if not processed_records:
#                     return self._finalize_batch(metrics, "api_failure")
            
#             metrics.total_raw_records = len(processed_records)
            
#             # Step 2: Store raw VR batch data
#             if processed_records:
#                 try:
#                     batch_storage_data = {
#                         "batch_id": batch_id,
#                         "date_range": date_range,
#                         "extraction_timestamp": datetime.now().isoformat(),
#                         "total_records": len(processed_records),
#                         "records": processed_records
#                     }
                    
#                     batch_path = await self.storage_manager.store_json_data(
#                         data=batch_storage_data,
#                         vr_id="batch",
#                         file_type="vr_api_batch"
#                     )
                    
#                     logger.info(f"Stored VR batch at: {batch_path}")
                    
#                 except Exception as e:
#                     logger.error(f"Failed to store batch: {str(e)}")
#                     await self.store_batch_logs(batch_id, "storage_error", {
#                         "stage": "batch_storage",
#                         "error": str(e)
#                     })
            
#             # Step 3: Validate records (minimal validation)
#             valid_records = []
            
#             for idx, vr_record in enumerate(processed_records):
#                 record_id = vr_record.get("id", f"unknown_{idx}")
                
#                 # Minimal validation
#                 is_valid, validation_error = self.validate_vr_record(vr_record, idx)
                
#                 if not is_valid:
#                     metrics.invalid_records += 1
#                     metrics.validation_errors.append({
#                         "index": idx,
#                         "id": record_id,
#                         "error": validation_error
#                     })
                    
#                     if "Duplicate" in validation_error:
#                         metrics.skipped_duplicates += 1
                    
#                     continue
                
#                 valid_records.append(vr_record)
#                 self.processed_vr_ids.add(str(record_id))
#                 metrics.valid_records += 1
            
#             # Log validation summary if there were errors
#             if metrics.validation_errors:
#                 await self.store_batch_logs(batch_id, "validation_summary", {
#                     "total_errors": len(metrics.validation_errors),
#                     "invalid_records": metrics.invalid_records
#                 })
            
#             # Step 4: Process valid records through workflow
#             logger.info(f"Processing {len(valid_records)} valid records through workflow")
            
#             workflow_results = []
            
#             for idx, vr_record in enumerate(valid_records):
#                 vr_id = str(vr_record.get("id"))
                
#                 # Progress logging every 10 records
#                 if idx % 10 == 0 and idx > 0:
#                     logger.info(f"Processing progress: {idx}/{len(valid_records)}")
                
#                 try:
#                     # Process record through workflow
#                     result = await process_single_vr_record(vr_record, batch_id=batch_id)
                    
#                     # Check workflow status
#                     workflow_status = result.get("workflow_status")
                    
#                     if workflow_status == WorkflowStatus.DBO_DECISION_READY:
#                         metrics.processed_successfully += 1
#                         status = "success"
#                     elif workflow_status == WorkflowStatus.ERROR:
#                         metrics.failed_processing += 1
#                         status = "error"
#                     else:
#                         metrics.failed_processing += 1
#                         status = "incomplete"
                    
#                     workflow_results.append({
#                         "vr_id": vr_id,
#                         "status": status,
#                         "workflow_status": str(workflow_status),
#                         "dbo_decision": result.get("dbo_action_decision", {}).get("overall_recommendation", ""),
#                         "storage_paths": result.get("storage_paths", {}),
#                         "processing_time": datetime.now().isoformat()
#                     })
                    
#                 except Exception as e:
#                     error_msg = f"Workflow error: {str(e)}"
#                     logger.error(f"VR ID {vr_id}: {error_msg}")
#                     logger.debug(f"Traceback: {traceback.format_exc()}")
                    
#                     metrics.failed_processing += 1
#                     metrics.processing_errors.append({
#                         "vr_id": vr_id,
#                         "error": error_msg,
#                         "error_type": type(e).__name__
#                     })
                    
#                     # Store failed record
#                     await self._store_failed_record(vr_record, error_msg, batch_id, "workflow_error")
                    
#                     workflow_results.append({
#                         "vr_id": vr_id,
#                         "status": "error",
#                         "error": error_msg,
#                         "processing_time": datetime.now().isoformat()
#                     })
                
#                 # Rate limiting to avoid overwhelming services
#                 await asyncio.sleep(0.5)
            
#             # Step 5: Store batch summary
#             batch_summary = {
#                 **metrics.to_dict(),
#                 "workflow_results": workflow_results,
#                 "status": "completed"
#             }
            
#             summary_path = await self._store_batch_summary(batch_summary)
#             logger.info(f"Batch processing completed. Summary at: {summary_path}")
            
#             # Store completion log
#             await self.store_batch_logs(batch_id, "completion", {
#                 "metrics": metrics.to_dict()["metrics"],
#                 "summary_path": summary_path
#             })
            
#             return batch_summary
            
#         except Exception as e:
#             # Catastrophic failure
#             logger.error(f"Batch processing failed: {str(e)}")
#             logger.debug(f"Traceback: {traceback.format_exc()}")
            
#             # Store error log
#             await self.store_batch_logs(batch_id, "catastrophic_error", {
#                 "error": str(e),
#                 "traceback": traceback.format_exc()
#             })
            
#             return self._finalize_batch(metrics, "catastrophic_failure", str(e))
    
#     async def _store_failed_record(self, vr_record: Dict, error: str, batch_id: str, error_type: str):
#         """Store failed record with detailed error information"""
#         try:
#             vr_id = str(vr_record.get("id", "unknown"))
            
#             failed_record_data = {
#                 "batch_id": batch_id,
#                 "timestamp": datetime.now().isoformat(),
#                 "error_type": error_type,
#                 "error": error,
#                 "vr_record": vr_record,
#                 "can_retry": True  # All workflow errors can be retried
#             }
            
#             await self.storage_manager.store_json_data(
#                 data=failed_record_data,
#                 vr_id=vr_id,
#                 file_type="failed_record"
#             )
            
#         except Exception as e:
#             logger.error(f"Failed to store failed record: {str(e)}")
    
#     async def _store_batch_summary(self, summary: Dict) -> str:
#         """Store batch processing summary"""
#         try:
#             date_folder = datetime.now().strftime("%Y%m%d")
#             filename = f"batch_summary_{summary['batch_id']}.json"
#             blob_path = f"agent-output/batch-summaries/{date_folder}/{filename}"
            
#             blob_client = self.storage_manager.blob_service_client.get_blob_client(
#                 container=self.storage_manager.container_name,
#                 blob=blob_path
#             )
#             blob_client.upload_blob(
#                 json.dumps(summary, indent=2),
#                 overwrite=True
#             )
            
#             return blob_path
            
#         except Exception as e:
#             logger.error(f"Failed to store batch summary: {str(e)}")
#             return ""
    
#     def _finalize_batch(self, metrics: BatchMetrics, status: str, error: Optional[str] = None) -> Dict:
#         """Finalize batch with error status"""
#         summary = metrics.to_dict()
#         summary["status"] = status
#         if error:
#             summary["error"] = error
#         return summary


# # Azure Function entry point
# async def main(date_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
#     """
#     Main entry point for Azure Function
    
#     Args:
#         date_range: Optional date range, defaults to previous day
#     """
#     try:
#         # Default to previous day if no date range provided
#         if not date_range:
#             yesterday = datetime.now() - timedelta(days=1)
#             date_range = {
#                 "start_date": yesterday.strftime("%Y-%m-%d"),
#                 "end_date": yesterday.strftime("%Y-%m-%d")
#             }
        
#         logger.info(f"Starting VR batch processing for: {date_range}")
        
#         # Create processor and run
#         processor = VRBatchProcessor()
#         result = await processor.process_batch(date_range)
        
#         logger.info(f"Batch processing completed with status: {result.get('status')}")
        
#         return result
        
#     except Exception as e:
#         logger.error(f"Fatal error in batch processor: {str(e)}")
#         return {
#             "status": "fatal_error",
#             "error": str(e),
#             "timestamp": datetime.now().isoformat()
#         }


# # For local testing
# if __name__ == "__main__":
#     # Example: Process specific date
#     test_date_range = {
#         "start_date": "2025-01-15",
#         "end_date": "2025-01-15"
#     }
    
#     result = asyncio.run(main(test_date_range))
#     print(json.dumps(result, indent=2))
# ```

# ## Key Changes Made:

# 1. **Removed VR Data Class Processing** - The API already returns processed data
# 2. **Removed Individual Record Timeout** - As requested
# 3. **Minimal Validation** - Only validates:
#    - VR ID (required for tracking/storage)
#    - Entity Type (required for workflow routing)
#    - Duplicate check
# 4. **No Field Requirements** - No validation for firstName, lastName, workplaceName etc.

# ---

# ## Task 2: Analysis of VR Record Data Handling in Workflow

# Looking at your entire workflow, here are the critical areas where VR record data is accessed:

# ### 1. **Search Strategy Determination** (`determine_okdb_search_strategy`)
# - Accesses: `entityTypeIco`, `individualOriginKeyEid`, `workplaceOriginKeyEid`
# - Fallbacks: `firstName`, `lastName`, `workplaceUsualName`, `country`, `city`
# - **Risk**: If origin keys are missing, falls back to name-based search

# ### 2. **OK DB Search** (`execute_okdb_search`)
# - When no results: Builds search requirements using:
#   - `countryCode`, `firstName`, `lastName`, `workplaceUsualName`, `city`, `specialityCode1`
# - **Risk**: All these fields might be empty/missing

# ### 3. **VR vs OK DB Analysis** (`analyze_vr_vs_okdb`)
# - Sends entire VR record to LLM for comparison
# - Overrides search requirements using same fields as above
# - **Risk**: LLM needs to handle missing fields gracefully

# ### 4. **Search Requirements Construction**
# - Your exact format expects these fields, but they might be missing
# - **Risk**: Search tools need to handle empty values

# ### Recommendations for Robustness:

# 1. **Use `.get()` with defaults everywhere**:
#    ```python
#    firstName = vr_record.get("firstName", "")
#    lastName = vr_record.get("lastName", "")
#    ```

# 2. **Update search requirements to handle missing data**:
#    ```python
#    search_requirements = {
#        "verification_needed": True,
#        "geographic_region": vr_record.get("countryCode", ""),
#        "firstName": vr_record.get("firstName", ""),
#        "lastName": vr_record.get("lastName", ""),
#        "workplaceName": vr_record.get("workplaceUsualName", ""),
#        "address": vr_record.get("city", vr_record.get("postalCity", "")),
#        "specialityCode": vr_record.get("specialityCode1", vr_record.get("specialityCode2", "")),
#        "entity_type": "ent_activity" if vr_record.get("entityTypeIco") == "ENT_ACTIVITY" else "ent_workplace"
#    }
#    ```

# 3. **Search tools should skip empty values**:
#    - If firstName and lastName are empty, search by workplace only
#    - If workplace is empty, search by name only
#    - If all are empty, log warning but continue

# 4. **LLM prompts should handle missing data**:
#    - Update prompts to explicitly state fields might be missing
#    - Ask LLM to work with available data only

# Your workflow can handle missing data gracefully with these considerations!
























"""
Production Batch Processor for VR Records
Designed for Azure Function deployment with comprehensive error handling and blob storage
"""

import asyncio
import json
import logging
import sys
import traceback
from datetime import datetime, timedelta
from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import os

# Import your workflow
from app.my_agent.agent import process_single_vr_record
from app.my_agent.utils.storage_integration import AgentStorageManager
from app.my_agent.utils.state import WorkflowStatus

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger(__name__)

# TODO: Import actual VR API client
# from your_vr_api import VRAPIClient


@dataclass
class BatchMetrics:
    """Track batch processing metrics"""
    batch_id: str
    start_time: datetime
    date_range: Dict[str, str]
    total_raw_records: int = 0
    valid_records: int = 0
    invalid_records: int = 0
    processed_successfully: int = 0
    failed_processing: int = 0
    skipped_duplicates: int = 0
    api_errors: int = 0
    validation_errors: List[Dict] = None
    processing_errors: List[Dict] = None
    
    def __post_init__(self):
        if self.validation_errors is None:
            self.validation_errors = []
        if self.processing_errors is None:
            self.processing_errors = []
    
    def to_dict(self) -> Dict:
        return {
            "batch_id": self.batch_id,
            "start_time": self.start_time.isoformat(),
            "end_time": datetime.now().isoformat(),
            "duration_seconds": (datetime.now() - self.start_time).total_seconds(),
            "date_range": self.date_range,
            "metrics": {
                "total_raw_records": self.total_raw_records,
                "valid_records": self.valid_records,
                "invalid_records": self.invalid_records,
                "processed_successfully": self.processed_successfully,
                "failed_processing": self.failed_processing,
                "skipped_duplicates": self.skipped_duplicates,
                "api_errors": self.api_errors,
                "success_rate": (self.processed_successfully / self.valid_records * 100) if self.valid_records > 0 else 0
            },
            "validation_errors": self.validation_errors[:100],  # Limit to 100 for storage
            "processing_errors": self.processing_errors[:100]   # Limit to 100 for storage
        }


class VRBatchProcessor:
    """Production-ready batch processor with comprehensive error handling"""
    
    def __init__(self):
        self.storage_manager = AgentStorageManager()
        # TODO: Initialize VR API client
        # self.vr_api_client = VRAPIClient()
        self.processed_vr_ids = set()  # Track processed IDs to avoid duplicates
        
    def validate_vr_record(self, record: Dict[str, Any], index: int) -> Tuple[bool, Optional[str]]:
        """
        Minimal validation - only check what's absolutely required for workflow
        Returns: (is_valid, error_message)
        """
        try:
            # Critical: Must have an ID for tracking and storage
            vr_id = record.get("validation.id")
            if not vr_id:
                return False, "Missing VR ID - required for processing"
            
            # Critical: Must have entity type for routing decisions
            entity_type = record.get("validation.entityTypeIco")
            if not entity_type:
                return False, "Missing entityTypeIco - required for workflow routing"
            
            # Validate entity type values
            valid_entity_types = ["ENT_ACTIVITY", "ENT_WORKPLACE"]
            if entity_type not in valid_entity_types:
                return False, f"Invalid entityTypeIco: {entity_type}"
            
            # Check for duplicate processing
            if str(vr_id) in self.processed_vr_ids:
                return False, f"Duplicate VR ID: {vr_id}"
            
            # All other fields are optional - workflow handles missing data gracefully
            return True, None
            
        except Exception as e:
            return False, f"Validation error: {str(e)}"
    
    async def fetch_vr_json_from_api(self, date_range: Dict[str, str]) -> Tuple[Optional[Dict], Optional[str]]:
        """
        Fetch VR JSON file from API - returns processed data as JSON
        Returns: (json_data, error_message)
        """
        try:
            logger.info(f"Fetching VR JSON for date range: {date_range}")
            
            # TODO: Replace with actual VR API call
            # The API returns a JSON file with all processed records
            # vr_json_data = await self.vr_api_client.get_records_json(
            #     start_date=date_range["start_date"],
            #     end_date=date_range["end_date"]
            # )
            
            # Mock implementation - replace with actual
            vr_json_data = {
                "extraction_date": datetime.now().isoformat(),
                "date_range": date_range,
                "total_records": 0,
                "records": []  # List of VR records
            }
            
            if not isinstance(vr_json_data, dict):
                return None, "VR API returned invalid JSON format"
            
            if "response" not in vr_json_data:
                return None, "VR JSON missing 'response' field"

            response_data = vr_json_data.get("response", {})
            if "results" not in response_data:
                return None, "VR JSON response missing 'results' field"

            # Check if API call was successful
            if not response_data.get("success", False):
                return None, f"VR API returned unsuccessful status: {response_data.get('status', 'UNKNOWN')}"
                        
            logger.info(f"Fetched VR JSON with {len(vr_json_data.get('records', []))} records")
            return vr_json_data, None
            
        except asyncio.TimeoutError:
            error_msg = "VR API timeout error"
            logger.error(error_msg)
            return None, error_msg
            
        except ConnectionError as e:
            error_msg = f"VR API connection error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
            
        except json.JSONDecodeError as e:
            error_msg = f"VR API returned invalid JSON: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
            
        except Exception as e:
            error_msg = f"VR API error: {str(e)}"
            logger.error(error_msg)
            return None, error_msg
    
    async def store_batch_logs(self, batch_id: str, log_type: str, log_data: Dict):
        """Store batch processing logs to blob storage"""
        try:
            log_entry = {
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat(),
                "log_type": log_type,
                "data": log_data
            }
            
            date_folder = datetime.now().strftime("%Y%m%d")
            filename = f"batch_log_{log_type}_{batch_id}_{datetime.now().strftime('%H%M%S')}.json"
            blob_path = f"agent-output/batch-logs/{date_folder}/{filename}"
           
            blob_client = self.storage_manager.blob_service_client.get_blob_client(
                container=self.storage_manager.container_name,
                blob=blob_path
            )
            blob_client.upload_blob(
                json.dumps(log_entry, indent=2),
                overwrite=True
            )
            
        except Exception as e:
            logger.error(f"Failed to store batch log: {str(e)}")
    
    async def process_batch(self, date_range: Dict[str, str]) -> Dict[str, Any]:
        """
        Main batch processing logic with comprehensive error handling
        """
        # Initialize batch
        batch_start_time = datetime.now()
        batch_timestamp = batch_start_time.strftime('%Y%m%d_%H%M%S')
        batch_id = f"batch_{date_range['start_date']}_{batch_timestamp}"
        
        # Initialize metrics
        metrics = BatchMetrics(
            batch_id=batch_id,
            start_time=batch_start_time,
            date_range=date_range
        )
        
        # Store batch start log
        await self.store_batch_logs(batch_id, "start", {
            "date_range": date_range,
            "environment": os.getenv("ENVIRONMENT", "Development"),
        })
        
        try:
            # Step 1: Fetch VR JSON from API
            vr_json_data, api_error = await self.fetch_vr_json_from_api(date_range)

            if api_error:
                metrics.api_errors += 1
                await self.store_batch_logs(batch_id, "api_error", {"error": api_error})
                
                if not vr_json_data:
                    return self._finalize_batch(metrics, "api_failure", api_error)
            
            # Extract records from JSON
            response_data = vr_json_data.get("response", {})
            vr_records = response_data.get("results", [])
            metrics.total_raw_records = len(vr_records)
            
            # Step 2: Store the entire VR JSON batch
            if vr_json_data:
                try:
                    # Add batch metadata to the JSON
                    vr_json_data["batch_id"] = batch_id
                    vr_json_data["stored_timestamp"] = datetime.now().isoformat()
                    
                    # Store the complete JSON
                    date_folder = datetime.now().strftime("%Y%m%d")
                    filename = f"vr_api_batch_{date_range['start_date']}_{date_range['end_date']}_{batch_timestamp}.json"
                    blob_path = f"agent-output/vr-api-batches/{date_folder}/{filename}"
                    
                    blob_client = self.storage_manager.blob_service_client.get_blob_client(
                        container=self.storage_manager.container_name,
                        blob=blob_path
                    )
                    blob_client.upload_blob(
                        json.dumps(vr_json_data, indent=2),
                        overwrite=True
                    )
                    
                    logger.info(f"Stored VR API batch JSON at: {blob_path}")
                    
                    # Store the path for reference
                    await self.store_batch_logs(batch_id, "vr_json_stored", {
                        "blob_path": blob_path,
                        "total_records": len(vr_records)
                    })
                    
                except Exception as e:
                    logger.error(f"Failed to store VR JSON: {str(e)}")
                    await self.store_batch_logs(batch_id, "storage_error", {
                        "stage": "vr_json_storage",
                        "error": str(e)
                    })
            
            # Step 3: Validate records
            valid_records = []
            
            if not vr_records:
                logger.warning("No records found in VR JSON")
                return self._finalize_batch(metrics, "no_records")
            
            for idx, vr_record in enumerate(vr_records):
                record_id = vr_record.get("validation.id", f"unknown_{idx}")
                
                # Minimal validation
                is_valid, validation_error = self.validate_vr_record(vr_record, idx)
                
                if not is_valid:
                    metrics.invalid_records += 1
                    metrics.validation_errors.append({
                        "index": idx,
                        "id": record_id,
                        "error": validation_error
                    })
                    
                    if "Duplicate" in validation_error:
                        metrics.skipped_duplicates += 1
                    
                    continue
                
                valid_records.append(vr_record)
                self.processed_vr_ids.add(str(record_id))
                metrics.valid_records += 1
            
            # Log validation summary
            if metrics.validation_errors:
                await self.store_batch_logs(batch_id, "validation_summary", {
                    "total_errors": len(metrics.validation_errors),
                    "invalid_records": metrics.invalid_records,
                    "duplicates": metrics.skipped_duplicates
                })
            
            logger.info(f"Validation complete: {len(valid_records)} valid records out of {len(vr_records)} total")
            
            # Step 4: Process valid records through workflow
            if not valid_records:
                logger.warning("No valid records to process")
                return self._finalize_batch(metrics, "no_valid_records")
            
            logger.info(f"Processing {len(valid_records)} valid records through workflow")
            
            workflow_results = []
            
            for idx, vr_record in enumerate(valid_records):
                vr_id = str(vr_record.get("validation.id"))
                
                # Progress logging
                if idx % 10 == 0 and idx > 0:
                    logger.info(f"Processing progress: {idx}/{len(valid_records)}")
                
                try:
                    # Process record through workflow
                    result = await process_single_vr_record(vr_record, batch_id=batch_id)
                    
                    # Check workflow status
                    workflow_status = result.get("workflow_status")
                    
                    if workflow_status == WorkflowStatus.DBO_DECISION_READY:
                        metrics.processed_successfully += 1
                        status = "success"
                    elif workflow_status == WorkflowStatus.ERROR:
                        metrics.failed_processing += 1
                        status = "error"
                    else:
                        metrics.failed_processing += 1
                        status = "incomplete"
                    
                    workflow_results.append({
                        "vr_id": vr_id,
                        "status": status,
                        "workflow_status": str(workflow_status),
                        "dbo_decision": result.get("dbo_action_decision", {}).get("overall_recommendation", ""),
                        "storage_paths": result.get("storage_paths", {}),
                        "processing_time": datetime.now().isoformat()
                    })
                    
                except Exception as e:
                    error_msg = f"Workflow error: {str(e)}"
                    logger.error(f"VR ID {vr_id}: {error_msg}")
                    logger.debug(f"Traceback: {traceback.format_exc()}")
                    
                    metrics.failed_processing += 1
                    metrics.processing_errors.append({
                        "vr_id": vr_id,
                        "error": error_msg,
                        "error_type": type(e).__name__
                    })
                    
                    # Store failed record
                    await self._store_failed_record(vr_record, error_msg, batch_id, "workflow_error")
                    
                    workflow_results.append({
                        "vr_id": vr_id,
                        "status": "error",
                        "error": error_msg,
                        "processing_time": datetime.now().isoformat()
                    })
                
                # Rate limiting
                await asyncio.sleep(0.5)
            
            # Step 5: Store batch summary
            batch_summary = {
                **metrics.to_dict(),
                "workflow_results": workflow_results,
                "status": "completed",
                "vr_json_metadata": {
                    "extraction_date": vr_json_data.get("extraction_date"),
                    "original_total_records": vr_json_data.get("total_records", len(vr_records))
                }
            }
            
            summary_path = await self._store_batch_summary(batch_summary)
            logger.info(f"Batch processing completed. Summary at: {summary_path}")
            
            # Store completion log
            await self.store_batch_logs(batch_id, "completion", {
                "metrics": metrics.to_dict()["metrics"],
                "summary_path": summary_path
            })
            
            return batch_summary
            
        except Exception as e:
            # Catastrophic failure
            logger.error(f"Batch processing failed: {str(e)}")
            logger.debug(f"Traceback: {traceback.format_exc()}")
            
            # Store error log
            await self.store_batch_logs(batch_id, "catastrophic_error", {
                "error": str(e),
                "traceback": traceback.format_exc()
            })
            
            return self._finalize_batch(metrics, "catastrophic_failure", str(e))
    
    async def _store_failed_record(self, vr_record: Dict, error: str, batch_id: str, error_type: str):
        """Store failed record with detailed error information"""
        try:
            vr_id = str(vr_record.get("id", "unknown"))
            
            failed_record_data = {
                "batch_id": batch_id,
                "timestamp": datetime.now().isoformat(),
                "error_type": error_type,
                "error": error,
                "vr_record": vr_record,
                "can_retry": True
            }
            
            await self.storage_manager.store_json_data(
                data=failed_record_data,
                vr_id=vr_id,
                file_type="failed_record"
            )
            
        except Exception as e:
            logger.error(f"Failed to store failed record: {str(e)}")
    
    async def _store_batch_summary(self, summary: Dict) -> str:
        """Store batch processing summary"""
        try:
            date_folder = datetime.now().strftime("%Y%m%d")
            filename = f"batch_summary_{summary['batch_id']}.json"
            blob_path = f"agent-output/batch-summaries/{date_folder}/{filename}"
            
            blob_client = self.storage_manager.blob_service_client.get_blob_client(
                container=self.storage_manager.container_name,
                blob=blob_path
            )
            blob_client.upload_blob(
                json.dumps(summary, indent=2),
                overwrite=True
            )
            
            return blob_path
            
        except Exception as e:
            logger.error(f"Failed to store batch summary: {str(e)}")
            return ""
    
    def _finalize_batch(self, metrics: BatchMetrics, status: str, error: Optional[str] = None) -> Dict:
        """Finalize batch with error status"""
        summary = metrics.to_dict()
        summary["status"] = status
        if error:
            summary["error"] = error
        return summary


# Azure Function entry point
async def main(date_range: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
    """
    Main entry point for Azure Function
    
    Args:
        date_range: Optional date range, defaults to previous day
    """
    try:
        # Default to previous day if no date range provided
        if not date_range:
            yesterday = datetime.now() - timedelta(days=1)
            date_range = {
                "start_date": yesterday.strftime("%Y-%m-%d"),
                "end_date": yesterday.strftime("%Y-%m-%d")
            }
        
        logger.info(f"Starting VR batch processing for: {date_range}")
        
        # Create processor and run
        processor = VRBatchProcessor()
        result = await processor.process_batch(date_range)
        
        logger.info(f"Batch processing completed with status: {result.get('status')}")
        
        return result
        
    except Exception as e:
        logger.error(f"Fatal error in batch processor: {str(e)}")
        return {
            "status": "fatal_error",
            "error": str(e),
            "timestamp": datetime.now().isoformat()
        }


# For local testing
if __name__ == "__main__":
    # Example: Process specific date
    test_date_range = {
        "start_date": "2025-01-15",
        "end_date": "2025-01-15"
    }
    
    result = asyncio.run(main(test_date_range))
    print(json.dumps(result, indent=2))
