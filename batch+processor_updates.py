```python
        # Store outside agent-output as discussed
        blob_path = f"vr-api-responses/{country_name}/{date_folder}/{filename}"
        
        # Use the same pattern as batch summary storage
        blob_client = self.blob_service_client.get_blob_client(
            container=self.container_name,
            blob=blob_path
        )
        
        blob_client.upload_blob(
            json.dumps(country_data, indent=2),
            overwrite=True
        )
        
        logger.info(f"Stored {country_name} API response to: {blob_path}")
        return blob_path
        
    except Exception as e:
        logger.error(f"Failed to store {country_name} API response: {str(e)}")
        raise
```

### Update the `process_batch` method to process countries separately:

```python
async def process_batch(self, date_range: Dict[str, str]) -> Dict[str, Any]:
    """Execute the complete batch processing workflow"""
    start_time = datetime.now()
    self.batch_id = f"batch_{start_time.strftime('%Y%m%d_%H%M%S')}"
    
    logger.info(f"Starting batch processing - ID: {self.batch_id}")
    
    try:
        # Step 1: Fetch VR data from API for both countries
        api_results, error = await self.fetch_vr_json_from_api(date_range)
        
        if error:
            return {
                "status": "failed",
                "error": error,
                "batch_id": self.batch_id
            }
        
        # Step 2: Process each country's records separately
        all_processing_results = {}
        total_processed = 0
        total_failed = 0
        
        for country_name, country_info in api_results["countries"].items():
            if not country_info["data"]:
                logger.warning(f"No data for {country_name}, skipping")
                continue
            
            logger.info(f"\nProcessing {country_name.upper()} records...")
            
            # Get the VR records for this country
            vr_records = country_info["data"].get("processed_vrs", [])
            
            if not vr_records:
                logger.warning(f"No VR records found for {country_name}")
                all_processing_results[country_name] = {
                    "processed": 0,
                    "failed": 0,
                    "records": []
                }
                continue
            
            # Process this country's records
            country_results = await self.process_vr_records(vr_records, country_name.upper())
            all_processing_results[country_name] = country_results
            
            total_processed += country_results["processed"]
            total_failed += country_results["failed"]
            
            logger.info(f"{country_name.upper()} processing complete: "
                       f"{country_results['processed']} success, "
                       f"{country_results['failed']} failed")
        
        # Step 3: Generate and store batch summary
        end_time = datetime.now()
        
        batch_summary = {
            "batch_id": self.batch_id,
            "date_range": date_range,
            "start_time": start_time.isoformat(),
            "end_time": end_time.isoformat(),
            "duration_seconds": (end_time - start_time).total_seconds(),
            "total_records": total_processed + total_failed,
            "processed_successfully": total_processed,
            "failed_processing": total_failed,
            "country_results": all_processing_results,
            "api_blob_paths": {
                country: info["blob_path"] 
                for country, info in api_results["countries"].items()
            }
        }
        
        # Store batch summary
        summary_path = await self.store_batch_summary(batch_summary)
        
        logger.info(f"Batch processing completed - Summary: {summary_path}")
        
        return {
            "status": "completed",
            "batch_id": self.batch_id,
            "summary": batch_summary
        }
        
    except Exception as e:
        logger.error(f"Batch processing failed: {str(e)}")
        return {
            "status": "failed",
            "error": str(e),
            "batch_id": self.batch_id
        }
```

### Remove these methods (no longer needed):

Remove:
- `_store_country_vr_data`
- `_store_combined_vr_data`
- `_upload_to_blob`

## Step 3: Verify Data Structure Compatibility

The data flow is now:

1. **getVR_dataclass.py** returns:
```python
{
    "processed_vrs": [
        {
            "vr_data": {
                "validation_id": 123,
                "validation_customerId": 456,
                "individual_firstName": "MARCO",
                // ... other fields with underscore notation
            }
        }
    ]
}
```

2. **batch_processor.py** converts field names:
```python
# From: validation_id
# To: validation.id (what agent expects)
```

## Summary of Final Changes:

### In batch_processor.py:
1. Added `process_vr_records` method that:
   - Processes each country's records separately
   - Converts field names from underscore to dot notation
   - Properly extracts VR data from wrapper

2. Updated `fetch_vr_json_from_api` to:
   - Store each country's data separately
   - Use existing blob storage pattern
   - Return structured results by country

3. Updated `process_batch` to:
   - Process each country separately
   - Track results by country
   - Store comprehensive batch summary

### Storage Structure:
```
vr-api-responses/  (outside agent-output)
├── italy/
│   └── 20250123/
│       └── vr_api_italy_20250123_143022.json
└── france/
    └── 20250123/
        └── vr_api_france_20250123_143025.json

agent-output/
├── batch-summaries/
│   └── 20250123/
│       └── batch_summary_batch_20250123_143000.json
└── vr-data/  (individual VR processing results)
    └── {vrId}/
        └── {YYYYMMDD}/
```

This approach:
- ✅ Processes each country separately
- ✅ Uses existing storage patterns
- ✅ Stores API responses outside agent-output
- ✅ Properly converts field names for agent compatibility
- ✅ Maintains data integrity throughout the flow
