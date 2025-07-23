## Azure Functions Files for Platform Engineer

Here are all the files you need to provide to your platform engineer for Azure Functions deployment:

### 1. **Folder Structure to Create**
```
azure_functions/
├── host.json
├── requirements.txt
├── VRBatchProcessor/
│   ├── __init__.py
│   └── function.json
```

### 2. **File Contents**

#### **host.json**
```json
{
  "version": "2.0",
  "logging": {
    "applicationInsights": {
      "samplingSettings": {
        "isEnabled": true,
        "excludedTypes": "Request"
      }
    },
    "logLevel": {
      "default": "Information",
      "Host.Results": "Information",
      "Function": "Information",
      "Host.Aggregator": "Warning"
    }
  },
  "extensionBundle": {
    "id": "Microsoft.Azure.Functions.ExtensionBundle",
    "version": "[4.*, 5.0.0)"
  },
  "functionTimeout": "00:30:00"
}
```

#### **requirements.txt**
```txt
azure-functions==1.18.0
azure-storage-blob==12.19.0
langchain==0.1.0
langchain-openai==0.0.5
langgraph==0.0.26
python-dotenv==1.0.0
pydantic==2.0.0
aiohttp==3.8.0
tenacity==8.2.0
pytz==2024.1
```

#### **VRBatchProcessor/function.json**
```json
{
  "scriptFile": "__init__.py",
  "bindings": [
    {
      "name": "mytimer",
      "type": "timerTrigger",
      "direction": "in",
      "schedule": "0 0 2 * * *",
      "runOnStartup": false,
      "useMonitor": true
    }
  ]
}
```

#### **VRBatchProcessor/__init__.py**
```python
import datetime
import logging
import asyncio
import sys
import os

# Add the app directory to Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
parent_dir = os.path.dirname(current_dir)
sys.path.insert(0, parent_dir)

import azure.functions as func
from app.my_agent.batch_processor import VRBatchProcessor

def main(mytimer: func.TimerRequest) -> None:
    """
    Azure Function for VR Batch Processing
    Runs daily at 2 AM to process previous day's VR records
    """
    utc_timestamp = datetime.datetime.utcnow().replace(
        tzinfo=datetime.timezone.utc).isoformat()
    
    logging.info('=' * 60)
    logging.info(f'VR Batch Processor Started at {utc_timestamp}')
    
    if mytimer.past_due:
        logging.info('The timer is past due!')
    
    try:
        # Initialize batch processor
        processor = VRBatchProcessor()
        
        # Process yesterday's records
        yesterday = datetime.datetime.now() - datetime.timedelta(days=1)
        date_range = {
            "start_date": yesterday.strftime("%Y-%m-%d"),
            "end_date": yesterday.strftime("%Y-%m-%d")
        }
        
        logging.info(f"Processing VR records for: {date_range}")
        
        # Run async batch processor
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(processor.process_batch(date_range))
        loop.close()
        
        logging.info(f"Batch processing completed: {result.get('status')}")
        
    except Exception as e:
        logging.error(f"Batch processing failed: {str(e)}")
        raise
```

### 3. **Environment Variables Needed**

Tell the platform engineer to set these in Azure Portal → Function App → Configuration:

```
AZURE_OPENAI_API_KEY=<your-key>
AZURE_OPENAI_ENDPOINT=https://<your-resource>.openai.azure.com/
AZURE_OPENAI_API_VERSION=2024-02-15-preview
AZURE_OPENAI_DEPLOYMENT_NAME=gpt-4

AZURE_STORAGE_CONNECTION_STRING=<storage-connection-string>

ENVIRONMENT=production
LOG_LEVEL=INFO

# If you have VR API credentials
VR_API_ENDPOINT=<your-vr-api-endpoint>
VR_API_KEY=<your-vr-api-key>

# If you have OK DB API credentials
OKDB_API_ENDPOINT=<your-okdb-api-endpoint>
OKDB_API_KEY=<your-okdb-api-key>
```

### 4. **Deployment Instructions for Platform Engineer**

```markdown
## Deployment Steps:

1. Create Azure Function App with:
   - Python 3.11 runtime
   - Linux OS
   - Consumption or Premium plan
   - Same region as storage account

2. Deploy the azure_functions folder using:
   - VS Code Azure Functions extension
   - Azure CLI: `func azure functionapp publish <app-name>`
   - Or ZIP deployment

3. Configure all environment variables in Application Settings

4. Ensure Function App has access to:
   - Storage Account (for blob storage)
   - OpenAI resource
   - Any other APIs

5. Enable Application Insights for monitoring

6. Test manually using "Code + Test" in Azure Portal
```

### 5. **What to Include in Deployment Package**

The platform engineer needs:
```
deployment_package/
├── azure_functions/          # The folder with all files above
├── app/
│   └── my_agent/            # Your entire my_agent folder
│       ├── batch_processor.py
│       ├── agents/
│       ├── utils/
│       ├── tools/
│       └── ... (all other files)
```

That's it! These are all the Azure Functions specific files needed. The platform engineer will handle the actual Azure setup and deployment.
