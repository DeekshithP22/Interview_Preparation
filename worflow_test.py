"""
Simple test script for complete VR workflow
"""

import asyncio
import logging
from main import process_single_vr_record

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


async def test_workflow():
    """Test the complete workflow with sample VR records"""
    
    # Test Case 1: Italian Individual Activity
    vr_record_1 = {
        "id": 1019000131693245,
        "entityTypeIco": "ENT_ACTIVITY",
        "countryCode": "IT",
        "individualOriginKeyEid": "WITO43596247",
        "firstName": "PAOLO",
        "lastName": "CORVISIERI",
        "specialityCode1": "18",
        "workplaceOriginKeyEid": "WITO62731655",
        "workplaceUsualName": "DISTRETTO SANITARIO FIUMICINO",
        "city": "VELLETRI",
        "postalCity": "VELLETRI",
        "requestComment": "Verify doctor at Velletri hospital"
    }
    
    logger.info("\n" + "="*50)
    logger.info("Testing VR Record 1 - Italian Individual")
    logger.info("="*50)
    
    try:
        result = await process_single_vr_record(vr_record_1)
        
        # Print results
        logger.info(f"\nWorkflow Status: {result.get('workflow_status')}")
        logger.info(f"Record Status: {result.get('record_status')}")
        
        if result.get('dbo_action_decision'):
            logger.info(f"\nDBO Decision:")
            logger.info(f"  - Action: {result['dbo_action_decision'].get('overall_recommendation')}")
            logger.info(f"  - Confidence: {result['dbo_action_decision'].get('decision_confidence')}")
        
        if result.get('search_confidence'):
            logger.info(f"\nSearch Results:")
            logger.info(f"  - Confidence: {result.get('search_confidence')}")
            logger.info(f"  - Tools Used: {result.get('selected_tools', [])}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")
    
    
    # Test Case 2: French Workplace
    vr_record_2 = {
        "id": 2020000456789,
        "entityTypeIco": "ENT_WORKPLACE",
        "countryCode": "FR",
        "workplaceOriginKeyEid": "WFRA98765432",
        "workplaceUsualName": "HOPITAL SAINT-LOUIS",
        "city": "PARIS",
        "postalCity": "PARIS",
        "requestComment": "Verify hospital information"
    }
    
    logger.info("\n" + "="*50)
    logger.info("Testing VR Record 2 - French Workplace")
    logger.info("="*50)
    
    try:
        result = await process_single_vr_record(vr_record_2)
        
        logger.info(f"\nWorkflow Status: {result.get('workflow_status')}")
        logger.info(f"Record Status: {result.get('record_status')}")
        
        if result.get('dbo_action_decision'):
            logger.info(f"\nDBO Decision: {result['dbo_action_decision'].get('overall_recommendation')}")
            
    except Exception as e:
        logger.error(f"Test failed: {str(e)}")


async def test_minimal():
    """Minimal test with just required fields"""
    
    minimal_vr = {
        "id": 123456,
        "entityTypeIco": "ENT_ACTIVITY",
        "countryCode": "IT",
        "firstName": "Test",
        "lastName": "User"
    }
    
    logger.info("\n" + "="*50)
    logger.info("Testing Minimal VR Record")
    logger.info("="*50)
    
    result = await process_single_vr_record(minimal_vr)
    logger.info(f"Result: {result.get('workflow_status')}")


if __name__ == "__main__":
    # Run main test
    asyncio.run(test_workflow())
    
    # Run minimal test
    # asyncio.run(test_minimal())
