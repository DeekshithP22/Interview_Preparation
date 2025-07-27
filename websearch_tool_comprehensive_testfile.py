#!/usr/bin/env python3
"""
Comprehensive test script for WebSearch Agent debugging
Shows EVERYTHING - raw results, validation process, filtered results
"""

import asyncio
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Configure detailed logging
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create logger for this test
test_logger = logging.getLogger("WebSearchTest")

# Import your WebSearch Agent (adjust import path as needed)
try:
    from websearch_agent_get_codes import WebSearchAgent
    print("✅ WebSearchAgent imported successfully")
except ImportError as e:
    print(f"❌ Failed to import WebSearchAgent: {e}")
    exit(1)

# Test data - Using your actual VR record data
TEST_HCP_DATA = {
    "firstName": "Simona",
    "lastName": "Spinoglio", 
    "workplaceName": "ATS DELLA BRIANZA",
    "address": "Monza",
    "specialtyCode": "123",
    "geographic_region": "IT",
    "entity_type": "ent_activity"
}

TEST_WORKPLACE_DATA = {
    "workplaceName": "ATS DELLA BRIANZA",
    "address": "Monza", 
    "geographic_region": "IT",
    "entity_type": "ent_workplace"
}

def print_separator(title: str, char: str = "=", width: int = 80):
    """Print a separator with title"""
    padding = (width - len(title) - 2) // 2
    print(f"\n{char * padding} {title} {char * padding}")

def print_raw_tavily_results(results: dict, search_name: str):
    """Print raw Tavily results before any processing"""
    print_separator(f"RAW TAVILY RESULTS - {search_name}", "🔍")
    
    raw_results = results.get("results", [])
    answer = results.get("answer", "")
    
    print(f"📊 Total Raw Results: {len(raw_results)}")
    print(f"💡 Tavily Answer: {answer[:200]}..." if answer else "💡 Tavily Answer: [Empty]")
    
    for i, result in enumerate(raw_results, 1):
        print(f"\n🔗 Raw Result {i}:")
        print(f"   URL: {result.get('url', 'N/A')}")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   Score: {result.get('score', 'N/A')}")
        
        content = result.get('content', '')
        if content:
            if len(content) > 300:
                print(f"   Content: {content[:300]}... [TRUNCATED - Full length: {len(content)} chars]")
            else:
                print(f"   Content: {content}")
        else:
            print(f"   Content: [EMPTY]")
        
        # Show raw_content if available
        raw_content = result.get('raw_content')
        if raw_content:
            print(f"   Raw Content: {str(raw_content)[:100]}...")

def validate_single_result_debug(agent: WebSearchAgent, result: dict, search_input: dict, result_index: int):
    """Debug validation for a single result with detailed output"""
    print(f"\n🧪 VALIDATING RESULT {result_index + 1}")
    print(f"   URL: {result.get('url', 'N/A')}")
    
    # Extract data for validation
    url = result.get("url", "")
    title = result.get("title", "")
    content = result.get("content", "")
    full_content = f"{title} {content}"
    
    first_name = search_input.get("firstName", "")
    last_name = search_input.get("lastName", "")
    address = search_input.get("address", "")
    entity_type = search_input.get("entity_type", "")
    
    print(f"   🎯 Validation Target:")
    print(f"      First Name: '{first_name}'")
    print(f"      Last Name: '{last_name}'")
    print(f"      Address: '{address}'")
    print(f"      Entity Type: '{entity_type}'")
    
    print(f"   📝 Content for Validation:")
    print(f"      Title: '{title}'")
    print(f"      Content: '{content[:200]}...'" if len(content) > 200 else f"      Content: '{content}'")
    print(f"      Full Content Length: {len(full_content)} chars")
    
    # Test individual validation components
    try:
        # Name validation
        name_match = agent.validator.calculate_exact_name_match(full_content, first_name, last_name)
        print(f"   ✅ Name Match Test: {name_match}")
        
        # Check individual name components
        content_lower = full_content.lower()
        first_lower = first_name.lower().strip()
        last_lower = last_name.lower().strip()
        full_name = f"{first_lower} {last_lower}"
        
        print(f"      - Full name '{full_name}' in content: {full_name in content_lower}")
        print(f"      - First name '{first_lower}' in content: {first_lower in content_lower}")
        print(f"      - Last name '{last_lower}' in content: {last_lower in content_lower}")
        
        # City validation
        city_match = agent.validator.validate_exact_city_match(full_content, address)
        print(f"   🌍 City Match Test: {city_match}")
        if address:
            address_lower = address.lower().strip()
            print(f"      - Address '{address_lower}' in content: {address_lower in content_lower}")
        
        # Geographic validation
        region = search_input.get("geographic_region", "IT")
        geo_match = agent.validator.validate_geographic_match(url, full_content, region)
        print(f"   🗺️  Geographic Match Test: {geo_match}")
        
        # Professional context validation (if applicable)
        try:
            prof_match = agent.validator.validate_professional_context(full_content, region)
            print(f"   👔 Professional Context Test: {prof_match}")
        except Exception as e:
            print(f"   👔 Professional Context Test: ERROR - {e}")
        
        # Full validation
        validation_result = agent.validator.validate_url(result, search_input)
        
        print(f"   🎯 FINAL VALIDATION RESULT:")
        print(f"      ✅ Is Valid: {validation_result.is_valid}")
        print(f"      🎯 Confidence: {validation_result.confidence_score}")
        print(f"      📋 Validation Reasons:")
        for reason in validation_result.validation_reasons:
            print(f"         - {reason}")
        print(f"      🌍 Geographic Match: {validation_result.geographic_match}")
        print(f"      👤 Name Match: {validation_result.name_match}")
        print(f"      🏢 Workplace Match: {validation_result.workplace_match}")
        
        return validation_result
        
    except Exception as e:
        print(f"   ❌ VALIDATION ERROR: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

def print_validation_summary(raw_results: List[dict], validated_results: List[dict], search_type: str):
    """Print before/after validation summary"""
    print_separator(f"VALIDATION SUMMARY - {search_type}", "📊")
    
    print(f"📥 BEFORE VALIDATION: {len(raw_results)} results")
    print(f"📤 AFTER VALIDATION: {len(validated_results)} results")
    print(f"🚫 FILTERED OUT: {len(raw_results) - len(validated_results)} results")
    
    if len(validated_results) > 0:
        print(f"\n✅ VALIDATED RESULTS:")
        for i, result in enumerate(validated_results, 1):
            validation = result.get("validation", {})
            print(f"   {i}. {result.get('url', 'N/A')}")
            print(f"      Valid: {validation.get('is_valid', False)}")
            print(f"      Confidence: {validation.get('confidence_score', 0)}")
    else:
        print(f"\n❌ NO RESULTS PASSED VALIDATION")

def print_final_results(results: dict, search_type: str):
    """Print final formatted results"""
    print_separator(f"FINAL RESULTS - {search_type}", "🎉")
    
    search_results = results.get("search_results", {})
    
    total_final = 0
    for result_type, data in search_results.items():
        results_list = data.get("results", [])
        total_final += len(results_list)
        print(f"\n📋 {result_type.replace('_', ' ').title()}: {len(results_list)} results")
        
        for i, result in enumerate(results_list, 1):
            print(f"   {i}. {result.get('url', 'N/A')}")
            print(f"      Title: {result.get('title', 'N/A')[:100]}...")
            print(f"      Score: {result.get('score', 'N/A')}")
    
    print(f"\n🎯 TOTAL FINAL RESULTS: {total_final}")
    
    # Print answers
    tavily_answer = results.get("tavily_answer", "")
    if tavily_answer:
        print(f"\n💡 Tavily Answer: {tavily_answer[:300]}...")
    
    llm_answer = results.get("llm_answer", "")
    if llm_answer:
        print(f"\n🤖 LLM Analysis: {llm_answer[:500]}...")

class DebugWebSearchAgent(WebSearchAgent):
    """WebSearchAgent with detailed debugging"""
    
    async def debug_search_hcp(self, doctor_info: Dict[str, Any]) -> Dict[str, Any]:
        """HCP search with detailed debugging at each step"""
        print_separator("STARTING HCP SEARCH WITH DEBUG", "🚀")
        
        # Get specialty info
        specialty_code = doctor_info.get("specialtyCode")
        region = doctor_info.get("geographic_region", "IT")
        specialty_name = None
        if specialty_code:
            specialty_name = self.get_specialty_name(specialty_code, region)
            print(f"🔬 Specialty: {specialty_code} → {specialty_name}")
        
        # Build search inputs
        search_input = {**doctor_info, "entity_type": "ent_activity"}
        print(f"🎯 Search Input: {json.dumps(search_input, indent=2)}")
        
        # Build payloads
        main_payload = self.payload_builder.build_hcp_main_payload(doctor_info, specialty_name)
        linkedin_payload = self.payload_builder.build_linkedin_payload(doctor_info, specialty_name)
        workplace_payload = self.payload_builder.build_workplace_validation_payload(doctor_info)
        
        print(f"\n📋 Search Payloads:")
        print(f"   Main: {json.dumps(main_payload, indent=2)}")
        print(f"   LinkedIn: {json.dumps(linkedin_payload, indent=2)}")
        print(f"   Workplace: {json.dumps(workplace_payload, indent=2)}")
        
        # Execute searches
        print_separator("EXECUTING CONCURRENT SEARCHES", "⚡")
        
        main_task = self.tavily_client.search_async(**main_payload)
        linkedin_task = self.tavily_client.search_async(**linkedin_payload)
        workplace_task = self.tavily_client.search_async(**workplace_payload)
        
        online_results, linkedin_results, workplace_site_results = await asyncio.gather(
            main_task, linkedin_task, workplace_task
        )
        
        # Show raw results
        print_raw_tavily_results(online_results, "MAIN SEARCH")
        print_raw_tavily_results(linkedin_results, "LINKEDIN SEARCH") 
        print_raw_tavily_results(workplace_site_results, "WORKPLACE SEARCH")
        
        # Debug validation for each search type
        print_separator("VALIDATING MAIN SEARCH RESULTS", "🔍")
        validated_main = []
        for i, result in enumerate(online_results.get("results", [])):
            validation_result = validate_single_result_debug(self, result, search_input, i)
            if validation_result:
                validated_result = {
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "validation": {
                        "is_valid": validation_result.is_valid,
                        "confidence_score": validation_result.confidence_score,
                        "validation_reasons": validation_result.validation_reasons,
                        "geographic_match": validation_result.geographic_match,
                        "name_match": validation_result.name_match,
                        "workplace_match": validation_result.workplace_match,
                    },
                }
                validated_main.append(validated_result)
        
        print_separator("VALIDATING LINKEDIN SEARCH RESULTS", "🔍")
        validated_linkedin = []
        for i, result in enumerate(linkedin_results.get("results", [])):
            validation_result = validate_single_result_debug(self, result, search_input, i)
            if validation_result:
                validated_result = {
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                    "validation": {
                        "is_valid": validation_result.is_valid,
                        "confidence_score": validation_result.confidence_score,
                        "validation_reasons": validation_result.validation_reasons,
                        "geographic_match": validation_result.geographic_match,
                        "name_match": validation_result.name_match,
                        "workplace_match": validation_result.workplace_match,
                    },
                }
                validated_linkedin.append(validated_result)
        
        # Print validation summaries
        print_validation_summary(online_results.get("results", []), validated_main, "MAIN SEARCH")
        print_validation_summary(linkedin_results.get("results", []), validated_linkedin, "LINKEDIN SEARCH")
        
        # Build final response
        clean_response = {
            "search_results": {
                "online_search": {"results": []},
                "linkedin_search": {"results": []},
                "workplace_search": {"results": []},
            },
            "tavily_answer": online_results.get("answer", "") or "",
            "llm_answer": "",
        }
        
        # Filter and add results
        print_separator("FILTERING RESULTS FOR FINAL RESPONSE", "📤")
        
        for result in validated_main:
            validation = result.get("validation", {})
            if validation.get("is_valid", False):
                print(f"✅ Adding main result: {result.get('url', 'N/A')}")
                clean_response["search_results"]["online_search"]["results"].append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                })
            else:
                print(f"❌ Rejecting main result: {result.get('url', 'N/A')} (is_valid: {validation.get('is_valid', False)})")
        
        for result in validated_linkedin:
            validation = result.get("validation", {})
            if validation.get("is_valid", False):
                print(f"✅ Adding LinkedIn result: {result.get('url', 'N/A')}")
                clean_response["search_results"]["linkedin_search"]["results"].append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                })
            else:
                print(f"❌ Rejecting LinkedIn result: {result.get('url', 'N/A')} (is_valid: {validation.get('is_valid', False)})")
        
        # Generate LLM analysis
        try:
            clean_response["llm_answer"] = await self._generate_llm_analysis_async(
                clean_response, doctor_info, specialty_name
            )
        except Exception as e:
            print(f"❌ LLM analysis failed: {e}")
            clean_response["llm_answer"] = f"LLM analysis failed: {str(e)}"
        
        return clean_response

async def test_comprehensive_hcp_search():
    """Test HCP search with complete debugging"""
    print_separator("COMPREHENSIVE HCP SEARCH TEST", "🚀")
    
    try:
        agent = DebugWebSearchAgent()
        start_time = datetime.now()
        
        results = await agent.debug_search_hcp(TEST_HCP_DATA)
        
        end_time = datetime.now()
        print(f"\n⏱️  Total search time: {(end_time - start_time).total_seconds():.2f} seconds")
        
        print_final_results(results, "HCP SEARCH")
        
        return results
        
    except Exception as e:
        print(f"❌ Comprehensive HCP search failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def test_validation_only():
    """Test validation with your exact data"""
    print_separator("ISOLATED VALIDATION TEST", "🧪")
    
    # Your exact Tavily result that should pass
    sample_result = {
        "url": "https://www.ats-brianza.it/component/ipnetregister/?task=import_download&catid=390&id=636&file=CV_SPINOGLIO_S_[390-636-1].pdf",
        "title": "[PDF] Curriculum Vitae Europass - ATS Brianza",
        "content": "Pagina 5 / 6 - Curriculum Vitae di. Simona Spinoglio ... • Conduzione laboratorio musicale e teatrale presso UILM Sezione di Monza dal 2018 ad ... della voce",
        "score": 0.61196244
    }
    
    try:
        agent = WebSearchAgent()
        search_input = TEST_HCP_DATA.copy()
        
        print(f"🎯 Testing with known good result:")
        print(f"   URL: {sample_result['url']}")
        print(f"   Content: {sample_result['content']}")
        
        validation_result = validate_single_result_debug(agent, sample_result, search_input, 0)
        
        if validation_result and validation_result.is_valid:
            print(f"\n✅ VALIDATION PASSED - This result should appear in final output")
        else:
            print(f"\n❌ VALIDATION FAILED - This is why you're getting zero results")
        
        return validation_result
        
    except Exception as e:
        print(f"❌ Validation test failed: {str(e)}")
        import traceback
        traceback.print_exc()
        return None

async def main():
    """Main comprehensive test"""
    print("🧪 COMPREHENSIVE WEBSEARCH AGENT DEBUG SUITE")
    print("=" * 80)
    
    # Test 1: Isolated validation
    await test_validation_only()
    
    # Test 2: Full search with debugging
    await test_comprehensive_hcp_search()
    
    print_separator("DEBUG SESSION COMPLETE", "🎉")

if __name__ == "__main__":
    asyncio.run(main())
