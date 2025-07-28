#!/usr/bin/env python3
"""
Comprehensive Test Suite for WebSearch Agent
Shows EVERYTHING - Raw results, validation process, scoring, final output
Complete end-to-end debugging visibility
"""

import asyncio
import json
import logging
import time
from datetime import datetime
from typing import Dict, Any, List
import traceback

# Configure detailed logging for maximum visibility
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)

# Create test logger
test_logger = logging.getLogger("WebSearchTest")

# Import your WebSearch Agent (adjust import path as needed)
try:
    from paste import WebSearchAgent  # Adjust this import path
    print("✅ WebSearchAgent imported successfully")
except ImportError as e:
    print(f"❌ Failed to import WebSearchAgent: {e}")
    print("Please update the import path to match your file structure")
    exit(1)

# ============================================================================
# TEST DATA - Real scenarios to test
# ============================================================================

# Test Case 1: Complete HCP data (your actual example)
TEST_HCP_COMPLETE = {
    "firstName": "Simona",
    "lastName": "Spinoglio", 
    "workplaceName": "ATS DELLA BRIANZA",
    "address": "Monza",
    "specialtyCode": "123",
    "geographic_region": "IT",
    "entity_type": "ent_activity"
}

# Test Case 2: HCP with missing specialty
TEST_HCP_NO_SPECIALTY = {
    "firstName": "Mario",
    "lastName": "Rossi",
    "workplaceName": "Ospedale San Raffaele",
    "address": "Milano", 
    "geographic_region": "IT",
    "entity_type": "ent_activity"
}

# Test Case 3: Common name (high collision risk)
TEST_HCP_COMMON_NAME = {
    "firstName": "Giovanni",
    "lastName": "Bianchi",
    "workplaceName": "Policlinico Gemelli",
    "address": "Roma",
    "geographic_region": "IT", 
    "entity_type": "ent_activity"
}

# Test Case 4: Workplace validation
TEST_WORKPLACE = {
    "workplaceName": "ATS DELLA BRIANZA",
    "address": "Monza",
    "geographic_region": "IT",
    "entity_type": "ent_workplace"
}

# All test cases
ALL_TEST_CASES = [
    ("Complete HCP Data", TEST_HCP_COMPLETE),
    ("HCP No Specialty", TEST_HCP_NO_SPECIALTY), 
    ("Common Name HCP", TEST_HCP_COMMON_NAME),
    ("Workplace Validation", TEST_WORKPLACE)
]

# ============================================================================
# DISPLAY UTILITIES - Make everything visible
# ============================================================================

def print_header(title: str, char: str = "=", width: int = 100):
    """Print a prominent header"""
    print(f"\n{char * width}")
    print(f"{title:^{width}}")
    print(f"{char * width}")

def print_section(title: str, char: str = "-", width: int = 80):
    """Print a section header"""
    print(f"\n{char * width}")
    print(f" {title}")
    print(f"{char * width}")

def print_subsection(title: str, char: str = "•", width: int = 60):
    """Print a subsection header"""
    print(f"\n{char * 3} {title} {char * (width - len(title) - 6)}")

def format_json(data: Any, max_length: int = 1000) -> str:
    """Format JSON with truncation for readability"""
    json_str = json.dumps(data, indent=2, ensure_ascii=False)
    if len(json_str) > max_length:
        return json_str[:max_length] + f"\n... [TRUNCATED - Full length: {len(json_str)} chars]"
    return json_str

def display_tavily_payload(payload: Dict[str, Any], search_type: str):
    """Display Tavily search payload"""
    print_subsection(f"{search_type} Search Payload")
    print(f"🎯 Query: {payload.get('query', 'N/A')}")
    print(f"🔍 Search Depth: {payload.get('search_depth', 'N/A')}")
    print(f"📊 Max Results: {payload.get('max_results', 'N/A')}")
    print(f"🌍 Country: {payload.get('country', 'N/A')}")
    
    # Show domains if present
    if 'include_domains' in payload:
        print(f"✅ Include Domains: {payload['include_domains']}")
    if 'exclude_domains' in payload:
        print(f"❌ Exclude Domains: {payload['exclude_domains']}")

def display_raw_tavily_results(results: Dict[str, Any], search_type: str):
    """Display raw Tavily results before any processing"""
    print_subsection(f"RAW TAVILY RESULTS - {search_type}")
    
    raw_results = results.get("results", [])
    answer = results.get("answer", "")
    
    print(f"📊 Total Raw Results: {len(raw_results)}")
    print(f"💡 Tavily Answer: {answer[:200]}..." if answer else "💡 Tavily Answer: [Empty]")
    
    for i, result in enumerate(raw_results[:5], 1):  # Show first 5
        print(f"\n🔗 Raw Result {i}:")
        print(f"   URL: {result.get('url', 'N/A')}")
        print(f"   Title: {result.get('title', 'N/A')}")
        print(f"   Score: {result.get('score', 'N/A')}")
        
        content = result.get('content', '')
        if content:
            if len(content) > 200:
                print(f"   Content: {content[:200]}... [Full length: {len(content)} chars]")
            else:
                print(f"   Content: {content}")
        else:
            print(f"   Content: [EMPTY]")

def display_validation_details(result: Dict[str, Any], index: int, search_input: Dict[str, Any]):
    """Display detailed validation process for a single result"""
    print(f"\n🧪 VALIDATION DETAILS - Result {index + 1}")
    print(f"   URL: {result.get('url', 'N/A')}")
    print(f"   Title: {result.get('title', 'N/A')}")
    print(f"   Tavily Score: {result.get('score', 0.0):.3f}")
    
    # Show validation results if present
    validation = result.get('validation', {})
    if validation:
        is_valid = validation.get('is_valid', False)
        confidence = validation.get('confidence_score', 0.0)
        reasons = validation.get('validation_reasons', [])
        
        print(f"   🎯 Final Decision: {'✅ VALID' if is_valid else '❌ INVALID'}")
        print(f"   📊 Confidence Score: {confidence:.3f}")
        print(f"   📝 Validation Reasons:")
        for reason in reasons:
            print(f"      - {reason}")
        print(f"   📈 Name Match: {validation.get('name_match', 'N/A')}")
        print(f"   🏢 Workplace Match: {validation.get('workplace_match', 'N/A')}")
        print(f"   🌍 Geographic Match: {validation.get('geographic_match', 'N/A')}")

def display_final_results_summary(results: Dict[str, Any], search_type: str):
    """Display final results summary"""
    print_section(f"FINAL RESULTS SUMMARY - {search_type}")
    
    search_results = results.get("search_results", {})
    total_final = 0
    
    for result_type, data in search_results.items():
        results_list = data.get("results", [])
        count = len(results_list)
        total_final += count
        
        print(f"\n📋 {result_type.replace('_', ' ').title()}: {count} results")
        
        if count > 0:
            for i, result in enumerate(results_list, 1):
                print(f"   {i}. Score: {result.get('score', 'N/A'):.3f} - {result.get('url', 'N/A')}")
                print(f"      Title: {result.get('title', 'N/A')[:80]}...")
        else:
            print(f"      No results found")
    
    print(f"\n🎯 TOTAL FINAL RESULTS: {total_final}")
    return total_final

def display_performance_metrics(start_time: float, end_time: float, api_calls: int):
    """Display performance metrics"""
    print_section("PERFORMANCE METRICS")
    duration = end_time - start_time
    print(f"⏱️  Total Duration: {duration:.2f} seconds")
    print(f"🔗 API Calls Made: {api_calls}")
    print(f"📊 Average Time per Call: {duration/max(api_calls, 1):.2f} seconds")

# ============================================================================
# ENHANCED WEBSEARCH AGENT WITH DEBUGGING
# ============================================================================

class DebugWebSearchAgent(WebSearchAgent):
    """WebSearchAgent with comprehensive debugging"""
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.api_call_count = 0
        self.debug_mode = True
        
    async def debug_search_hcp(self, doctor_info: Dict[str, Any]) -> Dict[str, Any]:
        """HCP search with complete debugging visibility"""
        print_header("STARTING COMPREHENSIVE HCP SEARCH DEBUG")
        
        self.api_call_count = 0
        search_start_time = time.time()
        
        # Display input data
        print_section("INPUT DATA ANALYSIS")
        print("🎯 Search Target:")
        for key, value in doctor_info.items():
            print(f"   {key}: {value}")
        
        # Get specialty mapping
        specialty_code = doctor_info.get("specialtyCode")
        region = doctor_info.get("geographic_region", "IT")
        specialty_name = None
        
        if specialty_code:
            print_subsection("Specialty Mapping")
            specialty_name = self.get_specialty_name(specialty_code, region)
            print(f"🔬 Specialty Code {specialty_code} → {specialty_name or 'Not Found'} (Region: {region})")
        
        # Build search payloads
        print_section("SEARCH PAYLOAD CONSTRUCTION")
        
        main_payload = self.payload_builder.build_hcp_main_payload(doctor_info, specialty_name)
        linkedin_payload = self.payload_builder.build_linkedin_payload(doctor_info, specialty_name)
        workplace_payload = self.payload_builder.build_workplace_validation_payload(doctor_info)
        
        display_tavily_payload(main_payload, "MAIN")
        display_tavily_payload(linkedin_payload, "LINKEDIN")  
        display_tavily_payload(workplace_payload, "WORKPLACE VALIDATION")
        
        # Execute searches with debugging
        print_header("EXECUTING CONCURRENT SEARCHES")
        
        print("🚀 Launching 3 concurrent Tavily searches...")
        concurrent_start = time.time()
        
        try:
            main_task = self.tavily_client.search_async(**main_payload)
            linkedin_task = self.tavily_client.search_async(**linkedin_payload)
            workplace_task = self.tavily_client.search_async(**workplace_validation_payload)
            
            online_results, linkedin_results, workplace_site_results = await asyncio.gather(
                main_task, linkedin_task, workplace_task
            )
            
            self.api_call_count += 3
            concurrent_end = time.time()
            print(f"✅ All searches completed in {concurrent_end - concurrent_start:.2f} seconds")
            
        except Exception as e:
            print(f"❌ Search execution failed: {str(e)}")
            traceback.print_exc()
            return {"error": str(e)}
        
        # Display raw results
        print_header("RAW SEARCH RESULTS ANALYSIS")
        display_raw_tavily_results(online_results, "MAIN SEARCH")
        display_raw_tavily_results(linkedin_results, "LINKEDIN SEARCH")
        display_raw_tavily_results(workplace_site_results, "WORKPLACE VALIDATION")
        
        # Validation phase
        print_header("VALIDATION PHASE")
        
        search_input = {**doctor_info, "entity_type": "ent_activity"}
        
        print_subsection("Validating Main Search Results")
        validated_main = await self._validate_results_async(online_results.get("results", []), search_input)
        
        print(f"📊 Main Search: {len(online_results.get('results', []))} raw → {len(validated_main)} validated")
        for i, result in enumerate(validated_main):
            display_validation_details(result, i, search_input)
        
        print_subsection("Validating LinkedIn Search Results")
        validated_linkedin = await self._validate_results_async(linkedin_results.get("results", []), search_input)
        
        print(f"📊 LinkedIn Search: {len(linkedin_results.get('results', []))} raw → {len(validated_linkedin)} validated")
        for i, result in enumerate(validated_linkedin):
            display_validation_details(result, i, search_input)
        
        # Workplace domain extraction
        print_section("WORKPLACE DOMAIN EXTRACTION")
        
        official_domain = self._extract_official_domain(
            workplace_site_results.get("results", []),
            doctor_info.get("workplaceName", "")
        )
        
        if official_domain:
            print(f"🏢 ✅ Official domain found: {official_domain}")
            
            # Targeted workplace search
            print_subsection("Targeted Workplace Search")
            targeted_payload = self.payload_builder.build_targeted_workplace_payload(doctor_info, official_domain)
            display_tavily_payload(targeted_payload, f"TARGETED ({official_domain})")
            
            targeted_results = await self.tavily_client.search_async(**targeted_payload)
            self.api_call_count += 1
            
            display_raw_tavily_results(targeted_results, f"TARGETED WORKPLACE ({official_domain})")
            
            validated_workplace = await self._validate_results_async(
                targeted_results.get("results", []), search_input
            )
            
            print(f"📊 Targeted Workplace: {len(targeted_results.get('results', []))} raw → {len(validated_workplace)} validated")
            
        else:
            print(f"🏢 ❌ No official domain found - using fallback search")
            
            fallback_payload = self.payload_builder.build_workplace_payload(doctor_info)
            display_tavily_payload(fallback_payload, "FALLBACK WORKPLACE")
            
            fallback_results = await self.tavily_client.search_async(**fallback_payload)
            self.api_call_count += 1
            
            validated_workplace = await self._validate_results_async(
                fallback_results.get("results", []), search_input
            )
            
            print(f"📊 Fallback Workplace: {len(fallback_results.get('results', []))} raw → {len(validated_workplace)} validated")
        
        # Response assembly
        print_header("RESPONSE ASSEMBLY")
        
        clean_response = {
            "search_results": {
                "online_search": {"results": []},
                "linkedin_search": {"results": []},
                "workplace_search": {"results": []},
            },
            "tavily_answer": online_results.get("answer", "") or "",
            "llm_answer": "",
        }
        
        # Assemble final results
        print_subsection("Assembling Final Results")
        
        # Main search results
        main_added = 0
        for result in validated_main:
            validation = result.get("validation", {})
            if validation.get("is_valid", False):
                clean_response["search_results"]["online_search"]["results"].append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                })
                main_added += 1
        print(f"📋 Main Search: Added {main_added} results")
        
        # LinkedIn results
        linkedin_added = 0
        for result in validated_linkedin:
            validation = result.get("validation", {})
            if validation.get("is_valid", False):
                clean_response["search_results"]["linkedin_search"]["results"].append({
                    "url": result.get("url", ""),
                    "title": result.get("title", ""),
                    "content": result.get("content", ""),
                    "score": result.get("score", 0.0),
                })
                linkedin_added += 1
        print(f"📋 LinkedIn Search: Added {linkedin_added} results")
        
        # Workplace results with official domain logic
        workplace_added = 0
        if official_domain:
            hcp_found_on_official_domain = False
            
            for result in validated_workplace:
                validation = result.get("validation", {})
                if validation.get("is_valid", False):
                    result_url = result.get("url", "")
                    if result_url:
                        from urllib.parse import urlparse
                        result_domain = urlparse(result_url).netloc
                        if result_domain.startswith("www."):
                            result_domain = result_domain[4:]
                        
                        if result_domain == official_domain or result_domain.endswith(f".{official_domain}"):
                            clean_response["search_results"]["workplace_search"]["results"].append({
                                "url": result.get("url", ""),
                                "title": result.get("title", ""),
                                "content": result.get("content", ""),
                                "score": result.get("score", 0.0),
                            })
                            hcp_found_on_official_domain = True
                            workplace_added += 1
            
            # Add official domain info if no HCP found
            if not hcp_found_on_official_domain:
                clean_response["search_results"]["workplace_search"]["results"].append({
                    "url": f"https://{official_domain}",
                    "title": f"Official Website - {doctor_info.get('workplaceName', '')}",
                    "content": f"Found official domain {official_domain} for {doctor_info.get('workplaceName', '')}, but no specific information about {doctor_info.get('firstName', '')} {doctor_info.get('lastName', '')} was found on this official website.",
                    "score": 0.5,
                })
                workplace_added += 1
                print(f"📋 Added official domain info (no HCP found on domain)")
        
        print(f"📋 Workplace Search: Added {workplace_added} results")
        
        # LLM Analysis
        print_section("LLM ANALYSIS GENERATION")
        
        try:
            llm_start = time.time()
            clean_response["llm_answer"] = await self._generate_llm_analysis_async(
                clean_response, doctor_info, specialty_name
            )
            llm_end = time.time()
            print(f"🤖 LLM analysis completed in {llm_end - llm_start:.2f} seconds")
            
        except Exception as e:
            print(f"❌ LLM analysis failed: {str(e)}")
            clean_response["llm_answer"] = f"LLM analysis failed: {str(e)}"
        
        # Final summary
        search_end_time = time.time()
        
        print_header("SEARCH COMPLETION SUMMARY")
        display_final_results_summary(clean_response, "HCP SEARCH")
        display_performance_metrics(search_start_time, search_end_time, self.api_call_count)
        
        return clean_response

# ============================================================================
# TEST EXECUTION FUNCTIONS
# ============================================================================

async def run_single_test(test_name: str, test_data: Dict[str, Any]):
    """Run a single comprehensive test"""
    print_header(f"EXECUTING TEST: {test_name}")
    
    try:
        # Initialize debug agent
        agent = DebugWebSearchAgent()
        
        # Determine test type
        entity_type = test_data.get("entity_type", "")
        
        if entity_type == "ent_activity" or ("firstName" in test_data and "lastName" in test_data):
            # HCP test
            results = await agent.debug_search_hcp(test_data)
        else:
            # Workplace test
            print("🏢 Executing workplace search...")
            results = await agent.search_workplace(test_data)
            display_final_results_summary(results, "WORKPLACE SEARCH")
        
        return results, True
        
    except Exception as e:
        print(f"❌ TEST FAILED: {str(e)}")
        traceback.print_exc()
        return None, False

async def run_quick_validation_test():
    """Run quick validation test with known data"""
    print_header("QUICK VALIDATION TEST")
    
    # Test validation with your exact data
    sample_result = {
        "url": "https://www.ats-brianza.it/component/ipnetregister/?task=import_download&catid=390&id=636&file=CV_SPINOGLIO_S_[390-636-1].pdf",
        "title": "[PDF] Curriculum Vitae Europass - ATS Brianza",
        "content": "Pagina 5 / 6 - Curriculum Vitae di. Simona Spinoglio ... • Conduzione laboratorio musicale e teatrale presso UILM Sezione di Monza dal 2018 ad ... della voce",
        "score": 0.61196244
    }
    
    try:
        agent = WebSearchAgent()
        search_input = TEST_HCP_COMPLETE.copy()
        
        print("🧪 Testing validation with known good result:")
        print(f"   URL: {sample_result['url']}")
        print(f"   Score: {sample_result['score']}")
        print(f"   Content: {sample_result['content'][:100]}...")
        
        validation_result = agent.validator.validate_url(sample_result, search_input)
        
        print(f"\n📊 Validation Results:")
        print(f"   ✅ Is Valid: {validation_result.is_valid}")
        print(f"   🎯 Confidence: {validation_result.confidence_score:.3f}")
        print(f"   📝 Reasons:")
        for reason in validation_result.validation_reasons:
            print(f"      - {reason}")
        
        return validation_result.is_valid
        
    except Exception as e:
        print(f"❌ Validation test failed: {str(e)}")
        traceback.print_exc()
        return False

async def run_all_tests():
    """Run complete test suite"""
    print_header("COMPREHENSIVE WEBSEARCH AGENT TEST SUITE", "🧪", 120)
    
    overall_start = time.time()
    test_results = []
    
    # Quick validation test first
    print("\n" + "="*100)
    validation_passed = await run_quick_validation_test()
    test_results.append(("Quick Validation", validation_passed))
    
    # Run all test cases
    for test_name, test_data in ALL_TEST_CASES:
        print("\n" + "="*100)
        result, success = await run_single_test(test_name, test_data)
        test_results.append((test_name, success))
        
        # Add delay between tests to be respectful to APIs
        await asyncio.sleep(2)
    
    # Final summary
    overall_end = time.time()
    
    print_header("FINAL TEST SUITE SUMMARY", "🎉", 120)
    
    print(f"⏱️  Total Suite Duration: {overall_end - overall_start:.2f} seconds")
    print(f"📊 Tests Executed: {len(test_results)}")
    
    passed = sum(1 for _, success in test_results if success)
    print(f"✅ Tests Passed: {passed}")
    print(f"❌ Tests Failed: {len(test_results) - passed}")
    
    print(f"\n📋 Detailed Results:")
    for test_name, success in test_results:
        status = "✅ PASSED" if success else "❌ FAILED"
        print(f"   {test_name}: {status}")
    
    if passed == len(test_results):
        print(f"\n🎉 ALL TESTS PASSED! Your WebSearch Agent is working perfectly!")
    else:
        print(f"\n⚠️  Some tests failed. Check the detailed output above for debugging.")

# ============================================================================
# MAIN EXECUTION
# ============================================================================

if __name__ == "__main__":
    print("🚀 Starting Comprehensive WebSearch Agent Test Suite")
    print("="*120)
    
    try:
        # Run the complete test suite
        asyncio.run(run_all_tests())
        
    except KeyboardInterrupt:
        print("\n⚠️  Test suite interrupted by user")
    except Exception as e:
        print(f"\n❌ Test suite failed with error: {str(e)}")
        traceback.print_exc()
    
    print("\n🏁 Test suite completed!")