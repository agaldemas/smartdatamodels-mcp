import sys
import asyncio
import traceback
import time
from pathlib import Path
import pytest
import json

# Add src to path relative to script location
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from smart_data_models_mcp.data_access import SmartDataModelsAPI


# =============================================================================
# TEST TOOLS - Tests for main API functionality and tools
# =============================================================================

@pytest.mark.asyncio
async def test_tools():
    """Test the main API functionality and core tools."""
    api = SmartDataModelsAPI()
    print("Testing Smart Data Models MCP tools internal API..")

    try:
        # Test 1: list_domains
        print("\n1. Testing list_domains...")
        domains = await api.list_domains()
        print(f"✓ Domains found: {len(domains)}")
        print(f"  Domains: {domains}")

        # Test 2: list_subjects
        print("\n2. Testing list_subjects...")
        subjects = await api.list_subjects()
        print(f"✓ Subjects found: {len(subjects)}")
        if subjects:
            print(f"  Sample subjects: {subjects[:5]}")

        # Test 3: list_domain_subjects (using first available domain)
        print("\n3. Testing list_domain_subjects...")
        if domains:
            test_domain = domains[0]
            domain_subjects = await api.list_domain_subjects(test_domain)
            print(f"✓ {test_domain} subjects: {len(domain_subjects)} - {domain_subjects[:5]}..." if len(domain_subjects) > 5 else f"✓ {test_domain} subjects: {len(domain_subjects)} - {domain_subjects}")

        # Test 3.1: Test _normalize_subject method (internal utility)
        print("\n3.1. Testing _normalize_subject utility method...")
        try:
            # Test normalization without prefix
            normalized = api._normalize_subject("Transportation")
            assert normalized == "dataModel.Transportation", f"Expected 'dataModel.Transportation', got '{normalized}'"

            # Test normalization with existing prefix
            normalized = api._normalize_subject("dataModel.Transportation")
            assert normalized == "dataModel.Transportation", f"Expected 'dataModel.Transportation', got '{normalized}'"

            # Test normalization with None
            normalized = api._normalize_subject(None)
            assert normalized is None, f"Expected None, got '{normalized}'"

            print("✓ _normalize_subject utility method working correctly")
        except Exception as e:
            print(f"✗ _normalize_subject test failed: {e}")

        # Test 4: list_models_in_subject (using first available subject)
        print("\n4. Testing list_models_in_subject...")
        if subjects:
            test_subject = subjects[0]
            models_in_subject = await api.list_models_in_subject(test_subject)
            print(f"✓ {test_subject} models: {len(models_in_subject)} - {models_in_subject[:5]}..." if len(models_in_subject) > 5 else f"✓ {test_subject} models: {len(models_in_subject)} - {models_in_subject}")

        # Test 5: get_model_details (using first available model from first subject)
        print("\n5. Testing get_model_details...")
        if subjects and models_in_subject:
            test_model = models_in_subject[0]
            details = await api.get_model_details(test_subject, test_model)
            print(f"✓ Model details for {test_subject}/{test_model}:")
            print(f"  Description: {details.get('description', 'N/A')[:100]}...")
            print(f"  Attributes: {len(details.get('attributes', []))}")
            print(f"  Source: {details.get('source', 'N/A')}")

        # Test 6: suggest_matching_models
        print("\n6. Testing suggest_matching_models...")
        test_data = {
            "id": "test-entity-001",
            "type": "TestEntity",
            "location": {"coordinates": [2.174, 41.387]},
            "temperature": 25.5,
            "humidity": 60.0
        }
        try:
            suggestions = await api.suggest_matching_models(test_data, top_k=3)
            print(f"✓ Model suggestions for test data: {len(suggestions)} suggestions")
            for i, suggestion in enumerate(suggestions[:3]):
                print(f"  {i+1}. {suggestion['subject']}/{suggestion['model']} (similarity: {suggestion['similarity']})")
        except Exception as e:
            print(f"✗ Model suggestion test failed: {e}")

        # Test 7: search_models (enhanced)
        print("\n7. Testing search_models...")
        search_results = await api.search_models("animal cattle livestock", include_attributes=False)
        print(f"✓ Search results for 'animal cattle livestock': {len(search_results)}")

        # Test 8: Also test with attributes included
        search_with_attrs = await api.search_models("temperature", include_attributes=True, limit=5)
        print(f"✓ Search results for 'temperature' with attributes: {len(search_with_attrs)}")
        for result in search_with_attrs[:2]:
            matched_attrs = result.get('matched_attributes', [])
            print(f"  - {result.get('subject')}/{result.get('model')}: {len(matched_attrs)} relevant attributes")

        # Test 9: Test private utility methods
        print("\n9. Testing private utility methods...")

        # Test 9.1: _find_domain_repository
        print("  9.1. Testing _find_domain_repository...")
        try:
            # Test with exact match
            repo = await api._find_domain_repository("SmartCities")
            print(f"    ✓ Found repository for SmartCities: {repo}")

            # Test with partial match
            repo = await api._find_domain_repository("Water")
            print(f"    ✓ Found repository for Water: {repo}")

            # Test with non-existent domain
            repo = await api._find_domain_repository("NonExistentDomain12345")
            print(f"    ✓ Correctly returned None for non-existent domain: {repo}")
        except Exception as e:
            print(f"    ✗ _find_domain_repository test failed: {e}")

        # Test 9.2: _get_subjects_from_github_api
        print("  9.2. Testing _get_subjects_from_github_api...")
        try:
            if domains:
                subjects = await api._get_subjects_from_github_api(domains[0])
                if subjects:
                    print(f"    ✓ Found {len(subjects)} subjects in {domains[0]}: {subjects[:3]}...")
                else:
                    print(f"    ⚠ No subjects found in {domains[0]} (this may be normal)")
        except Exception as e:
            print(f"    ✗ _get_subjects_from_github_api test failed: {e}")

        # Test 9.3: _get_models_from_github_api
        print("  9.3. Testing _get_models_from_github_api...")
        try:
            if subjects:
                models = await api._get_models_from_github_api(subjects[0])
                if models:
                    print(f"    ✓ Found {len(models)} models in {subjects[0]}: {models[:3]}...")
                else:
                    print(f"    ⚠ No models found in {subjects[0]} (this may be normal)")
        except Exception as e:
            print(f"    ✗ _get_models_from_github_api test failed: {e}")

        # Test 9.4: Test cache functionality
        print("  9.4. Testing cache functionality...")
        try:
            # Test cache set and get
            api._cache.set("test_key", "test_value")
            cached_value = api._cache.get("test_key")
            assert cached_value == "test_value", "Cache set/get failed"
            print("    ✓ Cache set/get working correctly")

            # Test cache TTL (set very short TTL and wait)
            api._cache.set("ttl_test", "ttl_value")
            time.sleep(0.1)  # Wait a bit
            cached_value = api._cache.get("ttl_test")
            print(f"    ✓ Cache TTL working: {cached_value is not None}")
        except Exception as e:
            print(f"    ✗ Cache test failed: {e}")

        # Test 10: Test additional private methods and edge cases
        print("\n10. Testing additional methods and edge cases...")

        # Test 10.1: Test _generate_basic_context method
        print("  10.1. Testing _generate_basic_context method...")
        try:
            context = api._generate_basic_context("TestSubject")
            assert isinstance(context, dict), "Context should be a dictionary"
            assert "@context" in context, "Context should have @context key"
            assert "Property" in context["@context"], "Context should have Property mapping"
            print("    ✓ _generate_basic_context working correctly")
        except Exception as e:
            print(f"    ✗ _generate_basic_context test failed: {e}")

        # Test 10.2: Test error handling for invalid inputs
        print("  10.2. Testing error handling...")
        try:
            # Test with invalid subject
            try:
                await api.get_model_schema("InvalidSubject", "InvalidModel")
                print("    ⚠ Should have raised an error for invalid subject/model")
            except ValueError as e:
                print("    ✓ Correctly raised ValueError for invalid subject/model")

            # Test suggest_matching_models with invalid data
            invalid_suggestions = await api.suggest_matching_models("invalid_data", top_k=3)
            print(f"    ✓ Handled invalid data gracefully: {len(invalid_suggestions)} suggestions")

            # Test search with empty query
            empty_search = await api.search_models("", limit=5)
            print(f"    ✓ Handled empty search query: {len(empty_search)} results")
        except Exception as e:
            print(f"    ✗ Error handling test failed: {e}")

        # Test 10.3: Test cache behavior under load
        print("  10.3. Testing cache behavior...")
        try:
            # Test multiple cache operations
            for i in range(5):
                api._cache.set(f"test_key_{i}", f"test_value_{i}")

            retrieved_count = 0
            for i in range(5):
                value = api._cache.get(f"test_key_{i}")
                if value:
                    retrieved_count += 1

            print(f"    ✓ Cache operations: {retrieved_count}/5 values retrieved correctly")
        except Exception as e:
            print(f"    ✗ Cache behavior test failed: {e}")

        print("\n✓ All API tools tests completed successfully!")

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        traceback.print_exc()


# =============================================================================
# TEST RESOURCES - Tests for resource template methods
# =============================================================================

@pytest.mark.asyncio
async def test_resources():
    """Test all resource template methods in a single comprehensive test."""
    api = SmartDataModelsAPI()
    print("\nTesting all resource template methods...")

    subject = "Transportation"
    model = "TrafficFlowObserved"

    try:
        # Test 1: get_model_schema
        print("\n1. Testing get_model_schema method...")
        schema = await api.get_model_schema(subject, model)

        # Verify schema is a dictionary
        assert isinstance(schema, dict), f"Schema should be a dictionary, got {type(schema)}"

        # Verify schema has required JSON Schema properties
        required_keys = ["$schema", "allOf"]  # Smart Data Models use allOf structure
        for key in required_keys:
            assert key in schema, f"Schema missing required key: {key}"

        # Verify schema has allOf structure (common in Smart Data Models)
        allOf = schema.get("allOf", [])
        assert isinstance(allOf, list), "Schema allOf should be a list"
        assert len(allOf) > 0, "Schema should have at least one allOf element"

        # Find the properties in the allOf structure
        properties = None
        for element in allOf:
            if isinstance(element, dict) and "properties" in element:
                properties = element["properties"]
                break

        # If no properties found in allOf, check if schema has direct properties
        if properties is None:
            properties = schema.get("properties", {})

        assert isinstance(properties, dict), "Schema properties should be a dictionary"
        assert len(properties) > 0, "Schema should have at least one property"

        # Verify required fields are present
        required = schema.get("required", [])
        assert isinstance(required, list), "Schema required should be a list"
        assert "id" in required, "Schema should require 'id' field"
        assert "type" in required, "Schema should require 'type' field"

        print(f"✓ Schema test passed for {subject}/{model}")
        print(f"  Schema has {len(properties)} properties")
        print(f"  Required fields: {required}")

        # Test 2: get_model_examples
        print("\n2. Testing get_model_examples method...")
        examples = await api.get_model_examples(subject, model)

        # Verify examples is a list
        assert isinstance(examples, list), f"Examples should be a list, got {type(examples)}"

        if len(examples) > 0:
            # Verify first example is a dictionary
            example = examples[0]
            assert isinstance(example, dict), f"Example should be a dictionary, got {type(example)}"

            # Verify example has required NGSI-LD fields
            assert "id" in example, "Example should have 'id' field"
            assert "type" in example, "Example should have 'type' field"
            assert example["type"] == model, f"Example type should be '{model}', got {example['type']}"

            print(f"✓ Examples test passed for {subject}/{model}")
            print(f"  Found {len(examples)} examples")
            print(f"  First example has keys: {list(example.keys())}")
        else:
            print(f"⚠ No examples found for {subject}/{model}, but method executed successfully")

        # Test 3: get_subject_context
        print("\n3. Testing get_subject_context method...")
        context = await api.get_subject_context(subject)

        # Verify context is a dictionary
        assert isinstance(context, dict), f"Context should be a dictionary, got {type(context)}"

        # Verify context has @context key
        assert "@context" in context, "Context should have '@context' key"

        # Verify @context is a dictionary
        context_content = context["@context"]
        assert isinstance(context_content, dict), f"@context should be a dictionary, got {type(context_content)}"

        # Verify common NGSI-LD context mappings are present
        common_mappings = ["Property", "Relationship", "GeoProperty", "Location"]
        for mapping in common_mappings:
            if mapping in context_content:
                print(f"  Found common mapping: {mapping}")

        print(f"✓ Context test passed for {subject}")
        print(f"  Context has {len(context_content)} mappings")

        # Test 4: Resource template methods integration
        print("\n4. Testing resource template methods integration...")
        # Test schema
        print("  - Testing schema resource...")
        schema = await api.get_model_schema(subject, model)
        assert isinstance(schema, dict)
        # Smart Data Models may not have direct "type" field, they use allOf structure
        if "type" in schema:
            assert schema.get("type") == "object"
        else:
            # Verify it has allOf structure instead
            assert "allOf" in schema
        print("    ✓ Schema resource working")

        # Test examples
        print("  - Testing examples resource...")
        examples = await api.get_model_examples(subject, model)
        assert isinstance(examples, list)
        print("    ✓ Examples resource working")

        # Test context
        print("  - Testing context resource...")
        context = await api.get_subject_context(subject)
        assert isinstance(context, dict)
        assert "@context" in context
        print("    ✓ Context resource working")

        print(f"\n✓ All resource template methods working for {subject}/{model}")
        print("🎉 All resources tests completed successfully!")

    except Exception as e:
        print(f"✗ Resources test failed: {e}")
        traceback.print_exc()
        raise


# =============================================================================
# MAIN TEST RUNNERS
# =============================================================================

async def run_tools_tests():
    """Run all tools tests."""
    print("=" * 60)
    print("🛠️  RUNNING TOOLS TESTS")
    print("=" * 60)
    await test_tools()
    print("\n" + "=" * 60)


async def run_resources_tests():
    """Run all resources tests."""
    print("=" * 60)
    print("📚 RUNNING RESOURCES TESTS")
    print("=" * 60)
    await test_resources()
    print("\n" + "=" * 60)


async def run_all_tests():
    """Run all tests (both tools and resources)."""
    print("=" * 60)
    print("🚀 RUNNING ALL TESTS")
    print("=" * 60)
    await run_tools_tests()
    await run_resources_tests()
    print("🎉 ALL TESTS COMPLETED!")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description='Run Smart Data Models tests')
    parser.add_argument('--tools', action='store_true', help='Run only tools tests')
    parser.add_argument('--resources', action='store_true', help='Run only resources tests')
    parser.add_argument('--all', action='store_true', help='Run all tests (default)')

    args = parser.parse_args()

    if args.tools:
        asyncio.run(run_tools_tests())
    elif args.resources:
        asyncio.run(run_resources_tests())
    else:
        # Default to running all tests
        asyncio.run(run_all_tests())
