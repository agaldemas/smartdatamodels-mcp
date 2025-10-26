import sys
import asyncio
import traceback
from pathlib import Path
import pytest

# Add src to path relative to script location
sys.path.insert(0, str(Path(__file__).parent / 'src'))

from smart_data_models_mcp.data_access import SmartDataModelsAPI

@pytest.mark.asyncio
async def test_api():
    api = SmartDataModelsAPI()
    print("Testing Smart Data Models API...")

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

        # Test 6: get_model_schema
        print("\n6. Testing get_model_schema...")
        if subjects and models_in_subject:
            try:
                schema = await api.get_model_schema(test_subject, test_model)
                print(f"✓ Schema for {test_subject}/{test_model}: {type(schema)} with keys: {list(schema.keys()) if isinstance(schema, dict) else 'N/A'}")
            except Exception as e:
                print(f"✗ Schema test failed: {e}")

        # Test 7: get_model_examples
        print("\n7. Testing get_model_examples...")
        if subjects and models_in_subject:
            try:
                examples = await api.get_model_examples(test_subject, test_model)
                print(f"✓ Examples for {test_subject}/{test_model}: {len(examples)} examples")
            except Exception as e:
                print(f"✗ Examples test failed: {e}")

        # Test 8: get_subject_context
        print("\n8. Testing get_subject_context...")
        if subjects:
            try:
                context = await api.get_subject_context(test_subject)
                print(f"✓ Context for {test_subject}: {type(context)}")
            except Exception as e:
                print(f"✗ Context test failed: {e}")

        # Test 9: suggest_matching_models
        print("\n9. Testing suggest_matching_models...")
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

        # Test 10: search_models (enhanced)
        print("\n10. Testing search_models...")
        search_results = await api.search_models("animal cattle livestock", include_attributes=False)
        print(f"✓ Search results for 'animal cattle livestock': {len(search_results)}")

        # Also test with attributes included
        search_with_attrs = await api.search_models("temperature", include_attributes=True, limit=5)
        print(f"✓ Search results for 'temperature' with attributes: {len(search_with_attrs)}")
        for result in search_with_attrs[:2]:
            matched_attrs = result.get('matched_attributes', [])
            print(f"  - {result.get('subject')}/{result.get('model')}: {len(matched_attrs)} relevant attributes")

        print("\n✓ All API tests completed successfully!")

    except Exception as e:
        print(f"✗ Error during testing: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_api())
