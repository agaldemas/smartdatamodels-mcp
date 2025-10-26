import sys
import asyncio
sys.path.insert(0, 'src')

from smart_data_models_mcp.data_access import SmartDataModelsAPI

async def test_api():
    api = SmartDataModelsAPI()
    print("Testing Smart Data Models API...")

    try:
        domains = await api.list_domains()
        print(f"✓ Domains found: {len(domains)}")
        print(f"  Domains: {domains}")

        subjects = await api.list_subjects()
        print(f"✓ Subjects found: {len(subjects)}")
        if subjects:
            print(f"  Sample subjects: {subjects[:5]}")

        # Test agrifood domain
        agrifood_subjects = await api.list_domain_subjects("Agrifood")
        print(f"✓ Agrifood subjects: {len(agrifood_subjects)} - {agrifood_subjects}")

        # Test search
        search_results = await api.search_models("animal cattle livestock", include_attributes=False)
        print(f"✓ Search results: {len(search_results)}")
        for result in search_results[:3]:
            print(f"  - {result.get('subject')}/{result.get('model')}: {result.get('description', '')}")

    except Exception as e:
        print(f"✗ Error: {e}")
        import traceback
        traceback.print_exc()

if __name__ == "__main__":
    asyncio.run(test_api())
