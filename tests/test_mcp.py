"""Basic integration tests for smart-data-models-mcp."""

import asyncio
import json
import sys
import os
import unittest
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

# Add src directory to path for local testing
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', 'src'))

from smart_data_models_mcp import (
    data_access,
    model_generator,
    model_validator,
)

# Import MCP server components for testing
try:
    from smart_data_models_mcp.server import (
        data_api,
        ngsi_generator,
        schema_validator,
        list_domains,
        list_subjects,
        list_domain_subjects,
        list_models_in_subject,
        search_data_models,
        get_model_details,
        validate_against_model,
        generate_ngsi_ld_from_json,
        suggest_matching_models,
        get_instructions,
        get_model_schema,
        get_model_examples,
        get_subject_context,
    )
    # Access the underlying functions from FunctionTool objects
    list_domains_func = list_domains.fn
    list_subjects_func = list_subjects.fn
    list_domain_subjects_func = list_domain_subjects.fn
    list_models_in_subject_func = list_models_in_subject.fn
    search_data_models_func = search_data_models.fn
    get_model_details_func = get_model_details.fn
    validate_against_model_func = validate_against_model.fn
    generate_ngsi_ld_from_json_func = generate_ngsi_ld_from_json.fn
    suggest_matching_models_func = suggest_matching_models.fn

    # Access resource functions
    get_instructions_func = get_instructions.fn
    get_model_schema_func = get_model_schema.fn
    get_model_examples_func = get_model_examples.fn
    get_subject_context_func = get_subject_context.fn

except ImportError:
    # Handle import errors gracefully for testing
    pass


class TestDataAccess(unittest.TestCase):
    """Test data access functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.api = data_access.SmartDataModelsAPI()

    def test_list_domains(self):
        """Test listing domains."""
        async def test_async():
            domains = await self.api.list_domains()
            # Should return known domains as fallback
            self.assertIsInstance(domains, list)
            self.assertGreater(len(domains), 0)
            # Should include expected domains
            self.assertIn("SmartCities", domains)
            self.assertIn("SmartEnergy", domains)
            self.assertIn("SmartAgrifood", domains)
            self.assertIn("SmartWater", domains)

        # Run the async test
        asyncio.run(test_async())


class TestModelGenerator(unittest.TestCase):
    """Test NGSI-LD model generation."""

    def setUp(self):
        """Set up test fixtures."""
        self.generator = model_generator.NGSILDGenerator()

    def test_generate_ngsi_ld_basic(self):
        """Test basic NGSI-LD entity generation."""
        test_data = {
            "temperature": 25.5,
            "humidity": 60
        }

        async def test_async():
            entity = await self.generator.generate_ngsi_ld(test_data)

            # Check basic structure
            self.assertIn("id", entity)
            self.assertIn("type", entity)
            self.assertIn("@context", entity)

            # Check that properties were created
            self.assertIn("temperature", entity)
            self.assertIn("humidity", entity)

            # Check property structure
            temp_prop = entity["temperature"]
            self.assertEqual(temp_prop["type"], "Property")
            self.assertEqual(temp_prop["value"], 25.5)

        asyncio.run(test_async())

    def test_generate_ngsi_ld_with_location(self):
        """Test NGSI-LD generation with geographic data."""
        test_data = {
            "name": "Test Sensor",
            "location": [-122.4194, 37.7749]
        }

        async def test_async():
            entity = await self.generator.generate_ngsi_ld(test_data)

            # Check location is converted to GeoProperty
            self.assertIn("location", entity)
            location_prop = entity["location"]
            self.assertEqual(location_prop["type"], "GeoProperty")
            self.assertIn("value", location_prop)

            # Check GeoJSON structure
            geo_value = location_prop["value"]
            self.assertEqual(geo_value["type"], "Point")
            self.assertEqual(geo_value["coordinates"], [-122.4194, 37.7749])

        asyncio.run(test_async())


class TestModelValidator(unittest.TestCase):
    """Test schema validation functionality."""

    def setUp(self):
        """Set up test fixtures."""
        self.validator = model_validator.SchemaValidator()

    def test_ngsi_ld_entity_validation(self):
        """Test NGSI-LD entity structure validation."""
        # Valid NGSI-LD entity
        valid_entity = {
            "id": "urn:ngsi-ld:WeatherObserved:001",
            "type": "WeatherObserved",
            "@context": "https://uri.etsi.org/ngsi-ld/v1.7/gsma-cim/common.jsonld",
            "temperature": {"type": "Property", "value": 25.5},
            "humidity": {"type": "Property", "value": 60}
        }

        async def test_async():
            is_valid, errors = await self.validator.validate_ngsi_ld_entity(valid_entity)

            self.assertTrue(is_valid)
            self.assertEqual(len(errors), 0)

        asyncio.run(test_async())

    def test_ngsi_ld_entity_validation_missing_id(self):
        """Test validation fails for entity missing required id."""
        invalid_entity = {
            "type": "WeatherObserved",
            "@context": "https://uri.etsi.org/ngsi-ld/v1.7/gsma-cim/common.jsonld"
        }

        async def test_async():
            is_valid, errors = await self.validator.validate_ngsi_ld_entity(invalid_entity)

            self.assertFalse(is_valid)
            self.assertIn("Missing required 'id' field", errors)

        asyncio.run(test_async())



class TestMCPTools(unittest.TestCase):
    """Test MCP tool functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the global instances used by the tools
        self.mock_data_api = MagicMock()
        self.mock_ngsi_generator = MagicMock()
        self.mock_schema_validator = MagicMock()

        # Make async methods return coroutines
        self.mock_data_api.list_domains = AsyncMock(return_value=["SmartCities", "SmartEnergy"])
        self.mock_data_api.list_subjects = AsyncMock(return_value=["dataModel.Weather", "dataModel.User"])
        self.mock_data_api.list_domain_subjects = AsyncMock(return_value=["dataModel.Weather", "dataModel.Sensor"])
        self.mock_data_api.list_models_in_subject = AsyncMock(return_value=["WeatherObserved", "AirQualityObserved"])
        self.mock_data_api.search_models = AsyncMock(return_value=[{"name": "WeatherObserved", "subject": "dataModel.Weather"}])
        self.mock_data_api.get_model_details = AsyncMock(return_value={"schema": {}, "examples": []})
        self.mock_data_api.suggest_matching_models = AsyncMock(return_value=[
            {"model": "WeatherObserved", "score": 0.95},
            {"model": "AirQualityObserved", "score": 0.85}
        ])
        self.mock_data_api.get_model_schema = AsyncMock(return_value={"type": "object", "properties": {"temperature": {"type": "number"}}})
        self.mock_data_api.get_model_examples = AsyncMock(return_value=[{"type": "WeatherObserved"}])
        self.mock_data_api.get_subject_context = AsyncMock(return_value={
            "@context": {"temperature": "https://uri.etsi.org/ngsi-ld/temperature"}
        })

        self.mock_ngsi_generator.generate_ngsi_ld = AsyncMock(return_value={
            "id": "urn:ngsi-ld:Test:001",
            "type": "Test",
            "@context": "https://uri.etsi.org/ngsi-ld/v1.7/gsma-cim/common.jsonld",
            "temperature": {"type": "Property", "value": 25.5},
            "humidity": {"type": "Property", "value": 60}
        })

        # Patch the global instances in the server module
        self.patches = [
            patch('smart_data_models_mcp.server.data_api', self.mock_data_api),
            patch('smart_data_models_mcp.server.ngsi_generator', self.mock_ngsi_generator),
            patch('smart_data_models_mcp.server.schema_validator', self.mock_schema_validator),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        """Clean up test fixtures."""
        for p in self.patches:
            p.stop()

    async def async_test_list_domains(self):
        """Test list_domains tool."""
        # Setup mock
        self.mock_data_api.list_domains.return_value = ["SmartCities", "SmartEnergy"]

        # Call the tool function directly
        result = await list_domains_func()

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertTrue(response["success"])
        self.assertEqual(response["domains"], ["SmartCities", "SmartEnergy"])
        self.assertEqual(response["count"], 2)
        self.mock_data_api.list_domains.assert_called_once()

    def test_list_domains(self):
        """Test list_domains tool wrapper."""
        asyncio.run(self.async_test_list_domains())

    async def async_test_list_domains_error(self):
        """Test list_domains tool with error."""
        # Setup mock to raise exception
        self.mock_data_api.list_domains.side_effect = Exception("API Error")

        # Call the tool function directly
        result = await list_domains_func()

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertFalse(response["success"])
        self.assertIn("API Error", response["error"])

    def test_list_domains_error(self):
        """Test list_domains tool error handling."""
        asyncio.run(self.async_test_list_domains_error())

    async def async_test_list_subjects(self):
        """Test list_subjects tool."""
        # Setup mock
        self.mock_data_api.list_subjects.return_value = ["dataModel.Weather", "dataModel.User"]

        # Call the tool function directly
        result = await list_subjects_func()

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertTrue(response["success"])
        self.assertEqual(response["subjects"], ["dataModel.Weather", "dataModel.User"])
        self.assertEqual(response["count"], 2)

    def test_list_subjects(self):
        """Test list_subjects tool wrapper."""
        asyncio.run(self.async_test_list_subjects())

    async def async_test_list_domain_subjects(self):
        """Test list_domain_subjects tool."""
        # Setup mock
        self.mock_data_api.list_domain_subjects.return_value = ["dataModel.Weather", "dataModel.Sensor"]

        # Call the tool function directly
        result = await list_domain_subjects_func(domain="SmartCities")

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertTrue(response["success"])
        self.assertEqual(response["domain"], "SmartCities")
        self.assertEqual(response["subjects"], ["dataModel.Weather", "dataModel.Sensor"])
        self.assertEqual(response["count"], 2)
        self.mock_data_api.list_domain_subjects.assert_called_once_with("SmartCities")

    def test_list_domain_subjects(self):
        """Test list_domain_subjects tool wrapper."""
        asyncio.run(self.async_test_list_domain_subjects())

    async def async_test_list_models_in_subject(self):
        """Test list_models_in_subject tool."""
        # Setup mock
        self.mock_data_api.list_models_in_subject.return_value = ["WeatherObserved", "AirQualityObserved"]

        # Call the tool function directly
        result = await list_models_in_subject_func(subject="dataModel.Environment")

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertTrue(response["success"])
        self.assertEqual(response["subject"], "dataModel.Environment")
        self.assertEqual(response["models"], ["WeatherObserved", "AirQualityObserved"])
        self.assertEqual(response["count"], 2)
        self.mock_data_api.list_models_in_subject.assert_called_once_with(subject="dataModel.Environment")

    def test_list_models_in_subject(self):
        """Test list_models_in_subject tool wrapper."""
        asyncio.run(self.async_test_list_models_in_subject())

    async def async_test_search_data_models(self):
        """Test search_data_models tool."""
        # Setup mock
        mock_results = [{"name": "WeatherObserved", "subject": "dataModel.Weather"}]
        self.mock_data_api.search_models.return_value = mock_results

        # Call the tool function directly
        result = await search_data_models_func(
            query="weather",
            domain="SmartCities",
            subject="dataModel.Weather",
            include_attributes=False
        )

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertTrue(response["success"])
        self.assertEqual(response["results"], mock_results)
        self.assertEqual(response["count"], 1)
        self.assertEqual(response["query"], "weather")
        self.mock_data_api.search_models.assert_called_once_with(
            query="weather",
            domain="SmartCities",
            subject="dataModel.Weather",
            include_attributes=False
        )

    def test_search_data_models(self):
        """Test search_data_models tool wrapper."""
        asyncio.run(self.async_test_search_data_models())

    async def async_test_get_model_details(self):
        """Test get_model_details tool."""
        # Setup mock
        mock_details = {"schema": {}, "examples": []}
        self.mock_data_api.get_model_details.return_value = mock_details

        # Call the tool function directly
        result = await get_model_details_func(model="WeatherObserved", subject="dataModel.Weather")

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertTrue(response["success"])
        self.assertEqual(response["model"], "WeatherObserved")
        self.assertEqual(response["subject"], "dataModel.Weather")
        self.assertEqual(response["details"], mock_details)
        self.mock_data_api.get_model_details.assert_called_once_with(
            subject="dataModel.Weather",
            model="WeatherObserved"
        )

    def test_get_model_details(self):
        """Test get_model_details tool wrapper."""
        asyncio.run(self.async_test_get_model_details())

    async def async_test_validate_against_model(self):
        """Test validate_against_model tool."""
        test_data = {"temperature": 25.5, "humidity": 60}

        # Call the tool function directly
        result = await validate_against_model_func(
            model="WeatherObserved",
            data=test_data,
            subject="dataModel.Weather"
        )

        # Parse JSON response
        response = json.loads(result)

        # Assertions (validation is disabled, always returns success)
        self.assertTrue(response["success"])
        self.assertEqual(response["model"], "WeatherObserved")
        self.assertEqual(response["subject"], "dataModel.Weather")
        self.assertTrue(response["is_valid"])
        self.assertEqual(response["errors"], [])
        self.assertEqual(response["data_keys"], ["temperature", "humidity"])

    def test_validate_against_model(self):
        """Test validate_against_model tool wrapper."""
        asyncio.run(self.async_test_validate_against_model())

    async def async_test_validate_against_model_invalid_json(self):
        """Test validate_against_model tool with invalid JSON."""
        # Call the tool function directly with invalid JSON string
        result = await validate_against_model_func(
            model="WeatherObserved",
            data='{"invalid": json}',
            subject="dataModel.Weather"
        )

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertFalse(response["success"])
        self.assertIn("Invalid JSON data", response["error"])

    def test_validate_against_model_invalid_json(self):
        """Test validate_against_model tool with invalid JSON."""
        asyncio.run(self.async_test_validate_against_model_invalid_json())

    async def async_test_generate_ngsi_ld_from_json(self):
        """Test generate_ngsi_ld_from_json tool."""
        test_data = {"temperature": 25.5, "humidity": 60}
        mock_entity = {
            "id": "urn:ngsi-ld:Test:001",
            "type": "Test",
            "@context": "https://uri.etsi.org/ngsi-ld/v1.7/gsma-cim/common.jsonld",
            "temperature": {"type": "Property", "value": 25.5},
            "humidity": {"type": "Property", "value": 60}
        }

        # Setup mock
        self.mock_ngsi_generator.generate_ngsi_ld.return_value = mock_entity

        # Call the tool function directly
        result = await generate_ngsi_ld_from_json_func(
            data=test_data,
            entity_type="Test",
            entity_id="urn:ngsi-ld:Test:001",
            context="https://uri.etsi.org/ngsi-ld/v1.7/gsma-cim/common.jsonld"
        )

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertTrue(response["success"])
        self.assertEqual(response["entity"], mock_entity)
        self.assertEqual(response["original_data_keys"], ["temperature", "humidity"])
        self.mock_ngsi_generator.generate_ngsi_ld.assert_called_once_with(
            data=test_data,
            entity_type="Test",
            entity_id="urn:ngsi-ld:Test:001",
            context="https://uri.etsi.org/ngsi-ld/v1.7/gsma-cim/common.jsonld"
        )

    def test_generate_ngsi_ld_from_json(self):
        """Test generate_ngsi_ld_from_json tool wrapper."""
        asyncio.run(self.async_test_generate_ngsi_ld_from_json())

    async def async_test_generate_ngsi_ld_from_json_invalid_json(self):
        """Test generate_ngsi_ld_from_json tool with invalid JSON."""
        # Call the tool function directly with invalid JSON string
        result = await generate_ngsi_ld_from_json_func(data='{"invalid": json}')

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertFalse(response["success"])
        self.assertIn("Invalid JSON data", response["error"])

    def test_generate_ngsi_ld_from_json_invalid_json(self):
        """Test generate_ngsi_ld_from_json tool with invalid JSON."""
        asyncio.run(self.async_test_generate_ngsi_ld_from_json_invalid_json())

    async def async_test_suggest_matching_models(self):
        """Test suggest_matching_models tool."""
        test_data = {"temperature": 25.5, "humidity": 60}
        mock_suggestions = [
            {"model": "WeatherObserved", "score": 0.95},
            {"model": "AirQualityObserved", "score": 0.85}
        ]

        # Setup mock
        self.mock_data_api.suggest_matching_models.return_value = mock_suggestions

        # Call the tool function directly
        result = await suggest_matching_models_func(data=test_data)

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertTrue(response["success"])
        self.assertEqual(response["suggestions"], mock_suggestions)
        self.assertEqual(response["data_keys"], ["temperature", "humidity"])
        self.mock_data_api.suggest_matching_models.assert_called_once_with(data=test_data)

    def test_suggest_matching_models(self):
        """Test suggest_matching_models tool wrapper."""
        asyncio.run(self.async_test_suggest_matching_models())

    async def async_test_suggest_matching_models_invalid_json(self):
        """Test suggest_matching_models tool with invalid JSON."""
        # Call the tool function directly with invalid JSON string
        result = await suggest_matching_models_func(data='{"invalid": json}')

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertFalse(response["success"])
        self.assertIn("Invalid JSON data", response["error"])

    def test_suggest_matching_models_invalid_json(self):
        """Test suggest_matching_models tool with invalid JSON."""
        asyncio.run(self.async_test_suggest_matching_models_invalid_json())


class TestMCPResources(unittest.TestCase):
    """Test MCP resource functions."""

    def setUp(self):
        """Set up test fixtures."""
        # Mock the global instances used by the resources
        self.mock_data_api = MagicMock()

        # Make async methods return coroutines
        self.mock_data_api.get_model_schema = AsyncMock(return_value={"type": "object", "properties": {"temperature": {"type": "number"}}})
        self.mock_data_api.get_model_examples = AsyncMock(return_value=[{"type": "WeatherObserved"}])
        self.mock_data_api.get_subject_context = AsyncMock(return_value={
            "@context": {"temperature": "https://uri.etsi.org/ngsi-ld/temperature"}
        })

        # Patch the global instances in the server module
        self.patches = [
            patch('smart_data_models_mcp.server.data_api', self.mock_data_api),
        ]
        for p in self.patches:
            p.start()

    def tearDown(self):
        """Clean up test fixtures."""
        for p in self.patches:
            p.stop()

    async def async_test_get_instructions(self):
        """Test get_instructions resource."""
        # Call the resource function directly
        result = await get_instructions_func()

        # Assertions
        self.assertIsInstance(result, str)
        self.assertGreater(len(result), 0)
        # Should contain key terms from the instructions
        self.assertIn("FIWARE Smart Data Models", result)
        self.assertIn("NGSI-LD", result)

    def test_get_instructions(self):
        """Test get_instructions resource wrapper."""
        asyncio.run(self.async_test_get_instructions())

    async def async_test_get_model_schema(self):
        """Test get_model_schema resource."""
        mock_schema = {"type": "object", "properties": {"temperature": {"type": "number"}}}

        # Setup mock
        self.mock_data_api.get_model_schema.return_value = mock_schema

        # Call the resource function directly
        result = await get_model_schema_func("dataModel.Weather", "WeatherObserved")

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertEqual(response["type"], "object")
        self.assertIn("properties", response)
        self.mock_data_api.get_model_schema.assert_called_once_with(
            subject="dataModel.Weather",
            model="WeatherObserved"
        )

    def test_get_model_schema(self):
        """Test get_model_schema resource wrapper."""
        asyncio.run(self.async_test_get_model_schema())

    async def async_test_get_model_schema_error(self):
        """Test get_model_schema resource with error."""
        # Setup mock to raise exception
        self.mock_data_api.get_model_schema.side_effect = Exception("Schema not found")

        # Call the resource function directly
        result = await get_model_schema_func("dataModel.Weather", "NonExistentModel")

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertIn("error", response)
        self.assertIn("Schema not found", response["error"])
        self.assertEqual(response["subject"], "dataModel.Weather")
        self.assertEqual(response["model"], "NonExistentModel")

    def test_get_model_schema_error(self):
        """Test get_model_schema resource error handling."""
        asyncio.run(self.async_test_get_model_schema_error())



    async def async_test_get_model_examples_error(self):
        """Test get_model_examples resource with error."""
        # Setup mock to raise exception
        self.mock_data_api.get_model_examples.side_effect = Exception("Examples not found")

        # Call the resource function directly
        result = await get_model_examples_func("dataModel.Weather", "NonExistentModel")

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertIn("error", response)
        self.assertIn("Examples not found", response["error"])
        self.assertEqual(response["subject"], "dataModel.Weather")
        self.assertEqual(response["model"], "NonExistentModel")

    def test_get_model_examples_error(self):
        """Test get_model_examples resource error handling."""
        asyncio.run(self.async_test_get_model_examples_error())

    async def async_test_get_subject_context(self):
        """Test get_subject_context resource."""
        mock_context = {
            "@context": {
                "temperature": "https://uri.etsi.org/ngsi-ld/temperature"
            }
        }

        # Setup mock
        self.mock_data_api.get_subject_context.return_value = mock_context

        # Call the resource function directly
        result = await get_subject_context_func("dataModel.Weather")

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertIn("@context", response)
        self.assertIn("temperature", response["@context"])
        self.mock_data_api.get_subject_context.assert_called_once_with(subject="dataModel.Weather")

    def test_get_subject_context(self):
        """Test get_subject_context resource wrapper."""
        asyncio.run(self.async_test_get_subject_context())

    async def async_test_get_subject_context_error(self):
        """Test get_subject_context resource with error."""
        # Setup mock to raise exception
        self.mock_data_api.get_subject_context.side_effect = Exception("Context not found")

        # Call the resource function directly
        result = await get_subject_context_func("dataModel.NonExistent")

        # Parse JSON response
        response = json.loads(result)

        # Assertions
        self.assertIn("error", response)
        self.assertIn("Context not found", response["error"])
        self.assertEqual(response["subject"], "dataModel.NonExistent")

    def test_get_subject_context_error(self):
        """Test get_subject_context resource error handling."""
        asyncio.run(self.async_test_get_subject_context_error())


if __name__ == "__main__":
    # Set up basic logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main()
