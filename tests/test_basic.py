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
            self.assertIn("Energy", domains)

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


class TestIntegration(unittest.TestCase):
    """Integration tests for the complete system."""

    def test_imports(self):
        """Test that all modules can be imported successfully."""
        try:
            from smart_data_models_mcp.server import app
            from smart_data_models_mcp.data_access import SmartDataModelsAPI
            from smart_data_models_mcp.model_generator import NGSILDGenerator
            from smart_data_models_mcp.model_validator import SchemaValidator

            # Check that classes can be instantiated
            api = SmartDataModelsAPI()
            generator = NGSILDGenerator()
            validator = SchemaValidator()

            # Basic type checks
            self.assertIsInstance(api, SmartDataModelsAPI)
            self.assertIsInstance(generator, NGSILDGenerator)
            self.assertIsInstance(validator, SchemaValidator)

        except ImportError as e:
            self.fail(f"Import failed: {e}")


if __name__ == "__main__":
    # Set up basic logging for tests
    import logging
    logging.basicConfig(level=logging.INFO)

    # Run tests
    unittest.main()
