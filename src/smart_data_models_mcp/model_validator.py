"""Schema validator for Smart Data Models.

This module provides functionality to validate data against Smart Data Model
schemas using JSON Schema validation.
"""

import json
import logging
from typing import Any, Dict, List, Optional, Tuple

import jsonschema
from jsonschema import validate, ValidationError, SchemaError

logger = logging.getLogger(__name__)


class SchemaValidator:
    """Validator for Smart Data Models using JSON Schema."""

    def __init__(self):
        self._schema_cache: Dict[str, Dict[str, Any]] = {}
        self._validator_cache: Dict[str, jsonschema.Draft7Validator] = {}

    async def validate_data(
        self,
        domain: str,
        model: str,
        data: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate data against a Smart Data Model schema.

        Args:
            domain: Domain name
            model: Model name
            data: Data to validate

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        if not isinstance(data, dict):
            return False, ["Data must be a dictionary"]

        try:
            # Get or load the schema
            schema = await self._get_schema(domain, model)
            if not schema:
                return False, ["Schema not found for model"]

            # Validate against schema
            errors = self._validate_against_schema(data, schema)
            is_valid = len(errors) == 0

            return is_valid, errors

        except Exception as e:
            logger.error(f"Validation error for {domain}/{model}: {e}")
            return False, [f"Validation error: {str(e)}"]

    async def _get_schema(self, domain: str, model: str) -> Optional[Dict[str, Any]]:
        """Get schema for a model, with caching."""
        cache_key = f"{domain}_{model}"

        # Check cache first
        if cache_key in self._schema_cache:
            return self._schema_cache[cache_key]

        # Import here to avoid circular imports
        from .data_access import SmartDataModelsAPI

        api = SmartDataModelsAPI()

        try:
            schema = await api.get_model_schema(domain=domain, model=model)
            if schema:
                self._schema_cache[cache_key] = schema
            return schema
        except Exception as e:
            logger.error(f"Failed to get schema for {domain}/{model}: {e}")
            return None

    def _validate_against_schema(self, data: Dict[str, Any], schema: Dict[str, Any]) -> List[str]:
        """Validate data against JSON schema and return error messages."""
        errors = []

        try:
            # Use Draft7Validator for better error handling
            if "properties" not in schema:
                return ["Invalid schema: missing properties"]

            validator = jsonschema.Draft7Validator(schema)

            # Collect all validation errors
            validation_errors = list(validator.iter_errors(data))

            for error in validation_errors:
                error_msg = self._format_validation_error(error)
                errors.append(error_msg)

        except SchemaError as e:
            errors.append(f"Schema validation error: {e}")
        except Exception as e:
            errors.append(f"Unexpected validation error: {e}")

        return errors

    def _format_validation_error(self, error: ValidationError) -> str:
        """Format a JSON Schema validation error into a readable message."""
        path = list(error.relative_path) if error.relative_path else []
        path_str = ".".join(str(p) for p in path) if path else "root"

        message = error.message

        # Make error messages more user-friendly
        if "is not of type" in message:
            expected_type = message.split("'")[1] if "'" in message else "unknown"
            message = f"'{path_str}' should be of type {expected_type}"
        elif "is a required property" in message:
            prop = message.split("'")[1] if "'" in message else "unknown"
            message = f"'{prop}' is required but missing"
        elif "Additional properties are not allowed" in message:
            message = f"'{path_str}' contains unexpected additional properties"
        elif "is not valid under any of the given schemas" in message:
            message = f"'{path_str}' does not match any expected format"
        elif error.validator == "enum":
            allowed = error.schema.get("enum", [])
            message = f"'{path_str}' must be one of: {', '.join(str(v) for v in allowed)}"
        elif error.validator == "minimum":
            min_val = error.schema.get("minimum")
            message = f"'{path_str}' must be >= {min_val}"
        elif error.validator == "maximum":
            max_val = error.schema.get("maximum")
            message = f"'{path_str}' must be <= {max_val}"
        elif error.validator == "pattern":
            pattern = error.schema.get("pattern", "")
            message = f"'{path_str}' does not match required pattern: {pattern}"
        elif error.validator == "minLength":
            min_len = error.schema.get("minLength")
            message = f"'{path_str}' must be at least {min_len} characters long"
        elif error.validator == "maxLength":
            max_len = error.schema.get("maxLength")
            message = f"'{path_str}' must not exceed {max_len} characters"
        elif error.validator == "minItems":
            min_items = error.schema.get("minItems")
            message = f"'{path_str}' must have at least {min_items} items"
        elif error.validator == "maxItems":
            max_items = error.schema.get("maxItems")
            message = f"'{path_str}' must not have more than {max_items} items"
        else:
            # Add path context to generic messages
            if path_str != "root":
                message = f"'{path_str}': {message}"

        return message

    async def validate_ngsi_ld_entity(
        self,
        entity: Dict[str, Any]
    ) -> Tuple[bool, List[str]]:
        """Validate an NGSI-LD entity structure.

        Args:
            entity: NGSI-LD entity to validate

        Returns:
            Tuple of (is_valid, list_of_error_messages)
        """
        errors = []

        # Check required NGSI-LD fields
        if "id" not in entity:
            errors.append("Missing required 'id' field")
        elif not isinstance(entity["id"], str):
            errors.append("'id' must be a string")
        elif not entity["id"].startswith("urn:ngsi-ld:"):
            errors.append("'id' must be a valid NGSI-LD URN")

        if "type" not in entity:
            errors.append("Missing required 'type' field")
        elif not isinstance(entity["type"], str):
            errors.append("'type' must be a string")

        # Check context
        if "@context" not in entity:
            errors.append("Missing '@context' field")
        else:
            context = entity["@context"]
            if not isinstance(context, (str, list)):
                errors.append("'@context' must be a string or array")
            elif isinstance(context, str) and not (context.startswith("http") or context.startswith("https")):
                errors.append("'@context' should be a valid URL")

        # Validate properties and relationships
        for key, value in entity.items():
            if key in ["id", "type", "@context"]:
                continue

            prop_errors = self._validate_ngsi_ld_property(key, value)
            errors.extend(prop_errors)

        return len(errors) == 0, errors

    def _validate_ngsi_ld_property(self, name: str, value: Any) -> List[str]:
        """Validate an NGSI-LD property structure."""
        errors = []

        if not isinstance(value, dict):
            return [f"'{name}' must be an NGSI-LD property object"]

        # Check for required 'type' field
        if "type" not in value:
            errors.append(f"'{name}' missing required 'type' field")
            return errors

        prop_type = value["type"]

        # Validate based on property type
        if prop_type == "Property":
            if "value" not in value:
                errors.append(f"'{name}' Property missing 'value' field")
        elif prop_type == "GeoProperty":
            if "value" not in value:
                errors.append(f"'{name}' GeoProperty missing 'value' field")
            else:
                # Basic GeoJSON validation
                geo_value = value["value"]
                if not isinstance(geo_value, dict):
                    errors.append(f"'{name}' GeoProperty value must be a GeoJSON object")
                elif "type" not in geo_value:
                    errors.append(f"'{name}' GeoProperty value missing GeoJSON 'type'")
        elif prop_type == "Relationship":
            if "object" not in value:
                errors.append(f"'{name}' Relationship missing 'object' field")
            else:
                obj = value["object"]
                if not isinstance(obj, str):
                    errors.append(f"'{name}' Relationship object must be a string")
                elif not obj.startswith("urn:ngsi-ld:"):
                    errors.append(f"'{name}' Relationship object should be a valid NGSI-LD URN")
        else:
            errors.append(f"'{name}' has unknown property type: {prop_type}")

        # Check for invalid NGSI-LD fields
        invalid_fields = set(value.keys()) - {"type", "value", "object", "observedAt", "datasetId", "instanceId", "unitCode", "accuracy"}
        if invalid_fields:
            errors.append(f"'{name}' contains invalid NGSI-LD fields: {', '.join(invalid_fields)}")

        return errors

    async def compare_data_to_model(
        self,
        domain: str,
        model: str,
        data: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Compare data structure to model and provide detailed analysis.

        Args:
            domain: Domain name
            model: Model name
            data: Data to compare

        Returns:
            Analysis results with matching info, missing fields, etc.
        """
        analysis = {
            "domain": domain,
            "model": model,
            "data_keys": list(data.keys()),
            "matching_attributes": [],
            "missing_attributes": [],
            "extra_attributes": [],
            "validation_errors": []
        }

        try:
            # Get model details
            from .data_access import SmartDataModelsAPI
            api = SmartDataModelsAPI()
            details = await api.get_model_details(domain, model)

            if "attributes" not in details:
                return analysis

            model_attrs = {attr["name"]: attr for attr in details["attributes"]}
            data_keys = set(data.keys())

            # Find matches and missing/extra fields
            for attr_name, attr_def in model_attrs.items():
                if attr_name in data_keys:
                    analysis["matching_attributes"].append({
                        "name": attr_name,
                        "expected_type": attr_def.get("type", "unknown"),
                        "data_type": type(data[attr_name]).__name__,
                        "required": attr_def.get("required", False)
                    })
                else:
                    required = attr_def.get("required", False)
                    if not details.get("required_attributes") or required:
                        analysis["missing_attributes"].append({
                            "name": attr_name,
                            "type": attr_def.get("type", "unknown"),
                            "description": attr_def.get("description", ""),
                            "required": required
                        })

            # Find extra attributes
            model_attr_names = set(model_attrs.keys())
            extra_keys = data_keys - model_attr_names
            for extra_key in extra_keys:
                analysis["extra_attributes"].append({
                    "name": extra_key,
                    "data_type": type(data[extra_key]).__name__
                })

            # Run validation to get errors
            is_valid, errors = await self.validate_data(domain, model, data)
            analysis["validation_errors"] = errors
            analysis["is_valid"] = is_valid

        except Exception as e:
            logger.error(f"Comparison error for {domain}/{model}: {e}")
            analysis["error"] = str(e)

        return analysis

    async def get_validation_statistics(self) -> Dict[str, Any]:
        """Get statistics about validation cache and performance."""
        return {
            "cached_schemas": len(self._schema_cache),
            "cached_validators": len(self._validator_cache)
        }
