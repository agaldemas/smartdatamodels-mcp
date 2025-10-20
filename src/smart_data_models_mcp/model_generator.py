"""NGSI-LD model generator.

This module provides functionality to generate NGSI-LD compliant entities
from arbitrary JSON data, optionally using Smart Data Models as templates.
"""

import json
import logging
import uuid
from datetime import datetime, timezone
from typing import Any, Dict, Optional, Union

logger = logging.getLogger(__name__)


class NGSILDGenerator:
    """Generator for NGSI-LD compliant entities."""

    def __init__(self):
        # Standard NGSI-LD context
        self.default_context = "https://uri.etsi.org/ngsi-ld/v1.7/gsma-cim/common.jsonld"

        # Known data types and their NGSI-LD property types
        self.type_mappings = {
            "string": "Property",
            "integer": "Property",
            "number": "Property",
            "boolean": "Property",
            "array": "Property",
            "object": "Property",
            "null": "Property"
        }

    async def generate_ngsi_ld(
        self,
        data: Dict[str, Any],
        entity_type: Optional[str] = None,
        entity_id: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate an NGSI-LD entity from arbitrary JSON data.

        Args:
            data: Input JSON data
            entity_type: Optional entity type (will try to infer from data)
            entity_id: Optional entity ID (will generate if not provided)
            context: Optional JSON-LD context URL

        Returns:
            NGSI-LD compliant entity
        """
        if not isinstance(data, dict):
            raise ValueError("Input data must be a dictionary")

        # Infer entity type if not provided
        if not entity_type:
            entity_type = self._infer_entity_type(data)

        # Generate entity ID if not provided
        if not entity_id:
            entity_id = self._generate_entity_id(entity_type)

        # Create the basic entity structure
        entity = {
            "id": entity_id,
            "type": entity_type,
        }

        # Add context if provided
        if context:
            entity["@context"] = context
        else:
            entity["@context"] = self.default_context

        # Convert each data field to NGSI-LD properties
        for key, value in data.items():
            if key in ["id", "type", "@context"]:
                # Skip NGSI-LD reserved fields
                continue

            try:
                property_def = self._convert_to_ngsi_property(key, value)
                if property_def:
                    entity[key] = property_def
            except Exception as e:
                logger.warning(f"Failed to convert property '{key}': {e}")
                # Add as basic property
                entity[key] = {"value": value}

        return entity

    def _infer_entity_type(self, data: Dict[str, Any]) -> str:
        """Infer entity type from data structure."""
        # Look for type hints in the data
        type_hints = ["type", "entityType", "entity_type", "model", "class"]

        for hint in type_hints:
            if hint in data and isinstance(data[hint], str):
                return data[hint].strip()

        # Look for common patterns in keys to infer type
        keys = set(data.keys())

        # Sensor patterns
        if any(k.lower() in ["value", "timestamp", "observation", "measurement"] for k in keys):
            return "Sensor"

        # Location patterns
        if any(k.lower() in ["location", "latitude", "longitude", "coordinates", "address"] for k in keys):
            return "Location"

        # Building patterns
        if any(k.lower() in ["building", "floor", "room"] for k in keys):
            return "Building"

        # Device patterns
        if any(k.lower() in ["device", "category", "status"] for k in keys):
            return "Device"

        # Vehicle patterns
        if any(k.lower() in ["vehicle", "license_plate", "model", "brand"] for k in keys):
            return "Vehicle"

        # Weather patterns
        if any(k.lower() in ["temperature", "humidity", "pressure", "wind"] for k in keys):
            return "WeatherObservation"

        # Default fallback
        return "Entity"

    def _generate_entity_id(self, entity_type: str) -> str:
        """Generate a unique NGSI-LD entity ID."""
        unique_id = str(uuid.uuid4())[:8]  # Use first 8 chars for brevity
        return f"urn:ngsi-ld:{entity_type}:{unique_id}"

    def _convert_to_ngsi_property(self, key: str, value: Any) -> Optional[Dict[str, Any]]:
        """Convert a data field to an NGSI-LD property.

        Args:
            key: Property name
            value: Property value

        Returns:
            NGSI-LD property definition or None if conversion fails
        """
        # Handle null values
        if value is None:
            return None

        # Detect property type and structure
        property_type = self._determine_property_type(value)

        # Handle special cases
        if key.lower() in ["location", "coordinates"]:
            return self._create_location_property(value)
        elif key.lower() in ["datecreated", "datemodified", "timestamp", "time"]:
            return self._create_datetime_property(value)
        elif key.lower().endswith("_relationship") or self._looks_like_relationship(value):
            return self._create_relationship_property(value)
        else:
            return self._create_value_property(value, property_type)

    def _determine_property_type(self, value: Any) -> str:
        """Determine the NGSI-LD property type from Python data type."""
        if isinstance(value, str):
            return "string"
        elif isinstance(value, bool):
            return "boolean"
        elif isinstance(value, int):
            return "integer"
        elif isinstance(value, float):
            return "number"
        elif isinstance(value, list):
            return "array"
        elif isinstance(value, dict):
            return "object"
        else:
            return "string"  # Default fallback

    def _create_value_property(self, value: Any, data_type: str) -> Dict[str, Any]:
        """Create a basic value property."""
        # Determine if this should be a GeoProperty
        if self._is_geographic_value(value):
            return self._create_geo_property(value)

        # Standard Property
        return {
            "type": "Property",
            "value": value
        }

    def _create_location_property(self, value: Any) -> Dict[str, Any]:
        """Create a Location property (GeoProperty)."""
        if isinstance(value, str):
            # Try to parse various location formats
            try:
                if value.startswith("POINT(") or value.startswith("POLYGON"):
                    return {
                        "type": "GeoProperty",
                        "value": self._parse_wkt(value)
                    }
                else:
                    coords = self._parse_coordinate_string(value)
                    return {
                        "type": "GeoProperty",
                        "value": {
                            "type": "Point",
                            "coordinates": coords
                        }
                    }
            except Exception:
                pass
        elif isinstance(value, dict):
            # Already a GeoJSON-like structure
            if "coordinates" in value or "type" in value:
                return {
                    "type": "GeoProperty",
                    "value": value
                }
        elif isinstance(value, list) and len(value) >= 2:
            # Coordinate array [longitude, latitude]
            return {
                "type": "GeoProperty",
                "value": {
                    "type": "Point",
                    "coordinates": value
                }
            }

        # Fallback to standard property
        return {
            "type": "Property",
            "value": value
        }

    def _create_geo_property(self, value: Any) -> Dict[str, Any]:
        """Create a GeoProperty from geographic data."""
        # Check if value looks like coordinates
        if isinstance(value, list) and len(value) >= 2:
            try:
                coords = [float(value[0]), float(value[1])]  # [longitude, latitude]
                return {
                    "type": "GeoProperty",
                    "value": {
                        "type": "Point",
                        "coordinates": coords
                    }
                }
            except (ValueError, IndexError):
                pass

        elif isinstance(value, dict):
            # Check for GeoJSON structure
            if "type" in value and value["type"] in ["Point", "Polygon", "LineString"]:
                return {
                    "type": "GeoProperty",
                    "value": value
                }

        # Not geographic, return as regular property
        return {
            "type": "Property",
            "value": value
        }

    def _create_datetime_property(self, value: Any) -> Dict[str, Any]:
        """Create a datetime/timestamp property."""
        # Try to parse various datetime formats
        if isinstance(value, str):
            try:
                # Try ISO format parsing
                if "T" in value or ("-" in value and ":" in value):
                    # Assume it's already in ISO format or parseable
                    return {
                        "type": "Property",
                        "value": value
                    }
            except Exception:
                pass
        elif isinstance(value, (int, float)):
            # Unix timestamp
            try:
                dt = datetime.fromtimestamp(value, tz=timezone.utc)
                return {
                    "type": "Property",
                    "value": dt.isoformat()
                }
            except Exception:
                pass

        # Fallback
        return {
            "type": "Property",
            "value": value
        }

    def _create_relationship_property(self, value: Any) -> Dict[str, Any]:
        """Create a Relationship property."""
        if isinstance(value, str):
            # Check if it's already a URN
            if value.startswith("urn:"):
                return {
                    "type": "Relationship",
                    "object": value
                }
            else:
                # Try to construct a URN from the string
                return {
                    "type": "Relationship",
                    "object": f"urn:ngsi-ld:Entity:{value}"
                }
        elif isinstance(value, dict):
            # Check for relationship object
            if "object" in value:
                return {
                    "type": "Relationship",
                    **value
                }

        # Not a clear relationship, return as property
        return {
            "type": "Property",
            "value": value
        }

    def _is_geographic_value(self, value: Any) -> bool:
        """Check if a value represents geographic data."""
        if isinstance(value, list):
            # Check for coordinate arrays [lon, lat] or [lon, lat, alt]
            if len(value) >= 2 and len(value) <= 3:
                try:
                    coords = [float(x) for x in value]
                    # Basic bounds checking for longitude/latitude
                    if len(coords) >= 2 and -180 <= coords[0] <= 180 and -90 <= coords[1] <= 90:
                        return True
                except (ValueError, TypeError):
                    pass
        elif isinstance(value, dict):
            # Check for GeoJSON structures
            if "type" in value and value["type"] in ["Point", "Polygon", "LineString", "MultiPoint", "MultiPolygon"]:
                return True
            if "coordinates" in value:
                return True

        return False

    def _parse_coordinate_string(self, coord_str: str) -> list:
        """Parse coordinate string formats."""
        # Remove brackets and split by comma
        coord_str = coord_str.strip("[]()")
        parts = [p.strip() for p in coord_str.split(",")]

        try:
            coords = [float(p) for p in parts[:2]]  # Take first two as lon, lat
            return coords
        except ValueError:
            raise ValueError(f"Cannot parse coordinates from: {coord_str}")

    def _parse_wkt(self, wkt: str) -> dict:
        """Parse Well-Known Text geometry (simplified)."""
        # Simple WKT parsing - POINT(lon lat) format
        if wkt.upper().startswith("POINT"):
            coords_str = wkt[5:].strip("()")
            parts = coords_str.split()
            try:
                coords = [float(p) for p in parts[:2]]
                return {
                    "type": "Point",
                    "coordinates": coords
                }
            except (ValueError, IndexError):
                pass

        # Return as-is for more complex WKT or unsupported formats
        return {"type": "Point", "coordinates": [0, 0]}  # Default fallback

    def _looks_like_relationship(self, value: Any) -> bool:
        """Check if a value looks like a relationship reference."""
        if isinstance(value, str):
            # Check for URN patterns
            if value.startswith("urn:ngsi-ld:") or value.startswith("urn:"):
                return True
            # Check for entity reference patterns
            if "://" in value or value.replace("-", "").replace("_", "").isalnum():
                return True
        elif isinstance(value, dict):
            # Check for relationship object structure
            return "object" in value or "Relationship" in str(value)

        return False

    async def generate_from_template(
        self,
        template_domain: str,
        template_model: str,
        data: Dict[str, Any],
        entity_id: Optional[str] = None,
        context: Optional[str] = None
    ) -> Dict[str, Any]:
        """Generate NGSI-LD entity using a Smart Data Model as template.

        Args:
            template_domain: Domain of the template model
            template_model: Name of the template model
            data: Data to populate the template
            entity_id: Optional entity ID
            context: Optional context URL

        Returns:
            NGSI-LD entity based on the template
        """
        # For now, implement basic template following
        # This could be extended to fully validate against schemas

        entity_data = data.copy()

        # Use template model as entity type
        entity_type = template_model

        # Generate entity
        entity = await self.generate_ngsi_ld(
            entity_data,
            entity_type=entity_type,
            entity_id=entity_id,
            context=context
        )

        # Mark that this was generated from a template
        entity["_generated_from_template"] = f"{template_domain}/{template_model}"

        return entity
