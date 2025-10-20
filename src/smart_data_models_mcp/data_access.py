"""Data access layer for Smart Data Models.

This module provides access to FIWARE Smart Data Models through the pysmartdatamodels
package and GitHub API, with caching and error handling.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests

try:
    import pysmartdatamodels as psdm
    PYSMARTDATAMODELS_AVAILABLE = True
except ImportError:
    psdm = None
    PYSMARTDATAMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


class Cache:
    """Simple in-memory cache with TTL."""

    def __init__(self, ttl_seconds: int = 3600):  # 1 hour default
        self._cache: Dict[str, Tuple[Any, float]] = {}
        self._ttl = ttl_seconds

    def get(self, key: str) -> Optional[Any]:
        """Get cached value if not expired."""
        if key in self._cache:
            value, timestamp = self._cache[key]
            if time.time() - timestamp < self._ttl:
                return value
            else:
                del self._cache[key]
        return None

    def set(self, key: str, value: Any) -> None:
        """Set cached value with current timestamp."""
        self._cache[key] = (value, time.time())


class SmartDataModelsAPI:
    """API wrapper for accessing Smart Data Models data."""

    # Known domains from the specification
    KNOWN_DOMAINS = [
        "SmartCities", "Agrifood", "Water", "Energy", "Logistics",
        "Robotics", "Sensoring", "Cross sector", "Health", "Destination",
        "Environment", "Aeronautics", "Manufacturing", "Incubated", "Harmonization"
    ]

    GITHUB_API_BASE = "https://api.github.com"
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com"
    SMART_DATA_MODELS_REPO = "smart-data-models"

    def __init__(self):
        self._cache = Cache(ttl_seconds=1800)  # 30 minutes
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": "smart-data-models-mcp/0.1.0"
        })

    async def _run_sync_in_thread(self, func, *args, **kwargs):
        """Run synchronous function in thread pool."""
        loop = asyncio.get_event_loop()
        return await loop.run_in_executor(None, func, *args, **kwargs)

    async def list_domains(self) -> List[str]:
        """List all available domains."""
        cache_key = "domains"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Get from pysmartdatamodels
        try:
            domains = await self._run_sync_in_thread(smd.list_all_domains)

            # Filter out any domains not in our known list for consistency
            filtered_domains = [d for d in domains if d in self.KNOWN_DOMAINS]
            if not filtered_domains:
                # Fallback to known domains if pymartdatamodels returns empty/unknown
                filtered_domains = self.KNOWN_DOMAINS

            self._cache.set(cache_key, filtered_domains)
            return filtered_domains

        except Exception as e:
            logger.warning(f"pysmartdatamodels domain listing failed: {e}")
            # Return known domains as fallback
            domains = self.KNOWN_DOMAINS
            self._cache.set(cache_key, domains)
            return domains

    async def list_models_in_domain(self, domain: str, limit: int = 50) -> List[str]:
        """List all models in a specific domain."""
        cache_key = f"domain_models_{domain}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached[:limit]

        try:
            # Use pysmartdatamodels to get domain data
            domain_data = await self._run_sync_in_thread(smd.load_all_datamodels_from_domain, domain)

            if domain_data and isinstance(domain_data, dict):
                models = list(domain_data.keys())[:limit]

                # Also try to get from GitHub API for more complete list
                try:
                    github_models = await self._get_models_from_github_api(domain)
                    if github_models:
                        models = list(set(models + github_models))[:limit]
                except Exception as e:
                    logger.debug(f"GitHub API fallback failed for {domain}: {e}")

                self._cache.set(cache_key, models)
                return models
            else:
                # Fallback to GitHub API
                models = await self._get_models_from_github_api(domain)
                if models:
                    models = models[:limit]
                    self._cache.set(cache_key, models)
                    return models

        except Exception as e:
            logger.warning(f"Failed to get models for domain {domain}: {e}")
            # Fallback to GitHub API
            try:
                models = await self._get_models_from_github_api(domain)
                if models:
                    models = models[:limit]
                    self._cache.set(cache_key, models)
                    return models
            except Exception as e2:
                logger.error(f"GitHub API fallback also failed for {domain}: {e2}")

        # Ultimate fallback: empty list
        return []

    async def _get_models_from_github_api(self, domain: str) -> Optional[List[str]]:
        """Get models from GitHub API repository contents."""
        try:
            # Check if domain repo exists
            domain_url = f"{self.GITHUB_API_BASE}/repos/{self.SMART_DATA_MODELS_REPO}/{domain}/contents"
            response = await self._run_sync_in_thread(
                self._session.get, domain_url, timeout=30
            )

            if response.status_code == 200:
                contents = response.json()
                models = []
                for item in contents:
                    if item.get("type") == "dir" and not item["name"].startswith("."):
                        models.append(item["name"])
                return models

        except Exception as e:
            logger.debug(f"GitHub API call failed for {domain}: {e}")

        return None

    async def search_models(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 20,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for models across domains."""
        cache_key = f"search_{query}_{domain}_{limit}_{include_attributes}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            # Use pysmartdatamodels search if available
            if hasattr(smd, 'find_datamodels'):
                search_results = await self._run_sync_in_thread(smd.find_datamodels, query, domain)

                if search_results:
                    # Convert to our format
                    results = []
                    for result in search_results[:limit]:
                        model_info = {
                            "domain": result.get("domain", "Unknown"),
                            "model": result.get("model", "Unknown"),
                            "name": result.get("name", ""),
                            "description": result.get("description", "")
                        }

                        if include_attributes and "attributes" in result:
                            model_info["attributes"] = result["attributes"]

                        results.append(model_info)

                    self._cache.set(cache_key, results)
                    return results

        except Exception as e:
            logger.debug(f"pysmartdatamodels search failed: {e}")

        # Fallback: simple text search across domains
        results = await self._simple_text_search(query, domain, limit, include_attributes)

        self._cache.set(cache_key, results)
        return results

    async def _simple_text_search(
        self,
        query: str,
        domain: Optional[str] = None,
        limit: int = 20,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Perform simple text search across models."""
        results = []
        query_lower = query.lower()

        domains_to_search = [domain] if domain else await self.list_domains()

        for domain_name in domains_to_search:
            try:
                models = await self.list_models_in_domain(domain_name, limit=100)
                for model_name in models:
                    try:
                        details = await self.get_model_details(domain_name, model_name)

                        # Check if query matches model name, description, or attributes
                        name_match = query_lower in model_name.lower()
                        desc_match = query_lower in details.get("description", "").lower()

                        attr_match = False
                        if include_attributes and "attributes" in details:
                            for attr in details["attributes"]:
                                if query_lower in attr.get("name", "").lower() or \
                                   query_lower in attr.get("description", "").lower():
                                    attr_match = True
                                    break

                        if name_match or desc_match or attr_match:
                            model_info = {
                                "domain": domain_name,
                                "model": model_name,
                                "name": model_name,
                                "description": details.get("description", "")
                            }

                            if include_attributes and "attributes" in details:
                                model_info["attributes"] = details["attributes"]

                            results.append(model_info)

                            if len(results) >= limit:
                                return results

                    except Exception as e:
                        logger.debug(f"Error checking model {domain_name}/{model_name}: {e}")
                        continue

            except Exception as e:
                logger.debug(f"Error searching domain {domain_name}: {e}")
                continue

        return results

    async def get_model_details(self, domain: str, model: str) -> Dict[str, Any]:
        """Get detailed information about a specific model."""
        cache_key = f"model_details_{domain}_{model}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            # Try pysmartdatamodels first
            details = await self._run_sync_in_thread(smd.description_attribute_metadata, domain, model)

            if details and isinstance(details, dict):
                processed_details = {
                    "domain": domain,
                    "model": model,
                    "description": details.get("description", ""),
                    "attributes": details.get("attributes", []),
                    "required_attributes": details.get("required", []),
                    "source": "pysmartdatamodels"
                }

                # Add more metadata if available
                if "author" in details:
                    processed_details["author"] = details["author"]
                if "license" in details:
                    processed_details["license"] = details["license"]

                self._cache.set(cache_key, processed_details)
                return processed_details

        except Exception as e:
            logger.debug(f"pysmartdatamodels details failed for {domain}/{model}: {e}")

        # Fallback: construct basic details from GitHub
        try:
            details = await self._get_basic_model_details_from_github(domain, model)
            if details:
                self._cache.set(cache_key, details)
                return details
        except Exception as e:
            logger.debug(f"GitHub details fallback failed for {domain}/{model}: {e}")

        # Ultimate fallback
        details = {
            "domain": domain,
            "model": model,
            "description": f"Smart Data Model for {model} in {domain} domain",
            "attributes": [],
            "source": "fallback"
        }

        self._cache.set(cache_key, details)
        return details

    async def _get_basic_model_details_from_github(self, domain: str, model: str) -> Optional[Dict[str, Any]]:
        """Get basic model details from GitHub repository."""
        try:
            # Try to get README for description
            readme_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_REPO}/{domain}/{model}/README.md"
            response = await self._run_sync_in_thread(
                self._session.get, readme_url, timeout=30
            )

            description = ""
            if response.status_code == 200:
                readme_content = response.text
                # Extract first paragraph as description
                lines = readme_content.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#") and len(line) > 10:
                        description = line
                        break

            # Try to get schema for attributes
            schema_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_REPO}/{domain}/{model}/schema.json"
            response = await self._run_sync_in_thread(
                self._session.get, schema_url, timeout=30
            )

            attributes = []
            if response.status_code == 200:
                try:
                    schema = response.json()

                    # Extract properties from schema
                    properties = schema.get("properties", {})
                    required = schema.get("required", [])

                    for prop_name, prop_def in properties.items():
                        if isinstance(prop_def, dict):
                            attr = {
                                "name": prop_name,
                                "type": prop_def.get("type", "string"),
                                "description": prop_def.get("description", ""),
                                "required": prop_name in required
                            }
                            attributes.append(attr)
                except json.JSONDecodeError:
                    pass

            return {
                "domain": domain,
                "model": model,
                "description": description or f"Smart Data Model for {model}",
                "attributes": attributes,
                "source": "github"
            }

        except Exception as e:
            logger.debug(f"GitHub API call failed for {domain}/{model}: {e}")
            return None

    async def get_model_schema(self, domain: str, model: str) -> Dict[str, Any]:
        """Get the JSON schema for a model."""
        cache_key = f"schema_{domain}_{model}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Get from GitHub
        schema_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_REPO}/{domain}/{model}/schema.json"
        try:
            response = await self._run_sync_in_thread(
                self._session.get, schema_url, timeout=30
            )

            if response.status_code == 200:
                schema = response.json()
                self._cache.set(cache_key, schema)
                return schema
            else:
                raise ValueError(f"Schema not found: HTTP {response.status_code}")

        except Exception as e:
            raise ValueError(f"Failed to fetch schema: {e}")

    async def get_model_examples(self, domain: str, model: str) -> List[Dict[str, Any]]:
        """Get example instances for a model."""
        cache_key = f"examples_{domain}_{model}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            # Try pysmartdatamodels first
            examples = await self._run_sync_in_thread(smd.ngsi_ld_example_generator, domain, model)

            if examples:
                if isinstance(examples, dict):
                    examples = [examples]
                elif isinstance(examples, list):
                    examples = examples

                self._cache.set(cache_key, examples)
                return examples

        except Exception as e:
            logger.debug(f"pysmartdatamodels examples failed for {domain}/{model}: {e}")

        # Fallback: try to get from GitHub examples
        try:
            examples = await self._get_examples_from_github(domain, model)
            if examples:
                self._cache.set(cache_key, examples)
                return examples
        except Exception as e:
            logger.debug(f"GitHub examples failed for {domain}/{model}: {e}")

        # Ultimate fallback: generate basic example
        example = await self._generate_basic_example(domain, model)
        examples = [example] if example else []

        self._cache.set(cache_key, examples)
        return examples

    async def _get_examples_from_github(self, domain: str, model: str) -> Optional[List[Dict[str, Any]]]:
        """Get examples from GitHub repository."""
        try:
            # Try various common example file names
            example_paths = [
                "example.json",
                "examples/example.json",
                "example.jsonld",
                "examples/example.jsonld"
            ]

            for path in example_paths:
                example_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_REPO}/{domain}/{model}/{path}"
                try:
                    response = await self._run_sync_in_thread(
                        self._session.get, example_url, timeout=30
                    )

                    if response.status_code == 200:
                        try:
                            example = response.json()
                            return [example] if isinstance(example, dict) else example
                        except json.JSONDecodeError:
                            continue
                except Exception:
                    continue

        except Exception as e:
            logger.debug(f"GitHub examples fetch failed for {domain}/{model}: {e}")

        return None

    async def _generate_basic_example(self, domain: str, model: str) -> Optional[Dict[str, Any]]:
        """Generate a basic example for a model."""
        try:
            details = await self.get_model_details(domain, model)

            if not details.get("attributes"):
                return None

            # Generate basic values for required attributes
            example = {
                "id": f"urn:ngsi-ld:{model}:001",
                "type": model
            }

            for attr in details["attributes"]:
                if attr.get("required", False) or not details.get("required_attributes"):
                    attr_name = attr["name"]
                    attr_type = attr.get("type", "string")

                    if attr_type == "string":
                        example[attr_name] = {"value": f"Sample {attr_name}"}
                    elif attr_type == "number" or attr_type == "integer":
                        example[attr_name] = {"value": 1.0 if attr_type == "number" else 1}
                    elif attr_type == "boolean":
                        example[attr_name] = {"value": True}
                    elif attr_type == "array":
                        example[attr_name] = {"value": []}
                    elif attr_type == "object":
                        example[attr_name] = {"value": {}}

            return example

        except Exception as e:
            logger.debug(f"Failed to generate basic example for {domain}/{model}: {e}")
            return None

    async def get_domain_context(self, domain: str) -> Dict[str, Any]:
        """Get the JSON-LD context for a domain."""
        cache_key = f"context_{domain}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Try to get from GitHub
        context_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_REPO}/{domain}/context.jsonld"
        try:
            response = await self._run_sync_in_thread(
                self._session.get, context_url, timeout=30
            )

            if response.status_code == 200:
                context = response.json()
                self._cache.set(cache_key, context)
                return context
            else:
                raise ValueError(f"Context not found: HTTP {response.status_code}")

        except Exception as e:
            # Generate a basic context
            context = self._generate_basic_context(domain)
            self._cache.set(cache_key, context)
            return context

    def _generate_basic_context(self, domain: str) -> Dict[str, Any]:
        """Generate a basic JSON-LD context for a domain."""
        return {
            "@context": {
                "GeoProperty": "https://uri.etsi.org/ngsi-ld/v1.7/commonTerms#GeoProperty",
                "Location": "https://uri.etsi.org/ngsi-ld/v1.7/commonTerms#Location",
                "Property": "https://uri.etsi.org/ngsi-ld/v1.7/commonTerms#Property",
                "Relationship": "https://uri.etsi.org/ngsi-ld/v1.7/commonTerms#Relationship",
                "dateCreated": "https://uri.etsi.org/ngsi-ld/v1.7/commonTerms#createdAt",
                "dateModified": "https://uri.etsi.org/ngsi-ld/v1.7/commonTerms#modifiedAt",
                "id": "@id",
                "type": "@type",
                "value": {
                    "@id": "https://uri.etsi.org/ngsi-ld/v1.7/commonTerms#hasValue",
                    "@type": "@json"
                }
            }
        }

    async def suggest_matching_models(self, data: Dict[str, Any], top_k: int = 5) -> List[Dict[str, Any]]:
        """Suggest models that match the provided data structure."""
        if not isinstance(data, dict):
            return []

        data_keys = set(data.keys())

        # Score models based on attribute overlap
        candidates = []

        domains = await self.list_domains()

        for domain in domains:
            try:
                models = await self.list_models_in_domain(domain, limit=100)

                for model_name in models[:50]:  # Limit to avoid too many requests
                    try:
                        details = await self.get_model_details(domain, model_name)

                        if "attributes" in details:
                            model_attrs = {attr["name"] for attr in details["attributes"]}
                            overlap = len(data_keys.intersection(model_attrs))
                            total_attrs = len(model_attrs.union(data_keys))

                            if total_attrs > 0:
                                similarity = overlap / total_attrs
                                if similarity > 0.1:  # Only include reasonable matches
                                    candidates.append({
                                        "domain": domain,
                                        "model": model_name,
                                        "similarity": round(similarity, 3),
                                        "matched_attributes": overlap,
                                        "total_attributes": len(model_attrs),
                                        "description": details.get("description", "")
                                    })

                    except Exception as e:
                        logger.debug(f"Error checking model {domain}/{model_name}: {e}")
                        continue

            except Exception as e:
                logger.debug(f"Error in domain {domain}: {e}")
                continue

        # Sort by similarity and return top k
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]
