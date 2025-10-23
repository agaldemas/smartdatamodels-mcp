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

    # Known subjects from the specification
    KNOWN_SUBJECTS = [
        "SmartCities", "Agrifood", "Water", "Energy", "Logistics",
        "Robotics", "Sensoring", "Cross sector", "Health", "Destination",
        "Environment", "Aeronautics", "Manufacturing", "Incubated", "Harmonization"
    ]
    GITHUB_API_BASE = "https://api.github.com/repos"
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com"
    SMART_DATA_MODELS_ORG = "smart-data-models" # Organization name

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

    async def list_subjects(self) -> List[str]:
        """List all available subjects.

        Args:
            None

        Returns:
            List[str]: A list of available Smart Data Model subjects.
        """
        cache_key = "subjects"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Get from pysmartdatamodels
        try:
            subjects = await self._run_sync_in_thread(psdm.list_all_subjects)
            # If pysmartdatamodels returns subjects, use them. Otherwise, fallback to KNOWN_SUBJECTS.
            if subjects:
                self._cache.set(cache_key, subjects)
                return subjects
            else:
                logger.warning("pysmartdatamodels returned no subjects. Falling back to KNOWN_SUBJECTS.")
                self._cache.set(cache_key, self.KNOWN_SUBJECTS)
                return self.KNOWN_SUBJECTS

        except Exception as e:
            logger.warning(f"pysmartdatamodels subject listing failed: {e}. Falling back to KNOWN_SUBJECTS.")
            self._cache.set(cache_key, self.KNOWN_SUBJECTS)
            return self.KNOWN_SUBJECTS

    async def list_models_in_subject(self, subject: str, limit: int = 50) -> List[str]:
        """List all models in a specific subject.

        Args:
            subject (str): The name of the subject to list models from.
            limit (int): The maximum number of models to return (default: 50).

        Returns:
            List[str]: A list of model names within the specified subject.
        """
        cache_key = f"subject_models_{subject}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached[:limit]

        try:
            # Use pysmartdatamodels to get subject data
            subject_data = await self._run_sync_in_thread(psdm.datamodels_subject, subject)

            if subject_data and isinstance(subject_data, list):
                models = subject_data[:limit]

                # Also try to get from GitHub API for more complete list
                try:
                    github_models = await self._get_models_from_github_api(subject)
                    if github_models:
                        models = list(set(models + github_models))[:limit]
                except Exception as e:
                    logger.debug(f"GitHub API fallback failed for {subject}: {e}")

                self._cache.set(cache_key, models)
                return models
            else:
                # Fallback to GitHub API
                models = await self._get_models_from_github_api(subject)
                if models:
                    models = models[:limit]
                    self._cache.set(cache_key, models)
                    return models

        except Exception as e:
            logger.warning(f"Failed to get models for subject {subject}: {e}")
            # Fallback to GitHub API
            try:
                models = await self._get_models_from_github_api(subject)
                if models:
                    models = models[:limit]
                    self._cache.set(cache_key, models)
                    return models
            except Exception as e2:
                logger.error(f"GitHub API fallback also failed for {subject}: {e2}")

        # Ultimate fallback: empty list
        return []

    async def _get_models_from_github_api(self, subject: str) -> Optional[List[str]]:
        """Get models from GitHub API repository contents."""
        try:
            repo_name = f"dataModel.{subject}"
            subject_url = f"{self.GITHUB_API_BASE}/{self.SMART_DATA_MODELS_ORG}/{repo_name}/contents"
            response = await self._run_sync_in_thread(
                self._session.get, subject_url, timeout=30
            )

            if response.status_code == 200:
                contents = response.json()
                models = []
                for item in contents:
                    if item.get("type") == "dir" and not item["name"].startswith("."):
                        models.append(item["name"])
                return models

        except Exception as e:
            logger.debug(f"GitHub API call failed for {subject}: {e}")

        return None

    async def search_models(
        self,
        query: str,
        subject: Optional[str] = None,
        limit: int = 20,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for models across subjects.

        Args:
            query (str): The search query (model name, attributes, or keywords).
            subject (Optional[str]): Limits the search to a specific subject.
            limit (int): The maximum number of results to return (default: 20).
            include_attributes (bool): Whether to include attribute details in the results (default: False).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a matching data model.
        """
        cache_key = f"search_{query}_{subject}_{limit}_{include_attributes}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            # Use pysmartdatamodels search if available
            if hasattr(psdm, 'find_datamodels'):
                search_results = await self._run_sync_in_thread(psdm.find_datamodels, query, subject)

                if search_results:
                    # Convert to our format
                    results = []
                    for result in search_results[:limit]:
                        model_info = {
                            "subject": result.get("subject", "Unknown"),
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

        # Fallback 1: Direct model name match across all subjects
        direct_match_results = await self._direct_model_name_search(query, subject, include_attributes)
        if direct_match_results:
            self._cache.set(cache_key, direct_match_results)
            return direct_match_results[:limit]

        # Fallback 2: simple text search across subjects
        results = await self._simple_text_search(query, subject, limit, include_attributes)

        self._cache.set(cache_key, results)
        return results

    async def _direct_model_name_search(
        self,
        query: str,
        subject: Optional[str] = None,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Attempt to find a model directly by its name across subjects."""
        results = []
        subjects_to_search = [subject] if subject else await self.list_subjects()

        for subject_name in subjects_to_search:
            try:
                details = await self.get_model_details(subject_name, query)
                if details:
                    model_info = {
                        "subject": subject_name,
                        "model": query,
                        "name": query,
                        "description": details.get("description", "")
                    }
                    if include_attributes and "attributes" in details:
                        model_info["attributes"] = details["attributes"]
                    results.append(model_info)
                    # If we find a direct match, we can stop searching
                    return results
            except Exception as e:
                logger.debug(f"Direct model name search failed for {subject_name}/{query}: {e}")
                continue
        return []

    async def _simple_text_search(
        self,
        query: str,
        subject: Optional[str] = None,
        limit: int = 20,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Perform simple text search across models."""
        results = []
        query_lower = query.lower()

        subjects_to_search = [subject] if subject else await self.list_subjects()

        for subject_name in subjects_to_search:
            try:
                models = await self.list_models_in_subject(subject_name, limit=100)
                for model_name in models:
                    try:
                        details = await self.get_model_details(subject_name, model_name)

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
                                "subject": subject_name,
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
                        logger.debug(f"Error checking model {subject_name}/{model_name}: {e}")
                        continue

            except Exception as e:
                logger.debug(f"Error searching subject {subject_name}: {e}")
                continue

        return results

    async def get_model_details(self, subject: str, model: str) -> Dict[str, Any]:
        """Get detailed information about a specific model.

        Args:
            subject (str): The name of the subject the model belongs to.
            model (str): The name of the model to retrieve details for.

        Returns:
            Dict[str, Any]: A dictionary containing detailed information about the model,
                            including its description, attributes, and source.
        """
        cache_key = f"model_details_{subject}_{model}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        details = None
        # Attempt 1: Try pysmartdatamodels with the provided subject
        try:
            details = await self._run_sync_in_thread(psdm.description_attribute_metadata, subject, model)
            if details and isinstance(details, dict):
                processed_details = {
                    "subject": subject,
                    "model": model,
                    "description": details.get("description", ""),
                    "attributes": details.get("attributes", []),
                    "required_attributes": details.get("required", []),
                    "source": "pysmartdatamodels"
                }
                if "author" in details:
                    processed_details["author"] = details["author"]
                if "license" in details:
                    processed_details["license"] = details["license"]
                self._cache.set(cache_key, processed_details)
                return processed_details
        except Exception as e:
            logger.debug(f"pysmartdatamodels details failed for {subject}/{model}: {e}")

        # Attempt 2: Fallback to GitHub API with the provided subject
        if not details:
            try:
                details = await self._get_basic_model_details_from_github(subject, model, repo_subject=subject)
                if details:
                    self._cache.set(cache_key, details)
                    return details
            except Exception as e:
                logger.debug(f"GitHub details fallback failed for {subject}/{model}: {e}")

        # Attempt 3: Infer subject from model name (e.g., WaterQualityObserved -> WaterQuality)
        if not details and "WaterQuality" in model:
            inferred_subject = "WaterQuality"
            logger.info(f"Attempting to infer subject for {model}: {inferred_subject}")
            try:
                details = await self._get_basic_model_details_from_github(subject, model, repo_subject=inferred_subject)
                if details:
                    details["subject"] = inferred_subject # Update subject to the inferred one
                    self._cache.set(cache_key, details)
                    return details
            except Exception as e:
                logger.debug(f"GitHub details fallback with inferred subject {inferred_subject}/{model} failed: {e}")
        elif not details and "Quality" in model: # General heuristic for other 'Quality' models
            inferred_subject = model.split('Quality')[0] + 'Quality'
            logger.info(f"Attempting to infer subject for {model}: {inferred_subject}")
            try:
                details = await self._get_basic_model_details_from_github(subject, model, repo_subject=inferred_subject)
                if details:
                    details["subject"] = inferred_subject # Update subject to the inferred one
                    self._cache.set(cache_key, details)
                    return details
            except Exception as e:
                logger.debug(f"GitHub details fallback with inferred subject {inferred_subject}/{model} failed: {e}")


        # Ultimate fallback
        details = {
            "subject": subject,
            "model": model,
            "description": f"Smart Data Model for {model} in {subject} subject",
            "attributes": [],
            "source": "fallback"
        }

        self._cache.set(cache_key, details)
        return details

    async def _get_basic_model_details_from_github(self, subject: str, model: str, repo_subject: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get basic model details from GitHub repository.
        repo_subject allows overriding the subject used for constructing the GitHub repository name.
        """
        try:
            actual_repo_subject = repo_subject if repo_subject else subject
            repo_name = f"dataModel.{actual_repo_subject}"

            # Try to get README for description
            readme_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_ORG}/{repo_name}/master/{model}/README.md"
            logger.debug(f"Fetching README from: {readme_url}")
            response = await self._run_sync_in_thread(
                self._session.get, readme_url, timeout=30
            )
            logger.debug(f"README response status: {response.status_code}")

            description = ""
            if response.status_code == 200:
                readme_content = response.text
                lines = readme_content.strip().split("\n")
                for line in lines:
                    line = line.strip()
                    if line and not line.startswith("#") and len(line) > 10:
                        description = line
                        break

            # Try to get schema for attributes
            schema_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_ORG}/{repo_name}/master/{model}/schema.json"
            logger.debug(f"Fetching schema from: {schema_url}")
            response = await self._run_sync_in_thread(
                self._session.get, schema_url, timeout=30
            )
            logger.debug(f"Schema response status: {response.status_code}")

            attributes = []
            if response.status_code == 200:
                try:
                    schema = response.json()
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
                    logger.warning(f"Failed to decode JSON schema from {schema_url}")
                    pass

            return {
                "subject": subject, # Keep original subject for consistency in results, but source is from repo_subject
                "model": model,
                "description": description or f"Smart Data Model for {model}",
                "attributes": attributes,
                "source": f"github ({actual_repo_subject})"
            }

        except Exception as e:
            logger.debug(f"GitHub API call failed for {actual_repo_subject}/{model}: {e}")
            return None

    async def get_model_schema(self, subject: str, model: str, repo_subject: Optional[str] = None) -> Dict[str, Any]:
        """Get the JSON schema for a model.

        Args:
            subject (str): The name of the subject the model belongs to.
            model (str): The name of the model to retrieve the schema for.
            repo_subject (Optional[str]): Overrides the subject for constructing the GitHub repository name.

        Returns:
            Dict[str, Any]: The JSON schema of the specified model.
        """
        cache_key = f"schema_{subject}_{model}_{repo_subject}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Get from GitHub
        actual_repo_subject = repo_subject if repo_subject else subject
        repo_name = f"dataModel.{actual_repo_subject}"
        schema_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_ORG}/{repo_name}/master/{model}/schema.json"
        try:
            response = await self._run_sync_in_thread(
                self._session.get, schema_url, timeout=30
            )

            if response.status_code == 200:
                schema = response.json()
                self._cache.set(cache_key, schema)
                return schema
            else:
                raise ValueError(f"Schema not found: HTTP {response.status_code} for {actual_repo_subject}/{model}")

        except Exception as e:
            raise ValueError(f"Failed to fetch schema for {actual_repo_subject}/{model}: {e}")

    async def get_model_examples(self, subject: str, model: str, repo_subject: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get example instances for a model.

        Args:
            subject (str): The name of the subject the model belongs to.
            model (str): The name of the model to retrieve examples for.
            repo_subject (Optional[str]): Overrides the subject for constructing the GitHub repository name.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing an example instance of the model.
        """
        cache_key = f"examples_{subject}_{model}_{repo_subject}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        try:
            # Try pysmartdatamodels first
            examples = await self._run_sync_in_thread(psdm.ngsi_ld_example_generator, subject, model)

            if examples:
                if isinstance(examples, dict):
                    examples = [examples]
                elif isinstance(examples, list):
                    examples = examples

                self._cache.set(cache_key, examples)
                return examples

        except Exception as e:
            logger.debug(f"pysmartdatamodels examples failed for {subject}/{model}: {e}")

        # Fallback: try to get from GitHub examples
        try:
            examples = await self._get_examples_from_github(subject, model, repo_subject=repo_subject)
            if examples:
                self._cache.set(cache_key, examples)
                return examples
        except Exception as e:
            logger.debug(f"GitHub examples failed for {subject}/{model}: {e}")

        # Ultimate fallback: generate basic example
        example = await self._generate_basic_example(subject, model)
        examples = [example] if example else []

        self._cache.set(cache_key, examples)
        return examples

    async def _get_examples_from_github(self, subject: str, model: str, repo_subject: Optional[str] = None) -> Optional[List[Dict[str, Any]]]:
        """Get examples from GitHub repository.
        repo_subject allows overriding the subject used for constructing the GitHub repository name.
        """
        try:
            actual_repo_subject = repo_subject if repo_subject else subject
            repo_name = f"dataModel.{actual_repo_subject}"
            # Try various common example file names
            example_paths = [
                "example.json",
                "examples/example.json",
                "example.jsonld",
                "examples/example.jsonld"
            ]

            for path in example_paths:
                example_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_ORG}/{repo_name}/master/{model}/{path}"
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
            logger.debug(f"GitHub examples fetch failed for {actual_repo_subject}/{model}: {e}")

        return None

    async def _generate_basic_example(self, subject: str, model: str) -> Optional[Dict[str, Any]]:
        """Generate a basic example for a model."""
        try:
            details = await self.get_model_details(subject, model)

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
            logger.debug(f"Failed to generate basic example for {subject}/{model}: {e}")
            return None

    async def get_subject_context(self, subject: str) -> Dict[str, Any]:
        """Get the JSON-LD context for a subject.

        Args:
            subject (str): The name of the subject to retrieve the context for.

        Returns:
            Dict[str, Any]: The JSON-LD context for the specified subject.
        """
        cache_key = f"context_{subject}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Try to get from GitHub
        repo_name = f"dataModel.{subject}"
        context_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_ORG}/{repo_name}/master/context.jsonld"
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
            context = self._generate_basic_context(subject)
            self._cache.set(cache_key, context)
            return context

    def _generate_basic_context(self, subject: str) -> Dict[str, Any]:
        """Generate a basic JSON-LD context for a subject."""
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
        """Suggest models that match the provided data structure.

        Args:
            data (Dict[str, Any]): The data structure (as a dictionary) to compare against models.
            top_k (int): The number of top matching models to return (default: 5).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing a suggested model,
                                   its similarity score, and matched attributes.
        """
        if not isinstance(data, dict):
            return []

        data_keys = set(data.keys())

        # Score models based on attribute overlap
        candidates = []

        subjects = await self.list_subjects()

        for subject in subjects:
            try:
                models = await self.list_models_in_subject(subject, limit=100)

                for model_name in models[:50]:  # Limit to avoid too many requests
                    try:
                        details = await self.get_model_details(subject, model_name)

                        if "attributes" in details:
                            model_attrs = {attr["name"] for attr in details["attributes"]}
                            overlap = len(data_keys.intersection(model_attrs))
                            total_attrs = len(model_attrs.union(data_keys))

                            if total_attrs > 0:
                                similarity = overlap / total_attrs
                                if similarity > 0.1:  # Only include reasonable matches
                                    candidates.append({
                                        "subject": subject,
                                        "model": model_name,
                                        "similarity": round(similarity, 3),
                                        "matched_attributes": overlap,
                                        "total_attributes": len(model_attrs),
                                        "description": details.get("description", "")
                                    })

                    except Exception as e:
                        logger.debug(f"Error checking model {subject}/{model_name}: {e}")
                        continue

            except Exception as e:
                logger.debug(f"Error in subject {subject}: {e}")
                continue

        # Sort by similarity and return top k
        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:top_k]
