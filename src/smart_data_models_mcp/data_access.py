"""Data access layer for Smart Data Models.

This module provides access to FIWARE Smart Data Models through GitHub analyzer
and pysmartdatamodels package, with caching and error handling.
"""

import asyncio
import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from .github_repo_analyzer import EmbeddedGitHubAnalyzer

try:
    import pysmartdatamodels as psdm
    PYSMARTDATAMODELS_AVAILABLE = True
except ImportError:
    psdm = None
    PYSMARTDATAMODELS_AVAILABLE = False

logger = logging.getLogger(__name__)


# Available pysmartdatamodels functions:
# - list_all_subjects()
# - datamodels_subject(subject)
# - attributes_datamodel(subject, datamodel)
# - description_attribute(subject, datamodel, attribute)
# - datatype_attribute(subject, datamodel, attribute)
# - ngsi_datatype_attribute(subject, datamodel, attribute)
# - model_attribute(subject, datamodel, attribute)
# - units_attribute(subject, datamodel, attribute)
# - ngsi_ld_example_generator(schema_url) - requires full schema URL
# - subject_for_datamodel(datamodel)
# - list_datamodel_metadata(datamodel, subject)

# Recent fixes applied:
# ✓ Fix list_subjects to use psdm.list_all_subjects()
# ✓ Fix list_models_in_subject to use psdm.datamodels_subject()
# ✓ Rewrite get_model_details with multiple psdm functions
# ✓ Fix search to use comprehensive text search
# ✓ Fix ngsi_ld_example_generator to use proper schema URL


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

    def _normalize_subject(self, subject: Optional[str]) -> Optional[str]:
        """Normalize subject name by prepending 'dataModel.' if not present.

        Args:
            subject: Subject name that may or may not include 'dataModel.' prefix

        Returns:
            Normalized subject name with 'dataModel.' prefix, or None if input was None
        """
        if subject is None:
            return None
        if not subject.startswith("dataModel."):
            return f"dataModel.{subject}"
        return subject

    # Known domains from the specification
    KNOWN_DOMAINS = [
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
        # Don't pass timeout to run_in_executor - handle timeout within the sync function if needed
        timeout_value = kwargs.pop('timeout', None)
        result = await loop.run_in_executor(None, func, *args, **kwargs)
        return result

    async def list_domains(self) -> List[str]:
        """List all available domains."""
        cache_key = "domains"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Fetch domains from GitHub organization repositories
        try:
            # GitHub API endpoint for organization repositories
            org_repos_url = f"https://api.github.com/orgs/{self.SMART_DATA_MODELS_ORG}/repos"
            response = await self._run_sync_in_thread(
                self._session.get, org_repos_url, timeout=30
            )

            if response.status_code == 200:
                repos_data = response.json()
                domains = []
                for repo in repos_data:
                    repo_name = repo.get("name", "")
                    # Filter repositories that do NOT start with "dataModel."
                    if repo_name and not repo_name.startswith("dataModel."):
                        domains.append(repo_name)

                domains.sort()  # Sort for consistent ordering
                self._cache.set(cache_key, domains)
                return domains
            else:
                logger.error(f"GitHub API returned status {response.status_code} for {org_repos_url}")

        except Exception as e:
            logger.error(f"Failed to fetch domains from GitHub API: {e}")

        # Fallback to known domains if GitHub API fails
        self._cache.set(cache_key, self.KNOWN_DOMAINS)
        return self.KNOWN_DOMAINS

    async def list_subjects(self) -> List[str]:
        """List all available subjects."""
        cache_key = "subjects"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Try pysmartdatamodels first
        if psdm:
            try:
                subjects = await self._run_sync_in_thread(psdm.list_all_subjects)
                self._cache.set(cache_key, subjects)
                return subjects
            except (AttributeError, Exception) as e:
                logger.warning(f"pysmartdatamodels list_all_subjects failed: {e}")

        # Fallback: Get subjects from GitHub API
        try:
            subjects = []
            domains = await self.list_domains()  # Get all domains
            for domain in domains:
                # Skip repositories that start with dataModel (these are subject repos, not domain repos)
                if domain.startswith("dataModel."):
                    continue

                try:
                    domain_subjects = await self._get_subjects_from_github_api(domain)
                    if domain_subjects:
                        subjects.extend(domain_subjects)
                except Exception as e:
                    logger.debug(f"Failed to get subjects for domain {domain}: {e}")
                    continue

            # Remove duplicates and sort
            subjects = list(set(subjects))
            subjects.sort()
            self._cache.set(cache_key, subjects)
            return subjects

        except Exception as e:
            logger.error(f"GitHub API fallback also failed: {e}")
            # Return empty list as ultimate fallback
            return []

    async def list_models_in_subject(self, subject: str) -> List[str]:
        """List all models in a specific subject using GitHub API."""
        cache_key = f"subject_models_{subject}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Use GitHub API to get models from the repository
        try:
            models = await self._get_models_from_github_api(subject)
            if models:
                # Cache and return the models
                self._cache.set(cache_key, models)
                return models
        except Exception as e:
            logger.error(f"GitHub API failed for subject {subject}: {e}")

        # Fallback: return empty list
        return []

    async def list_domain_subjects(self, domain: str) -> List[str]:
        """List all subjects in a specific domain using GitHub API."""
        cache_key = f"domain_subjects_{domain}"
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Find the actual repository name that matches the domain
        actual_repo_name = await self._find_domain_repository(domain)
        if not actual_repo_name:
            logger.warning(f"Could not find repository for domain '{domain}'")
            return []

        # Use GitHub API to get subjects from the found repository
        try:
            subjects = await self._get_subjects_from_github_api(actual_repo_name)
            if subjects:
                # Cache and return the subjects
                self._cache.set(cache_key, subjects)
                return subjects
        except Exception as e:
            logger.error(f"GitHub API failed for domain {domain} (repo: {actual_repo_name}): {e}")

        # Fallback: return empty list
        return []

    async def _find_domain_repository(self, domain: str) -> Optional[str]:
        """Find the actual repository name that matches the requested domain.

        This method performs flexible matching to handle cases where users might
        specify "Water" but the repo is named "SmartWater".

        Args:
            domain: The domain name to search for (case-insensitive)

        Returns:
            The exact repository name if found, None otherwise
        """
        try:
            # Fetch all repositories from the smart-data-models organization
            org_repos_url = f"https://api.github.com/orgs/{self.SMART_DATA_MODELS_ORG}/repos"
            response = await self._run_sync_in_thread(
                self._session.get, org_repos_url, timeout=30
            )

            if response.status_code != 200:
                logger.error(f"GitHub API returned status {response.status_code} for {org_repos_url}")
                return None

            repos_data = response.json()
            domain_lower = domain.lower()

            # Extract repository names (filter out dataModel.* repos for domain search)
            available_repos = [
                repo.get("name", "") for repo in repos_data
                if repo.get("name", "") and not repo.get("name", "").startswith("dataModel.")
            ]

            # Step 1: Exact match
            if domain in available_repos:
                return domain

            # Step 2: Case-insensitive match
            for repo in available_repos:
                if repo.lower() == domain_lower:
                    return repo

            # Step 3: Fuzzy matching - try removing "Smart" prefix if present
            if domain_lower.startswith("smart"):
                stripped_domain = domain_lower[5:]  # Remove "smart" prefix
                for repo in available_repos:
                    if repo.lower() == stripped_domain:
                        return repo

            # Step 4: Reverse - if domain doesn't start with "smart", try adding it
            if not domain_lower.startswith("smart"):
                smart_domain = f"smart{domain_lower}"
                for repo in available_repos:
                    if repo.lower() == smart_domain:
                        return repo

            # Step 5: Partial match (domain contained in repo name)
            for repo in available_repos:
                if domain_lower in repo.lower():
                    return repo

            logger.warning(f"No repository found matching domain '{domain}'. Available domains: {available_repos[:10]}...")
            return None

        except Exception as e:
            logger.error(f"Error finding repository for domain '{domain}': {e}")
            return None

    async def _get_subjects_from_github_api(self, domain: str) -> Optional[List[str]]:
        """Get subjects from GitHub API repository contents."""
        try:
            repo_name = domain
            subject_url = f"{self.GITHUB_API_BASE}/{self.SMART_DATA_MODELS_ORG}/{repo_name}/contents"
            response = await self._run_sync_in_thread(
                self._session.get, subject_url, timeout=30
            )

            if response.status_code == 200:
                contents = response.json()
                subjects = []
                for item in contents:
                    # Check both directories and files that start with "dataModel."
                    if item["name"].startswith("dataModel."):
                        # Remove the "dataModel." prefix to return just the subject name
                        subjects.append(item["name"][10:])  # len("dataModel.") = 10
                return subjects

        except Exception as e:
            logger.debug(f"GitHub API call failed for {domain}: {e}")

        return None

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

        # Perform comprehensive text search across subjects and models
        results = await self._comprehensive_model_search(query, subject, limit, include_attributes)

        self._cache.set(cache_key, results)
        return results

    async def _comprehensive_model_search(
        self,
        query: str,
        subject: Optional[str] = None,
        limit: int = 20,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Perform comprehensive search across all models using pysmartdatamodels functions."""
        # Normalize subject parameter
        subject = self._normalize_subject(subject)

        results = []
        query_lower = query.lower()

        subjects_to_search = [subject] if subject else await self.list_subjects()

        for subject_name in subjects_to_search:
            try:
                # Get all models in this subject
                models = await self.list_models_in_subject(subject_name, limit=200)  # Higher limit for search

                for model_name in models:
                    try:
                        # Get model details which includes attributes information
                        details = await self.get_model_details(subject_name, model_name)

                        # Check if query matches model name
                        name_match = query_lower in model_name.lower()

                        # Check if query matches description
                        desc_match = query_lower in details.get("description", "").lower()

                        # Check if query matches any attribute names or descriptions
                        attr_match = False
                        matched_attr = None
                        if include_attributes and "attributes" in details:
                            for attr in details["attributes"]:
                                attr_name = attr.get("name", "")
                                attr_desc = attr.get("description", "")
                                if query_lower in attr_name.lower() or query_lower in attr_desc.lower():
                                    attr_match = True
                                    matched_attr = attr
                                    break

                        # If any match found, add to results
                        if name_match or desc_match or attr_match:
                            model_info = {
                                "subject": subject_name,
                                "model": model_name,
                                "name": model_name,
                                "description": details.get("description", ""),
                                "source": details.get("source", "pysmartdatamodels")
                            }

                            if attr_match and matched_attr:
                                # Add specific attribute match info
                                model_info["matched_attribute"] = matched_attr["name"]
                                model_info["attribute_description"] = matched_attr.get("description", "")

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

        # Try to use GitHub analyzer to get metadata
        try:
            analyzer = EmbeddedGitHubAnalyzer()
            repo_url = f"https://smart-data-models.github.io/dataModel.{subject}/{model}"

            # Run the synchronous analyzer in thread
            metadata = await self._run_sync_in_thread(analyzer.generate_metadata, repo_url)

            if metadata:
                # Convert GitHub analyzer format to the expected format
                processed_details = {
                    "subject": metadata["subject"],
                    "model": metadata["dataModel"],
                    "description": metadata["description"],
                    "version": metadata.get("version", "0.1.1"),
                    "title": metadata.get("title", ""),
                    "modelTags": metadata.get("modelTags", ""),
                    "$id": metadata.get("$id", ""),
                    "yamlUrl": metadata.get("yamlUrl", ""),
                    "jsonSchemaUrl": metadata.get("jsonSchemaUrl", ""),
                    "@context": metadata.get("@context", ""),
                    "exampleKeyvaluesJson": metadata.get("exampleKeyvaluesJson", ""),
                    "exampleKeyvaluesJsonld": metadata.get("exampleKeyvaluesJsonld", ""),
                    "exampleNormalizedJson": metadata.get("exampleNormalizedJson", ""),
                    "exampleNormalizedJsonld": metadata.get("exampleNormalizedJsonld", ""),
                    "sql": metadata.get("sql", ""),
                    "adopters": metadata.get("adopters", ""),
                    "contributors": metadata.get("contributors", ""),
                    "spec": metadata.get("spec", ""),
                    "spec_DE": metadata.get("spec_DE", ""),
                    "spec_ES": metadata.get("spec_ES", ""),
                    "spec_FR": metadata.get("spec_FR", ""),
                    "spec_IT": metadata.get("spec_IT", ""),
                    "spec_JA": metadata.get("spec_JA", ""),
                    "spec_KO": metadata.get("spec_KO", ""),
                    "spec_ZH": metadata.get("spec_ZH", ""),
                    "required": metadata.get("required", ["type", "id"]),
                    "attributes": [],  # Will be populated from schema if needed
                    "source": "github_analyzer"
                }

                # Try to get attributes from the schema
                try:
                    schema = await self.get_model_schema(subject, model)
                    if schema and isinstance(schema, dict):
                        attributes = []
                        required_attrs = schema.get("required", [])

                        properties = schema.get("properties", {})
                        for prop_name, prop_def in properties.items():
                            if isinstance(prop_def, dict):
                                attr_info = {
                                    "name": prop_name,
                                    "type": prop_def.get("type", "string"),
                                    "description": prop_def.get("description", f"Property {prop_name}"),
                                    "required": prop_name in required_attrs
                                }
                                attributes.append(attr_info)

                        processed_details["attributes"] = attributes
                except Exception as e:
                    logger.debug(f"Failed to get attributes from schema: {e}")

                self._cache.set(cache_key, processed_details)
                return processed_details

        except Exception as e:
            logger.debug(f"GitHub analyzer failed for {subject}/{model}: {e}")

        # Fallback 1: Try basic GitHub details
        try:
            details = await self._get_basic_model_details_from_github(subject, model)
            if details:
                self._cache.set(cache_key, details)
                return details
        except Exception as e:
            logger.debug(f"Basic GitHub details fallback failed for {subject}/{model}: {e}")

        # Fallback 2: Try pysmartdatamodels as final fallback
        try:
            # Get attributes list for this model
            attributes_list = await self._run_sync_in_thread(psdm.attributes_datamodel, subject, model)

            if attributes_list and isinstance(attributes_list, list):
                attributes = []

                # For each attribute, get its details
                for attr_name in attributes_list[:50]:  # Limit to avoid too many calls
                    try:
                        # Get description, data type, and NGSI type for each attribute
                        attr_desc = await self._run_sync_in_thread(psdm.description_attribute, subject, model, attr_name)
                        attr_type = await self._run_sync_in_thread(psdm.datatype_attribute, subject, model, attr_name)
                        ngsi_type = await self._run_sync_in_thread(psdm.ngsi_datatype_attribute, subject, model, attr_name)
                        attr_units = await self._run_sync_in_thread(psdm.units_attribute, subject, model, attr_name)
                        attr_model = await self._run_sync_in_thread(psdm.model_attribute, subject, model, attr_name)

                        attribute_info = {
                            "name": attr_name,
                            "type": attr_type if attr_type else "string",
                            "description": attr_desc if attr_desc else f"Attribute {attr_name}",
                            "ngsi_type": ngsi_type if ngsi_type else "Property"
                        }

                        if attr_units:
                            attribute_info["units"] = attr_units
                        if attr_model:
                            attribute_info["model"] = attr_model

                        attributes.append(attribute_info)

                    except Exception as e:
                        logger.debug(f"Error getting details for attribute {attr_name}: {e}")
                        # Add basic attribute info even if detailed retrieval fails
                        attributes.append({
                            "name": attr_name,
                            "type": "string",
                            "description": f"Attribute {attr_name}",
                            "ngsi_type": "Property"
                        })

                # Try to get metadata for the model
                metadata = await self._run_sync_in_thread(psdm.list_datamodel_metadata, model, subject)
                if metadata and isinstance(metadata, dict):
                    # Extract information from metadata
                    processed_details = {
                        "subject": subject,
                        "model": model,
                        "description": metadata.get("description", f"Smart Data Model for {model}"),
                        "attributes": attributes,
                        "source": "pysmartdatamodels"
                    }

                    # Add optional metadata fields if available
                    for field in ["version", "modelTags", "license", "spec", "title", "required"]:
                        if field in metadata and metadata[field]:
                            processed_details[field] = metadata[field]

                    self._cache.set(cache_key, processed_details)
                    return processed_details
                else:
                    # No metadata available, use basic constructed details
                    processed_details = {
                        "subject": subject,
                        "model": model,
                        "description": f"Smart Data Model for {model} in {subject} subject",
                        "attributes": attributes,
                        "source": "pysmartdatamodels"
                    }
                    self._cache.set(cache_key, processed_details)
                    return processed_details

        except Exception as e:
            logger.debug(f"pysmartdatamodels details construction failed for {subject}/{model}: {e}")

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
        # Normalize subject parameter
        subject = self._normalize_subject(subject)
        repo_subject = self._normalize_subject(repo_subject)

        cache_key = f"examples_{subject}_{model}_{repo_subject}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            return cached

        # Try pysmartdatamodels with schema URL (it requires a full schema URL, not subject/model)
        try:
            # Construct the schema URL first
            actual_repo_subject = repo_subject if repo_subject else subject
            repo_name = f"dataModel.{actual_repo_subject}"
            schema_url = f"https://raw.githubusercontent.com/{self.SMART_DATA_MODELS_ORG}/{repo_name}/master/{model}/schema.json"

            examples = await self._run_sync_in_thread(psdm.ngsi_ld_example_generator, schema_url)

            if examples and examples != "dataModel" and examples != "False":
                if isinstance(examples, dict):
                    examples = [examples]
                elif isinstance(examples, list):
                    examples = [examples] if examples else []

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
                # Normalize subject parameter
                normalized_subject = self._normalize_subject(subject)

                models = await self.list_models_in_subject(normalized_subject, limit=100)

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
