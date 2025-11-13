"""Data access layer for Smart Data Models.

This module provides access to FIWARE Smart Data Models through GitHub analyzer
and pysmartdatamodels package, with caching and error handling.
"""

import asyncio
import json
import logging
import os
import time
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import urljoin

import requests
from dotenv import load_dotenv
from .github_repo_analyzer import EmbeddedGitHubAnalyzer


from pysmartdatamodels.pysmartdatamodels import (
    list_all_subjects, datamodels_subject, attributes_datamodel,
    description_attribute, datatype_attribute, ngsi_datatype_attribute,
    units_attribute, model_attribute, ngsi_ld_example_generator,
    subject_for_datamodel, list_datamodel_metadata, load_all_attributes,
    list_all_datamodels
)

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
# ✓ Fix list_subjects to use list_all_subjects()
# ✓ Fix list_models_in_subject to use datamodels_subject()
# ✓ Rewrite get_model_details with multiple psdm functions
# ✓ Fix search to use comprehensive text search
# ✓ Fix ngsi_ld_example_generator to use proper schema URL
# ✓ Optimize search_models with GitHub Code Search-first strategy (performance improvement)
# ✓ Enhance search with additional pysmartdatamodels functions (load_all_attributes, list_all_datamodels, subject_for_datamodel)
# ✓ Fix non-existent load_all_datamodels function calls - replaced with available functions


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
        "SmartCities", "SmartAgrifood", "SmartWater", "SmartEnergy", "SmartLogistics",
        "SmartRobotics", "Smart-Sensoring", "Cross sector", "SmartHealth", "SmartDestination",
        "SmartEnvironment", "SmartAeronautics", "SmartManufacturing", "Incubated", "Harmonization"
    ]
    GITHUB_API_BASE = "https://api.github.com/repos"
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com"
    SMART_DATA_MODELS_ORG = "smart-data-models" # Organization name

    def __init__(self):
        # Load environment variables from .env file
        load_dotenv()

        self._cache = Cache(ttl_seconds=1800)  # 30 minutes
        self._session = requests.Session()

        # Set up base headers
        headers = {
            "User-Agent": "smart-data-models-mcp/0.1.0"
        }

        # Add GitHub token if available
        github_token = os.getenv("GITHUB_READ_TOKEN")
        if github_token:
            headers["Authorization"] = f"token {github_token}"
            logger.info("GitHub token loaded and will be used for API requests")
        else:
            logger.warning("No GitHub token found in environment - API rate limits may apply")

        self._session.headers.update(headers)

    async def _run_sync_in_thread(self, func, *args, **kwargs):
        """Run synchronous function in thread pool."""
        loop = asyncio.get_event_loop()
        # Don't pass timeout to run_in_executor - handle timeout within the sync function if needed
        timeout_value = kwargs.pop('timeout', None)
        result = await loop.run_in_executor(None, func, *args, **kwargs)
        return result

    async def list_domains(self) -> List[str]:
        """List all available domains."""
        logger.info("List Domains: Starting - using GitHub API to fetch domain list")
        cache_key = "domains"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("List Domains: Returning cached data")
            return cached

        # Fetch domains from GitHub organization repositories with pagination
        try:
            domains = []
            page = 1

            while True:
                # GitHub API endpoint for organization repositories with pagination
                org_repos_url = f"https://api.github.com/orgs/{self.SMART_DATA_MODELS_ORG}/repos?page={page}&per_page=100"
                response = await self._run_sync_in_thread(
                    self._session.get, org_repos_url, timeout=30
                )

                if response.status_code == 200:
                    repos_data = response.json()

                    # If no more repos, break
                    if not repos_data:
                        break

                    for repo in repos_data:
                        repo_name = repo.get("name", "")
                        # Filter repositories that do NOT start with "dataModel."
                        if repo_name and not repo_name.startswith("dataModel."):
                            domains.append(repo_name)

                    # Check if there are more pages (if we got a full page, there might be more)
                    if len(repos_data) < 100:
                        break

                    page += 1  # Go to next page
                else:
                    logger.error(f"GitHub API returned status {response.status_code} for {org_repos_url}")
                    raise Exception(f"GitHub API error: {response.status_code}")

            domains.sort()  # Sort for consistent ordering
            logger.info(f"List Domains: Retrieved {len(domains)} domains total from {page} pages")
            self._cache.set(cache_key, domains)
            return domains

        except Exception as e:
            logger.error(f"Failed to fetch domains from GitHub API: {e}")

        # Fallback to known domains if GitHub API fails
        self._cache.set(cache_key, self.KNOWN_DOMAINS)
        return self.KNOWN_DOMAINS

    async def list_subjects(self) -> List[str]:
        """List all available subjects using GitHub API with pysmartdatamodels fallback."""
        logger.info("List Subjects: Starting - using GitHub API with pysmartdatamodels fallback")
        cache_key = "subjects"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("List Subjects: Returning cached data")
            return cached

        # Get subjects from GitHub API
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
            logger.error(f"GitHub API failed: {e}")
            # Fallback to pysmartdatamodels
            try:
                logger.info("List Subjects: Falling back to pysmartdatamodels")
                psdm_subjects = await self._run_sync_in_thread(list_all_subjects)
                if psdm_subjects and isinstance(psdm_subjects, list):
                    # Normalize subjects (remove dataModel. prefix if present)
                    normalized_subjects = []
                    for subject in psdm_subjects:
                        if subject.startswith("dataModel."):
                            normalized_subjects.append(subject[10:])  # Remove "dataModel." prefix
                        else:
                            normalized_subjects.append(subject)

                    # Remove duplicates and sort
                    normalized_subjects = list(set(normalized_subjects))
                    normalized_subjects.sort()
                    self._cache.set(cache_key, normalized_subjects)
                    return normalized_subjects
            except Exception as e:
                logger.error(f"pysmartdatamodels fallback failed: {e}")

            # Return empty list as ultimate fallback
            return []

    async def list_models_in_subject(self, subject: str) -> List[str]:
        """List all models in a specific subject using GitHub API.

        Args:
            subject: Subject name (may or may not include 'dataModel.' prefix)
        """
        logger.info("List Models in Subject: Starting - subject='%s' - using GitHub API", subject)

        # Denormalize subject name for internal use
        if subject.startswith("dataModel."):
            subject = subject[10:]  # Remove "dataModel." prefix

        cache_key = f"subject_models_{subject}"
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("List Models in Subject: Returning cached data - %s models", len(cached) if cached else 0)
            return cached

        # Use GitHub API to get models from the repository
        try:
            logger.debug("List Models in Subject: Calling _get_models_from_github_api for subject '%s'", subject)
            models = await self._get_models_from_github_api(subject)
            if models:
                logger.info("List Models in Subject: GitHub API success - %s models found for subject '%s'", len(models), subject)
                # Cache and return the models
                self._cache.set(cache_key, models)
                return models
            else:
                logger.warning("List Models in Subject: No models found for subject '%s' via GitHub API", subject)
        except Exception as e:
            logger.error(f"GitHub API failed for subject {subject}: {e}")

        # Fallback: return empty list
        logger.info("List Models in Subject: Fallback - returning empty list for subject '%s'", subject)
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
        domain: Optional[str] = None,
        subject: Optional[str] = None,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Search for models across subjects using optimized strategy.

        Performance optimization strategy:
        1. First try GitHub Code Search API (fast, comprehensive, leverages GitHub's search infrastructure)
        2. Fallback to pysmartdatamodels if no results found

        Args:
            query (str): The search query (model name, attributes, or keywords).
            domain (Optional[str]): Limits the search to a specific domain.
            subject (Optional[str]): Limits the search to a specific subject.
            include_attributes (bool): Whether to include attribute details in the results (default: False).

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each representing a matching data model.
        """
        logger.info("Search Models: Starting - query='%s', domain='%s', subject='%s', include_attributes=%s - using GitHub Code Search-first strategy", query, domain, subject, include_attributes)
        cache_key = f"search_{query}_{domain}_{subject}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Search Models: Returning cached data - %s results", len(cached) if cached else 0)
            return cached

        # Use optimized GitHub Code Search-first strategy
        results = await self._github_code_search_first_search(query, domain, subject, include_attributes)

        logger.info("Search Models: Search completed - %s results found", len(results) if results else 0)
        self._cache.set(cache_key, results)
        return results

    async def _github_code_search_first_search(
        self,
        query: str,
        domain: Optional[str] = None,
        subject: Optional[str] = None,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Optimized search strategy: GitHub Code Search first, then pysmartdatamodels as fallback.

        Performance optimization strategy:
        1. Use GitHub Code Search API first (fast, comprehensive, leverages GitHub's search infrastructure)
        2. Fallback to pysmartdatamodels if no results found
        """
        logger.info("GitHub Code Search-first search: query='%s', domain='%s', subject='%s'", query, domain, subject)

        # Step 1: Try GitHub Code Search API first (fast, leverages GitHub's infrastructure)
        github_results = await self._search_github_with_code_api(query, domain, subject, include_attributes)

        if github_results:
            logger.info("GitHub Code Search found %d results - returning without fallback", len(github_results))
            return github_results

        # Step 2: No results from GitHub Code Search, fallback to pysmartdatamodels
        logger.info("No results from GitHub Code Search - falling back to pysmartdatamodels")
        psdm_results = await self._pysmartdatamodels_first_search(query, domain, subject, include_attributes)

        return psdm_results

    async def _pysmartdatamodels_first_search(
        self,
        query: str,
        domain: Optional[str] = None,
        subject: Optional[str] = None,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Optimized search strategy: pysmartdatamodels first, then GitHub excluding existing models.

        Performance optimization strategy:
        1. Use pysmartdatamodels list_all_datamodels() and list_all_subjects() for fast local search
        2. Filter results by query, domain, and subject criteria
        3. Use additional pysmartdatamodels functions (load_all_attributes, subject_for_datamodel) as fallback
        4. Only fallback to GitHub API if no results found, excluding models already in pysmartdatamodels
        """
        logger.info("PySmartDataModels-first search: query='%s', domain='%s', subject='%s'", query, domain, subject)

        # Step 1: Try pysmartdatamodels first (fast, local data)
        psdm_results = await self._search_with_pysmartdatamodels(query, domain, subject, include_attributes)

        if psdm_results:
            logger.info("PySmartDataModels search found %d results - returning without fallback", len(psdm_results))
            return psdm_results

        # Step 2: Try additional pysmartdatamodels functions as fallback
        logger.info("No results from primary PySmartDataModels search - trying additional functions")
        additional_psdm_results = await self._search_with_additional_pysmartdatamodels_functions(query, domain, subject, include_attributes)

        if additional_psdm_results:
            logger.info("Additional PySmartDataModels functions found %d results - returning without GitHub fallback", len(additional_psdm_results))
            return additional_psdm_results

        # Step 3: No results from pysmartdatamodels, fallback to GitHub but exclude existing models
        logger.info("No results from PySmartDataModels functions - falling back to GitHub API (excluding existing models)")
        github_results = await self._search_github_excluding_pysmartdatamodels(query, domain, subject, include_attributes)

        return github_results

    async def _search_with_additional_pysmartdatamodels_functions(
        self,
        query: str,
        domain: Optional[str] = None,
        subject: Optional[str] = None,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Search using additional pysmartdatamodels functions as fallback.

        Uses load_all_attributes(), list_all_datamodels(), and subject_for_datamodel()
        when primary pysmartdatamodels search doesn't find results.
        """

        try:
            query_lower = query.lower()
            results = []

            # Strategy 1: Use load_all_attributes() for comprehensive attribute search
            logger.debug("Additional PySmartDataModels: Trying load_all_attributes() for comprehensive attribute search")
            try:
                all_attributes = await self._run_sync_in_thread(load_all_attributes)

                if all_attributes and isinstance(all_attributes, list):
                    logger.debug(f"Additional PySmartDataModels: Found {len(all_attributes)} attributes to search through")

                    # Group attributes by model for efficient processing
                    model_matches = {}

                    for attr in all_attributes:
                        try:
                            attr_repo = attr.get("repoName", "")
                            attr_model = attr.get("dataModel", "")
                            attr_name = attr.get("property", "").lower()
                            attr_desc = attr.get("description", "").lower()

                            # Check if attribute matches query
                            if query_lower in attr_name or query_lower in attr_desc:
                                model_key = f"{attr_repo}:{attr_model}"

                                if model_key not in model_matches:
                                    model_matches[model_key] = {
                                        "subject": attr_repo,
                                        "model": attr_model,
                                        "matched_attributes": [],
                                        "relevance_score": 0,
                                        "matched_parts": []
                                    }

                                # Add attribute match
                                model_matches[model_key]["matched_attributes"].append({
                                    "name": attr.get("property", ""),
                                    "description": attr.get("description", ""),
                                    "type": attr.get("type", "string")
                                })

                                # Increase relevance score
                                model_matches[model_key]["relevance_score"] += 1.5
                                if "attributes" not in model_matches[model_key]["matched_parts"]:
                                    model_matches[model_key]["matched_parts"].append("attributes")

                        except Exception as e:
                            logger.debug(f"Error processing attribute: {e}")
                            continue

                    # Convert model matches to results
                    for model_key, match_data in model_matches.items():
                        try:
                            # Get model description using description_attribute
                            description = await self._run_sync_in_thread(
                                description_attribute,
                                match_data["subject"],
                                match_data["model"],
                                "id"  # Use 'id' as it's always present
                            )

                            result = {
                                "subject": match_data["subject"],
                                "model": match_data["model"],
                                "description": description or f"Smart Data Model for {match_data['model']}",
                                "relevance_score": round(match_data["relevance_score"], 2),
                                "matched_parts": match_data["matched_parts"],
                                "source": "pysmartdatamodels_attributes"
                            }

                            if include_attributes and match_data["matched_attributes"]:
                                result["attributes"] = match_data["matched_attributes"]

                            results.append(result)

                        except Exception as e:
                            logger.debug(f"Error processing model match {model_key}: {e}")
                            continue

                    if results:
                        logger.info(f"Additional PySmartDataModels: load_all_attributes() found {len(results)} results")
                        results.sort(key=lambda x: x["relevance_score"], reverse=True)
                        return results[:20]  # Limit results

            except Exception as e:
                logger.debug(f"load_all_attributes() search failed: {e}")

            # Strategy 2: Use list_all_datamodels() for direct model name search
            logger.debug("Additional PySmartDataModels: Trying list_all_datamodels() for direct model name search")
            try:
                all_models = await self._run_sync_in_thread(list_all_datamodels)

                if all_models and isinstance(all_models, list):
                    logger.debug(f"Additional PySmartDataModels: Found {len(all_models)} models to search through")

                    for model_name in all_models:
                        if query_lower in model_name.lower():
                            # Try to find subject for this model
                            try:
                                subjects = await self._run_sync_in_thread(subject_for_datamodel, model_name)

                                if subjects and isinstance(subjects, list):
                                    for subject_name in subjects:
                                        try:
                                            # Get description
                                            description = await self._run_sync_in_thread(
                                                description_attribute, subject_name, model_name, "id"
                                            )

                                            result = {
                                                "subject": subject_name,
                                                "model": model_name,
                                                "description": description or f"Smart Data Model for {model_name}",
                                                "relevance_score": 3.0,  # High score for direct model name match
                                                "matched_parts": ["name"],
                                                "source": "pysmartdatamodels_models"
                                            }

                                            results.append(result)

                                        except Exception as e:
                                            logger.debug(f"Error processing model {subject_name}/{model_name}: {e}")
                                            continue

                            except Exception as e:
                                logger.debug(f"subject_for_datamodel failed for {model_name}: {e}")
                                continue

                    if results:
                        logger.info(f"Additional PySmartDataModels: list_all_datamodels() found {len(results)} results")
                        results.sort(key=lambda x: x["relevance_score"], reverse=True)
                        return results[:20]  # Limit results

            except Exception as e:
                logger.debug(f"list_all_datamodels() search failed: {e}")

            # Strategy 3: Use subject_for_datamodel() for reverse lookup
            logger.debug("Additional PySmartDataModels: Trying subject_for_datamodel() for reverse lookup")
            try:
                # Try to find if query might be a model name
                subjects = await self._run_sync_in_thread(subject_for_datamodel, query)

                if subjects and isinstance(subjects, list):
                    logger.debug(f"Additional PySmartDataModels: subject_for_datamodel() found subjects: {subjects}")

                    for subject_name in subjects:
                        try:
                            # Get description
                            description = await self._run_sync_in_thread(
                                description_attribute, subject_name, query, "id"
                            )

                            result = {
                                "subject": subject_name,
                                "model": query,
                                "description": description or f"Smart Data Model for {query}",
                                "relevance_score": 3.0,  # High score for exact model match
                                "matched_parts": ["name"],
                                "source": "pysmartdatamodels_reverse"
                            }

                            results.append(result)

                        except Exception as e:
                            logger.debug(f"Error processing reverse lookup result {subject_name}/{query}: {e}")
                            continue

                    if results:
                        logger.info(f"Additional PySmartDataModels: subject_for_datamodel() found {len(results)} results")
                        return results

            except Exception as e:
                logger.debug(f"subject_for_datamodel() search failed: {e}")

            logger.info("Additional PySmartDataModels functions search completed - no results found")
            return results

        except Exception as e:
            logger.warning(f"Additional PySmartDataModels functions search failed: {e}")
            return []

    async def _search_with_pysmartdatamodels(
        self,
        query: str,
        domain: Optional[str] = None,
        subject: Optional[str] = None,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Search using pysmartdatamodels local data (fast, no network calls).

        Uses list_all_datamodels() and list_all_subjects() for comprehensive local search.
        """

        try:
            # Get all model names for quick filtering
            all_model_names = await self._run_sync_in_thread(list_all_datamodels)

            if not all_model_names:
                logger.debug("No model names found in pysmartdatamodels")
                return []

            results = []
            query_lower = query.lower()

            # Determine subjects to search
            subjects_to_search = []
            if subject:
                # Use specific subject
                normalized_subject = self._normalize_subject(subject)
                subjects_to_search = [normalized_subject]
            else:
                # Search all subjects (domain filtering not supported without load_all_datamodels)
                try:
                    all_subjects = await self._run_sync_in_thread(list_all_subjects)
                    subjects_to_search = all_subjects if all_subjects else []
                except Exception as e:
                    logger.debug(f"Failed to get subjects from pysmartdatamodels: {e}")
                    subjects_to_search = []

            logger.debug(f"Searching {len(subjects_to_search)} subjects from pysmartdatamodels")

            # Search through each subject
            for subject_name in subjects_to_search:
                try:
                    # Get models for this subject
                    models_in_subject = await self._run_sync_in_thread(datamodels_subject, subject_name)

                    if not models_in_subject:
                        continue

                    # Filter models by query
                    for model_name in models_in_subject:
                        if model_name in all_model_names:  # Safety check
                            relevance_score = 0
                            matched_parts = []

                            # Check model name match
                            if query_lower in model_name.lower():
                                relevance_score += 3.0
                                matched_parts.append("name")

                            # Check subject match (if searching multiple subjects)
                            if len(subjects_to_search) > 1 and query_lower in subject_name.lower():
                                relevance_score += 1.5
                                matched_parts.append("subject")

                            # Get description if we have a potential match
                            if relevance_score > 0:
                                try:
                                    description = await self._run_sync_in_thread(
                                        description_attribute, subject_name, model_name, "id"
                                    )
                                    if description and query_lower in description.lower():
                                        relevance_score += 1.0
                                        matched_parts.append("description")
                                except Exception:
                                    pass  # Description lookup failed, continue

                                # Get attributes if requested and we have a match
                                attributes = []
                                if include_attributes and relevance_score > 0:
                                    try:
                                        attr_names = await self._run_sync_in_thread(
                                            attributes_datamodel, subject_name, model_name
                                        )
                                        if attr_names:
                                            for attr_name in attr_names:
                                                if query_lower in attr_name.lower():
                                                    relevance_score += 1.5
                                                    matched_parts.append("attributes")
                                                attributes.append({
                                                    "name": attr_name,
                                                    "type": "string",  # Default, could be enhanced
                                                    "description": ""
                                                })
                                    except Exception:
                                        pass  # Attribute lookup failed, continue

                                if relevance_score > 0:
                                    model_info = {
                                        "subject": subject_name,
                                        "model": model_name,
                                        "description": description or f"Smart Data Model for {model_name}",
                                        "relevance_score": round(relevance_score, 2),
                                        "matched_parts": matched_parts,
                                        "source": "pysmartdatamodels"
                                    }

                                    if include_attributes and attributes:
                                        model_info["attributes"] = attributes

                                    results.append(model_info)

                except Exception as e:
                    logger.debug(f"Error searching subject {subject_name}: {e}")
                    continue

            # Sort by relevance score
            results.sort(key=lambda x: x["relevance_score"], reverse=True)
            logger.info(f"PySmartDataModels search found {len(results)} results")
            return results

        except Exception as e:
            logger.warning(f"PySmartDataModels search failed: {e}")
            return []

    async def _search_github_with_code_api(
        self,
        query: str,
        domain: Optional[str] = None,
        subject: Optional[str] = None,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Search GitHub using the global code search API for efficient model discovery.

        This method searches for each term in the query separately to improve search results,
        then combines and deduplicates the findings.

        Args:
            query: Search query string
            domain: Optional domain filter
            subject: Optional subject filter
            include_attributes: Whether to include attribute details

        Returns:
            List of model dictionaries with search results
        """
        try:
            logger.info("GitHub Code Search: Starting - query='%s', domain='%s', subject='%s'", query, domain, subject)

            # Split query into individual terms for separate searches
            query_terms = [term.strip() for term in query.split() if term.strip()]
            if not query_terms:
                logger.warning("No valid query terms found")
                return []

            logger.debug(f"Searching for {len(query_terms)} individual terms: {query_terms}")

            # Perform separate searches for each term
            all_results = []
            max_results_per_term = 30  # Limit per term to avoid overwhelming
            per_page = 20  # Smaller page size for individual term searches

            for term in query_terms:
                logger.debug(f"Searching for term: '{term}'")
                term_results = await self._search_single_term_github(term, domain, subject, include_attributes, max_results_per_term, per_page)
                all_results.extend(term_results)

            # Remove duplicates based on subject:model combination and aggregate relevance scores
            model_aggregates = {}
            for result in all_results:
                key = f"{result['subject']}:{result['model']}"
                if key not in model_aggregates:
                    model_aggregates[key] = {
                        "result": result,
                        "term_matches": 1,
                        "total_score": result["relevance_score"]
                    }
                else:
                    # Aggregate scores for models found by multiple terms
                    model_aggregates[key]["term_matches"] += 1
                    model_aggregates[key]["total_score"] += result["relevance_score"]

            # Convert back to results with adjusted scores
            unique_results = []
            for aggregate in model_aggregates.values():
                result = aggregate["result"].copy()
                term_matches = aggregate["term_matches"]
                total_score = aggregate["total_score"]

                # Boost score for models matching multiple terms
                if term_matches > 1:
                    result["relevance_score"] = round(total_score * (1 + 0.2 * (term_matches - 1)), 2)  # 20% boost per additional term match
                    result["matched_terms"] = term_matches
                else:
                    result["relevance_score"] = round(total_score, 2)

                unique_results.append(result)

            # Sort by relevance score (descending)
            unique_results.sort(key=lambda x: x["relevance_score"], reverse=True)

            # Limit final results
            max_final_results = 50
            final_results = unique_results[:max_final_results]

            logger.info(f"GitHub Code Search: Completed - searched {len(query_terms)} terms, found {len(final_results)} unique models")
            return final_results

        except Exception as e:
            logger.error(f"GitHub code search failed: {e}")
            return []

    async def _search_single_term_github(
        self,
        term: str,
        domain: Optional[str],
        subject: Optional[str],
        include_attributes: bool,
        max_results: int,
        per_page: int
    ) -> List[Dict[str, Any]]:
        """Search GitHub for a single term.

        Args:
            term: Single search term
            domain: Optional domain filter
            subject: Optional subject filter
            include_attributes: Whether to include attribute details
            max_results: Maximum results to return for this term
            per_page: Results per page for GitHub API

        Returns:
            List of search results for this term
        """
        try:
            # Build search query for GitHub API
            search_query = f"org:{self.SMART_DATA_MODELS_ORG}"

            # Add subject filter if specified
            if subject:
                # Normalize subject name
                normalized_subject = self._normalize_subject(subject)
                if normalized_subject.startswith("dataModel."):
                    subject_short = normalized_subject[10:]  # Remove "dataModel." prefix
                    search_query += f" repo:{self.SMART_DATA_MODELS_ORG}/dataModel.{subject_short}"

            # Add the search term
            search_query += f" {term}"

            logger.debug(f"GitHub search query for term '{term}': {search_query}")

            results = []

            # GitHub search API pagination
            page = 1

            while len(results) < max_results:
                try:
                    # GitHub code search API endpoint
                    search_url = "https://api.github.com/search/code"
                    params = {
                        "q": search_query,
                        "type": "code",
                        "per_page": min(per_page, max_results - len(results)),
                        "page": page
                    }

                    response = await self._run_sync_in_thread(
                        lambda: self._session.get(search_url, params=params, timeout=30)
                    )

                    if response.status_code == 403:
                        logger.warning("GitHub API rate limit exceeded - consider using GITHUB_READ_TOKEN")
                        break
                    elif response.status_code != 200:
                        logger.error(f"GitHub search API returned status {response.status_code}: {response.text}")
                        break

                    search_data = response.json()
                    items = search_data.get("items", [])

                    if not items:
                        break  # No more results

                    # Process search results for this term
                    page_results = await self._process_github_search_results(items, term, include_attributes)
                    results.extend(page_results)

                    # Check if we have enough results or if there are more pages
                    if len(results) >= max_results or len(items) < per_page:
                        break

                    page += 1

                    # Safety check to avoid infinite loops
                    if page > 5:  # Fewer pages per term
                        logger.warning(f"GitHub search for term '{term}' exceeded 5 pages, stopping")
                        break

                except Exception as e:
                    logger.error(f"Error during GitHub search API call for term '{term}' (page {page}): {e}")
                    break

            logger.debug(f"Found {len(results)} results for term '{term}'")
            return results[:max_results]

        except Exception as e:
            logger.error(f"GitHub search failed for term '{term}': {e}")
            return []

    async def _process_github_search_results(
        self,
        search_items: List[Dict[str, Any]],
        original_query: str,
        include_attributes: bool
    ) -> List[Dict[str, Any]]:
        """Process GitHub search API results to extract model information.

        Args:
            search_items: List of items from GitHub search API response
            original_query: The original search query for relevance scoring
            include_attributes: Whether to include attribute details

        Returns:
            List of processed model results
        """
        results = []
        query_lower = original_query.lower()
        processed_models = set()  # Track processed models to avoid duplicates

        for item in search_items:
            try:
                repo_info = item.get("repository", {})
                repo_name = repo_info.get("name", "")

                # Only process dataModel.* repositories
                if not repo_name.startswith("dataModel."):
                    continue

                # Extract subject from repository name
                subject = repo_name
                if not subject.startswith("dataModel."):
                    continue

                # Extract model from path
                path = item.get("path", "")
                path_parts = path.split("/")

                # Look for model directory in path (e.g., "TrafficFlowObserved/README.md" or "TrafficFlowObserved/schema.json")
                model_name = None
                if len(path_parts) >= 2:
                    # Check if first part looks like a model directory
                    potential_model = path_parts[0]
                    if potential_model and not potential_model.startswith(".") and len(potential_model) > 2:
                        model_name = potential_model

                if not model_name:
                    continue

                # Skip if we've already processed this model in this batch
                model_key = f"{subject}:{model_name}"
                if model_key in processed_models:
                    continue
                processed_models.add(model_key)

                # Get model details and file content for better matching
                try:
                    model_details = await self._get_basic_model_details_from_github(subject, model_name, subject)
                    if not model_details:
                        # Fallback: create basic model info
                        model_details = {
                            "subject": subject,
                            "model": model_name,
                            "description": f"Smart Data Model for {model_name}",
                            "attributes": [],
                            "source": f"github_code_search"
                        }

                    # Get file content to search for query terms - extract relative path within model directory
                    # path format: "TrafficFlowObserved/README.md" -> relative_path: "README.md"
                    # path format: "TrafficFlowObserved/doc/spec.md" -> relative_path: "doc/spec.md"
                    # path format: "TrafficFlowObserved/examples/example.json" -> relative_path: "examples/example.json"
                    if '/' in path:
                        # Remove model name prefix to get relative path within model directory
                        path_parts = path.split('/')
                        if len(path_parts) > 1 and path_parts[0] == model_name:
                            relative_path = '/'.join(path_parts[1:])
                        else:
                            # Fallback to just filename if path structure is unexpected
                            relative_path = path_parts[-1]
                    else:
                        relative_path = path

                    file_content = await self._get_file_content_from_github(subject, model_name, relative_path)
                    content_lower = file_content.lower() if file_content else ""

                    # Calculate relevance score based on query match
                    relevance_score = 0
                    matched_parts = []

                    # Model name match
                    if query_lower in model_name.lower():
                        relevance_score += 3.0
                        matched_parts.append("name")

                    # Description match
                    desc = model_details.get("description", "")
                    if query_lower in desc.lower():
                        relevance_score += 1.0
                        matched_parts.append("description")

                    # File content match (search in README, schema, etc.)
                    if query_lower in content_lower:
                        relevance_score += 2.0
                        matched_parts.append("content")

                    # Attribute matches
                    if include_attributes:
                        attributes = model_details.get("attributes", [])
                        for attr in attributes:
                            attr_name = attr.get("name", "").lower()
                            attr_desc = attr.get("description", "").lower()
                            if query_lower in attr_name or query_lower in attr_desc:
                                relevance_score += 1.5
                                matched_parts.append("attributes")
                                break

                    # Only include if there's some relevance
                    if relevance_score > 0:
                        model_result = {
                            "subject": subject,
                            "model": model_name,
                            "description": model_details.get("description", ""),
                            "relevance_score": round(relevance_score, 2),
                            "matched_parts": matched_parts,
                            "source": "github_code_search"
                        }

                        if include_attributes and model_details.get("attributes"):
                            model_result["attributes"] = model_details["attributes"]

                        results.append(model_result)

                except Exception as e:
                    logger.debug(f"Failed to get details for model {subject}/{model_name}: {e}")
                    continue

            except Exception as e:
                logger.debug(f"Error processing search item: {e}")
                continue

        return results

    async def _search_github_excluding_pysmartdatamodels(
        self,
        query: str,
        domain: Optional[str] = None,
        subject: Optional[str] = None,
        include_attributes: bool = False
    ) -> List[Dict[str, Any]]:
        """Search GitHub API but exclude models that already exist in pysmartdatamodels.

        This ensures we don't duplicate results and focus on potentially newer/missing models.
        Uses GitHub's global code search API for efficiency instead of repo-by-repo enumeration.
        """
        try:
            # Get list of models that already exist in pysmartdatamodels
            existing_models = set()
            try:
                # Get all subjects and then get models for each subject
                all_subjects = await self._run_sync_in_thread(list_all_subjects)
                if all_subjects:
                    for subject_name in all_subjects:
                        try:
                            models_in_subject = await self._run_sync_in_thread(datamodels_subject, subject_name)
                            if models_in_subject:
                                for model_name in models_in_subject:
                                    existing_models.add(f"{subject_name}:{model_name}")
                        except Exception as e:
                            logger.debug(f"Failed to get models for subject {subject_name}: {e}")
                            continue
            except Exception as e:
                logger.debug(f"Failed to get existing models from pysmartdatamodels: {e}")

            logger.info(f"Excluding {len(existing_models)} existing models from GitHub search")

            # Use new efficient GitHub code search API instead of comprehensive repo search
            all_github_results = await self._search_github_with_code_api(query, domain, subject, include_attributes)

            # Filter out models that already exist in pysmartdatamodels
            filtered_results = []
            for result in all_github_results:
                model_key = f"{result['subject']}:{result['model']}"
                if model_key not in existing_models:
                    # Mark as GitHub-only result
                    result["source"] = f"github (not in pysmartdatamodels)"
                    filtered_results.append(result)

            logger.info(f"GitHub search (excluding pysmartdatamodels) found {len(filtered_results)} additional results")
            return filtered_results

        except Exception as e:
            logger.warning(f"GitHub search excluding pysmartdatamodels failed: {e}")
            return []













    async def get_model_details(self, subject: str, model: str) -> Dict[str, Any]:
        """Get detailed information about a specific model.

        Args:
            subject (str): The name of the subject the model belongs to.
            model (str): The name of the model to retrieve details for.

        Returns:
            Dict[str, Any]: A dictionary containing detailed information about the model,
                            including its description, attributes, and source.
        """
        logger.info("Get Model Details: Starting - subject='%s', model='%s' - strategies: GitHub analyzer -> GitHub basic -> pysmartdatamodels -> fallback", subject, model)
        cache_key = f"model_details_{subject}_{model}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Get Model Details: Returning cached data for %s/%s", subject, model)
            return cached

        # Normalize subject for consistent processing
        normalized_subject = self._normalize_subject(subject)

        # Try to use GitHub analyzer to get metadata
        try:
            logger.debug("Get Model Details: Attempting GitHub analyzer for %s/%s", normalized_subject, model)
            analyzer = EmbeddedGitHubAnalyzer()
            # Remove 'dataModel.' prefix for analyzer URL construction
            analyzer_subject = normalized_subject[10:] if normalized_subject.startswith("dataModel.") else normalized_subject
            repo_url = f"https://smart-data-models.github.io/dataModel.{analyzer_subject}/{model}"

            # Run the synchronous analyzer in thread
            metadata = await self._run_sync_in_thread(analyzer.generate_metadata, repo_url)

            if metadata:
                logger.info("Get Model Details: GitHub analyzer success for %s/%s - source: github_analyzer", normalized_subject, model)
                # Convert GitHub analyzer format to the expected format
                processed_details = {
                    "subject": normalized_subject,  # Use normalized subject
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
                    schema = await self.get_model_schema(normalized_subject, model)
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
            logger.debug(f"GitHub analyzer failed for {normalized_subject}/{model}: {e}")
            logger.info("Get Model Details: GitHub analyzer failed for %s/%s - trying fallback GitHub basic", normalized_subject, model)

        # Fallback 1: Try basic GitHub details
        try:
            details = await self._get_basic_model_details_from_github(normalized_subject, model)
            if details:
                logger.info("Get Model Details: GitHub basic success for %s/%s - source: %s", normalized_subject, model, details.get("source", "unknown"))
                self._cache.set(cache_key, details)
                return details
        except Exception as e:
            logger.debug(f"Basic GitHub details fallback failed for {normalized_subject}/{model}: {e}")
            logger.info("Get Model Details: GitHub basic failed for %s/%s - trying pysmartdatamodels", normalized_subject, model)

        # Fallback 2: Try pysmartdatamodels as final fallback
        try:
            # Import the functions directly
            from pysmartdatamodels import (
                attributes_datamodel, description_attribute, datatype_attribute,
                ngsi_datatype_attribute, units_attribute, model_attribute,
                list_datamodel_metadata
            )

            # Get attributes list for this model
            attributes_list = await self._run_sync_in_thread(attributes_datamodel, normalized_subject, model)

            if attributes_list and isinstance(attributes_list, list):
                attributes = []

                # For each attribute, get its details
                for attr_name in attributes_list[:50]:  # Limit to avoid too many calls
                    try:
                        # Get description, data type, and NGSI type for each attribute
                        attr_desc = await self._run_sync_in_thread(description_attribute, normalized_subject, model, attr_name)
                        attr_type = await self._run_sync_in_thread(datatype_attribute, normalized_subject, model, attr_name)
                        ngsi_type = await self._run_sync_in_thread(ngsi_datatype_attribute, normalized_subject, model, attr_name)
                        attr_units = await self._run_sync_in_thread(units_attribute, normalized_subject, model, attr_name)
                        attr_model = await self._run_sync_in_thread(model_attribute, normalized_subject, model, attr_name)

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
                metadata = await self._run_sync_in_thread(list_datamodel_metadata, model, normalized_subject)
                if metadata and isinstance(metadata, dict):
                    # Extract information from metadata
                    processed_details = {
                        "subject": normalized_subject,
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
                        "subject": normalized_subject,
                        "model": model,
                        "description": f"Smart Data Model for {model} in {normalized_subject} subject",
                        "attributes": attributes,
                        "source": "pysmartdatamodels"
                    }
                    self._cache.set(cache_key, processed_details)
                    return processed_details

        except Exception as e:
            logger.debug(f"pysmartdatamodels details construction failed for {normalized_subject}/{model}: {e}")

        # Ultimate fallback
        details = {
            "subject": normalized_subject,
            "model": model,
            "description": f"Smart Data Model for {model} in {normalized_subject} subject",
            "attributes": [],
            "source": "fallback"
        }

        self._cache.set(cache_key, details)
        return details

    async def _get_file_content_from_github(self, subject: str, model: str, file_path: str) -> Optional[str]:
        """Get the content of a specific file from GitHub repository.

        Args:
            subject: Subject name (may include 'dataModel.' prefix)
            model: Model name
            file_path: Path to the file within the model directory

        Returns:
            File content as string, or None if not found
        """
        try:
            # Remove 'dataModel.' prefix if present to construct correct repo name
            actual_subject = subject.replace('dataModel.', '') if subject.startswith('dataModel.') else subject
            repo_name = f"dataModel.{actual_subject}"

            # Construct the raw GitHub URL for the file
            file_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_ORG}/{repo_name}/master/{model}/{file_path}"
            logger.debug(f"Fetching file content from: {file_url}")

            response = await self._run_sync_in_thread(
                self._session.get, file_url, timeout=30
            )

            if response.status_code == 200:
                return response.text
            else:
                logger.debug(f"File not found at {file_url} (status: {response.status_code})")
                return None

        except Exception as e:
            logger.debug(f"Failed to get file content for {subject}/{model}/{file_path}: {e}")
            return None

    async def _get_basic_model_details_from_github(self, subject: str, model: str, repo_subject: Optional[str] = None) -> Optional[Dict[str, Any]]:
        """Get basic model details from GitHub repository.
        repo_subject allows overriding the subject used for constructing the GitHub repository name.
        """
        try:
            actual_repo_subject = repo_subject if repo_subject else subject
            # Remove 'dataModel.' prefix if present to construct correct repo name
            if actual_repo_subject.startswith("dataModel."):
                actual_repo_subject = actual_repo_subject[10:]
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

        # Get from GitHub - strip 'dataModel.' prefix if present for repo name construction
        actual_repo_subject = repo_subject if repo_subject else subject
        actual_repo_subject = actual_repo_subject.replace('dataModel.', '') if actual_repo_subject.startswith('dataModel.') else actual_repo_subject
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
        logger.info("Get Model Examples: Starting - subject='%s', model='%s' - strategies: pysmartdatamodels -> GitHub -> generation", subject, model)
        # Normalize subject parameter
        subject = self._normalize_subject(subject)
        repo_subject = self._normalize_subject(repo_subject)

        cache_key = f"examples_{subject}_{model}_{repo_subject}"

        # Try cache first
        cached = self._cache.get(cache_key)
        if cached:
            logger.debug("Get Model Examples: Returning cached data for %s/%s", subject, model)
            return cached

        # Try pysmartdatamodels with schema URL (it requires a full schema URL, not subject/model)
        try:
            logger.debug("Get Model Examples: Attempting pysmartdatamodels for %s/%s", subject, model)
            # Construct the schema URL first
            actual_repo_subject = repo_subject if repo_subject else subject
            repo_name = f"dataModel.{actual_repo_subject}"
            schema_url = f"https://raw.githubusercontent.com/{self.SMART_DATA_MODELS_ORG}/{repo_name}/master/{model}/schema.json"

            examples = await self._run_sync_in_thread(ngsi_ld_example_generator, schema_url)

            if examples and examples != "dataModel" and examples != "False":
                logger.info("Get Model Examples: pysmartdatamodels success for %s/%s - %s examples generated", subject, model, len(examples) if isinstance(examples, list) else 1)
                if isinstance(examples, dict):
                    examples = [examples]
                elif isinstance(examples, list):
                    examples = [examples] if examples else []

                self._cache.set(cache_key, examples)
                return examples

        except Exception as e:
            logger.debug(f"pysmartdatamodels examples failed for {subject}/{model}: {e}")
            logger.info("Get Model Examples: pysmartdatamodels failed for %s/%s - trying GitHub", subject, model)

        # Fallback: try to get from GitHub examples
        try:
            # Denormalize subject for GitHub API call (remove dataModel. prefix)
            denormalized_subject = subject[10:] if subject.startswith("dataModel.") else subject
            examples = await self._get_examples_from_github(denormalized_subject, model, repo_subject=None)
            if examples:
                logger.info("Get Model Examples: GitHub success for %s/%s - %s examples found", subject, model, len(examples))
                self._cache.set(cache_key, examples)
                return examples
        except Exception as e:
            logger.debug(f"GitHub examples failed for {subject}/{model}: {e}")
            logger.info("Get Model Examples: GitHub failed for %s/%s - trying basic generation", subject, model)

        # Ultimate fallback: generate basic example
        example = await self._generate_basic_example(subject, model)
        examples = [example] if example else []
        if examples:
            logger.info("Get Model Examples: Basic generation successful for %s/%s", subject, model)
        else:
            logger.warning(f"Get Model Examples: Basic generation failed for {subject}/{model} - no examples available")

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
                "examples/example.json",
                "examples/example.jsonld"
            ]

            for path in example_paths:
                example_url = f"{self.GITHUB_RAW_BASE}/{self.SMART_DATA_MODELS_ORG}/{repo_name}/master/{model}/{path}"
                logger.debug(f"Trying example URL: {example_url}")
                try:
                    response = await self._run_sync_in_thread(
                        self._session.get, example_url, timeout=30
                    )
                    logger.debug(f"Response status for {example_url}: {response.status_code}")

                    if response.status_code == 200:
                        try:
                            example = response.json()
                            logger.info(f"Found valid example at {path}")
                            return [example] if isinstance(example, dict) else example
                        except json.JSONDecodeError as e:
                            logger.debug(f"Failed to parse JSON from {example_url}: {e}")
                            continue
                except Exception as e:
                    logger.debug(f"HTTP error for {example_url}: {e}")
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
                    # Skip id and type as they are already set at the top level
                    if attr_name in ["id", "type"]:
                        continue

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

        # Try to get from GitHub - strip 'dataModel.' prefix if present for repo name construction
        actual_subject = subject.replace('dataModel.', '') if subject.startswith('dataModel.') else subject
        repo_name = f"dataModel.{actual_subject}"
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

    async def suggest_matching_models(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Suggest models that match the provided data structure using enhanced algorithm with GitHub pre-filtering.

        Enhanced matching strategy with performance optimization:
        1. Extract entity type and attribute names from input data
        2. Use existing GitHub search functions to find models containing these attributes (fast pre-filtering)
        3. For promising models, use pysmartdatamodels functions for detailed analysis:
           - attributes_datamodel() for attribute lists
           - description_attribute() for model descriptions
           - subject_for_datamodel() for subject lookup
           - load_all_attributes() for comprehensive attribute search
        4. Apply enhanced semantic matching with weighted scoring
        5. Return top matches with detailed similarity analysis

        This approach avoids scanning all ~500+ models by pre-filtering with existing GitHub functions.

        Args:
            data (Dict[str, Any]): The data structure (as a dictionary) to compare against models.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, each containing a suggested model,
                                   its similarity score, and matched attributes.
        """
        if not isinstance(data, dict):
            return []

        data_keys = set(data.keys())

        # Early return for empty data - no meaningful matches possible
        if not data_keys:
            return []

        logger.info("Suggest Matching Models: Using enhanced algorithm with GitHub pre-filtering")

        # Extract entity type for exact matching boost
        entity_type = data.get("type", "").strip()
        logger.debug(f"Entity type from data: '{entity_type}'")

        # Step 1: Use existing GitHub search to pre-filter promising models
        logger.info("Step 1: Pre-filtering models using existing GitHub functions")
        candidate_models = await self._prefilter_models_with_existing_github(data_keys, entity_type)

        if not candidate_models:
            logger.warning("No candidate models found via GitHub search, falling back to limited pysmartdatamodels search")
            # Fallback: try some common subjects
            candidate_models = await self._fallback_model_candidates(data_keys, entity_type)

        logger.info(f"Found {len(candidate_models)} candidate models for detailed analysis")

        # Step 2: Detailed analysis of candidate models using pysmartdatamodels
        logger.info("Step 2: Detailed analysis using pysmartdatamodels functions")
        detailed_results = await self._analyze_candidate_models(candidate_models, data, data_keys, entity_type)

        # Step 3: Sort by similarity and return top results
        detailed_results.sort(key=lambda x: x["similarity"], reverse=True)
        final_results = detailed_results[:10]

        logger.info(f"Suggest Matching Models: Analyzed {len(candidate_models)} models, found {len(final_results)} matching models")
        return final_results

    async def _enhanced_score_model_for_suggestion(self, model_name: str, subject: str, data: Dict[str, Any], data_keys: set, entity_type: str) -> Dict[str, Any]:
        """Enhanced scoring for model suggestion with improved attribute matching and semantic analysis.

        Args:
            model_name: Name of the model to score
            subject: Subject the model belongs to
            data: Original input data dictionary
            data_keys: Set of attribute names from the input data
            entity_type: Entity type from data (for exact matching boost)

        Returns:
            Dict with model info and enhanced similarity score, or empty dict if no match
        """
        try:
            # Get attributes for this model
            model_attrs_list = await self._run_sync_in_thread(attributes_datamodel, subject, model_name)

            if not model_attrs_list or not isinstance(model_attrs_list, list):
                return {}

            model_attrs = set(model_attrs_list)

            # Initialize scoring components
            base_similarity = 0
            type_boost = 0
            semantic_score = 0
            fuzzy_score = 0

            # 1. Exact type matching boost (highest priority)
            if entity_type and model_name.lower() == entity_type.lower():
                type_boost = 5.0  # Massive boost for exact type match
                logger.debug(f"Exact type match boost: {model_name} matches '{entity_type}'")

            # 2. Enhanced semantic matching for common attribute patterns
            matched_count = 0
            semantic_matches = self._calculate_semantic_matches(data_keys, model_attrs)
            semantic_score = semantic_matches['score']
            matched_count = semantic_matches['count']

            # 3. Calculate base attribute overlap similarity (exact matches)
            exact_overlap = len(data_keys.intersection(model_attrs))
            total_attrs = len(model_attrs.union(data_keys))

            if total_attrs == 0:
                return {}

            base_similarity = exact_overlap / total_attrs

            # 4. Fuzzy matching with improved algorithm
            fuzzy_matches = self._calculate_fuzzy_matches(data_keys, model_attrs)
            fuzzy_score = fuzzy_matches * 0.3  # Reduced weight for fuzzy matches

            # 5. Model name semantic matching (check if model name contains relevant keywords)
            model_name_score = self._calculate_model_name_relevance(model_name, data_keys)
            model_name_score *= 0.5  # Weight for model name relevance

            # 6. Calculate final similarity score with improved weighting
            final_similarity = (
                base_similarity * 2.0 +      # Exact matches (highest weight)
                type_boost +                 # Type matching boost
                semantic_score +             # Semantic matches
                fuzzy_score +                # Fuzzy matches
                model_name_score             # Model name relevance
            )

            # Only return if there's reasonable similarity (lower threshold for enhanced algorithm)
            if final_similarity > 0.1:  # Increased threshold for better quality
                # Get description for the model
                try:
                    description = await self._run_sync_in_thread(
                        description_attribute, subject, model_name, "id"
                    )
                    if not description:
                        description = f"Smart Data Model for {model_name}"
                except Exception:
                    description = f"Smart Data Model for {model_name}"

                result = {
                    "subject": subject,
                    "model": model_name,
                    "similarity": round(final_similarity, 3),
                    "matched_attributes": matched_count,  # Use semantic matches count
                    "total_attributes": len(model_attrs),
                    "description": description,
                    "source": "pysmartdatamodels"
                }

                logger.debug(f"Enhanced scoring result for {subject}/{model_name}: similarity={result['similarity']}, base={base_similarity:.3f}, type_boost={type_boost}, semantic={semantic_score:.3f}, fuzzy={fuzzy_score:.3f}, model_name={model_name_score:.3f}")

                return result

        except Exception as e:
            logger.debug(f"Error in enhanced scoring for model {model_name}: {e}")

        return {}

    def _calculate_semantic_matches(self, data_keys: set, model_attrs: set) -> Dict[str, float]:
        """Calculate semantic matches using common attribute patterns and synonyms."""
        score = 0
        matched_groups = set()

        # Define semantic groups for common attributes
        semantic_groups = {
            'temperature': {'temperature', 'temp', 'airTemperature', 'waterTemperature', 'soilTemperature'},
            'humidity': {'humidity', 'relativeHumidity', 'airHumidity', 'moisture'},
            'pressure': {'pressure', 'atmosphericPressure', 'barometricPressure', 'airPressure'},
            'wind': {'windSpeed', 'windDirection', 'windGust', 'wind'},
            'precipitation': {'precipitation', 'rain', 'rainfall', 'snow', 'hail'},
            'location': {'location', 'position', 'coordinates', 'latitude', 'longitude', 'address'},
            'time': {'dateObserved', 'timestamp', 'dateCreated', 'dateModified', 'time'},
            'id': {'id', 'identifier', 'name', 'code'},
            'type': {'type', 'category', 'class'}
        }

        # Find which semantic groups are present in the data
        data_groups = set()
        for data_attr in data_keys:
            data_attr_lower = data_attr.lower()
            for group_name, synonyms in semantic_groups.items():
                if data_attr_lower in synonyms or any(syn in data_attr_lower for syn in synonyms):
                    data_groups.add(group_name)
                    break

        # For each semantic group present in data, check if model has matching attributes
        for group_name in data_groups:
            group_attrs = semantic_groups[group_name]
            model_has_match = any(
                any(synonym in model_attr.lower() for synonym in group_attrs)
                for model_attr in model_attrs
            )

            if model_has_match:
                score += 1.0  # One point per matched semantic group
                matched_groups.add(group_name)

        return {'score': score, 'count': len(matched_groups)}

    def _calculate_fuzzy_matches(self, data_keys: set, model_attrs: set) -> float:
        """Calculate fuzzy matches using string similarity."""
        fuzzy_score = 0

        for data_attr in data_keys:
            data_attr_lower = data_attr.lower()
            best_match_score = 0

            for model_attr in model_attrs:
                model_attr_lower = model_attr.lower()

                # Skip exact matches (already counted in base similarity)
                if data_attr_lower == model_attr_lower:
                    continue

                # Calculate similarity score
                if data_attr_lower in model_attr_lower or model_attr_lower in data_attr_lower:
                    # Containment match
                    match_score = 0.8
                elif len(set(data_attr_lower) & set(model_attr_lower)) / len(set(data_attr_lower) | set(model_attr_lower)) > 0.6:
                    # High character overlap
                    match_score = 0.6
                else:
                    # Check for common prefixes/suffixes
                    common_prefix = 0
                    for i in range(1, min(len(data_attr_lower), len(model_attr_lower)) + 1):
                        if data_attr_lower[:i] == model_attr_lower[:i]:
                            common_prefix = i
                        else:
                            break

                    if common_prefix >= 3:  # At least 3 characters in common prefix
                        match_score = min(0.5, common_prefix / len(data_attr_lower))
                    else:
                        continue

                best_match_score = max(best_match_score, match_score)

            fuzzy_score += best_match_score

        return fuzzy_score

    def _calculate_model_name_relevance(self, model_name: str, data_keys: set) -> float:
        """Calculate relevance based on model name containing data attribute keywords."""
        model_lower = model_name.lower()
        relevance_score = 0

        # Keywords that suggest weather/environmental data
        weather_keywords = {'weather', 'climate', 'meteorological', 'atmospheric', 'environmental', 'air', 'sensor'}

        # Check if model name contains weather-related keywords
        for keyword in weather_keywords:
            if keyword in model_lower:
                relevance_score += 0.5
                break

        # Check if model name contains data attribute names
        for data_attr in data_keys:
            if data_attr.lower() in model_lower:
                relevance_score += 0.3

        return relevance_score

    async def _prefilter_models_with_existing_github(self, data_keys: set, entity_type: str) -> List[Tuple[str, str]]:
        """Pre-filter models using existing GitHub functions to find promising candidates.

        Uses existing GitHub functions like _get_basic_model_details_from_github and
        _get_file_content_from_github to efficiently find models that might match the data.
        """
        candidates = []
        processed_models = set()

        # Strategy 1: Search for models containing the entity type in their name
        if entity_type:
            logger.debug(f"Searching for models with entity type: {entity_type}")
            # Get some common subjects to search in
            try:
                subjects = await self.list_subjects()
                # Limit to first few subjects for performance
                for subject in subjects[:5]:
                    try:
                        models = await self.list_models_in_subject(subject)
                        for model_name in models:
                            if entity_type.lower() in model_name.lower():
                                if (subject, model_name) not in processed_models:
                                    candidates.append((subject, model_name))
                                    processed_models.add((subject, model_name))
                    except Exception as e:
                        logger.debug(f"Error searching subject {subject}: {e}")
                        continue
            except Exception as e:
                logger.debug(f"Error getting subjects: {e}")

        # Strategy 2: Search for models containing data attribute names using existing GitHub functions
        for attr in data_keys:
            if len(attr) < 3:  # Skip very short attribute names
                continue

            logger.debug(f"Searching for models containing attribute: {attr}")
            # Use existing GitHub functions to find models with this attribute
            # Get some subjects and check their models
            try:
                subjects = await self.list_subjects()
                for subject in subjects[:3]:  # Limit subjects for performance
                    try:
                        models = await self.list_models_in_subject(subject)
                        for model_name in models[:10]:  # Limit models per subject
                            if (subject, model_name) in processed_models:
                                continue

                            # Use existing GitHub function to get model details
                            try:
                                details = await self._get_basic_model_details_from_github(subject, model_name)
                                if details and details.get("attributes"):
                                    model_attrs = {attr_info["name"] for attr_info in details["attributes"]}
                                    if attr in model_attrs:
                                        candidates.append((subject, model_name))
                                        processed_models.add((subject, model_name))
                            except Exception as e:
                                logger.debug(f"Error checking model {subject}/{model_name}: {e}")
                                continue
                    except Exception as e:
                        logger.debug(f"Error in subject {subject}: {e}")
                        continue
            except Exception as e:
                logger.debug(f"Error getting subjects for attribute {attr}: {e}")

        # Remove duplicates (shouldn't be any due to processed_models set, but just in case)
        unique_candidates = list(set(candidates))

        logger.info(f"GitHub pre-filtering found {len(unique_candidates)} candidate models")
        return unique_candidates

    async def _analyze_candidate_models(self, candidate_models: List[Tuple[str, str]], data: Dict[str, Any], data_keys: set, entity_type: str) -> List[Dict[str, Any]]:
        """Analyze candidate models using pysmartdatamodels for detailed scoring."""
        results = []

        for subject, model_name in candidate_models:
            try:
                result = await self._enhanced_score_model_for_suggestion(model_name, subject, data, data_keys, entity_type)
                if result:
                    results.append(result)
            except Exception as e:
                logger.debug(f"Error analyzing candidate model {subject}/{model_name}: {e}")
                continue

        return results

    async def _fallback_model_candidates(self, data_keys: set, entity_type: str) -> List[Tuple[str, str]]:
        """Fallback method to generate some candidate models when GitHub search fails."""
        candidates = []

        # Try some common subjects that might contain weather/environmental data
        common_subjects = ["dataModel.Environment", "dataModel.Weather", "dataModel.Sensor"]

        for subject in common_subjects:
            try:
                models = await self.list_models_in_subject(subject)
                # Add first few models from each subject
                for model_name in models[:5]:
                    candidates.append((subject, model_name))
            except Exception as e:
                logger.debug(f"Error getting models for subject {subject}: {e}")
                continue

        logger.info(f"Fallback generated {len(candidates)} candidate models")
        return candidates

    async def _fallback_suggest_matching_models(self, data: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Fallback implementation using the original inefficient approach.

        This method is kept for compatibility when pysmartdatamodels is not available.
        """
        logger.warning("Using fallback suggest_matching_models (inefficient GitHub scanning)")

        data_keys = set(data.keys())
        candidates = []

        # Original inefficient approach - scan all subjects and models
        subjects = await self.list_subjects()

        for subject in subjects[:10]:  # Limit subjects in fallback
            try:
                normalized_subject = self._normalize_subject(subject)
                models = await self.list_models_in_subject(normalized_subject)

                for model_name in models[:20]:  # Limit models per subject in fallback
                    try:
                        details = await self.get_model_details(subject, model_name)

                        if "attributes" in details:
                            model_attrs = {attr["name"] for attr in details["attributes"]}
                            overlap = len(data_keys.intersection(model_attrs))
                            total_attrs = len(model_attrs.union(data_keys))

                            if total_attrs > 0:
                                similarity = overlap / total_attrs
                                if similarity > 0.1:
                                    candidates.append({
                                        "subject": subject,
                                        "model": model_name,
                                        "similarity": round(similarity, 3),
                                        "matched_attributes": overlap,
                                        "total_attributes": len(model_attrs),
                                        "description": details.get("description", ""),
                                        "source": "fallback_github"
                                    })

                    except Exception as e:
                        logger.debug(f"Error checking model {subject}/{model_name}: {e}")
                        continue

            except Exception as e:
                logger.debug(f"Error in subject {subject}: {e}")
                continue

        candidates.sort(key=lambda x: x["similarity"], reverse=True)
        return candidates[:10]
