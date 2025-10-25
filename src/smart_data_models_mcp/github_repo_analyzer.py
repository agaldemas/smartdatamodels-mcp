#!/usr/bin/env python3
"""
GitHub Repository Analyzer for Smart Data Models

This tool analyzes a GitHub repository from the smart-data-models organization
and generates metadata JSON based on the repository structure and contents.

Usage:
    python github_repo_analyzer.py "https://smart-data-models.github.io/dataModel.User/Activity"
"""

import json
import sys
from urllib.parse import urlparse
from typing import Dict, Any, Optional
import requests


# GitHub analyzer functionality for Smart Data Models

import json
import requests
from typing import Dict, Any, Optional
from urllib.parse import urlparse


class GitHubRepoAnalyzer:
    """Analyzer for smart-data-models GitHub repositories."""

    GITHUB_API_BASE = "https://api.github.com"
    GITHUB_RAW_BASE = "https://raw.githubusercontent.com"

    def __init__(self):
        self.session = requests.Session()

    def parse_url(self, repo_url: str) -> Dict[str, str]:
        """
        Parse the smart-data-models repo URL to extract subject and dataModel.

        Args:
            repo_url: URL like "https://smart-data-models.github.io/dataModel.User/Activity"

        Returns:
            Dict with 'subject' and 'dataModel' keys
        """
        parsed = urlparse(repo_url)
        if 'smart-data-models.github.io' not in repo_url:
            raise ValueError("Invalid smart-data-models URL format")

        # For URL like smart-data-models.github.io/dataModel.User/Activity
        # The path is /dataModel.User/Activity
        path_parts = parsed.path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("URL path too short")

        subject = path_parts[0]  # e.g., "dataModel.User"
        data_model = path_parts[1]  # e.g., "Activity"

        return {
            'subject': subject,
            'dataModel': data_model
        }

    def get_repo_contents(self, subject: str, path: str = "") -> Optional[Dict[str, Any]]:
        """
        Get repository contents using GitHub API.

        Args:
            subject: Repository name (e.g., "dataModel.User")
            path: Path within repository (optional)

        Returns:
            Repository contents or None if not found
        """
        url = f"{self.GITHUB_API_BASE}/repos/smart-data-models/{subject}/contents"
        if path:
            url += f"/{path}"

        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.json()
        except requests.exceptions.RequestException as e:
            print(f"Error accessing GitHub API: {e}")
            return None

    def get_file_content(self, subject: str, path: str) -> Optional[str]:
        """
        Get raw file content from GitHub.

        Args:
            subject: Repository name
            path: File path within repository

        Returns:
            File content as string or None if not found
        """
        url = f"{self.GITHUB_RAW_BASE}/smart-data-models/{subject}/main/{path}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException:
            return None

    def extract_description(self, subject: str, data_model: str) -> str:
        """
        Extract description from schema.json file.

        Args:
            subject: Repository name
            data_model: Data model name

        Returns:
            Description from schema.json or default
        """
        schema_path = f"{data_model}/schema.json"
        content = self.get_file_content(subject, schema_path)

        if content:
            try:
                schema = json.loads(content)
                return schema.get('description', f'Information on {data_model} data')
            except json.JSONDecodeError:
                pass

        return f"Information on the {data_model} performed by an entity"

    def extract_version(self, subject: str, data_model: str) -> str:
        """
        Extract version from schema.json or VERSION file.

        Args:
            subject: Repository name
            data_model: Data model name

        Returns:
            Version string or default "0.1.0"
        """
        # First try to get from schema.json
        schema_path = f"{data_model}/schema.json"
        content = self.get_file_content(subject, schema_path)

        if content:
            try:
                schema = json.loads(content)
                if 'version' in schema:
                    return schema['version']
            except json.JSONDecodeError:
                pass

        # Try VERSION file
        version_content = self.get_file_content(subject, "VERSION")
        if version_content:
            return version_content.strip()

        return "0.1.1"

    def generate_metadata(self, repo_url: str) -> Dict[str, Any]:
        """
        Generate metadata JSON for the given repository URL.

        Args:
            repo_url: GitHub repository URL

        Returns:
            Metadata dictionary
        """
        # Parse URL to get subject and dataModel
        parsed = self.parse_url(repo_url)
        subject = parsed['subject']
        data_model = parsed['dataModel']

        # Base URL for constructing other URLs
        base_url = f"https://raw.githubusercontent.com/smart-data-models/{subject}/master"

        # Generate metadata
        metadata = {
            "subject": subject,
            "dataModel": data_model,
            "version": self.extract_version(subject, data_model),
            "modelTags": "",
            "title": f"Smart Data Model - {subject.replace('dataModel.', '')} {data_model} schema",
            "$id": f"https://smart-data-models.github.io/{subject}/{data_model}/schema.json",
            "description": self.extract_description(subject, data_model),
            "required": [
                "type",
                "id"
            ],
            "yamlUrl": f"{base_url}/{data_model}/model.yaml",
            "jsonSchemaUrl": f"{base_url}/{data_model}/schema.json",
            "@context": f"{base_url}/context.jsonld",
            "exampleKeyvaluesJson": f"{base_url}/{data_model}/examples/example.json",
            "exampleKeyvaluesJsonld": f"{base_url}/{data_model}/examples/example.jsonld",
            "exampleNormalizedJson": f"{base_url}/{data_model}/examples/example-normalized.json",
            "exampleNormalizedJsonld": f"{base_url}/{data_model}/examples/example-normalized.jsonld",
            "sql": f"{base_url}/{data_model}/schema.sql",
            "adopters": f"{base_url}/{data_model}/ADOPTERS.yaml",
            "contributors": f"{base_url}/CONTRIBUTORS.yaml",
            "spec": f"{base_url}/{data_model}/doc/spec.md",
            "spec_DE": f"{base_url}/{data_model}/doc/spec_DE.md",
            "spec_ES": f"{base_url}/{data_model}/doc/spec_ES.md",
            "spec_FR": f"{base_url}/{data_model}/doc/spec_FR.md",
            "spec_IT": f"{base_url}/{data_model}/doc/spec_IT.md",
            "spec_JA": f"{base_url}/{data_model}/doc/spec_JA.md",
            "spec_KO": f"{base_url}/{data_model}/doc/spec_KO.md",
            "spec_ZH": f"{base_url}/{data_model}/doc/spec_ZH.md"
        }

        return metadata


class EmbeddedGitHubAnalyzer:
    """Embedded GitHub repository analyzer to avoid import dependencies."""

    def __init__(self):
        self.session = requests.Session()

    def parse_url(self, repo_url: str) -> Dict[str, str]:
        """Parse smart-data-models URL to extract subject and dataModel."""
        from urllib.parse import urlparse

        if 'smart-data-models.github.io' not in repo_url:
            raise ValueError("Invalid smart-data-models URL format")

        path_parts = urlparse(repo_url).path.strip('/').split('/')
        if len(path_parts) < 2:
            raise ValueError("URL path too short")

        return {
            'subject': path_parts[0],
            'dataModel': path_parts[1]
        }

    def get_file_content(self, subject: str, path: str) -> Optional[str]:
        """Get raw file content from GitHub."""
        url = f"https://raw.githubusercontent.com/smart-data-models/{subject}/main/{path}"
        try:
            response = self.session.get(url)
            response.raise_for_status()
            return response.text
        except requests.exceptions.RequestException:
            return None

    def extract_description(self, subject: str, data_model: str) -> str:
        """Extract description from schema.json file."""
        schema_path = f"{data_model}/schema.json"
        content = self.get_file_content(subject, schema_path)

        if content:
            try:
                import json
                schema = json.loads(content)
                return schema.get('description', f'Information on {data_model} data')
            except json.JSONDecodeError:
                pass

        return f"Information on the {data_model} performed by an entity"

    def extract_version(self, subject: str, data_model: str) -> str:
        """Extract version from schema.json or VERSION file."""
        schema_path = f"{data_model}/schema.json"
        content = self.get_file_content(subject, schema_path)

        if content:
            try:
                import json
                schema = json.loads(content)
                if 'version' in schema:
                    return schema['version']
            except json.JSONDecodeError:
                pass

        version_content = self.get_file_content(subject, "VERSION")
        if version_content:
            return version_content.strip()

        return "0.1.1"

    def generate_metadata(self, repo_url: str) -> Dict[str, Any]:
        """Generate metadata JSON for the given repository URL."""
        parsed = self.parse_url(repo_url)
        subject = parsed['subject']
        data_model = parsed['dataModel']

        base_url = f"https://raw.githubusercontent.com/smart-data-models/{subject}/master"

        metadata = {
            "subject": subject,
            "dataModel": data_model,
            "version": self.extract_version(subject, data_model),
            "modelTags": "",
            "title": f"Smart Data Model - {subject.replace('dataModel.', '')} {data_model} schema",
            "$id": f"https://smart-data-models.github.io/{subject}/{data_model}/schema.json",
            "description": self.extract_description(subject, data_model),
            "required": [
                "type",
                "id"
            ],
            "yamlUrl": f"{base_url}/{data_model}/model.yaml",
            "jsonSchemaUrl": f"{base_url}/{data_model}/schema.json",
            "@context": f"{base_url}/context.jsonld",
            "exampleKeyvaluesJson": f"{base_url}/{data_model}/examples/example.json",
            "exampleKeyvaluesJsonld": f"{base_url}/{data_model}/examples/example.jsonld",
            "exampleNormalizedJson": f"{base_url}/{data_model}/examples/example-normalized.json",
            "exampleNormalizedJsonld": f"{base_url}/{data_model}/examples/example-normalized.jsonld",
            "sql": f"{base_url}/{data_model}/schema.sql",
            "adopters": f"{base_url}/{data_model}/ADOPTERS.yaml",
            "contributors": f"{base_url}/CONTRIBUTORS.yaml",
            "spec": f"{base_url}/{data_model}/doc/spec.md",
            "spec_DE": f"{base_url}/{data_model}/doc/spec_DE.md",
            "spec_ES": f"{base_url}/{data_model}/doc/spec_ES.md",
            "spec_FR": f"{base_url}/{data_model}/doc/spec_FR.md",
            "spec_IT": f"{base_url}/{data_model}/doc/spec_IT.md",
            "spec_JA": f"{base_url}/{data_model}/doc/spec_JA.md",
            "spec_KO": f"{base_url}/{data_model}/doc/spec_KO.md",
            "spec_ZH": f"{base_url}/{data_model}/doc/spec_ZH.md"
        }

        return metadata


def main():
    """Main entry point."""
    if len(sys.argv) != 2:
        print("Usage: python github_repo_analyzer.py <repo_url>")
        print('Example: python github_repo_analyzer.py "https://smart-data-models.github.io/dataModel.User/Activity"')
        sys.exit(1)

    repo_url = sys.argv[1]

    analyzer = GitHubRepoAnalyzer()

    try:
        metadata = analyzer.generate_metadata(repo_url)
        print(json.dumps(metadata, indent=2))
    except ValueError as e:
        print(f"Error: {e}")
        sys.exit(1)
    except Exception as e:
        print(f"Unexpected error: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
