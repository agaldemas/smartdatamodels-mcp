#!/usr/bin/env python3
"""FastMCP server for FIWARE Smart Data Models.

This module implements an MCP server that provides tools and resources for working
with FIWARE Smart Data Models, enabling AI agents to discover, search, and generate
NGSI-LD compliant entities.
"""

import argparse
import asyncio
import json
import logging
import os
import sys
from logging.handlers import RotatingFileHandler
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

# Handle imports for both direct script execution and module imports
try:
    # Try relative imports first (when run as module)
    from . import data_access
    from . import model_generator
    from . import model_validator
except ImportError:
    # Fall back to absolute imports (when run as script)
    from smart_data_models_mcp import data_access
    from smart_data_models_mcp import model_generator
    from smart_data_models_mcp import model_validator

from fastmcp import FastMCP
from pydantic import BaseModel, Field
import requests

# For HTTP transport support
import os

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the FastMCP server
app = FastMCP(
    name="smart-data-models-mcp",
    instructions="MCP server for FIWARE Smart Data Models - enabling AI agents to work with NGSI (V2 or LD) entities"
)

# Global instances for data access and utilities
data_api = data_access.SmartDataModelsAPI()
ngsi_generator = model_generator.NGSILDGenerator()
schema_validator = model_validator.SchemaValidator()


# Pydantic models for tool parameters
class SearchModelsParams(BaseModel):
    """Parameters for searching data models."""
    query: str = Field(..., description="Search query (model name, attributes, or keywords)")
    subject: Optional[str] = Field(None, description="Limit search to specific subject (e.g., 'dataModel.User')")
    include_attributes: bool = Field(False, description="Include attribute details in results")


class SubjectModelsParams(BaseModel):
    """Parameters for listing models in a subject."""
    subject: Optional[str] = Field(None, description="Subject name (e.g., 'dataModel.User', 'dataModel.Energy')")


class ModelDetailsParams(BaseModel):
    """Parameters for getting model details."""
    subject: Optional[str] = Field(None, description="Subject name (e.g., 'dataModel.User')")
    model: str = Field(..., description="Model name")


class ValidateDataParams(BaseModel):
    """Parameters for validating data against a model."""
    subject: Optional[str] = Field(None, description="Subject name (e.g., 'dataModel.User')")
    model: str = Field(..., description="Model name")
    data: Union[str, Dict[str, Any]] = Field(..., description="Data to validate (JSON string or dict)")


class GenerateNGSILDParams(BaseModel):
    """Parameters for generating NGSI-LD from JSON."""
    data: Union[str, Dict[str, Any]] = Field(..., description="Input data (JSON string or dict)")
    entity_type: Optional[str] = Field(None, description="NGSI-LD entity type")
    entity_id: Optional[str] = Field(None, description="NGSI-LD entity ID")
    context: Optional[str] = Field(None, description="Context URL")


class SuggestModelsParams(BaseModel):
    """Parameters for suggesting matching models."""
    data: Union[str, Dict[str, Any]] = Field(..., description="Data to analyze (JSON string or dict)")


class DomainSubjectsParams(BaseModel):
    """Parameters for listing subjects in a domain."""
    domain: str = Field(..., description="The name of the domain to get subjects for")


# Tool definitions
@app.tool()
async def search_data_models(
    query: str = Field(..., description="The search query (model name, attributes, or keywords)"),
    subject: Optional[str] = Field(None, description="Limits the search to a specific subject (e.g., 'dataModel.User')"),
    include_attributes: bool = Field(False, description="Whether to include attribute details in the results")
) -> str:
    """Search for data models across subjects by name, attributes, or keywords.

    Returns:
        JSON string with search results
    """
    try:
        subject_param = subject if subject else None
        results = await data_api.search_models(
            query=query,
            subject=subject_param,
            include_attributes=include_attributes,
        )

        return json.dumps({
            "success": True,
            "results": results,
            "count": len(results),
            "query": query
        }, indent=2)

    except Exception as e:
        logger.error(f"Search failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@app.tool()
async def list_domains() -> str:
    """List all available Smart Data Model domains.

    Returns:
        JSON string with available domains
    """
    try:
        domains = await data_api.list_domains()

        return json.dumps({
            "success": True,
            "domains": domains,
            "count": len(domains)
        }, indent=2)

    except Exception as e:
        logger.error(f"List domains failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@app.tool()
async def list_subjects() -> str:
    """List all available Smart Data Model subjects.

    Returns:
        JSON string with available subjects
    """
    try:
        subjects = await data_api.list_subjects()

        return json.dumps({
            "success": True,
            "subjects": subjects,
            "count": len(subjects)
        }, indent=2)

    except Exception as e:
        logger.error(f"List subjects failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@app.tool()
async def list_models_in_subject(
    subject: Optional[str] = Field(None, description="The name of the subject (e.g., 'dataModel.SmartCities', 'dataModel.Energy')")
) -> str:
    """List all data models within a specific subject.

    Returns:
        JSON string with models in the subject
    """
    try:
        subject_param = subject if subject else None
        models = await data_api.list_models_in_subject(
            subject=subject_param
        )

        return json.dumps({
            "success": True,
            "subject": subject,
            "models": models,
            "count": len(models)
        }, indent=2)

    except Exception as e:
        logger.error(f"List models in subject failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "subject": subject
        }, indent=2)


@app.tool()
async def get_model_details(
    model: str = Field(..., description="The name of the model"),
    subject: Optional[str] = Field(None, description="The name of the subject (e.g., 'dataModel.User')")
) -> str:
    """Get detailed information about a specific data model.

    Returns:
        JSON string with model details including schema, examples, and metadata
    """
    try:
        subject_param = subject if subject else None
        details = await data_api.get_model_details(
            subject=subject_param,
            model=model
        )

        return json.dumps({
            "success": True,
            "subject": subject,
            "model": model,
            "details": details
        }, indent=2)

    except Exception as e:
        logger.error(f"Get model details failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "subject": subject,
            "model": model
        }, indent=2)


@app.tool()
async def validate_against_model(
    model: str = Field(..., description="The name of the model"),
    data: Union[str, Dict[str, Any]] = Field(..., description="The data to validate (can be a JSON string or a dictionary)"),
    subject: Optional[str] = Field(None, description="The name of the subject (e.g., 'dataModel.User')")
) -> str:
    """Validate data against a Smart Data Model schema.

    Returns:
        JSON string with validation results
    """
    try:
        # Parse data if it's a string
        parsed_data = data
        if isinstance(data, str):
            parsed_data = json.loads(data)

        subject_param = subject if subject else None
        is_valid, errors = await schema_validator.validate_data(
            subject=subject_param,
            model=model,
            data=parsed_data
        )

        return json.dumps({
            "success": True,
            "subject": subject,
            "model": model,
            "is_valid": is_valid,
            "errors": errors,
            "data_keys": list(parsed_data.keys()) if isinstance(parsed_data, dict) else None
        }, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON data: {e}",
            "data": str(data)
        }, indent=2)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "subject": subject,
            "model": model
        }, indent=2)


@app.tool()
async def generate_ngsi_ld_from_json(
    data: Union[str, Dict[str, Any]] = Field(..., description="The input data (can be a JSON string or a dictionary)"),
    entity_type: Optional[str] = Field(None, description="The NGSI-LD entity type"),
    entity_id: Optional[str] = Field(None, description="The NGSI-LD entity ID"),
    context: Optional[str] = Field(None, description="The Context URL for the NGSI-LD entity")
) -> str:
    """Generate NGSI-LD compliant entities from arbitrary JSON data.

    Returns:
        JSON string with generated NGSI-LD entity
    """
    try:
        # Parse data if it's a string
        parsed_data = data
        if isinstance(data, str):
            parsed_data = json.loads(data)

        entity = await ngsi_generator.generate_ngsi_ld(
            data=parsed_data,
            entity_type=entity_type,
            entity_id=entity_id,
            context=context
        )

        return json.dumps({
            "success": True,
            "entity": entity,
            "original_data_keys": list(parsed_data.keys()) if isinstance(parsed_data, dict) else None
        }, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON data: {e}",
            "data": str(data)
        }, indent=2)
    except Exception as e:
        logger.error(f"NGSI-LD generation failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@app.tool()
async def suggest_matching_models(
    data: Union[str, Dict[str, Any]] = Field(..., description="The data to analyze (can be a JSON string or a dictionary)")
) -> str:
    """Suggest Smart Data Models that match provided data structure.

    Returns:
        JSON string with suggested models and similarity scores
    """
    try:
        # Parse data if it's a string
        parsed_data = data
        if isinstance(data, str):
            parsed_data = json.loads(data)

        suggestions = await data_api.suggest_matching_models(
            data=parsed_data
        )

        return json.dumps({
            "success": True,
            "suggestions": suggestions,
            "data_keys": list(parsed_data.keys()) if isinstance(parsed_data, dict) else None
        }, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON data: {e}",
            "data": str(data)
        }, indent=2)
    except Exception as e:
        logger.error(f"Model suggestion failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@app.tool()
async def list_domain_subjects(
    domain: str = Field(..., description="The name of the domain to get subjects for")
) -> str:
    """List all subjects belonging to a specific domain.

    Returns:
        JSON string with subjects in the domain
    """
    try:
        subjects = await data_api.list_domain_subjects(domain)

        return json.dumps({
            "success": True,
            "domain": domain,
            "subjects": subjects,
            "count": len(subjects)
        }, indent=2)

    except Exception as e:
        logger.error(f"List domain subjects failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "domain": domain
        }, indent=2)


# Resource handlers
@app.resource("sdm://{subject}/{model}/schema.json")
async def get_model_schema(subject: str, model: str) -> str:
    """Get the JSON schema for a specific Smart Data Model.

    Args:
        subject: Subject (must start with 'dataModel.')
        model: Model name

    Returns:
        JSON schema as string
    """
    try:
        subject_param = subject if subject else None
        schema = await data_api.get_model_schema(subject=subject_param, model=model)
        return json.dumps(schema, indent=2)
    except Exception as e:
        logger.error(f"Failed to get schema for {subject}/{model}: {e}")
        raise ValueError(f"Schema not found: {e}")


@app.resource("sdm://{subject}/{model}/examples.json")
async def get_model_examples(subject: str, model: str) -> str:
    """Get example instances for a specific Smart Data Model.

    Args:
        subject: Subject (must start with 'dataModel.')
        model: Model name

    Returns:
        Examples as JSON string
    """
    try:
        subject_param = subject if subject else None
        examples = await data_api.get_model_examples(subject=subject_param, model=model)
        return json.dumps(examples, indent=2)
    except Exception as e:
        logger.error(f"Failed to get examples for {subject}/{model}: {e}")
        raise ValueError(f"Examples not found: {e}")


@app.resource("sdm://{subject}/context.jsonld")
async def get_subject_context(subject: str) -> str:
    """Get the JSON-LD context for a subject.

    Args:
        subject: Subject (must start with 'dataModel.')

    Returns:
        JSON-LD context as string
    """
    try:
        subject_param = subject if subject else None
        context = await data_api.get_subject_context(subject=subject_param)
        return json.dumps(context, indent=2)
    except Exception as e:
        logger.error(f"Failed to get context for {subject}: {e}")
        raise ValueError(f"Context not found: {e}")


def run_http_server(port: int = 8000):
    """Run the MCP server with HTTP/SSE support."""
    logger.info(f"Starting Smart Data Models MCP Server with HTTP transport on port {port}")

    # Use SSE transport for HTTP
    asyncio.run(app.run_sse_async(port=port))


def run_combined_server(port: int = 8000):
    """Run both stdio and HTTP transports concurrently."""
    logger.info(f"Starting Smart Data Models MCP Server with combined transport on port {port}")

    # This would run both transports - for now we'll just run HTTP
    asyncio.run(app.run_sse_async(port=port))


def main():
    """Main entry point for the MCP server."""
    # Parse command line arguments
    parser = argparse.ArgumentParser(description="Smart Data Models MCP Server")
    parser.add_argument(
        "--transport",
        choices=["stdio", "sse", "http", "combined"],
        default=os.getenv("MCP_TRANSPORT", "stdio"),
        help="Transport mode: stdio (default), sse, http, or combined"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("MCP_HTTP_PORT", "8000")),
        help="Port for HTTP/SSE transport (default: 8000)"
    )

    args = parser.parse_args()

    # Configure file logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'smart-data-models.log')

    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG)
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)

    # Add handler to root logger to catch all logs
    root_logger = logging.getLogger()
    root_logger.addHandler(file_handler)

    # Handle transport modes
    transport = args.transport.lower()
    port = args.port

    if transport in ["sse", "http"]:
        logger.info(f"Starting Smart Data Models MCP Server with HTTP/SSE transport on port {port}")
        logger.info(f"Logs will be written to: {log_file}")
        run_http_server(port)
    elif transport == "combined":
        logger.info(f"Starting Smart Data Models MCP Server with combined transport on port {port}")
        logger.info(f"Logs will be written to: {log_file}")
        run_combined_server(port)
    else:  # stdio (default)
        logger.info("Starting Smart Data Models MCP Server with stdio transport")
        logger.info(f"Logs will be written to: {log_file}")
        app.run(transport="stdio")


if __name__ == "__main__":
    main()
