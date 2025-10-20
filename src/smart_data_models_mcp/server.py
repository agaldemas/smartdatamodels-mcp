#!/usr/bin/env python3
"""FastMCP server for FIWARE Smart Data Models.

This module implements an MCP server that provides tools and resources for working
with FIWARE Smart Data Models, enabling AI agents to discover, search, and generate
NGSI-LD compliant entities.
"""

import asyncio
import json
import logging
import os
import sys
from typing import Any, Dict, List, Optional, Union
from urllib.parse import urlparse

# Add the src directory to the path so we can import modules
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '../..'))

import data_access
import model_generator
import model_validator

from fastmcp import FastMCP
from pydantic import BaseModel, Field
import requests

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger(__name__)

# Initialize the FastMCP server
app = FastMCP(
    name="smart-data-models-mcp",
    instructions="MCP server for FIWARE Smart Data Models - enabling AI agents to work with NGSI-LD entities"
)

# Global instances for data access and utilities
data_api = data_access.SmartDataModelsAPI()
ngsi_generator = model_generator.NGSILDGenerator()
schema_validator = model_validator.SchemaValidator()


# Pydantic models for tool parameters
class SearchModelsParams(BaseModel):
    """Parameters for searching data models."""
    query: str = Field(..., description="Search query (model name, attributes, or keywords)")
    domain: Optional[str] = Field(None, description="Limit search to specific domain")
    limit: int = Field(20, description="Maximum number of results to return")
    include_attributes: bool = Field(False, description="Include attribute details in results")


class DomainModelsParams(BaseModel):
    """Parameters for listing models in a domain."""
    domain: str = Field(..., description="Domain name (e.g., 'SmartCities', 'Energy')")
    limit: int = Field(50, description="Maximum number of models to return")


class ModelDetailsParams(BaseModel):
    """Parameters for getting model details."""
    domain: str = Field(..., description="Domain name")
    model: str = Field(..., description="Model name")


class ValidateDataParams(BaseModel):
    """Parameters for validating data against a model."""
    domain: str = Field(..., description="Domain name")
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
    top_k: int = Field(5, description="Number of top matches to return")


# Tool definitions
@app.tool()
async def search_data_models(params: SearchModelsParams) -> str:
    """Search for data models across domains by name, attributes, or keywords.

    Args:
        params: Search parameters including query, domain filter, and options

    Returns:
        JSON string with search results
    """
    try:
        results = await data_api.search_models(
            query=params.query,
            domain=params.domain,
            limit=params.limit,
            include_attributes=params.include_attributes,
        )

        return json.dumps({
            "success": True,
            "results": results,
            "count": len(results),
            "query": params.query
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
async def list_models_in_domain(params: DomainModelsParams) -> str:
    """List all data models within a specific domain.

    Args:
        params: Domain and limit parameters

    Returns:
        JSON string with models in the domain
    """
    try:
        models = await data_api.list_models_in_domain(
            domain=params.domain,
            limit=params.limit
        )

        return json.dumps({
            "success": True,
            "domain": params.domain,
            "models": models,
            "count": len(models)
        }, indent=2)

    except Exception as e:
        logger.error(f"List models in domain failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "domain": params.domain
        }, indent=2)


@app.tool()
async def get_model_details(params: ModelDetailsParams) -> str:
    """Get detailed information about a specific data model.

    Args:
        params: Domain and model identifiers

    Returns:
        JSON string with model details including schema, examples, and metadata
    """
    try:
        details = await data_api.get_model_details(
            domain=params.domain,
            model=params.model
        )

        return json.dumps({
            "success": True,
            "domain": params.domain,
            "model": params.model,
            "details": details
        }, indent=2)

    except Exception as e:
        logger.error(f"Get model details failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "domain": params.domain,
            "model": params.model
        }, indent=2)


@app.tool()
async def validate_against_model(params: ValidateDataParams) -> str:
    """Validate data against a Smart Data Model schema.

    Args:
        params: Validation parameters including domain, model, and data

    Returns:
        JSON string with validation results
    """
    try:
        # Parse data if it's a string
        data = params.data
        if isinstance(data, str):
            data = json.loads(data)

        is_valid, errors = await schema_validator.validate_data(
            domain=params.domain,
            model=params.model,
            data=data
        )

        return json.dumps({
            "success": True,
            "domain": params.domain,
            "model": params.model,
            "is_valid": is_valid,
            "errors": errors,
            "data_keys": list(data.keys()) if isinstance(data, dict) else None
        }, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON data: {e}",
            "data": str(params.data)
        }, indent=2)
    except Exception as e:
        logger.error(f"Validation failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "domain": params.domain,
            "model": params.model
        }, indent=2)


@app.tool()
async def generate_ngsi_ld_from_json(params: GenerateNGSILDParams) -> str:
    """Generate NGSI-LD compliant entities from arbitrary JSON data.

    Args:
        params: Generation parameters including data and optional entity metadata

    Returns:
        JSON string with generated NGSI-LD entity
    """
    try:
        # Parse data if it's a string
        data = params.data
        if isinstance(data, str):
            data = json.loads(data)

        entity = await ngsi_generator.generate_ngsi_ld(
            data=data,
            entity_type=params.entity_type,
            entity_id=params.entity_id,
            context=params.context
        )

        return json.dumps({
            "success": True,
            "entity": entity,
            "original_data_keys": list(data.keys()) if isinstance(data, dict) else None
        }, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON data: {e}",
            "data": str(params.data)
        }, indent=2)
    except Exception as e:
        logger.error(f"NGSI-LD generation failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


@app.tool()
async def suggest_matching_models(params: SuggestModelsParams) -> str:
    """Suggest Smart Data Models that match provided data structure.

    Args:
        params: Parameters including data and number of suggestions

    Returns:
        JSON string with suggested models and similarity scores
    """
    try:
        # Parse data if it's a string
        data = params.data
        if isinstance(data, str):
            data = json.loads(data)

        suggestions = await data_api.suggest_matching_models(
            data=data,
            top_k=params.top_k
        )

        return json.dumps({
            "success": True,
            "suggestions": suggestions,
            "data_keys": list(data.keys()) if isinstance(data, dict) else None
        }, indent=2)

    except json.JSONDecodeError as e:
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON data: {e}",
            "data": str(params.data)
        }, indent=2)
    except Exception as e:
        logger.error(f"Model suggestion failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e)
        }, indent=2)


# Resource handlers
@app.resource("sdm://{domain}/{model}/schema.json")
async def get_model_schema(domain: str, model: str) -> str:
    """Get the JSON schema for a specific Smart Data Model.

    Args:
        domain: Domain name
        model: Model name

    Returns:
        JSON schema as string
    """
    try:
        schema = await data_api.get_model_schema(domain=domain, model=model)
        return json.dumps(schema, indent=2)
    except Exception as e:
        logger.error(f"Failed to get schema for {domain}/{model}: {e}")
        raise ValueError(f"Schema not found: {e}")


@app.resource("sdm://{domain}/{model}/examples.json")
async def get_model_examples(domain: str, model: str) -> str:
    """Get example instances for a specific Smart Data Model.

    Args:
        domain: Domain name
        model: Model name

    Returns:
        Examples as JSON string
    """
    try:
        examples = await data_api.get_model_examples(domain=domain, model=model)
        return json.dumps(examples, indent=2)
    except Exception as e:
        logger.error(f"Failed to get examples for {domain}/{model}: {e}")
        raise ValueError(f"Examples not found: {e}")


@app.resource("sdm://{domain}/context.jsonld")
async def get_domain_context(domain: str) -> str:
    """Get the JSON-LD context for a domain.

    Args:
        domain: Domain name

    Returns:
        JSON-LD context as string
    """
    try:
        context = await data_api.get_domain_context(domain=domain)
        return json.dumps(context, indent=2)
    except Exception as e:
        logger.error(f"Failed to get context for {domain}: {e}")
        raise ValueError(f"Context not found: {e}")


def main():
    """Main entry point for the MCP server."""
    # Run the stdio server
    logger.info("Starting Smart Data Models MCP Server with stdio transport")
    asyncio.run(app.run_stdio_async())


if __name__ == "__main__":
    main()
