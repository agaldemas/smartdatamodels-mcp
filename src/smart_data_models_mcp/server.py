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
    subject: Optional[str] = Field(None, description="Limit search to specific subject or domain")
    limit: int = Field(20, description="Maximum number of results to return")
    include_attributes: bool = Field(False, description="Include attribute details in results")


class SubjectModelsParams(BaseModel):
    """Parameters for listing models in a subject."""
    subject: Optional[str] = Field(None, description="Subject or domain name (e.g., 'SmartCities', 'Energy')")
    limit: int = Field(50, description="Maximum number of models to return")


class ModelDetailsParams(BaseModel):
    """Parameters for getting model details."""
    subject: Optional[str] = Field(None, description="Subject or domain name")
    model: str = Field(..., description="Model name")


class ValidateDataParams(BaseModel):
    """Parameters for validating data against a model."""
    subject: Optional[str] = Field(None, description="Subject or domain name")
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
async def search_data_models(params: SearchModelsParams = Field(..., description=SearchModelsParams.__doc__)) -> str:
    """Search for data models across subjects by name, attributes, or keywords.

    Args:
        params: An object containing the search parameters.
            - `query` (str): The search query (model name, attributes, or keywords).
            - `subject` (Optional[str]): Limits the search to a specific subject or domain.
            - `limit` (int): The maximum number of results to return (default: 20).
            - `include_attributes` (bool): Whether to include attribute details in the results (default: False).

    Returns:
        JSON string with search results
    """
    try:
        results = await data_api.search_models(
            query=params.query,
            subject=params.subject,
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
async def list_models_in_subject(params: SubjectModelsParams = Field(..., description=SubjectModelsParams.__doc__)) -> str:
    """List all data models within a specific subject.

    Args:
        params: An object containing the subject and limit parameters.
            - `subject` (str): The name of the subject or domain (e.g., 'SmartCities', 'Energy').
            - `limit` (int): The maximum number of models to return (default: 50).

    Returns:
        JSON string with models in the subject
    """
    try:
        subject_param = f"dataModel.{params.subject}" if params.subject else None
        models = await data_api.list_models_in_subject(
            subject=subject_param,
            limit=params.limit
        )

        return json.dumps({
            "success": True,
            "subject": params.subject,
            "models": models,
            "count": len(models)
        }, indent=2)

    except Exception as e:
        logger.error(f"List models in subject failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "subject": params.subject
        }, indent=2)


@app.tool()
async def get_model_details(params: ModelDetailsParams = Field(..., description=ModelDetailsParams.__doc__)) -> str:
    """Get detailed information about a specific data model.

    Args:
        params: An object containing the subject and model identifiers.
            - `subject` (str): The name of the subject or domain.
            - `model` (str): The name of the model.

    Returns:
        JSON string with model details including schema, examples, and metadata
    """
    try:
        subject_param = f"dataModel.{params.subject}" if params.subject else None
        details = await data_api.get_model_details(
            subject=subject_param,
            model=params.model
        )

        return json.dumps({
            "success": True,
            "subject": params.subject,
            "model": params.model,
            "details": details
        }, indent=2)

    except Exception as e:
        logger.error(f"Get model details failed: {e}")
        return json.dumps({
            "success": False,
            "error": str(e),
            "subject": params.subject,
            "model": params.model
        }, indent=2)


@app.tool()
async def validate_against_model(params: ValidateDataParams = Field(..., description=ValidateDataParams.__doc__)) -> str:
    """Validate data against a Smart Data Model schema.

    Args:
        params: An object containing the validation parameters.
            - `subject` (str): The name of the subject or domain.
            - `model` (str): The name of the model.
            - `data` (Union[str, Dict[str, Any]]): The data to validate (can be a JSON string or a dictionary).

    Returns:
        JSON string with validation results
    """
    try:
        # Parse data if it's a string
        data = params.data
        if isinstance(data, str):
            data = json.loads(data)

        subject_param = f"dataModel.{params.subject}" if params.subject else None
        is_valid, errors = await schema_validator.validate_data(
            subject=subject_param,
            model=params.model,
            data=data
        )

        return json.dumps({
            "success": True,
            "subject": params.subject,
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
            "subject": params.subject,
            "model": params.model
        }, indent=2)


@app.tool()
async def generate_ngsi_ld_from_json(params: GenerateNGSILDParams = Field(..., description=GenerateNGSILDParams.__doc__)) -> str:
    """Generate NGSI-LD compliant entities from arbitrary JSON data.

    Args:
        params: An object containing the generation parameters.
            - `data` (Union[str, Dict[str, Any]]): The input data (can be a JSON string or a dictionary).
            - `entity_type` (Optional[str]): The NGSI-LD entity type.
            - `entity_id` (Optional[str]): The NGSI-LD entity ID.
            - `context` (Optional[str]): The Context URL for the NGSI-LD entity.

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
async def suggest_matching_models(params: SuggestModelsParams = Field(..., description=SuggestModelsParams.__doc__)) -> str:
    """Suggest Smart Data Models that match provided data structure.

    Args:
        params: An object containing the suggestion parameters.
            - `data` (Union[str, Dict[str, Any]]): The data to analyze (can be a JSON string or a dictionary).
            - `top_k` (int): The number of top matching models to return (default: 5).

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
@app.resource("sdm://{subject}/{model}/schema.json")
async def get_model_schema(subject: str, model: str) -> str:
    """Get the JSON schema for a specific Smart Data Model.

    Args:
        subject: Subject or domain name
        model: Model name

    Returns:
        JSON schema as string
    """
    try:
        subject_param = f"dataModel.{subject}" if subject else None
        schema = await data_api.get_model_schema(subject=subject_param, model=model)
        return json.dumps(schema, indent=2)
    except Exception as e:
        logger.error(f"Failed to get schema for {subject}/{model}: {e}")
        raise ValueError(f"Schema not found: {e}")


@app.resource("sdm://{subject}/{model}/examples.json")
async def get_model_examples(subject: str, model: str) -> str:
    """Get example instances for a specific Smart Data Model.

    Args:
        subject: Subject or domain name
        model: Model name

    Returns:
        Examples as JSON string
    """
    try:
        subject_param = f"dataModel.{subject}" if subject else None
        examples = await data_api.get_model_examples(subject=subject_param, model=model)
        return json.dumps(examples, indent=2)
    except Exception as e:
        logger.error(f"Failed to get examples for {subject}/{model}: {e}")
        raise ValueError(f"Examples not found: {e}")


@app.resource("sdm://{subject}/context.jsonld")
async def get_subject_context(subject: str) -> str:
    """Get the JSON-LD context for a subject.

    Args:
        subject: Subject or domain name

    Returns:
        JSON-LD context as string
    """
    try:
        subject_param = f"dataModel.{subject}" if subject else None
        context = await data_api.get_subject_context(subject=subject_param)
        return json.dumps(context, indent=2)
    except Exception as e:
        logger.error(f"Failed to get context for {subject}: {e}")
        raise ValueError(f"Context not found: {e}")


def main():
    """Main entry point for the MCP server."""
    # Run the stdio server
    logger.info("Starting Smart Data Models MCP Server with stdio transport")
    asyncio.run(app.run_stdio_async())


if __name__ == "__main__":
    main()
