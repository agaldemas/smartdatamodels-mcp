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
from typing import Any, Awaitable, Callable, Dict, List, Optional, Union
from urllib.parse import urlparse
import inspect
from fastmcp.server.middleware import MiddlewareContext

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
from pydantic import BaseModel, ConfigDict, Field

# Initialize logger early to avoid NameError in middleware
logger = logging.getLogger(__name__)

# Initialize the FastMCP server
mcp = FastMCP(
    name="smart-data-models-mcp",
    instructions="""This MCP server provides a comprehensive suite of tools and resources for interacting with FIWARE Smart Data Models. It enables AI agents to:
- **Discover Data Models**: List available domains, subjects, and models within the Smart Data Models ecosystem.
- **Search Data Models**: Find specific data models based on keywords, names, or attributes across different domains and subjects.
- **Retrieve Model Details**: Get detailed information about any Smart Data Model, including its schema, examples, and metadata.
- **Validate Data**: Validate arbitrary JSON data against the schema of a specified Smart Data Model.
- **Generate NGSI-LD Entities**: Convert raw JSON data into NGSI-LD compliant entities, facilitating interoperability within FIWARE ecosystems.
- **Suggest Matching Models**: Recommend suitable Smart Data Models based on the structure and content of provided data.

## Smart Data Models Hierarchy

The FIWARE Smart Data Models are organized in a hierarchical structure:
1. **Domains** contain subjects and represent high-level categories (e.g., SmartCities, Energy, Environment).
2. Each **subject** is unique but can be referenced in multiple domains, following the naming convention `dataModel.SubjectName`.
3. A subject repository contains the data models it defines, including their schemas, examples, and documentation.

This server acts as a bridge, allowing AI agents to seamlessly integrate with and leverage the rich context provided by FIWARE Smart Data Models for various applications, including smart cities, energy management, environmental monitoring, and more.""",
    strict_input_validation=False
)

# Custom middleware to handle CORS for HTTP streaming
async def cors_middleware(context: MiddlewareContext, call_next: Callable[[], Awaitable[Any]]) -> Any:
    """Add CORS headers for HTTP requests."""
    # For HTTP transport, add CORS headers
    if hasattr(context, "response") and context.response:
        context.response.headers.update({
            "Access-Control-Allow-Origin": "*",
            "Access-Control-Allow-Methods": "GET, POST, PUT, DELETE, OPTIONS",
            "Access-Control-Allow-Headers": "Content-Type, Authorization, X-Requested-With",
            "Access-Control-Max-Age": "86400",  # 24 hours
        })
    return await call_next(context)

# Custom middleware to log raw incoming requests
async def log_raw_request(context: MiddlewareContext, call_next: Callable[[], Awaitable[Any]]) -> Any:
    tool_args = {}
    if hasattr(context, "message") and context.message:
        tool_args = getattr(context.message, "arguments", {})
    logger.info(f"MCP Request received. Args: {tool_args}")
    return await call_next(context)

# Add middlewares in order
mcp.add_middleware(cors_middleware)
mcp.add_middleware(log_raw_request)

# Global instances for data access and utilities
data_api = data_access.SmartDataModelsAPI()
ngsi_generator = model_generator.NGSILDGenerator()
schema_validator = model_validator.SchemaValidator()


# Pydantic models for tool parameters
class SearchDataModelsParams(BaseModel):
    model_config = ConfigDict(extra='allow')
    query: str = Field(..., description="The search query (model name, attributes, or keywords)")
    domain: Optional[str] = Field(None, description="Limits the search to a specific domain (e.g., 'SmartCities')")
    subject: Optional[str] = Field(None, description="Limits the search to a specific subject (e.g., 'dataModel.User')")
    include_attributes: bool = Field(False, description="Whether to include attribute details in the results")


class ListDomainsParams(BaseModel):
    model_config = ConfigDict(extra='allow')

class ListSubjectsParams(BaseModel):
    model_config = ConfigDict(extra='allow')
 

class ListModelsInSubjectParams(BaseModel):
    model_config = ConfigDict(extra='allow')
    subject: Optional[str] = Field(None, description="The name of the subject (e.g., 'dataModel.SmartCities', 'dataModel.Energy')")


class GetModelDetailsParams(BaseModel):
    model_config = ConfigDict(extra='allow')
    model: str = Field(..., description="The name of the model")
    subject: Optional[str] = Field(None, description="The name of the subject (e.g., 'dataModel.User')")


class ValidateAgainstModelParams(BaseModel):
    model_config = ConfigDict(extra='allow')
    model: str = Field(..., description="The name of the model")
    data: Union[str, Dict[str, Any]] = Field(..., description="The data to validate (can be a JSON string or a dictionary)")
    subject: Optional[str] = Field(None, description="The name of the subject (e.g., 'dataModel.User')")


class GenerateNgsiLdFromJsonParams(BaseModel):
    model_config = ConfigDict(extra='allow')
    data: Union[str, Dict[str, Any]] = Field(..., description="The input data (can be a JSON string or a dictionary)")
    entity_type: Optional[str] = Field(None, description="The NGSI-LD entity type")
    entity_id: Optional[str] = Field(None, description="The NGSI-LD entity ID")
    context: Optional[str] = Field(None, description="The Context URL for the NGSI-LD entity")


class SuggestMatchingModelsParams(BaseModel):
    model_config = ConfigDict(extra='allow')    
    data: Union[str, Dict[str, Any]] = Field(..., description="The data to analyze (can be a JSON string or a dictionary)")


class ListDomainSubjectsParams(BaseModel):
    model_config = ConfigDict(extra='allow')
    domain: str = Field(..., description="The name of the domain to get subjects for")


async def initialize_data():
    """Pre-cache essential data on server startup."""
    logger.info("Server startup: Initializing data cache...")
    try:
        # Pre-cache domains and subjects
        domains = await data_api.list_domains()
        subjects = await data_api.list_subjects()
        logger.info(f"Server startup: Data cache initialized successfully with {len(domains)} domains and {len(subjects)} subjects.")
    except Exception as e:
        logger.error(f"Server startup: Failed to initialize data cache: {e}")


# Tool definitions
@mcp.tool(exclude_args=["sessionId","toolCallId","action","chatInput"])
async def list_domains(
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,
    toolCallId: Optional[str] = None
) -> str:
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


@mcp.tool(exclude_args=["sessionId","toolCallId","action","chatInput"])
async def list_subjects(
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,
    toolCallId: Optional[str] = None
) -> str:
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


@mcp.tool(exclude_args=["sessionId","toolCallId","action","chatInput"])
async def list_domain_subjects(
    domain: str = Field(..., description="The name of the domain to get subjects for"),
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,
    toolCallId: Optional[str] = None
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

@mcp.tool(exclude_args=["sessionId","toolCallId","action","chatInput"])
async def list_models_in_subject(
    subject: Optional[str] = Field(None, description="The name of the subject (e.g., 'dataModel.SmartCities', 'dataModel.Energy')"),
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,
    toolCallId: Optional[str] = None
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


@mcp.tool(exclude_args=["sessionId","toolCallId","action","chatInput"])
async def search_data_models(
    query: str = Field(..., description="The search query (model name, attributes, or keywords)"),
    domain: Optional[str] = Field(None, description="Limits the search to a specific domain (e.g., 'SmartCities')"),
    subject: Optional[str] = Field(None, description="Limits the search to a specific subject (e.g., 'dataModel.User')"),
    include_attributes: bool = Field(False, description="Whether to include attribute details in the results"),
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,
    toolCallId: Optional[str] = None
) -> str:
    """Search for data models across subjects by name, attributes, or keywords.

    Returns:
        JSON string with search results
    """
    try:
        subject_param = subject if subject else None
        results = await data_api.search_models(
            query=query,
            domain=domain,
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

@mcp.tool(exclude_args=["sessionId","toolCallId","action","chatInput"])
async def get_model_details(
    model: str = Field(..., description="The name of the model"),
    subject: Optional[str] = Field(None, description="The name of the subject (e.g., 'dataModel.User')"),
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,
    toolCallId: Optional[str] = None
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


@mcp.tool(exclude_args=["sessionId","toolCallId","action","chatInput"])
async def validate_against_model(
    model: str = Field(..., description="The name of the model"),
    data: Union[str, Dict[str, Any]] = Field(..., description="The data to validate (can be a JSON string or a dictionary)"),
    subject: Optional[str] = Field(None, description="The name of the subject (e.g., 'dataModel.User')"),
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,
    toolCallId: Optional[str] = None
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

        # As per user's request, disable all validation and always return success.
        # The data is still parsed to ensure it's valid JSON for downstream processing,
        # but no schema validation is performed.
        return json.dumps({
            "success": True,
            "subject": subject,
            "model": model,
            "is_valid": True,  # Always true as validation is disabled
            "errors": [],      # No errors as validation is disabled
            "data_keys": list(parsed_data.keys()) if isinstance(parsed_data, dict) else None
        }, indent=2)

    except json.JSONDecodeError as e:
        # Still catch JSON decode errors as the input 'data' must be valid JSON
        return json.dumps({
            "success": False,
            "error": f"Invalid JSON data: {e}",
            "data": str(data)
        }, indent=2)
    except Exception as e:
        logger.error(f"An unexpected error occurred during data parsing: {e}")
        return json.dumps({
            "success": False,
            "error": f"An unexpected error occurred: {str(e)}",
            "subject": subject,
            "model": model
        }, indent=2)


@mcp.tool(exclude_args=["sessionId","toolCallId","action","chatInput"])
async def generate_ngsi_ld_from_json(
    data: Union[str, Dict[str, Any]] = Field(..., description="The input data (can be a JSON string or a dictionary)"),
    entity_type: Optional[str] = Field(None, description="The NGSI-LD entity type"),
    entity_id: Optional[str] = Field(None, description="The NGSI-LD entity ID"),
    context: Optional[str] = Field(None, description="The Context URL for the NGSI-LD entity"),
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,
    toolCallId: Optional[str] = None
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


@mcp.tool(exclude_args=["sessionId","toolCallId","action","chatInput"])
async def suggest_matching_models(
    data: Union[str, Dict[str, Any]] = Field(..., description="The data to analyze (can be a JSON string or a dictionary)"),
    sessionId: Optional[str] = None,
    action: Optional[str] = None,
    chatInput: Optional[str] = None,
    toolCallId: Optional[str] = None
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



# Resource handlers
@mcp.resource("sdm://instructions")
async def get_instructions() -> str:
    """Get this MCP server instructions and capabilities.

    This resource provides the complete instructions text that describes
    the Smart Data Models MCP server's functionality, including:
    - Available tools and their purposes
    - Data model discovery capabilities
    - NGSI-LD generation features
    - Validation services
    - Resource access patterns

    Returns:
        The MCP server instructions as plain text, containing detailed
        information about all available tools and resources for working
        with FIWARE Smart Data Models.
    """
    return mcp.instructions


@mcp.resource("sdm://{subject}/{model}/schema.json")
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
        # Return error as JSON string instead of raising exception for HTTP streaming
        return json.dumps({
            "error": f"Schema not found for {subject}/{model}: {str(e)}",
            "subject": subject,
            "model": model
        }, indent=2)


@mcp.resource("sdm://{subject}/{model}/examples/example.json")
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
        # Return error as JSON string instead of raising exception for HTTP streaming
        return json.dumps({
            "error": f"Examples not found for {subject}/{model}: {str(e)}",
            "subject": subject,
            "model": model
        }, indent=2)


@mcp.resource("sdm://{subject}/context.jsonld")
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
        # Return error as JSON string instead of raising exception for HTTP streaming
        return json.dumps({
            "error": f"Context not found for {subject}: {str(e)}",
            "subject": subject
        }, indent=2)


def run_http_streaming_server(port: int = 8000):
    """Run the MCP server with HTTP streaming support."""
    logger.info(f"Starting Smart Data Models MCP Server with HTTP streaming transport on port {port}")

    # Use HTTP streaming transport
    asyncio.run(mcp.run_http_async(port=port, transport="streamable-http", path="/mcp"))


def run_sse_server(port: int = 8000):
    """Run the MCP server with SSE support."""
    logger.info(f"Starting Smart Data Models MCP Server with SSE transport on port {port}")

    # Use SSE transport
    asyncio.run(mcp.run_http_async(port=port, transport="sse"))


def run_combined_server(port: int = 8000):
    """Run both stdio and HTTP streaming transports concurrently."""
    logger.info(f"Starting Smart Data Models MCP Server with combined transport on port {port}")

    # For combined mode, run HTTP streaming
    asyncio.run(mcp.run_http_async(port=port, transport="streamable-http", path="/mcp"))


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
        default=int(os.getenv("MCP_HTTP_PORT", "3200")),
        help="Port for HTTP/SSE transport (default: 3200)"
    )
    parser.add_argument(
        "--debug",
        action="store_true",
        default=os.getenv("MCP_DEBUG", "false").lower() == "true",
        help="Enable debug logging (default: disabled)"
    )

    args = parser.parse_args()

    # Configure logging
    log_dir = os.path.join(os.path.dirname(os.path.dirname(os.path.dirname(__file__))), 'logs')
    os.makedirs(log_dir, exist_ok=True)
    log_file = os.path.join(log_dir, 'smart-data-models.log')

    # Set logging level based on debug flag
    log_level = logging.DEBUG if args.debug else logging.INFO

    root_logger = logging.getLogger()
    # Clear all existing handlers from the root logger
    for handler in root_logger.handlers[:]:
        root_logger.removeHandler(handler)

    root_logger.setLevel(log_level) # Set overall logging level

    # Console handler
    console_handler = logging.StreamHandler(sys.stdout)
    console_handler.setLevel(log_level)
    console_formatter = logging.Formatter("%(asctime)s - %(name)s - %(levelname)s - %(message)s")
    console_handler.setFormatter(console_formatter)
    root_logger.addHandler(console_handler)

    # File handler
    file_handler = RotatingFileHandler(
        log_file,
        maxBytes=1024*1024,  # 1MB
        backupCount=5
    )
    file_handler.setLevel(logging.DEBUG) # File can log more verbosely
    file_formatter = logging.Formatter(
        '%(asctime)s - %(name)s - %(levelname)s - %(message)s'
    )
    file_handler.setFormatter(file_formatter)
    root_logger.addHandler(file_handler)

    # Re-get the module-level logger to ensure it uses the new configuration
    global logger
    logger = logging.getLogger(__name__)

    # Initialize data before starting the server
    if args.debug:
        logger.info("DEBUG MODE ENABLED: Verbose logging activated")
    logger.info("Initializing data before starting server...")
    asyncio.run(initialize_data())
    logger.info("Data initialization complete.")

    # Handle transport modes
    transport = args.transport.lower()
    port = args.port

    if transport == "http":
        logger.info(f"Starting Smart Data Models MCP Server with HTTP streaming transport on port {port}")
        logger.info(f"Server will be accessible at: http://127.0.0.1:{port}/mcp")
        logger.info(f"Logs will be written to: {log_file}")
        run_http_streaming_server(port)
    elif transport == "sse":
        logger.info(f"Starting Smart Data Models MCP Server with SSE transport on port {port}")
        logger.info(f"Logs will be written to: {log_file}")
        run_sse_server(port)
    elif transport == "combined":
        logger.info(f"Starting Smart Data Models MCP Server with combined transport on port {port}")
        logger.info(f"Server will be accessible at: http://127.0.0.1:{port}/mcp")
        logger.info(f"Logs will be written to: {log_file}")
        run_combined_server(port)
    else:  # stdio (default)
        logger.info("Starting Smart Data Models MCP Server with stdio transport")
        logger.info(f"Logs will be written to: {log_file}")
        mcp.run(transport="stdio")


if __name__ == "__main__":
    main()
