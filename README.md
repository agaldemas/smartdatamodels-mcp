# Smart Data Models MCP Server

A Model Context Protocol (MCP) server that provides AI agents with access to FIWARE Smart Data Models, enabling seamless integration with NGSI-LD compliant IoT platforms.

## Overview

This MCP server allows AI agents to:

- **Discover** existing Smart Data Models across 15+ domains (Smart Cities, Energy, Agriculture, etc.)
- **Search** models by name, attributes, or keywords
- **Generate** NGSI-LD compliant entities from arbitrary JSON data
- **Validate** data against Smart Data Model schemas
- **Access** model schemas, examples, and JSON-LD contexts
- **Explore** domains and models for integration planning

## Features

### 🔍 Discovery & Search
- Browse all available domains (Smart Cities, Energy, Logistics, etc.)
- List models within specific domains
- Search models by name, attributes, or keywords across domains
- Get detailed model information including schemas and examples

### ⚡ NGSI-LD Generation
- Convert arbitrary JSON data to NGSI-LD compliant entities
- Automatic entity type inference from data structure
- Intelligent property type detection (Property, GeoProperty, Relationship)
- Geographic data recognition and GeoJSON generation

### ✅ Validation
- Validate data against Smart Data Model schemas
- Comprehensive error reporting with user-friendly messages
- NGSI-LD entity structure validation
- Data-model compatibility analysis

### 📊 Resources
- **Server Instructions**: `sdm://instructions` - Get the MCP server instructions and capabilities
- Direct access to JSON schemas: `sdm://domain/model/schema.json`
- Model examples: `sdm://domain/model/examples.json`
- Domain contexts: `sdm://domain/context.jsonld`

## Installation

### Prerequisites
- Python 3.9 or later
- UV package manager (install from https://github.com/astral-sh/uv) **OR** pip (Python package manager)

### Install from Source (UV - Recommended)
```bash
git clone https://github.com/agaldemas/smartdatamodels-mcp
cd smart-data-models-mcp
uv sync
```

### Install from Source (Alternative with pip)
```bash
git clone https://github.com/agaldemas/smartdatamodels-mcp
cd smart-data-models-mcp
pip install -e .
```

### Install from PyPI (when published)
```bash
uv tool install smart-data-models-mcp
```

**Note:** For PyPI installation with pip, use `pip install smart-data-models-mcp` (when published).

## Configuration

### GitHub Token Configuration

For optimal performance and to avoid rate limiting, you can configure a GitHub personal access token:

1. **Create a GitHub Token:**
   - Go to GitHub Settings → Developer settings → Personal access tokens → Tokens (classic)
   - Click "Generate new token (classic)"
   - Select minimal scopes (no scopes required for public repository access)
   - Copy the generated token

2. **Configure Environment Variable:**
   - Create a `.env` file in your project directory
   - Add the following line:

   ```env
   GITHUB_READ_TOKEN=ghp_your_token_here
   ```

3. **Alternative: Set System Environment Variable:**
   ```bash
   export GITHUB_READ_TOKEN=ghp_your_token_here
   ```

The server automatically loads the token from the `.env` file or environment variables and uses it for GitHub API requests. If no token is provided, requests will work but may be rate-limited.

### Logging Configuration

The server writes detailed logs to help with troubleshooting and monitoring. Log files are automatically created in the project's `logs/` directory:

```
logs/
└── smart-data-models.log
```

**Log Configuration:**
- Location: `logs/smart-data-models.log` (relative to project root)
- Format: Timestamp - Logger Name - Level - Message
- Rotation: 10MB file size with 5 backups
- Levels: DEBUG (file), INFO (console)

You can check the logs for detailed information about:
- API requests to GitHub
- Pysmartdatamodels operations
- Caching behavior
- Error conditions and troubleshooting details

### Server Launch Commands

#### Launch Server on Specific Port (SSE Mode)

To run the server in SSE mode on a specific port, use the following command:

```bash
# Using UV (recommended)
uv run python src/smart_data_models_mcp/server.py --transport sse --port 3200

# Using pip
python src/smart_data_models_mcp/server.py --transport sse --port 3200
```

**Available command-line options:**
- `--transport`: Transport mode (`stdio` or `sse`, default: `stdio`)
- `--port`: Port number for SSE mode (default: 3200)
- `--host`: Host address (default: `127.0.0.1`)
- `--help`: Show help message

**Examples:**
```bash
cd ~/Documents/mcp/smartdatamodels-mcp
# Run in stdio mode (default)
uv run python src/smart_data_models_mcp/server.py

# Run in SSE mode on port 3200
uv run python src/smart_data_models_mcp/server.py --transport sse --port 3200


uv run --active python src/smart_data_models_mcp/server.py --transport sse --port 3200

# Run in SSE mode on different host and port
uv run python src/smart_data_models_mcp/server.py --transport sse --host 0.0.0.0 --port 8080

# Show help
uv run python src/smart_data_models_mcp/server.py --help
```


### MCP Server Configuration

The Smart Data Models MCP server supports two transport modes:

- **stdio**: Standard input/output mode (default) - recommended for most use cases
- **sse**: Server-Sent Events mode - for web-based integrations and custom setups

#### STDIO Mode (Default)

The stdio mode is the standard way to run the MCP server and is recommended for most AI assistant integrations.

### Cline MCP Server Configuration

To configure the smart-data-models-mcp server for use with Cline, add the following to your Cline MCP settings file:

**Location:** `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`

```json
{
  "mcpServers": {
    "smart-data-models": {
      "autoApprove": [
      ],
      "disabled": false,
      "timeout": 60,
      "type": "stdio",
      "command": "python3",
      "args": [
        "src/smart_data_models_mcp/server.py"
      ],
      "cwd": "/Users/alaingaldemas/Documents/mcp/smartdatamodels-mcp/src"
    }
  }
}
```

#### HTTP Streamable Mode
```json
{
  "mcpServers": {
    "smart-data-models-http": {
      "autoApprove": [],
      "disabled": false,
      "type": "streamableHttp",
      "timeout": 180,
      "url": "http://127.0.0.1:3210/mcp"
    }
  }
}
```

### Claude Desktop Configuration

If you prefer to use the server with Claude Desktop, add the following to your Claude Desktop configuration file:

**macOS:** `~/Library/Application Support/Claude/claude_desktop_config.json`
**Windows:** `%APPDATA%/Claude/claude_desktop_config.json`

```json
{
  "mcpServers": {
    "smart-data-models": {
      "command": "python",
      "args": ["src/smart_data_models_mcp/server.py"],
      "cwd": "<path>//smartdatamodels-mcp",
      "env": {}
    }
  }
}
```
**Note: this configuration remains to verify**

#### SSE Mode Configuration

For web-based integrations or when you need to run the server as a web service, you can use the SSE (Server-Sent Events) transport mode. Add the following configuration to your MCP settings:

```json
{
  "mcpServers": {
    "smart-data-models": {
      "disabled": false,
      "timeout": 180,
      "type": "sse",
      "url": "http://127.0.0.1:3200/sse"
    }
  }
}
```

**Note:** The increased timeout (180 seconds) is specifically configured for the `search_data_models` tool, which can take time as it examines the smart-data-models repositories to find matching models.
#### local n8n usage configuration
this mcp server can also be used with n8n through an mcp client node
the Endpoint should be:
   - `http://localhost:3200/sse`
   - `http://host.docker.internal:3200/sse` if the n8n is under docker

- choose `sse`as Server Transport



### Installation Steps

1. **Navigate to the smart-data-models-mcp directory:**
   ```bash
   cd <path>/p/smart-data-models-mcp
   ```

2. **Install in development mode:**
   ```bash
   uv sync
   ```

3. **Configure MCP Server:**
   - Open the Cline MCP settings file: `~/Library/Application Support/Code/User/globalStorage/saoudrizwan.claude-dev/settings/cline_mcp_settings.json`
   - Add the smart-data-models server configuration
   - Restart Cline to load the new server

4. **Test Configuration:**
   - Ask Cline: "What domains are available in Smart Data Models?"
   - If it shows domain information, the server is working correctly

## Usage

### Basic Commands

After configuration, restart Claude Desktop to load the MCP server. You can then ask Claude to:

#### Discover Models
```
"Show me all available domains in Smart Data Models"
"What models are available in the Smart Cities domain?"
"Find data models related to weather or climate"
```

#### Get Model Details
```
"Get detailed information about the WeatherObserved model"
"Show me the schema for a Building model"
"What are some examples of a Device model?"
```

#### Generate NGSI-LD Entities
```
"Convert this JSON data to NGSI-LD format: {...}"
"Create an NGSI-LD entity for a weather sensor"
"Generate a Building entity from this data: {...}"
```

#### Validate Data
```
"Validate this data against the WeatherObserved model"
"Is this JSON valid for a Device entity?"
"Check if my data matches the Building schema"
```

### Example Interactions

#### 1. Domain Exploration
**User:** "What domains are available in Smart Data Models?"

**Claude:** Shows available domains and provides interactive options to explore models within each domain.

#### 2. Model Discovery
**User:** "Find weather-related data models"

**Claude:** Searches across domains and returns WeatherObserved, WeatherForecast, and other weather models with descriptions.

#### 3. Data Generation
**User:** "Convert this sensor data to NGSI-LD:"
```json
{
  "temperature": 25.5,
  "humidity": 60,
  "location": [-122.4194, 37.7749],
  "timestamp": "2025-01-15T10:30:00Z"
}
```

**Claude:** Generates a properly formatted NGSI-LD entity with appropriate property types.

#### 4. Validation
**User:** "Validate this Building data against the Smart Data Model"

**Claude:** Checks the data structure and provides detailed validation results with suggestions for fixes.

## Supported Domains

- Smart Cities
- Agrifood
- Water
- Energy
- Logistics
- Robotics
- Sensoring
- Cross sector
- Health
- Destination
- Environment
- Aeronautics
- Manufacturing
- Incubated
- Harmonization

## MCP Tools

### list_domains
List all available Smart Data Model domains.

**Returns:** JSON string with available domains and count

### list_subjects
List all available Smart Data Model subjects.

**Returns:** JSON string with available subjects and count

### list_domain_subjects
List all subjects belonging to a specific domain.

**Parameters:**
- `domain`: The name of the domain to get subjects for

**Returns:** JSON string with subjects in the domain and count

### list_models_in_subject
List all data models within a specific subject.

**Parameters:**
- `subject`: The name of the subject (e.g., 'dataModel.SmartCities', 'dataModel.Energy')

**Returns:** JSON string with models in the subject and count

### search_data_models
Search for data models across subjects by name, attributes, or keywords.

**Parameters:**
- `query`: The search query (model name, attributes, or keywords)
- `domain`: Optional domain filter (e.g., 'SmartCities')
- `subject`: Optional subject filter (e.g., 'dataModel.User')
- `include_attributes`: Whether to include attribute details in the results

**Returns:** JSON string with search results, count, and query info

### get_model_details
Get detailed information about a specific data model.

**Parameters:**
- `model`: The name of the model
- `subject`: Optional subject name (e.g., 'dataModel.User')

**Returns:** JSON string with model details including schema, examples, and metadata

### validate_against_model
Validate data against a Smart Data Model schema.

**Parameters:**
- `model`: The name of the model
- `data`: The data to validate (can be a JSON string or a dictionary)
- `subject`: Optional subject name (e.g., 'dataModel.User')

**Returns:** JSON string with validation results (currently always returns success)

### generate_ngsi_ld_from_json
Generate NGSI-LD compliant entities from arbitrary JSON data.

**Parameters:**
- `data`: The input data (can be a JSON string or a dictionary)
- `entity_type`: Optional NGSI-LD entity type
- `entity_id`: Optional NGSI-LD entity ID
- `context`: Optional Context URL for the NGSI-LD entity

**Returns:** JSON string with generated NGSI-LD entity

### suggest_matching_models
Suggest Smart Data Models that match provided data structure.

**Parameters:**
- `data`: The data to analyze (can be a JSON string or a dictionary)

**Returns:** JSON string with suggested models and similarity scores

## MCP Resources

### sdm://instructions
Get this MCP server instructions and capabilities.

**Returns:** The MCP server instructions as plain text, containing detailed information about all available tools and resources for working with FIWARE Smart Data Models.

### sdm://{subject}/{model}/schema.json
Get the JSON schema for a specific Smart Data Model.

**Parameters:**
- `subject`: Subject (must start with 'dataModel.')
- `model`: Model name

**Returns:** JSON schema as string

### sdm://{subject}/{model}/examples/example.json
Get example instances for a specific Smart Data Model.

**Parameters:**
- `subject`: Subject (must start with 'dataModel.')
- `model`: Model name

**Returns:** Examples as JSON string

### sdm://{subject}/context.jsonld
Get the JSON-LD context for a subject.

**Parameters:**
- `subject`: Subject (must start with 'dataModel.')

**Returns:** JSON-LD context as string

## Technical Details

### Data Sources
- **pysmartdatamodels**: Official Python package for Smart Data Models
- **GitHub API**: Direct access to model repositories
- **Fallback mechanisms**: Multiple data sources for reliability

### Caching
- 30-minute TTL for domain/model listings
- In-memory schema and example caching
- Automatic cache invalidation

### Error Handling
- Graceful degradation when services are unavailable
- Clear error messages for developers and AI agents
- Validation error formatting for better user experience

### Performance
- Async/await patterns for concurrent operations
- Efficient caching reduces API calls
- Streaming responses for large datasets

## Development

### Setup Development Environment
```bash
# Clone and install in development mode (UV)
git clone https://github.com/agaldemas/smartdatamodels-mcp
cd smart-data-models-mcp
uv sync --dev

# OR: Clone and install in development mode (pip alternative)
git clone https://github.com/agaldemas/smartdatamodels-mcp
cd smart-data-models-mcp
pip install -e .[test]

# Alternative: Install with all development tools (pip)
pip install -e .[test,dev]  # Assuming equivalent optional dependencies are configured
```

**Development Commands (UV):**
```bash
# Run tests
uv run pytest

# Run with debugging (stdio mode)
uv run python src/smart_data_models_mcp/server.py --transport stdio

# Run with debugging (SSE mode)
uv run python src/smart_data_models_mcp/server.py --transport sse --port 3200
```

**Development Commands (pip alternative):**
```bash
# Activate virtual environment (if using venv)
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Run tests
pytest

# Run with debugging (stdio mode)
python src/smart_data_models_mcp/server.py --transport stdio

# Run with debugging (SSE mode)
python src/smart_data_models_mcp/server.py --transport sse --port 3200
```

### Architecture

```
smart_data_models_mcp/
├── server.py           # FastMCP server and tool definitions
├── data_access.py      # Smart Data Models API integration
├── model_generator.py  # NGSI-LD generation logic
├── model_validator.py  # Schema validation logic
└── __init__.py
```

### Testing

The project includes a comprehensive test suite to ensure all functionality works correctly. Tests are located in the `tests/` directory and include integration tests and API validation.

#### Running Tests

**Testing with UV (Recommended):**
```bash
# Install test dependencies
uv sync --dev

# Run all tests
uv run pytest

# Run with coverage
uv run pytest --cov=smart_data_models_mcp --cov-report=html

# Run specific test files
uv run pytest tests/test_mcp.py          # MCP server tools and resources tests
uv run pytest tests/test_data_access.py  # Integration tests for data access layer

# Run tests with verbose output
uv run pytest -v tests/
```

**Testing with pip (Alternative):**
```bash
# Install test dependencies
pip install -e .[test]

# Run all tests
pytest

# Run with coverage (requires pytest-cov)
pytest --cov=smart_data_models_mcp --cov-report=html

# Run specific test files
pytest tests/test_mcp.py          # MCP server tools and resources tests
pytest tests/test_data_access.py  # Integration tests for data access layer

# Run tests with verbose output
pytest -v tests/
```

#### Test Files

- **`tests/test_mcp.py`**: Comprehensive tests for MCP server tools and resources
  - Tests all 9 MCP tools (list_domains, list_subjects, search_data_models, etc.)
  - Tests all 4 MCP resources (instructions, schema, examples, context)
  - Tests error handling and edge cases
  - Validates JSON response formats and API integration
  - Tests both successful operations and failure scenarios

- **`tests/test_data_access.py`**: Integration tests for the data access layer
  - Tests SmartDataModelsAPI class functionality
  - Tests domain and subject listing with GitHub API integration
  - Tests model discovery, details retrieval, and caching
  - Tests schema, example, and context fetching
  - Tests search functionality with multiple strategies
  - Validates API responses, error handling, and performance

- **`tests/test_suggest_matching_models.py`**: Specialized tests for model suggestion functionality
  - Tests similarity scoring and attribute matching
  - Tests MCP server integration for model suggestions
  - Validates suggestion ranking and response structure
  - Tests edge cases and error handling

#### Manual Testing

You can run additional integration tests to verify your setup. All tests are designed to properly import the project modules when run through pytest.

Expected output from the test suite should show all tests passing with coverage reports and API validation.

## Contributing

1. Fork the repository
2. Create a feature branch
3. Implement your changes with tests
4. Ensure all tests pass
5. Submit a pull request

### Development Guidelines
- Follow PEP 8 style guidelines
- Add type hints for new functions
- Include comprehensive docstrings
- Write unit tests for new functionality
- Update documentation for API changes

## License

MIT License - see LICENSE file for details.

## Links

- [FIWARE Smart Data Models](https://github.com/smart-data-models/)
- [NGSI-LD Specification](https://www.etsi.org/deliver/etsi_gs/CIM/001_099/009/01.07.01_60/gs_cim009v010701p.pdf)
- [Model Context Protocol](https://modelcontextprotocol.io/)
- [FastMCP](https://github.com/jlowin/fastmcp)

## Troubleshooting

### Common Issues

**Server doesn't start**
- Check Python version (3.9+ required)
- Verify dependencies are installed
- Check Claude Desktop logs

**Models not found**
- Verify domain and model names are correct
- Check internet connectivity for GitHub API
- Try restarting Claude Desktop

**Validation errors**
- Ensure data is valid JSON
- Check schema compatibility
- Review error messages for specific issues

### Getting Help
1. Check the [Issues](../../issues) page
2. Review the [documentation](https://github.com/smart-data-models/)
3. Contact the FIWARE community

---

*Built with ❤️ for the FIWARE ecosystem and AI agent integration.*

## 🤖 AI Contributors

Special thanks to our AI collaborators:
- **Cline**: AI Assistant for code development, project structure, and automated testing (AlaingClineBot)

---

<p align="center">
  <img src="img/cline-scenes_02.avif" alt="Cline Development Scene" style="width: 50%;">
</p>
