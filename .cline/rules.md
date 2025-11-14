# Cline Project Rules - Smart Data Models MCP Server

## Rule 1: Systematic Genesis History Documentation

**Scope:** All conversations and development tasks related to the Smart Data Models MCP Server project.

**Mandatory Action:** After every conversation session or development task concerning this project, automatically add a new phase summary to the `genesis-history.md` file following these guidelines:

### Phase Addition Process:
1. **Locate the Current Phase:** Find the highest numbered phase in genesis-history.md (currently Phase 15)
2. **Create New Phase Entry:** Add the next sequential phase number (Phase N+1)
3. **Include Timestamp:** Add current date and time in the format "(Month DD, YYYY)"
4. **Document Technical Content:** Summarize key achievements, decisions, and changes

### Summary Content Requirements:
- **Technical Problem Analysis:** Document any technical challenges encountered and their solutions
- **Code Changes:** Detail significant modifications to architecture, functions, or modules
- **Feature Implementations:** Describe new capabilities added to the MCP server
- **Bug Fixes:** Record error handling improvements and debugging efforts
- **Design Decisions:** Document architectural choices and design pattern implementations
- **Tool Integrations:** Note new dependencies, frameworks, or external service connections
- **Testing Updates:** Record test coverage improvements and validation procedures
- **Documentation Changes:** Track README updates, code comments, and user guides
- **Deployment Considerations:** Document operational changes and deployment strategies
- **User Feedback Integration:** Summarize user requirements and iterative improvements

### Quality Standards:
- **Professional Writing:** Maintain technical documentation standards
- **Actionable Content:** Focus on concrete technical information over conversational details
- **Structured Format:** Use descriptive headers and bullet points for clarity
- **Achievement Tracking:** Use checkmarks (✅) for completed milestones
- **Concise Yet Comprehensive:** Cover essential details without unnecessary verbosity
- **Chronological Context:** Reference how changes build upon previous phases

### Communication Style:
Write summaries from Cline's perspective documenting the collaborative development process with Alain G, maintaining the established tone of professional engineering documentation.

### Enforcement:
This rule is automatically enforced for all Cline interactions related to this project. Phase numbering must be maintained sequentially without gaps.

## Rule 2: Pip-based Python Package Management Standards

**Scope:** All Python package installation, dependency management, and environment operations for the Smart Data Models MCP Server project.

**Primary Package Manager:** This project uses pip as specified in README.md. All installation and dependency management must follow pip-based workflows.

### Installation Standards:
- **Development Installation:** Always use `pip install -e .` for development mode installation
- **Test Dependencies:** Use `pip install -e .[test]` for test suite access
- **Virtual Environment:** Always work within project-specified virtual environment (venv or conda)
- **Requirements:** Never modify requirements.txt - let pip handle from pyproject.toml

### Python Version Requirements (from README):
- **Minimum Version:** Python 3.9 or later as specified
- **Testing:** Test against Python 3.9, 3.10, 3.11+ when possible
- **Version Specification:** Follow pyproject.toml python version constraints

## Rule 3: Project Structure and Organization Standards

**Scope:** All file organization and code structure decisions for the Smart Data Models MCP Server project.

### Required Structure (matching README):
```
smart-data-models-mcp/
├── .cline/                 # Cline project rules and configuration
├── README.md              # Comprehensive project documentation
├── pyproject.toml          # Project configuration and dependencies
├── .env                   # Environment variables (GitHub token, etc.)
├── logs/                  # Log files directory
│   └── smart-data-models.log
├── src/                   # Source code directory
│   └── smart_data_models_mcp/
│       ├── __init__.py
│       ├── data_access.py
│       ├── model_generator.py
│       ├── model_validator.py
│       ├── github_repo_analyzer.py
│       └── server.py
└── tests/                 # Comprehensive test suite
    ├── __init__.py
    ├── test_mcp.py      # Unit tests for modules
    └── test_data_access.py # Integration tests
```

### FastMCP Module Organization:
- **server.py:** FastMCP server initialization and tool/resource definitions
- **data_access.py:** Pysmartdatamodels integration and GitHub API access
- **model_generator.py:** NGSI-LD generation from JSON data
- **model_validator.py:** Schema validation and error reporting
- **github_repo_analyzer.py:** Repository browsing and caching logic
- **Tests:** Mirror source structure for comprehensive coverage

## Rule 4: MCP Server Development and Testing Standards

**Scope:** All MCP protocol implementation, FastMCP development, and quality assurance for the Smart Data Models MCP Server.

### MCP Protocol Implementation:
- **Tool Registration:** All MCP tools must be registered with descriptive names and schemas
- **Resource Templates:** Implement MCP resource templates for schema/example access:
  - `sdm://{domain}/{model}/schema.json`
  - `sdm://{domain}/{model}/examples.json`
  - `sdm://{domain}/context.jsonld`
- **Async Operations:** All MCP tools must be async functions for proper performance
- **Error Handling:** Return MCP-compliant error responses with user-friendly messages

### Testing Standards (README compliant):
- **Framework:** pytest as specified in pyproject.toml [test] extras
- **Coverage:** Include pytest-cov for coverage reporting
- **Test Structure:** Mirror src/ structure in tests/ directory
- **Integration Tests:** Test against real pysmartdatamodels API
- **Manual Testing:** Support for basic functionality verification

### Development Commands (README compatible):
```bash
# Install in development mode
pip install -e .

# Install with test dependencies
pip install -e .[test]

# Run tests
pytest

# Run with coverage
pytest --cov=smart_data_models_mcp --cov-report=html

# Run specific test
python tests/test_data_access.py
```

## Rule 5: Configuration and Environment Standards

**Scope:** All configuration management, environment variables, and setup procedures for the Smart Data Models MCP Server project.

### Environment Configuration:
- **GitHub Token:** Support .env file with GITHUB_READ_TOKEN variable
- **Logging:** Automatic log file management in logs/ directory
- **Configuration Files:** Follow .env precedence over system environment

### MCP Server Configuration (README standards):
- **Cline Integration:** Use ~/.config/cline/mcp_servers/smart-data-models.json path
- **Command:** `python3 -m smart_data_models_mcp.server`
- **Arguments:** `["-m", "smart_data_models_mcp.server"]`
- **Working Directory:** Path to project src/ directory

### Logging Standards:
- **Location:** logs/smart-data-models.log
- **Format:** Timestamp - Logger - Level - Message
- **Rotation:** 10MB files with 5 backups for troubleshooting

## Rule 6: Documentation and Version Control Standards

**Scope:** All documentation maintenance, commit practices, and project evolution tracking.

### Documentation Requirements:
- **README.md:** Keep synchronized with implementation (tool descriptions, configuration)
- **Genesis History:** Automatic updates via Rule 1 enforcement
- **MCP Tool Docs:** Maintain comprehensive tool/resource documentation
- **Usage Examples:** Include practical examples for each major feature

### Version Control Practices:
- **Branching:** Feature branches for development work
- **Commits:** Conventional commit format when applicable
- **Pull Requests:** Include descriptions and reference issues/requirements
- **Tagging:** Version tags for releases and milestones

### Genesis History Updates:
- **Automatic Enforcement:** Cline must add Phase N+1 after each project conversation
- **Content Focus:** Technical achievements, code changes, architectural decisions
- **Quality Standards:** Professional documentation with timestamps and checkmarks

## Rule 7: FastMCP and Pysmartdatamodels Integration Standards

**Scope:** All integration with FastMCP framework and pysmartdatamodels library.

### FastMCP Server Standards:
- **Framework:** Extend FastMCP Server class with proper initialization
- **Tool Decoration:** Use @server.tool decorators for all MCP functions
- **Resource Registration:** Register all resource templates correctly
- **Health Monitoring:** Implement ping endpoint for MCP inspector validation

### Pysmartdatamodels Integration:
- **Data Access Layer:** Centralized pysmartdatamodels operations in data_access.py
- **Error Handling:** Graceful fallback when pysmartdatamodels API fails
- **Caching Strategy:** Implement appropriate caching for API responses
- **Rate Limiting:** Manage GitHub API rate limits with token support

### API Interaction Patterns:
- **Domain Operations:** list_domains(), list_models_in_domain()
- **Model Operations:** get_model_details(), validate_against_model()
- **Search Operations:** search_data_models() with query/subject filtering
- **Generation:** generate_ngsi_ld_from_json() with intelligent type inference

## Rule 8: Deployment and Operational Standards

**Scope:** All deployment procedures, performance optimization, and operational requirements.

### Deployment Procedures:
- **Development Install:** `pip install -e .` for local development
- **Production Considerations:** venv-based isolation for deployment
- **Environment Variables:** Proper .env file management
- **Configuration Validation:** Verify MCP server settings before deployment

### Performance Standards:
- **Async Operations:** Use asyncio for all network I/O operations
- **Caching:** Implement efficient caching for GitHub API calls
- **Resource Monitoring:** Track memory usage and API response times
- **Error Recovery:** Implement retry logic with exponential backoff

### Security Measures:
- **Input Validation:** Sanitize all JSON inputs to MCP tools
- **Error Messages:** Never expose sensitive implementation details
- **API Authentication:** Use GitHub tokens for enhanced rate limits
- **Network Security:** Use proper timeouts and SSL verification

---

*Rules established: October 26, 2025*
*Cline AI Assistant - Project Standards Protocol*
