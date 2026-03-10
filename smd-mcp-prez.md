# Smart Data Models MCP Server
## Connecting AI Agents to the FIWARE Ecosystem

---

# What is it?
- A **Model Context Protocol (MCP)** server
- Bridges AI agents with **FIWARE Smart Data Models**
- Provides seamless access to standardized IoT data structures
- Built with Python and the FastMCP framework

---

# Why Smart Data Models?
- **15+ Domains**: Smart Cities, Energy, Agriculture, etc.
- **Consistency**: Standardized attributes and relationships
- **Interoperability**: NGSI-LD compliant schemas
- **Proven**: Community-driven and industry-vetted

---

# Key Features
- **Discover**: Browse domains and list subjects
- **Search**: Find models by keywords or attributes
- **Generate**: Create NGSI-LD entities from raw JSON
- **Validate**: Check data against official schemas
- **Resources**: Direct access to schemas and examples

---

# Organization Structure
<div align="center">
  <img src="img/sdm-organization.drawio.png" alt="SDM Organization" style="max-height: 400px;">
</div>

- **Domains** contain **Subjects**
- **Subjects** contain **Models**
- **Models** use **Properties**

---

# How it Works
1. **Tool Access**: AI agents call tools like `search_data_models`
2. **Data Fetching**: Server connects to GitHub API and pysmartdatamodels
3. **Processing**: Intelligent inference of NGSI-LD types (GeoProperty, Relationship)
4. **Response**: Structured JSON for the AI agent to consume

---

# Installation
### Using UV (Recommended)
```bash
git clone https://github.com/agaldemas/smartdatamodels-mcp
cd smart-data-models-mcp
uv sync
```

### Using pip
```bash
pip install smart-data-models-mcp
```

---

# Configuration (Cline)
Add this to `cline_mcp_settings.json`:
```json
{
  "mcpServers": {
    "smart-data-models": {
      "type": "stdio",
      "command": "python3",
      "args": ["src/smart_data_models_mcp/server.py"],
      "cwd": "/path/to/smartdatamodels-mcp"
    }
  }
}
```

---

# Usage with AI Agent
Once configured, you can ask your agent:
- *"Show me all available domains in Smart Data Models"*
- *"Find data models related to weather"*
- *"Convert this sensor data to NGSI-LD format: {...}"*
- *"Validate this Building data against the official schema"*

---

# Technical Benefits
- **Asynchronous**: High performance with async/await
- **Caching**: 30-minute TTL to respect GitHub API limits
- **Robust**: Error handling and graceful degradation
- **Flexible**: Supports `stdio`, `sse`, and `http` transport modes

---

# Get Started
### Explore the Smart Data Models ecosystem today!

**GitHub**: agaldemas/smartdatamodels-mcp
**Official SDM**: github.com/smart-data-models

*Built with ❤️ for AI Interoperability*
