# Smart Data Models MCP Server
## Connecting AI Agents to the FIWARE Ecosystem

---

# What is it?
- A **Model Context Protocol (MCP)** server
- Bridges AI agents with **FIWARE Smart Data Models**
- Provides seamless access to standardized IoT data structures
- Built with **Python 3.9+** and the **FastMCP** framework
- Supports **Standard I/O**, **SSE**, and **HTTP Streaming** transports

---

# Why Smart Data Models?
- **15+ Domains**: Smart Cities, Energy, Agriculture, Water, etc.
- **Consistency**: Standardized attributes and relationships
- **Interoperability**: NGSI-LD compliant schemas
- **Proven**: Community-driven, industry-vetted, and backed by FIWARE
- **Global Standard**: Used worldwide for digital twin synchronization

---

# Key Features
- **🔍 Discover**: Browse 15+ domains and hundreds of subjects
- **🔎 Search**: Advanced search by keywords, attributes, or model names
- **✨ Generate**: Intelligent conversion of raw JSON to NGSI-LD entities
- **✅ Validate**: Real-time schema validation with detailed error reporting
- **💡 Suggest**: Smart model recommendations based on data structure analysis
- **📚 Resources**: Direct access to JSON schemas, examples, and LD-Contexts

---

# Organization Structure
<div align="center">
  <img src="img/sdm-organization.drawio.png" alt="SDM Organization" style="max-height: 400px;">
</div>

- **Domains** (e.g., SmartCities) contain **Subjects**
- **Subjects** (e.g., Mobility) contain **Models**
- **Models** (e.g., BikeHireStation) use **Properties**

---

# How it Works
1. **Tool Access**: AI agents call specialized tools (e.g., `search_data_models`)
2. **Data Fetching**: Multi-source strategy (GitHub API + `pysmartdatamodels`)
3. **Processing**: Intelligent inference of NGSI-LD types (GeoProperty, Relationship)
4. **Caching**: Robust in-memory caching with 30-minute TTL
5. **Response**: Structured JSON optimized for AI consumption

---

# Available MCP Tools
- `list_domains` / `list_subjects`: Explore the hierarchy
- `search_data_models`: Find the right model for your use case
- `get_model_details`: Deep dive into schemas and examples
- `validate_against_model`: Ensure data compliance
- `generate_ngsi_ld_from_json`: Automate NGSI-LD mapping
- `suggest_matching_models`: AI-driven model discovery

---

# MCP Resources
Accessible via URIs:
- **Instructions**: `sdm://instructions`
- **Schemas**: `sdm://{subject}/{model}/schema.json`
- **Examples**: `sdm://{subject}/{model}/examples/example.json`
- **Contexts**: `sdm://{subject}/context.jsonld`

---

# Installation
### Using UV (Recommended)
```bash
git clone https://github.com/agaldemas/smartdatamodels-mcp
cd smart-data-models-mcp
uv sync
```

### From TestPyPI
```bash
pip install --index-url https://test.pypi.org/simple/ smart-data-models-mcp
```

---

# Configuration (Cline / Claude)
### STDIO Mode
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
### SSE Mode (for n8n / Web)
```json
{
  "type": "sse",
  "url": "http://localhost:3200/sse"
}
```

---

# Usage with AI Agent
Once configured, you can ask your agent:
- *"Lister tous les domaines disponibles dans Smart Data Models"*
- *"Trouver des modèles liés à la qualité de l'air"*
- *"Générer une entité NGSI-LD à partir de ces données de capteur : {...}"*
- *"Valider ce fichier Building.json par rapport au schéma officiel"*
- *"Quels modèles correspondraient le mieux à cette structure de données ?"*

---

# Technical Benefits
- **Asynchronous**: High performance with Python `asyncio`
- **Smart Mapping**: Automatically detects `GeoProperty` (GeoJSON) and `Relationship`
- **Rate Limit Friendly**: Uses `GITHUB_READ_TOKEN` to avoid API limits
- **Robust**: Detailed logging and comprehensive test suite (`pytest`)
- **Flexible**: Easy integration with Cline, Claude Desktop, and n8n

---

# Get Started
### Empower your AI agents with standardized data today!

**GitHub**: agaldemas/smartdatamodels-mcp
**Official SDM**: github.com/smart-data-models

*Built with ❤️ for AI Interoperability and the FIWARE Ecosystem*
