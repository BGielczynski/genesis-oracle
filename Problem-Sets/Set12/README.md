# Problem Set 12: The Great Spreeland Logistics Sync

Source: https://unisvet.github.io/ams-course-2026/week.html?id=12

Week 12 topic: Multi-Agent Systems (MAS) and Complex Dynamics

Deadline: July 19, 2026 at 23:59  
Weight: 10% of module grade

## Scenario

In 2026, the Spreeland region is facing a major logistics bottleneck. Local
organic farmers need to get their harvest to Cottbus, but river levels are
fluctuating, and several bridges are under automated maintenance.

The objective is to design a Multi-Agent System that uses the new Agentic
Protocol Stack to orchestrate a resilient supply chain.

Concept visual: [assets/img/week12_logistics.png](assets/img/week12_logistics.png)

## Exercise 1: Protocol Architecture Design

Which tool for which job?

Describe how you would implement the following interactions using MCP, A2A,
UCP, AP2, A2UI, and AG-UI:

- Infrastructure Discovery: How does the "Dispatch Agent" check real-time bridge
  status from the City's PostgreSQL database?
- Expert Consultation: How does the agent find and query a specialized
  "Weather-Predictor Agent" built by a different team?
- Secure Fulfillment: How do you ensure the agent's wholesale purchase of 2 tons
  of gherkins is both authorized by the owner and cryptographically verifiable?
- Dynamic Visualization: How do you present a live delivery dashboard to the user
  without writing custom React/Flutter code?

## Exercise 2: Implementing the Dispatcher Swarm

Using `google-adk`, implement a starter script for the `SpreelandDispatcher`.
The agent must discover tools via MCP and communicate its reasoning to the user
via a streaming interaction.

Reference snippet:

```python
from google.adk.agents import Agent
from google.adk.tools.mcp_tool import McpToolset
from google.adk.tools.mcp_tool.mcp_session_manager import StdioConnectionParams
from mcp import StdioServerParameters

# 1. Connect to Infrastructure Data via MCP
infra_tools = McpToolset(connection_params=StdioConnectionParams(
    server_params=StdioServerParameters(
        command="npx", args=["-y", "@spreeland/bridge-mcp-server"],
        env={"API_KEY": "SPREE_2026_SECRET"})
))

# 2. Define the Dispatcher Agent
dispatcher = Agent(
    name="Spreeland_Dispatcher",
    instruction="""You coordinate logistics in Spreeland.
    1. Check bridge status.
    2. Negotiate with A2A supplier agents.
    3. Output status via A2UI cards.""",
    tools=[infra_tools]
)

# 3. Handle Interactive Streaming (AG-UI pattern)
# ... Implementation required for exercise submission
```

## Submission Requirements

- Design Document: A PDF explaining the choice of protocols for each
  sub-exercise.
- Python Source: A working script demonstrating MCP tool discovery and A2A
  negotiation logic.
- UI Schema: A JSON file containing a sample `A2UI` message that renders a
  delivery status card.

## Extracted Files

- [problemset.html](problemset.html): original HTML extracted from the course
  Problem Set tab.
- [assets/img/week12_logistics.png](assets/img/week12_logistics.png): referenced
  image asset used by the original problem set.

## Completed Submission Artifacts

- [docs/Set12_Design_Document.md](docs/Set12_Design_Document.md): protocol
  architecture and design-document source.
- [src/spreeland_dispatcher.py](src/spreeland_dispatcher.py): runnable mock
  dispatcher demonstrating MCP discovery, A2A negotiation, UCP/AP2 approval and
  verification, AG-UI event streaming, and A2UI card generation.
- [schemas/delivery_status_card.json](schemas/delivery_status_card.json): A2UI
  delivery status card payload.
- [.env.example](.env.example): optional local configuration template.
- [.gitignore](.gitignore): excludes `.env`, runtime state, and Python cache
  files inside Set12.

## Run

From the repository root:

```powershell
python Problem-Sets\Set12\src\spreeland_dispatcher.py
python Problem-Sets\Set12\src\spreeland_dispatcher.py --json
python Problem-Sets\Set12\src\spreeland_dispatcher.py --write-card Problem-Sets\Set12\schemas\delivery_status_card.json
```

The default path uses deterministic mock infrastructure and does not require an
API key. For real ADK/MCP integration, copy `.env.example` to `.env` and install
the optional MCP runtime package expected by the assignment snippet.
