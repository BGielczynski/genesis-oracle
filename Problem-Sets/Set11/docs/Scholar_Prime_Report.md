# Scholar-Prime Report

## Exercise 1

Science-Skills is cloned locally into `Problem-Sets/Set11/science-skills/` and
ignored by Git because it is an external dependency. The OpenAlex verification
output should be stored in `docs/openalex_author_resolution.txt`.

## Exercise 2

`cognitive_core/agent.py` defines `scholar_prime` and exposes it as
`root_agent` for the ADK Web UI. The agent uses the assignment model
`gemini-3.5-flash` by default and can be overridden with
`SCHOLAR_PRIME_MODEL`.

## Exercise 3

The `search_arxiv(query: str, max_results: int = 5) -> str` tool calls the
Science-Skills arXiv CLI and returns compact JSON with source metadata. The tool
is bound directly to Scholar-Prime through the ADK `tools` list.

The Web UI evidence is consolidated in
`docs/screenshots/scholar_prime_webui_combined.png`.

## Exercise 4

`scripts/extract_simulation_parameters.py` runs the search, selects the top
paper, extracts material parameters from its abstract, and writes
`docs/simulation_parameters.json`.
