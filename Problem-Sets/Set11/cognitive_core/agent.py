from __future__ import annotations

import os

from google.adk.agents.llm_agent import Agent

from .tools import extract_parameters_from_text, search_arxiv


scholar_prime = Agent(
    model=os.getenv("SCHOLAR_PRIME_MODEL", "gemini-3.5-flash"),
    name="scholar_prime",
    description=(
        "An academic research agent specialized in querying scientific "
        "databases and extracting material parameters."
    ),
    instruction=(
        "You are Scholar-Prime, a precise academic research agent for "
        "simulation modeling. Search scientific literature with the available "
        "tools, compare paper relevance from titles and abstracts, extract "
        "material parameters and formulas only when supported by the source "
        "text, and always report DOI or arXiv IDs when available. If a DOI is "
        "missing, state that explicitly instead of inventing one."
    ),
    tools=[search_arxiv, extract_parameters_from_text],
)

root_agent = scholar_prime

