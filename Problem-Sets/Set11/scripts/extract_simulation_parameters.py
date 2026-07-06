from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any

SET_ROOT = Path(__file__).resolve().parents[1]
if str(SET_ROOT) not in sys.path:
    sys.path.insert(0, str(SET_ROOT))

from cognitive_core.tools import extract_parameters_from_text, search_arxiv


DEFAULT_QUERY = "thermodynamic simulation parameters for advanced fission reactors"
MOCK_ABSTRACT = (
    "This paper studies Uranium Dioxide (UO2) reactor fuel material. "
    "The reported density is 10.97 g/cm3, the melting point is 3120 K, "
    "and the baseline thermal conductivity is 3.5 W/(m*K). "
    "Reference DOI: 10.1016/j.jnucmat.2019.01.001."
)


def _parse_search_output(output: str) -> dict[str, Any]:
    try:
        return json.loads(output)
    except json.JSONDecodeError as exc:
        raise ValueError(f"search_arxiv returned non-JSON output: {exc}") from exc


def run_pipeline(
    query: str, max_results: int, output_path: Path, use_mock: bool = False
) -> dict[str, Any]:
    """Run literature search, extract parameters from the top abstract, and save JSON."""

    if use_mock:
        extracted = extract_parameters_from_text(MOCK_ABSTRACT)
        payload = {
            "status": "success",
            "query": query,
            "agent_name": "scholar_prime",
            "source_paper": {
                "title": "Mock validation excerpt for UO2 material parameters",
                "authors": [],
                "published": None,
                "arxiv_id": None,
                "doi": extracted.get("doi"),
                "pdf_url": None,
            },
            "extraction": extracted,
            "note": "Generated with --mock because the live arXiv endpoint is optional for local validation.",
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    search_result = _parse_search_output(search_arxiv(query, max_results=max_results))
    if search_result.get("status") == "error":
        payload = {
            "status": "search_error",
            "query": query,
            "agent_name": "scholar_prime",
            "message": search_result.get("message", "Unknown search error."),
            "parameters": [],
            "source_paper": None,
        }
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        return payload

    papers = search_result.get("papers", [])
    if not papers:
        payload = {
            "status": "no_results",
            "query": query,
            "agent_name": "scholar_prime",
            "parameters": [],
            "source_paper": None,
        }
    else:
        top_paper = papers[0]
        abstract = top_paper.get("summary") or ""
        extracted = extract_parameters_from_text(abstract)
        payload = {
            "status": "success",
            "query": query,
            "agent_name": "scholar_prime",
            "source_paper": {
                "title": top_paper.get("title"),
                "authors": top_paper.get("authors", []),
                "published": top_paper.get("published"),
                "arxiv_id": top_paper.get("id"),
                "doi": top_paper.get("doi"),
                "pdf_url": top_paper.get("pdf_url"),
            },
            "extraction": extracted,
        }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return payload


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run the Scholar-Prime parameter extraction pipeline."
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--max-results", type=int, default=5)
    parser.add_argument(
        "--output",
        type=Path,
        default=SET_ROOT / "docs" / "simulation_parameters.json",
    )
    parser.add_argument(
        "--mock",
        action="store_true",
        help="Use a deterministic local excerpt instead of the live arXiv API.",
    )
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    result = run_pipeline(args.query, args.max_results, args.output, args.mock)
    print(json.dumps(result, indent=2))
