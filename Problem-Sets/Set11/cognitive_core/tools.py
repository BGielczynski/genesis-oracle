from __future__ import annotations

import json
import os
import re
import signal
import subprocess
import sys
from json import JSONDecoder
from pathlib import Path
from typing import Any


SET_ROOT = Path(__file__).resolve().parents[1]
SCIENCE_SKILLS_DIR = Path(
    os.getenv("SCIENCE_SKILLS_DIR", SET_ROOT / "science-skills")
).expanduser()


def _skill_dir(*candidate_names: str) -> Path:
    for name in candidate_names:
        candidate = SCIENCE_SKILLS_DIR / "skills" / name
        if candidate.exists():
            return candidate
    names = ", ".join(candidate_names)
    raise FileNotFoundError(
        f"Could not find any Science-Skills folder matching: {names}. "
        f"Expected base path: {SCIENCE_SKILLS_DIR}"
    )


def _last_json_object(text: str) -> dict[str, Any]:
    """Return the last JSON object printed by a CLI command."""

    decoder = JSONDecoder()
    last: dict[str, Any] | None = None
    for index, char in enumerate(text):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(text[index:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            last = value
    if last is None:
        raise ValueError("No JSON object found in CLI output.")
    return last


def _run_command(
    command: list[str], cwd: Path, timeout: int = 30
) -> subprocess.CompletedProcess[str]:
    creationflags = 0
    if sys.platform == "win32":
        creationflags = subprocess.CREATE_NEW_PROCESS_GROUP

    process = subprocess.Popen(
        command,
        cwd=cwd,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        creationflags=creationflags,
    )
    try:
        stdout, stderr = process.communicate(timeout=timeout)
    except subprocess.TimeoutExpired:
        if sys.platform == "win32":
            subprocess.run(
                ["taskkill", "/F", "/T", "/PID", str(process.pid)],
                capture_output=True,
                text=True,
                check=False,
            )
        else:
            process.send_signal(signal.SIGTERM)
            try:
                process.wait(timeout=5)
            except subprocess.TimeoutExpired:
                process.kill()
        raise

    return subprocess.CompletedProcess(command, process.returncode, stdout, stderr)


def search_arxiv(query: str, max_results: int = 5) -> str:
    """
    Search arXiv for research papers through the Science-Skills CLI.

    Args:
        query: Search phrase or arXiv query expression.
        max_results: Maximum number of papers to return.

    Returns:
        A compact JSON string containing titles, authors, abstracts, arXiv IDs,
        DOI fields when present, and PDF URLs.
    """

    query = query.strip()
    if not query:
        return json.dumps({"status": "error", "message": "query must not be empty"})

    max_results = max(1, min(int(max_results), 10))
    try:
        arxiv_dir = _skill_dir("literature_search_arxiv", "literature-search-arxiv")
    except FileNotFoundError as exc:
        return json.dumps({"status": "error", "message": str(exc)}, indent=2)

    script = arxiv_dir / "scripts" / "search_arxiv.py"
    if not script.exists():
        return json.dumps(
            {"status": "error", "message": f"arXiv CLI script not found: {script}"},
            indent=2,
        )

    command = [
        "uv",
        "run",
        str(script),
        "--query",
        query,
        "--max_results",
        str(max_results),
        "--sort_by",
        "relevance",
    ]
    try:
        result = _run_command(command, cwd=arxiv_dir)
    except subprocess.TimeoutExpired:
        return json.dumps(
            {
                "status": "error",
                "message": "arXiv Science-Skills CLI timed out after 30 seconds.",
            },
            indent=2,
        )
    except Exception as exc:
        return json.dumps({"status": "error", "message": str(exc)}, indent=2)

    if result.returncode != 0:
        return json.dumps(
            {
                "status": "error",
                "message": result.stderr.strip() or result.stdout.strip(),
            },
            indent=2,
        )

    try:
        data = _last_json_object(result.stdout)
    except ValueError:
        data = {"status": "success", "raw_output": result.stdout}

    papers = data.get("papers", [])
    compact_papers = []
    for paper in papers[:max_results]:
        compact_papers.append(
            {
                "id": paper.get("id"),
                "title": paper.get("title"),
                "authors": paper.get("authors", []),
                "published": paper.get("published"),
                "summary": paper.get("summary"),
                "doi": paper.get("doi"),
                "pdf_url": paper.get("pdf_url"),
            }
        )

    return json.dumps(
        {
            "status": data.get("status", "success"),
            "results_count": len(compact_papers),
            "papers": compact_papers,
        },
        indent=2,
    )


def extract_parameters_from_text(text: str) -> dict[str, Any]:
    """
    Extract material parameters from scientific text.

    Args:
        text: Abstract or paper excerpt that may contain material properties.

    Returns:
        A dictionary containing extracted parameter records and source metadata.
    """

    doi_match = re.search(r"\b10\.\d{4,9}/[-._;()/:A-Z0-9]+\b", text, re.IGNORECASE)
    material_match = re.search(
        r"\b(UO2|Uranium Dioxide|uranium dioxide|SiC|zirconium alloy|steel|graphite)\b",
        text,
    )

    patterns = [
        (
            "thermal_conductivity",
            r"(?:thermal conductivity|conductivity)\D{0,40}"
            r"(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>W\s*/?\s*\(?m\s*\*?\s*K\)?|W\s*m-1\s*K-1)",
        ),
        (
            "melting_point",
            r"(?:melting point|melts? at)\D{0,40}(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>K|Kelvin|C|°C)",
        ),
        (
            "density",
            r"(?:density)\D{0,40}(?P<value>\d+(?:\.\d+)?)\s*(?P<unit>g/cm3|g\s*cm-3|kg/m3|kg\s*m-3)",
        ),
    ]

    parameters: list[dict[str, Any]] = []
    for name, pattern in patterns:
        for match in re.finditer(pattern, text, re.IGNORECASE):
            start = max(0, match.start() - 80)
            end = min(len(text), match.end() + 80)
            parameters.append(
                {
                    "parameter": name,
                    "value": float(match.group("value")),
                    "unit": match.group("unit").replace(" ", ""),
                    "evidence": text[start:end].strip(),
                    "confidence": "medium",
                }
            )

    return {
        "material": material_match.group(0) if material_match else None,
        "doi": doi_match.group(0) if doi_match else None,
        "parameters": parameters,
        "extracted_from_text_preview": text[:300],
    }
