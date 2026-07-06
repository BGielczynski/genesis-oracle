# Problem Set 11: Scholar-Prime

This folder contains the Week 11 solution scaffold for the Scholar-Prime
literature retrieval and parameter extraction agent.

## External Dependency

The Google DeepMind Science-Skills repository is required by the assignment, but
it is intentionally not committed into this repository. Clone it locally into
this folder before running the notebook or tools:

```powershell
cd C:\GitHub\genesis-oracle\Problem-Sets\Set11
git clone https://github.com/google-deepmind/science-skills.git
```

The local checkout is ignored through the root `.gitignore` rule:

```text
Problem-Sets/Set11/science-skills/
```

The current Science-Skills repository uses underscore folder names such as
`literature_search_openalex` and `literature_search_arxiv`. The local wrapper
code supports both those real paths and the hyphenated paths shown in the
assignment text.

## Exercise 1: OpenAlex Verification

From `Problem-Sets/Set11`, run:

```powershell
cd science-skills\skills\literature_search_openalex
uv run scripts\openalex_cli.py resolve authors "Geoffrey Hinton"
```

Save the console output in:

```text
docs/openalex_author_resolution.txt
```

## Exercise 2 and 3: ADK Agent

The ADK agent is defined in:

```text
cognitive_core/agent.py
```

Create a local environment file for your Gemini API key:

```powershell
Copy-Item cognitive_core\.env.example cognitive_core\.env
```

Then edit `cognitive_core\.env` and set:

```env
GOOGLE_API_KEY="your-google-ai-studio-api-key"
SCHOLAR_PRIME_MODEL="gemini-3.5-flash"
```

Start the ADK Web UI from `Problem-Sets/Set11`:

```powershell
uv run adk web
```

Select `cognitive_core` in the UI and test:

```text
Scholar-Prime, search arXiv for papers on 'thermodynamic simulation parameters for advanced fission reactors'. Identify the most relevant paper and summarize its abstract.
```

Store the required screenshot in:

```text
docs/screenshots/
```

## Exercise 4: Parameter Extraction Pipeline

Run:

```powershell
uv run python scripts\extract_simulation_parameters.py
```

The script writes:

```text
docs/simulation_parameters.json
```

The output includes the source paper metadata, DOI when available, arXiv ID, and
structured extracted parameters.

## Model Configuration

The assignment model is `gemini-3.5-flash`. To use a temporary fallback without
editing code, change `SCHOLAR_PRIME_MODEL` in `cognitive_core\.env` or set it in
the shell:

```powershell
$env:SCHOLAR_PRIME_MODEL="gemini-2.5-flash"
```

To use a Science-Skills checkout outside this folder:

```powershell
$env:SCIENCE_SKILLS_DIR="C:\GitHub\science-skills"
```
