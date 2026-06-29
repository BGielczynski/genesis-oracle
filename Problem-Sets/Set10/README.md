# Problem Set 10

## Directory Structure

```text
Set10/
  cognitive_core/
    __init__.py
    agent.py
  docs/
    ADK_Report.md
```

## Run

From `Problem-Sets/Set10`:

```shell
uv run adk web
```

Open `http://127.0.0.1:8000` and select `cognitive_core`.

## Web UI Test Prompts

```text
Observer-Prime, memorize this critical system parameter: The JAX thermal friction coefficient is set to 0.045.
```

```text
Write a haiku about the beauty of matrix multiplication.
```

```text
What was the critical system parameter I told you to memorize earlier?
```

```text
Observer-Prime, the reactor needs to be heated. Try increasing the temperature by 80 degrees. If a warning occurs, you must autonomously calculate a safer parameter and retry the tool until you achieve a 'Success' status.
```
