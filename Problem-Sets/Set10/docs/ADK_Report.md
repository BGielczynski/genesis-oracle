# ADK Report

The ADK replaces the manual Week 9 while-loop with a native agent runtime that keeps the perception, action, and result-checking cycle inside the framework. State-tracking is handled by the web session, so the agent can recall earlier reactor parameters without hand-maintained chat arrays. Tool calling is also cleaner because Python functions with type hints and docstrings become callable tools directly, avoiding the raw JSON parsing and function-call dispatch code from the previous Mandelbrot agent.
