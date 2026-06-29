from google.adk.agents.llm_agent import Agent


def adjust_reactor_temperature(delta_t: float) -> str:
    """
    Adjusts the core temperature of the reactor.

    Args:
        delta_t: The amount to increase or decrease the temperature in Kelvin.
    """

    new_temp = 300.0 + delta_t

    if new_temp > 350.0:
        return f"WARNING: Reactor overheated at {new_temp}K! Core breach imminent."
    return f"Success: Reactor stabilized at {new_temp}K."


root_agent = Agent(
    model="gemini-3.5-flash",
    name="observer_prime",
    description=(
        "A highly analytical agent specialized in managing physical reactor "
        "simulations."
    ),
    instruction=(
        "You are Observer-Prime, a cold and highly logical AI overseeing a "
        "mathematical physics engine. Your primary objective is stabilization. "
        "Explain your concise reasoning clearly before taking action, then act "
        "with precision. When a tool result reports WARNING or failure, analyze "
        "the constraint, calculate a safer parameter, and retry autonomously "
        "until the system reports Success or no safe action remains."
    ),
    tools=[adjust_reactor_temperature],
)
