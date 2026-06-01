"""Weather tool (placeholder implementation)."""

from .base import Tool


def get_weather(city: str) -> dict:
    """Return the current weather for a city.

    Placeholder — replace with a real weather API call.
    """

    return {
        "city": city,
        "temperature_c": 18,
        "condition": "partly cloudy",
    }


weather_tool = Tool(
    name="get_weather",
    description="Get current weather for a city.",
    parameters={
        "type": "object",
        "properties": {
            "city": {
                "type": "string",
                "description": "City name, for example Vilnius",
            }
        },
        "required": ["city"],
    },
    fn=get_weather,
)
