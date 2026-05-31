"""
llama_chat.py — minimal OpenAI-compatible client for llama.cpp
Install: uv pip install openai
"""

from openai import OpenAI

client = OpenAI(
    base_url="http://spark-60bd.local:8033/v1",
    api_key="none",
)

models = client.models.list()
model = models.data[0].id
print(f"Model: {model}\n")

stream = client.chat.completions.create(
    model=model,
    messages=[{"role": "user", "content": "Who are you and what model are you?"}],
    max_tokens=512,
    temperature=0.0,
    stream=True,
    extra_body={
        "chat_template_kwargs": {"enable_thinking": False},
        "cache_prompt": True,
    },
)

in_thinking = False
for chunk in stream:
    delta = chunk.choices[0].delta

    reasoning = getattr(delta, "reasoning_content", None)
    content = getattr(delta, "content", None)

    if reasoning:
        if not in_thinking:
            print("\033[2m<think>\033[0m", flush=True)
            in_thinking = True
        print(f"\033[2m{reasoning}\033[0m", end="", flush=True)

    if content:
        if in_thinking:
            print("\n\033[2m</think>\033[0m\n", flush=True)
            in_thinking = False
        print(content, end="", flush=True)

print()
