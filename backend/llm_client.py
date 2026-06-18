"""
llm_client.py — optional LLM synthesis layer.
Auto-detects Anthropic vs OpenAI from available env keys.
If no key: callers fall back to showing raw chunks.
"""

import os
import logging

logger = logging.getLogger("ragforge.llm")

SYSTEM_PROMPT = """You are a precise document assistant.
Answer the user's question using ONLY the provided context chunks.
If the answer is not in the context, say so clearly.
Be concise. Cite the source doc_id when referencing specific facts."""


def _build_user_message(query: str, context: str) -> str:
    return f"""Context chunks:
{context}

---
Question: {query}

Answer:"""


async def stream_answer(query: str, context: str, streaming: bool = False):
    """
    If streaming=False: returns full answer string.
    If streaming=True: async generator yielding tokens.
    """
    anthropic_key = os.getenv("ANTHROPIC_API_KEY")
    openai_key    = os.getenv("OPENAI_API_KEY")

    if anthropic_key:
        return await _anthropic(query, context, anthropic_key, streaming)
    elif openai_key:
        return await _openai(query, context, openai_key, streaming)
    else:
        raise RuntimeError("No LLM API key configured")


async def _anthropic(query: str, context: str, key: str, streaming: bool):
    import anthropic
    client = anthropic.AsyncAnthropic(api_key=key)

    if not streaming:
        msg = await client.messages.create(
            model="claude-haiku-4-5-20251001",  # fastest + cheapest
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(query, context)}],
        )
        return msg.content[0].text

    async def _gen():
        async with client.messages.stream(
            model="claude-haiku-4-5-20251001",
            max_tokens=1024,
            system=SYSTEM_PROMPT,
            messages=[{"role": "user", "content": _build_user_message(query, context)}],
        ) as stream:
            async for token in stream.text_stream:
                yield token

    return _gen()


async def _openai(query: str, context: str, key: str, streaming: bool):
    from openai import AsyncOpenAI
    client = AsyncOpenAI(api_key=key)

    if not streaming:
        resp = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_message(query, context)},
            ],
            max_tokens=1024,
        )
        return resp.choices[0].message.content

    async def _gen():
        stream = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user",   "content": _build_user_message(query, context)},
            ],
            max_tokens=1024,
            stream=True,
        )
        async for chunk in stream:
            delta = chunk.choices[0].delta.content
            if delta:
                yield delta

    return _gen()
