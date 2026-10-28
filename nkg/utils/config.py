import os
from typing import Optional

import dspy


def configure_dspy(
        model: str,
        api_base: str,
        api_key: str,
        temperature: Optional[float] = None,
        max_tokens: int = 4096,
        cache: bool = False,
        num_retries: int = 1,
) -> None:
    """Configure DSPy's language model for index construction.

    model is a LiteLLM routing string ("deepseek/<model>" for the DeepSeek API, "openai/<model>" for an
    OpenAI-compatible server such as vLLM). temperature=None sends no temperature (provider default)."""
    kwargs: dict = dict(
        model=model,
        api_key=api_key,
        api_base=api_base,
        cache=cache,
        temperature=temperature,
        max_tokens=max_tokens,
        model_type="chat",
        num_retries=num_retries,
    )
    # DeepSeek: no extended thinking tokens
    if model.startswith("deepseek/"):
        kwargs["extra_body"] = {"thinking": {"type": "disabled"}}

    # HTTP connection pool for litellm's synchronous calls. litellm builds its httpx clients without limits, i.e.
    # httpx's default of 100 connections, so any number of worker threads above ~100 just queues for a connection.
    # Size the pool with LLM_MAX_CONNECTIONS (default 2500); set before the first call creates a cached client.
    import httpx
    import litellm
    max_conn = int(os.getenv("LLM_MAX_CONNECTIONS") or 2500)
    if getattr(litellm, "_nkg_max_connections", 0) < max_conn:
        limits = httpx.Limits(max_connections=max_conn, max_keepalive_connections=max_conn)
        litellm.sync_transport = httpx.HTTPTransport(limits=limits)
        litellm.client_session = httpx.Client(limits=limits, follow_redirects=True)
        client_cache = getattr(litellm, "in_memory_llm_clients_cache", None)
        if client_cache is not None and hasattr(client_cache, "flush_cache"):
            client_cache.flush_cache()  # drop clients created with the old pool
        litellm._nkg_max_connections = max_conn

    lm = dspy.LM(**kwargs)
    dspy.configure(lm=lm)
    dspy.settings.configure(lm=lm, temperature=temperature)
