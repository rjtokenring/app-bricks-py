# SPDX-FileCopyrightText: Copyright (C) Arduino s.r.l. and/or its affiliated companies
#
# SPDX-License-Identifier: MPL-2.0

"""Guards the sampling parameters an app can reach through `LargeLanguageModel`.

The brick names only `temperature` in its signature, but forwards `**kwargs` up to
`CloudLLM` and on to the LangChain client, so the other sampling controls are reachable
too: `top_p` as a standard chat-completions field, and llama.cpp's non-OpenAI ones
(`top_k`, `min_p`) through `extra_body`, which the OpenAI SDK merges into the top level
of the request body. That is what the README documents for tuning the small models
served on an UnoQ, so the pass-through is asserted here rather than assumed — a
`model_kwargs`-style rewrite anywhere in the chain would silently drop these instead of
sending them.

No value is imposed by the brick: built without them, a request must carry none of the
three and leave the inference server on its own defaults.
"""

import pytest

import arduino.app_bricks.cloud_llm.cloud_llm as cloud_llm_module
import arduino.app_bricks.llm.local_llm as local_llm_module
from arduino.app_bricks.llm.local_llm import LargeLanguageModel

LLAMACPP_MODEL = "llamacpp:Qwen3.5-0.8B-Q4_0"
GENIE_MODEL = "genie:qwen3_4b_instruct_2507"


class FakeModel:
    """Stand-in for the LangChain model, recording the kwargs it was built with."""

    def __init__(self, **kwargs):
        self.kwargs = kwargs
        # Read back by CloudLLM to build the reasoning client and by `list_models`.
        self.extra_body = kwargs.get("extra_body")
        self.openai_api_base = kwargs.get("base_url")
        self.openai_api_key = kwargs.get("api_key")
        self.max_tokens = kwargs.get("max_tokens")

    def model_copy(self, update=None):
        return self


@pytest.fixture
def built(monkeypatch):
    """Builds the brick against a recording model factory, with no runner reachable.

    Returns a callable taking the `LargeLanguageModel` arguments and giving back the
    kwargs the factory received, which are the ones the LangChain client is built with.
    """
    monkeypatch.setattr(local_llm_module, "resolve_address", lambda host: host)
    # The canary that lists the runner's models must not reach the network.
    monkeypatch.setattr(LargeLanguageModel, "list_models", lambda self: [])

    def build(**kwargs):
        recorded = {}

        def factory(model_name, **model_kwargs):
            recorded.update(model_kwargs)
            return FakeModel(**model_kwargs)

        monkeypatch.setattr(cloud_llm_module, "model_factory", factory)
        LargeLanguageModel(**kwargs)
        return recorded

    return build


@pytest.mark.parametrize("model", [LLAMACPP_MODEL, GENIE_MODEL])
def test_no_sampling_is_sent_unless_asked_for(model, built):
    """Left alone, the brick imposes nothing and the runner keeps its own cutoffs."""
    kwargs = built(model=model)

    assert "top_p" not in kwargs
    assert "extra_body" not in kwargs
    # `temperature` is the one the brick does name, and it keeps being forwarded.
    assert kwargs["temperature"] == 0.7


@pytest.mark.parametrize("model", [LLAMACPP_MODEL, GENIE_MODEL])
def test_sampling_kwargs_reach_the_model(model, built):
    """`top_p` and `extra_body` survive the hand-off down to the LangChain client."""
    kwargs = built(model=model, temperature=0.3, top_p=0.9, extra_body={"top_k": 20, "min_p": 0.1})

    assert kwargs["temperature"] == 0.3
    assert kwargs["top_p"] == 0.9
    assert kwargs["extra_body"] == {"top_k": 20, "min_p": 0.1}
