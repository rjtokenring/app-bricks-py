# Large Language Model (LLM) Brick

The Large Language Model (LLM) Brick provides a simple Python® interface for chatting with a locally hosted Large Language Model on the board. It lets Arduino® App Lab applications send text prompts and receive generated text from models such as Qwen, Llama, or Gemma running through the on-device model service.

## Overview

The LLM Brick is designed for applications that need conversational AI without sending prompts to the cloud. It wraps the local model runner behind a unified chat-style API, so the same code works whether the underlying model is served by the `genie` or the `llamacpp` service.

Use this Brick when your application needs to answer questions, follow instructions, call tools, or hold a conversation entirely on the device. It supports synchronous one-shot responses, real-time token streaming, optional chain-of-thought reasoning, tool calling, and conversational memory that can be persisted across restarts.

The Brick uses the model configured for `arduino:llm` in Arduino App Lab.

## Features

- **Local conversational AI**: Sends prompts to an LLM running on the board through the local model service, no cloud API key required.
- **Multiple runners**: Works with both `genie:*` and `llamacpp:*` models through the same interface.
- **Synchronous responses**: Uses `chat()` when the application needs the full answer before continuing.
- **Streaming responses**: Uses `chat_stream()` to display generated text as it arrives, ideal for responsive UIs.
- **Reasoning streaming**: Uses `chat_stream_reasoning()` to separate the model's chain-of-thought from the final answer (llama.cpp models only).
- **Tool calling**: Registers Python functions as tools the model can invoke to fetch data or perform actions.
- **Conversation memory**: Keeps recent chat history with `with_memory()` and can persist it across restarts.
- **Advanced access**: Exposes the underlying LangChain chat model through `get_client()` for custom integrations.

## Prerequisites

- A supported board (`unoq` or `ventunoq`) with a local model runner available.
- A compatible LLM model downloaded and configured in Arduino App Lab.
- The `arduino:llm` Brick added to the application from App Lab.
- The `genie` and/or `llamacpp` model service, depending on the selected model.

**Note:** The LLM runs locally through the board model service, so cloud inference and cloud API keys are not required for normal use.

## Code example and usage

### Basic Chat

This example sends a single prompt and waits for the complete model response.

```python
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_utils import App

llm = LargeLanguageModel()


def ask_prompt():
    prompt = "Hi, what can you do as an AI assistant?"
    print(llm.chat(prompt))
    raise StopIteration


App.run(ask_prompt)
```

### Stream a Response

Use `chat_stream()` when a web UI or terminal interface should show text as the model generates it.

```python
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_utils import App

llm = LargeLanguageModel()


def ask_prompt():
    prompt = "Hi, what can you do as an AI assistant?"
    for chunk in llm.chat_stream(prompt):
        print(chunk, end="", flush=True)
    print()
    raise StopIteration


App.run(ask_prompt)
```

### Tool Calling

Register Python functions as tools using the `tool` decorator. The model decides when to call them and incorporates their results into its response.

```python
from arduino.app_bricks.llm import LargeLanguageModel, tool
from arduino.app_utils import App


@tool
def get_current_weather(location: str) -> str:
    """
    Get the current weather in a given location.
    The output is a string with a summary of the weather.
    """
    if "turin" in location.lower():
        return "The current weather in Turin is 8°C and rainy."
    return f"Sorry, I do not have real-time weather data for {location}."


llm = LargeLanguageModel(tools=[get_current_weather])


def ask_prompt():
    prompt = "What is the weather like in Turin?"
    print(llm.chat(prompt))
    raise StopIteration


App.run(ask_prompt)
```

### Streaming Reasoning

Use `chat_stream_reasoning()` to keep the model's internal reasoning separate from the final answer. Each yielded item is a `ReasoningChunk` (chain-of-thought) or a `ContentChunk` (final answer). This is supported only for llama.cpp models.

```python
from arduino.app_bricks.llm import LargeLanguageModel, REASONING_BUDGET_MEDIUM
from arduino.app_bricks.cloud_llm import ReasoningChunk, ContentChunk
from arduino.app_utils import App

llm = LargeLanguageModel(model="llamacpp:Qwen3.5-0.8B-Q4_0")


def ask_prompt():
    prompt = "If a train travels 60 km in 45 minutes, what is its speed in km/h?"
    for chunk in llm.chat_stream_reasoning(prompt, reasoning_effort=REASONING_BUDGET_MEDIUM):
        if isinstance(chunk, ReasoningChunk):
            print(f"[thinking] {chunk.content}", end="", flush=True)
        elif isinstance(chunk, ContentChunk):
            print(chunk.content, end="", flush=True)
    print()
    raise StopIteration


App.run(ask_prompt)
```

### Persistent Conversation Memory

Use `with_memory()` when follow-up prompts should keep recent context. Pass a `SQLMessagePersistence` backend to retain the conversation across restarts.

```python
from arduino.app_bricks.cloud_llm import SQLMessagePersistence
from arduino.app_bricks.dbstorage_sqlstore import SQLStore
from arduino.app_bricks.llm import LargeLanguageModel
from arduino.app_utils import App

db = SQLStore("llm_persistent_demo.db")
db.start()

llm = LargeLanguageModel(
    system_prompt="You are a helpful assistant.",
).with_memory(
    max_messages=10,
    persistence=SQLMessagePersistence(sql_store=db, thread_id="llm-demo-conversation"),
)


def ask_prompt():
    prompt = "Hi, what can you do as an AI assistant?"
    print(llm.chat(prompt))
    raise StopIteration


App.run(ask_prompt)
```

## Configuration

The Brick is initialized with the following parameters:

| Parameter | Type | Default | Description |
| :-- | :-- | :-- | :-- |
| `model` | `str` | App Lab configured model | Local model identifier configured for `arduino:llm` in App Lab (e.g. `genie:qwen3_4b_instruct_2507` or `llamacpp:Qwen3.5-0.8B-Q4_0`). |
| `system_prompt` | `str` | `""` | System-level instruction that defines the assistant persona and constraints. |
| `temperature` | `float` \| `None` | `0.7` | Controls randomness. Lower values are more deterministic; higher values are more creative. |
| `max_tokens` | `int` | `512` | Maximum number of tokens to generate in the response. |
| `timeout` | `int` \| `None` | `None` | Maximum time in seconds to wait for a response. |
| `tools` | `list[Callable]` | `None` | Optional LangChain-compatible tool functions the model can call. |
| `**kwargs` | `dict` | `{}` | Additional keyword arguments passed to the underlying model constructor. |

### Reasoning Budgets

For llama.cpp models, `chat_stream_reasoning()` accepts a `reasoning_effort` integer token budget that controls how much the model reasons. The following constants are exported for convenience:

| Constant | Value | Description |
| :-- | :-- | :-- |
| `REASONING_BUDGET_UNRESTRICTED` | `-1` | Unrestricted reasoning. |
| `REASONING_BUDGET_OFF` | `0` | Reasoning disabled. |
| `REASONING_BUDGET_LOW` | `64` | Low reasoning budget. |
| `REASONING_BUDGET_MEDIUM` | `512` | Medium reasoning budget. |
| `REASONING_BUDGET_HIGH` | `2048` | High reasoning budget. |

Genie models do not support reasoning streaming and raise a `NotImplementedError` when `chat_stream_reasoning()` is called.

## Methods

- **`chat(message, images=None)`**: Sends a prompt and returns the complete generated response as a string. Blocks until generation is finished.
- **`chat_stream(message, images=None)`**: Sends a prompt and yields generated text chunks as they arrive.
- **`chat_stream_reasoning(message, images=None, reasoning_effort=None)`**: Streams both the model's reasoning (chain-of-thought) and its final answer, yielding `ReasoningChunk` and `ContentChunk` items. Supported only for llama.cpp models.
- **`stop_stream()`**: Requests cancellation of the active streaming response.
- **`with_memory(max_messages=DEFAULT_MEMORY, persistence=None)`**: Enables conversational memory for the instance. `persistence=None`/`False` keeps memory in-process, `True` enables a default persistence backend, and a `MessagePersistence` instance gives full control.
- **`clear_memory()`**: Clears the active conversation history.
- **`list_models()`**: Returns the list of local model identifiers exposed by the model service.
- **`get_client()`**: Returns the underlying LangChain `BaseChatModel` instance.

## Troubleshooting

### Model not found

**Fix:** Verify that the selected LLM model is downloaded and available in App Lab. If you override the model, make sure the identifier matches a model exposed by the local `genie` or `llamacpp` service and uses the correct provider prefix (`genie:` or `llamacpp:`).

### Response generation fails with a memory error

**Fix:** Reduce `max_tokens`, close other running applications, and restart the app. Large models can require significant memory during model loading and inference.

### Empty or generic responses

**Fix:** Use a more specific prompt, adjust the `temperature`, or provide a clear `system_prompt` to guide the model's behavior.

