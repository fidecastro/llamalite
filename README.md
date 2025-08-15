# LlamaLite

A simple, minimalistic, streamlined multimodal client for [llama.cpp](https://github.com/ggerganov/llama.cpp).

## Overview

LlamaLite provides a clean Python interface for interacting with llama.cpp servers, supporting both text and image inputs. It leverages the OpenAI Python SDK while adapting it for use with local llama.cpp instances.

## Features

- **Simple API**: Easy-to-use client with minimal setup
- **Multimodal Support**: Handle both text and image inputs
- **Chat History**: Maintain conversation context
- **OpenAI Compatible**: Uses familiar OpenAI SDK patterns
- **Local First**: Designed for local llama.cpp servers

## Installation

```bash
pip install openai httpx pillow
```

## Quick Start

```python
from llamalite import LlamaLiteClient
from PIL import Image

# Initialize client (assumes llama.cpp server running on localhost:8080)
client = LlamaLiteClient()

# Simple text chat
response = client.chat(
    prompt="What is the capital of France?",
    system_prompt="You are a helpful assistant."
)
print(response.choices[0].message.content)

# Chat with image
image = Image.open("photo.jpg")
response = client.chat(
    prompt="What do you see in this image?",
    images=[image]
)
print(response.choices[0].message.content)
```

## Prerequisites

You need a running llama.cpp server with the `--api` flag enabled:

```bash
./llama-server --model your-model.gguf --api --port 8080
```

For multimodal support, use a vision-capable model like LLaVA.

## Usage

### Basic Text Chat

```python
client = LlamaLiteClient()

response = client.chat(
    prompt="Tell me a joke",
    system_prompt="You are a comedian."
)
```

### Conversation with History

```python
history = [
    {"role": "user", "content": "My name is Alice"},
    {"role": "assistant", "content": "Hello Alice! Nice to meet you."}
]

response = client.chat(
    prompt="What's my name?",
    chat_history=history
)
```

### Multimodal Chat

```python
from PIL import Image

image = Image.open("image.jpg")
response = client.chat(
    prompt="Describe this image",
    images=[image]
)
```

## Configuration

```python
# Custom server URL
client = LlamaLiteClient(base_url="http://your-server:8080/v1")

# Additional parameters
response = client.chat(
    prompt="Hello",
    model="gpt-4",  # Model name (informational)
    temperature=0.7,
    max_tokens=100
)
```

## License

MIT License
