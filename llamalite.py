import base64
import io
from typing import List, Optional, Dict, Any

import httpx
from openai import OpenAI
from PIL import Image

class LlamaLiteClient:
    """
    A client for interacting with a llama.cpp server's chat completions endpoint.

    This client facilitates sending text and image-based prompts to a llama.cpp
    server, managing chat history, and handling responses. It is designed to
    work with the OpenAI Python library, adapting its interface for use with
    the llama.cpp server.

    Attributes:
        client (OpenAI): An instance of the OpenAI client, configured to
                         communicate with the llama.cpp server.
    """

    def __init__(self, base_url: str = "http://localhost:8080/v1"):
        """
        Initializes the LlamaLiteClient.

        Args:
            base_url (str): The base URL of the llama.cpp server. This should
                            point to the server's API endpoint, typically ending
                            in '/v1'.
        """
        # Explicitly create an httpx.Client to pass to the OpenAI client.
        # This can prevent a TypeError related to an unexpected 'proxies'
        # argument that may occur with certain versions of the openai and
        # httpx libraries.
        http_client = httpx.Client(
            base_url=base_url,
            limits=httpx.Limits(max_connections=100, max_keepalive_connections=20),
        )

        self.client = OpenAI(
            base_url=base_url,
            api_key="sk-no-key-required",  # API key is not required for local server.
            http_client=http_client,
        )


    @staticmethod
    def _pil_image_to_base64(image: Image.Image) -> str:
        """
        Converts a PIL Image object to a Base64 encoded string.

        This is a helper method to format images into the required string
        representation for the chat completions API.

        Args:
            image (Image.Image): The PIL Image object to be converted.

        Returns:
            str: A Base64 encoded string representation of the image.
        """
        with io.BytesIO() as buffered:
            # Determine the image format, defaulting to PNG if not available.
            img_format = image.format if image.format else "PNG"
            image.save(buffered, format=img_format)
            img_bytes = buffered.getvalue()
        
        return base64.b64encode(img_bytes).decode("utf-8")

    def chat(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        images: Optional[List[Image.Image]] = None,
        chat_history: Optional[List[Dict[str, Any]]] = None,
        model: str = "gpt-4",
        **kwargs,
    ) -> Dict[str, Any]:
        """
        Sends a request to the chat completions endpoint of the llama.cpp server.

        This method constructs and sends a request containing the user's prompt,
        optional system message, images, and existing chat history. It then
        returns the server's response.

        Args:
            prompt (str): The user's text prompt.
            system_prompt (Optional[str]): An optional system-level instruction
                                           for the model.
            images (Optional[List[Image.Image]]): A list of PIL Image objects
                                                  to be sent with the prompt.
            chat_history (Optional[List[Dict[str, Any]]]): An existing list of
                                                           messages representing
                                                           the conversation history.
            model (str): The model to use for the completion. While this is a
                         required parameter for the OpenAI library, the actual
                         model used is determined by the llama.cpp server
                         configuration.
            **kwargs: Additional keyword arguments to be passed to the
                      OpenAI client's `chat.completions.create` method.

        Returns:
            Dict[str, Any]: The response from the llama.cpp server, structured as
                            an OpenAI API chat completion object.
        """
        messages = chat_history.copy() if chat_history else []

        # Add the system prompt to the message history if provided.
        if system_prompt and not any(m["role"] == "system" for m in messages):
            messages.insert(0, {"role": "system", "content": system_prompt})

        # Prepare the content for the user's message, including text and images.
        user_content = [{"type": "text", "text": prompt}]
        if images:
            for image in images:
                base64_image = self._pil_image_to_base64(image)
                user_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}"
                    }
                })

        messages.append({"role": "user", "content": user_content})

        # Send the request to the server.
        completion = self.client.chat.completions.create(
            model=model,
            messages=messages,
            stream=False,  # As per requirement, no streaming.
            **kwargs
        )

        return completion

if __name__ == '__main__':
    # This is an example of how to use the LlamaLiteClient.
    # Make sure you have a llama.cpp server running.
    
    # Initialize the client.
    # Replace with your server's URL if it's not running on the default port.
    llama_client = LlamaLiteClient(base_url="http://localhost:8080/v1")

    # --- Example 1: Simple text prompt ---
    print("--- Running Example 1: Simple text prompt ---")
    try:
        response = llama_client.chat(
            prompt="What is the capital of France?",
            system_prompt="You are a helpful assistant."
        )
        # Extract and print the content of the response.
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {e}")
    print("-" * 50)

    # --- Example 2: Continuing a conversation ---
    print("--- Running Example 2: Continuing a conversation ---")
    try:
        # Start a conversation history.
        conversation_history = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is the capital of France?"},
            {"role": "assistant", "content": "The capital of France is Paris."}
        ]
        
        # Ask a follow-up question.
        response = llama_client.chat(
            prompt="And what is its population?",
            chat_history=conversation_history
        )
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {e}")
    print("-" * 50)

    # --- Example 3: Text and image prompt ---
    # This example requires an image file named 'example_image.png' in the same
    # directory as this script.
    print("--- Running Example 3: Text and image prompt ---")
    try:
        # Create a dummy image for demonstration purposes.
        # In a real scenario, you would load an image from a file or another source.
        try:
            # Try to open an existing image file.
            img = Image.open("example_image.png")
        except FileNotFoundError:
            print("Creating a dummy image because 'example_image.png' was not found.")
            # If the file doesn't exist, create a simple one.
            img = Image.new('RGB', (200, 100), color = (73, 109, 137))
            # You might want to add text to the image to make it more interesting.
            from PIL import ImageDraw
            d = ImageDraw.Draw(img)
            d.text((10,10), "Hello, LLaMA!", fill=(255,255,0))
            img.save("example_image.png")
            print("'example_image.png' created.")

        response = llama_client.chat(
            prompt="What do you see in this image?",
            images=[img]
        )
        print("Response:", response.choices[0].message.content)
    except Exception as e:
        print(f"An error occurred: {e}")
    print("-" * 50)
