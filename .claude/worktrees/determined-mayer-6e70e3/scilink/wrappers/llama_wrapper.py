import io
from PIL import Image
import base64
from types import SimpleNamespace
from llama_cpp import Llama

class LocalLlamaModel():
    """
    This class pretends to be a GenerativeModel.
    """
    def __init__(self, model_path = "../gemma3_27B_QAT_local/gemma-3-27b-it-q4_0.gguf", n_ctx = 3000):
        self.model = Llama(model_path = model_path, n_gpu_layers = -1, n_ctx=8096, verbose= False)
        
    def generate_content(self, contents, generation_config = None, safety_settings = None):
        message = self.prompt_parser(contents)
        response= self.model.create_chat_completion(message, max_tokens=8096)
        choice = response['choices'][0]
        response_text = choice['message']['content']
        finish_reason = choice.get('finish_reason', 'stop')
        response_text = self.fix_json_format(response_text)
        
        #Llama_cpp's finish rason is a string instead of int. Let's map them as int.
        finish_reason_map = {
            "stop": 1,
            "length": 0
        }
        mapped_finish_reason = finish_reason_map.get(finish_reason, -1)

        # Build Gemini-compatible dummy candidates list
        candidate = SimpleNamespace(
            content=response_text,
            finish_reason=mapped_finish_reason
        )
        candidates = [candidate]

        final_response = SimpleNamespace(
            text=response_text,
            candidates=candidates
        )

        return final_response

    def fix_json_format(self, response_text):
        # Remove the additional characters
        if "json" in response_text:
            print("Warning: Removing unexpected 'json' from LLM output.")
            response_text = response_text.replace("json", "")

        if "```" in response_text:
            print("Warning: Removing Markdown code block backticks (``` ) from LLM output.")
            response_text = response_text.replace("```", "")
        return response_text
        
    def prompt_parser(self, genaiList):
        """
        Transform google.generativeai style prompt_parts into llama_cpp messages.
        Treat everything as user messages. Image parts will be converted to 
        {"type": "image", "image": <PIL Image>} format.

        Returns:
            messages: list of dicts in llama_cpp chat format.
        """


        messages = []
        user_messages = []

        for x in genaiList:
            if isinstance(x, str):
                # Plain text parts become text entries
                user_messages.append({"type": "text", "text": x})

            elif isinstance(x, dict):
                # Check for image data
                if x.get("mime_type", "").startswith("image/") and "data" in x:
                    try:
                        # Convert bytes data to PIL image for llama_cpp
                        image_bytes = x["data"]
                        image = Image.open(io.BytesIO(image_bytes))
                        user_messages.append({"type": "image", "image": image})
                    except Exception as e:
                        self.logger.warning(f"Failed to parse image part: {e}")
                        user_messages.append({"type": "text", "text": "\n(Image part parsing failed.)"})
                else:
                    # Any other dict is converted to string text
                    user_messages.append({"type": "text", "text": str(x)})

            else:
                # Any other type is cast to string as fallback
                user_messages.append({"type": "text", "text": str(x)})

        # Combine into final messages list
        messages.append({
            "role": "user",
            "content": user_messages
        })

        # Return parsed messages to be used in model.create_chat_completion
        return messages
