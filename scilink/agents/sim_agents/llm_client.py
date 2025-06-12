import google.generativeai as genai
import logging


class LLMClient:
    def __init__(self, api_key: str, model_name: str):
        if not api_key:
            raise ValueError("API key not provided to LLMClient.")
        self.model_name = model_name
        try:
            # Configuration should ideally happen once externally,
            # but doing it here is okay for simplicity if this is the only client
            genai.configure(api_key=api_key)
            self.model = genai.GenerativeModel(self.model_name)
            logging.info(f"LLMClient initialized with model: {self.model_name}")
        except Exception as e:
            logging.exception(f"Error configuring GenerativeAI client for {model_name}: {e}")
            raise

    def generate_with_tools(self, prompt: str, tools: list, generation_config=None):
        logging.info("Sending request to LLM...")
        logging.debug(f"Prompt length: {len(prompt)} chars")
        try:
            response = self.model.generate_content(
                prompt,
                tools=tools,
                generation_config=generation_config
            )
            logging.debug(f"LLM Raw Response: {response}")
            return response
        except Exception as e:
            logging.exception(f"Error during LLM content generation: {e}")
            raise

    def send_function_response(self, tool_name: str, response_dict: dict, tools: list):
        logging.info(f"Sending function response for '{tool_name}' back to LLM.")
        logging.debug(f"Function response data: {response_dict}")
        try:
            function_response_proto = genai.protos.FunctionResponse(...) # Adapt as needed
            content_to_send = genai.protos.Content(...) # Adapt as needed
            response = self.model.generate_content(content_to_send, tools=tools)
            logging.debug(f"LLM Raw Response after function result: {response}")
            return response
        except Exception as e:
            logging.exception(f"Error sending function response for tool '{tool_name}': {e}")
            raise