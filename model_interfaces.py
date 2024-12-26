from typing import List, Optional
import openai
from anthropic import Anthropic
import google.generativeai as genai
from mistralai.client import MistralClient
from groq import Groq
from bon_agent import ModelInterface

class OpenAIInterface(ModelInterface):
    def __init__(self, api_key: str, model: str = "gpt-4"):
        self.client = openai.Client(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, n: int, temperature: float) -> List[str]:
        response = self.client.chat.completions.create(
            model=self.model,
            messages=[{"role": "user", "content": prompt}],
            n=n,
            temperature=temperature
        )
        return [choice.message.content for choice in response.choices]
    
    def get_logprobs(self, prompt: str, completion: str) -> float:
        # Note: Chat models don't provide direct logprobs access
        # This is a placeholder implementation
        return 0.0

class AnthropicInterface(ModelInterface):
    def __init__(self, api_key: str):
        self.client = Anthropic(api_key=api_key)
    
    def generate(self, prompt: str, n: int, temperature: float) -> List[str]:
        responses = []
        for _ in range(n):
            response = self.client.messages.create(
                model="claude-3-sonnet",
                max_tokens=1000,
                temperature=temperature,
                messages=[{"role": "user", "content": prompt}]
            )
            responses.append(response.content)
        return responses
    
    def get_logprobs(self, prompt: str, completion: str) -> float:
        # Note: Anthropic doesn't provide direct logprobs access
        # This is a placeholder implementation
        return 0.0

class GeminiInterface(ModelInterface):
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-pro')
    
    def generate(self, prompt: str, n: int, temperature: float) -> List[str]:
        responses = []
        for _ in range(n):
            response = self.model.generate_content(
                prompt,
                generation_config=genai.GenerationConfig(
                    temperature=temperature
                )
            )
            responses.append(response.text)
        return responses
    
    def get_logprobs(self, prompt: str, completion: str) -> float:
        # Note: Gemini doesn't provide direct logprobs access
        return 0.0

class MistralInterface(ModelInterface):
    def __init__(self, api_key: str, model: str = "mistral-medium"):
        self.client = MistralClient(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, n: int, temperature: float) -> List[str]:
        responses = []
        for _ in range(n):
            response = self.client.chat(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            responses.append(response.choices[0].message.content)
        return responses
    
    def get_logprobs(self, prompt: str, completion: str) -> float:
        # Note: Mistral doesn't provide direct logprobs access
        return 0.0

class GroqInterface(ModelInterface):
    def __init__(self, api_key: str, model: str = "mixtral-8x7b-32768"):
        self.client = Groq(api_key=api_key)
        self.model = model
    
    def generate(self, prompt: str, n: int, temperature: float) -> List[str]:
        responses = []
        for _ in range(n):
            response = self.client.chat.completions.create(
                model=self.model,
                messages=[{"role": "user", "content": prompt}],
                temperature=temperature
            )
            responses.append(response.choices[0].message.content)
        return responses
    
    def get_logprobs(self, prompt: str, completion: str) -> float:
        # Note: Groq doesn't provide direct logprobs access
        return 0.0
