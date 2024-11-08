from typing import List, Type
from openai import OpenAI
import instructor
from pydantic import BaseModel

class EssayEvaluator:
    def __init__(self, system_message_content: str, model: str, response_model: Type[BaseModel]):
        self.client = instructor.from_openai(OpenAI())
        self.system_message_content = system_message_content
        self.model = model
        self.response_model = response_model

    def evaluate_essay(self, data: str) -> BaseModel:
        response = self.client.chat.completions.create(
            model=self.model,
            response_model=self.response_model,
            messages=[
                {
                    "role": "system",
                    "content": self.system_message_content
                },
                {
                    "role": "user",
                    "content": data,
                },
            ],
            timeout=60 
        )
        return response