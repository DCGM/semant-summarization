import os


class Config:
    def __init__(self):
        self.PRODUCTION = os.getenv("PRODUCTION", False)
        self.JSON_SORT_KEYS = False
        self.OPENAI_KEY = os.getenv("OPENAI_KEY")
        self.OPENAI_URL = os.getenv("OPENAI_URL", None)
        self.OLLAMA_KEY = os.getenv("OLLAMA_KEY")
        self.OLLAMA_URL = os.getenv("OLLAMA_URL")


config = Config()
