from typing import Optional


class Preprocessor:

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def read_file(self, filepath: str) -> str:
        with open(filepath, "r", encoding=self.encoding) as f:
            return f.read()

    def write_file(self, filepath: str, text: str) -> None:
        with open(filepath, "w", encoding=self.encoding) as f:
            f.write(text)

    def preprocess(self, text: str) -> str:
        normalized = text.replace("\r\n", "\n").replace("\r", "\n")
        return normalized

    def reverse_preprocess(self, text: str) -> str:
        return text

    def get_config(self) -> dict:
        return {
            "encoding": self.encoding,
        }
