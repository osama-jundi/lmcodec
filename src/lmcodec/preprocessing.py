from typing import Optional


class Preprocessor:

    def __init__(self, encoding: str = "utf-8"):
        self.encoding = encoding

    def read_file(self, filepath: str) -> str:
        # newline="" preserves original line endings (\r\n, \r, \n) as-is
        with open(filepath, "r", encoding=self.encoding, newline="") as f:
            return f.read()

    def write_file(self, filepath: str, text: str) -> None:
        # newline="" writes exactly what we give it — no CRLF translation
        with open(filepath, "w", encoding=self.encoding, newline="") as f:
            f.write(text)

    def preprocess(self, text: str) -> tuple[str, dict]:
        """
        Normalize line endings to \\n for the language model.
        Returns the normalized text plus a dict of info needed to reverse the change.

        Bug 2 fix: previous version was not reversible, so CRLF/CR files were
        silently converted to LF after compress+decompress.
        """
        # Build a compact record of every line-ending position in the original
        # so we can reconstruct it exactly at decompression time.
        endings = []   # list of ('rn', 'r', 'n') in order
        i = 0
        normalized_parts = []
        while i < len(text):
            ch = text[i]
            if ch == "\r":
                if i + 1 < len(text) and text[i + 1] == "\n":
                    endings.append("rn")
                    normalized_parts.append("\n")
                    i += 2
                else:
                    endings.append("r")
                    normalized_parts.append("\n")
                    i += 1
            elif ch == "\n":
                endings.append("n")
                normalized_parts.append("\n")
                i += 1
            else:
                normalized_parts.append(ch)
                i += 1

        normalized = "".join(normalized_parts)

        # Only store info if there is a mix; otherwise we can reconstruct from a
        # single marker. This keeps metadata small for typical files.
        unique = set(endings)
        if len(unique) <= 1:
            info = {
                "mode": "uniform",
                "ending": next(iter(unique)) if unique else "n",
            }
        else:
            info = {
                "mode": "mixed",
                "sequence": endings,
            }

        return normalized, info

    def reverse_preprocess(self, text: str, info: Optional[dict] = None) -> str:
        """
        Reverse the line-ending normalization using the stored info dict.
        If info is None (legacy files or no preprocessing recorded), the text
        is returned unchanged.
        """
        if info is None:
            return text

        mode = info.get("mode", "uniform")

        if mode == "uniform":
            ending = info.get("ending", "n")
            if ending == "n":
                return text
            replacement = {"rn": "\r\n", "r": "\r", "n": "\n"}[ending]
            return text.replace("\n", replacement)

        # mixed: walk the text and replace each \n with its recorded ending
        sequence = info["sequence"]
        out = []
        idx = 0
        for ch in text:
            if ch == "\n":
                if idx >= len(sequence):
                    # defensive: fall back to \n if sequence ran out
                    out.append("\n")
                else:
                    tag = sequence[idx]
                    out.append({"rn": "\r\n", "r": "\r", "n": "\n"}[tag])
                    idx += 1
            else:
                out.append(ch)
        return "".join(out)

    def get_config(self) -> dict:
        return {
            "encoding": self.encoding,
        }
