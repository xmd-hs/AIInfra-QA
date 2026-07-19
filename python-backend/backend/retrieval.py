import math
from collections import Counter
from .text import tokenize


class BM25Retriever:
    def __init__(self, k1: float = 1.5, b: float = 0.75):
        self.k1, self.b = k1, b

    def search(self, query: str, chunks: list[dict], limit: int = 6) -> list[dict]:
        terms = list(dict.fromkeys(tokenize(query)))
        if not terms or not chunks:
            return []
        documents = [(chunk, tokenize(chunk["text"])) for chunk in chunks]
        average_length = sum(len(tokens) for _, tokens in documents) / len(documents) or 1
        frequencies = {term: sum(term in tokens for _, tokens in documents) for term in terms}
        results = []
        for chunk, tokens in documents:
            counts = Counter(tokens)
            score = 0.0
            for term in terms:
                tf = counts[term]
                if not tf:
                    continue
                df = frequencies[term]
                idf = math.log(1 + (len(documents) - df + 0.5) / (df + 0.5))
                score += idf * (tf * (self.k1 + 1)) / (tf + self.k1 * (1 - self.b + self.b * len(tokens) / average_length))
            coverage = sum(term in counts for term in terms) / len(terms)
            if score > 0:
                results.append({**chunk, "score": score * (0.75 + coverage * 0.25), "coverage": coverage})
        return sorted(results, key=lambda item: item["score"], reverse=True)[:limit]
