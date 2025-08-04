from typing import List, Tuple

from FlagEmbedding import FlagReranker


class Reranker:

    def __init__(self):
        self.bge_reranker = FlagReranker("BAAI/bge-reranker-v2-m3", use_fp16=True)

    def rerank(self, text, similar_terms, top_k: int = 5) -> Tuple[List[str], List[str]]:
        bge_results = self.bge_rerank(text, similar_terms)
        parsed, raw = self.parse_results(bge_results, top_k)

        return parsed, raw

    def bge_rerank(self, text, similar_terms) -> Tuple[List[str], List[str]]:

        combinations = [[text, term["text"]] for term in similar_terms]
        scores = self.bge_reranker.compute_score(combinations, normalize=True)

        results = [
            {"text": term["text"], "score": score, "distance": term["distance"]}
            for term, score in zip(similar_terms, scores)
        ]
        return results

    def parse_results(self, results, top_k: int = 5) -> Tuple[List[str], List[str]]:

        # Round scores to 3 decimals
        for item in results:
            item["score"] = round(item["score"], 3)

        # Filter and sort the data
        sorted_data = sorted(results, key=lambda x: (-x["score"], x["distance"]))[:top_k]

        raw_list = [item["text"] for item in sorted_data]

        # Parse the list of dicts into the desired format
        parsed_list = [f"[Score: {item['score']}, Distance: {item['distance']}] {item['text']}" for item in sorted_data]

        return parsed_list, raw_list
