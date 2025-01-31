"""
Deanonymizer
"""
import logging
from presidio_anonymizer import DeanonymizeEngine
from presidio_anonymizer.entities.engine.result import OperatorResult
from typing import List
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np

class Deanonymizer:
    """
    The Deanonymizer class for replacing anonymized values with the original PII values based on provided items.
    """

    def __init__(self) -> None:
        self.deanonymizer = DeanonymizeEngine()
        self.logger = logging.getLogger("presidio-deanonymizer")
    
    def _calculate_similarity(self, text1: str, text2: str) -> float:
        """
        Calculate cosine similarity between two strings.
        """
        vectorizer = TfidfVectorizer(analyzer='char_wb', ngram_range=(2, 3))
        tfidf_matrix = vectorizer.fit_transform([text1, text2])
        cosine_sim = cosine_similarity(tfidf_matrix[0:1], tfidf_matrix[1:2])[0][0]
        return cosine_sim
    
    def deanonymize(self, anonymized: str, similarity_threshold : int = 0.7) -> str:
        """
        Replace anonymized values with the original PII values based on provided items.
        """
        text = anonymized.text
        originals = [item.to_dict()['original_text'] for item in anonymized.items]
        modifieds = [item.to_dict()['text'] for item in anonymized.items]

        for modified, original in zip(modifieds, originals):
            start_idx = 0
            while start_idx < len(text):
                match_idx = text.find(modified, start_idx)
                if match_idx == -1:
                    break

                # Calculate similarity to ensure match
                similarity_score = self._calculate_similarity(text[match_idx:match_idx + len(modified)], modified)
                if similarity_score >= similarity_threshold:
                    text = text[:match_idx] + original + text[match_idx + len(modified):]
                    start_idx = match_idx + len(original)
                else:
                    start_idx = match_idx + len(modified)

        return text