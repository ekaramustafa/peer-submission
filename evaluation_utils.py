from rouge_score import rouge_scorer
from typing import List, Dict
from transformers import pipeline
import logging

from dataclasses import dataclass

## for rouge tests - contextual integrity tests
@dataclass
class RougeScores:
    original: str
    anonymized: str
    rouge_1: float
    rouge_2: float
    rouge_3: float

def calculate_rouge_n(reference, hypothesis, n=1):
    """
    Calculate ROUGE-N score between a reference and hypothesis text.

    :param reference: The reference text (ground truth).
    :param hypothesis: The generated text to evaluate.
    :param n: The 'N' for ROUGE-N (e.g., 1 for ROUGE-1, 2 for ROUGE-2, etc.).
    :return: A dictionary with precision, recall, and F1-score for ROUGE-N.
    """
    scorer = rouge_scorer.RougeScorer([f'rouge{n}'], use_stemmer=True)
    scores = scorer.score(reference, hypothesis)
    return scores[f'rouge{n}']



## for cosine similarity tests - contextual integrity tests
@dataclass
class CosineScores:
    original: str
    anonymized: str
    cosine_similarity: float


# for denanonymization attack
@dataclass
class DeanonymizationResult:
    original_entity: str
    anonymized_entity: str
    masked_text: str
    predictions: List[Dict[str, float]]
    entity_type: str
    is_correct: bool

class BertDeanonymizer:
    """
    Deanonymizer that attempts to recover original entities using BERT masked language modeling
    """
    
    def __init__(self, model_name="bert-base-uncased", top_k=10, use_context=True, context_window=10):
        self.unmasker = pipeline('fill-mask', model=model_name)
        self.supported_unmaskers = {
            "tr" : pipeline('fill-mask', model="dbmdz/bert-base-turkish-cased"),
            "en" : pipeline('fill-mask', model="bert-base-uncased")
        }
        self.top_k = top_k
        self.use_context = use_context
        self.context_window = context_window
        self.logger = logging.getLogger("bert-deanonymizer")

    def attack(self, original_text: str, anonymized_text: str, analyzer_results, language: str) -> List[DeanonymizationResult]:
        """
        Attempt to deanonymize entities by comparing original positions with anonymized text
        
        Args:
            original_text: The original text with entities
            anonymized_text: The anonymized version of the text
            analyzer_results: Original entity positions and types from analyzer
            
        Returns:
            List of DeanonymizationResult containing attack results
        """
        results = []

        unmasker = self.supported_unmaskers.get(language, self.unmasker)
        
        for entity in analyzer_results:
            try:
                # Get original entity information
                start, end = entity.start, entity.end
                original_entity = original_text[start:end]
                entity_type = entity.entity_type
                
                # Get anonymized entity at same position
                anonymized_entity = anonymized_text[start:end]
                
                # Create masked text for prediction
                if self.use_context:
                    context_start = max(0, start - self.context_window)
                    context_end = min(len(anonymized_text), end + self.context_window)
                    masked_text = (
                        anonymized_text[context_start:start] + 
                        unmasker.tokenizer.mask_token + 
                        anonymized_text[end:context_end]
                    )
                else:
                    masked_text = (
                        anonymized_text[:start] + 
                        unmasker.tokenizer.mask_token + 
                        anonymized_text[end:]
                    )
                
                # Skip if text is just the mask token
                if masked_text == unmasker.tokenizer.mask_token:
                    continue
                    
                # Get predictions
                predictions = unmasker(masked_text)
                top_predictions = predictions[:self.top_k]
                
                # Check if original entity is in predictions
                is_correct = original_entity.lower() in [
                    p['token_str'].lower() for p in top_predictions
                ]
                
                results.append(DeanonymizationResult(
                    original_entity=original_entity,
                    anonymized_entity=anonymized_entity,
                    masked_text=masked_text,
                    predictions=top_predictions,
                    entity_type=entity_type,
                    is_correct=is_correct
                ))
                
            except Exception as e:
                self.logger.error(f"Error processing entity: {str(e)}")
                continue
                
        return results