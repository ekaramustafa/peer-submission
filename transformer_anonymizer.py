from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline
from typing import List, Dict, Optional
from presidio_anonymizer.entities import (
    EngineResult,
    OperatorConfig,
    OperatorResult,
    PIIEntity,
)
class TransformerAnonymizer:
    """
    Simple anonymizer that uses BERT to replace text with contextually appropriate substitutes
    """

    def __init__(self, model_name: str = "bert-base-uncased"):
        self.model_name = model_name
        self.supported_unmaskers = {
            'tr' : pipeline('fill-mask', model="dbmdz/bert-base-turkish-cased"),
            'en' : pipeline('fill-mask', model="bert-base-uncased")
        }
        self.unmasker = pipeline('fill-mask', model=model_name)
        self.mask_token = self.unmasker.tokenizer.mask_token

    def anonymize(self, text: str, entities: List[Dict], language) -> str:
        """
        Anonymize text by replacing entities with BERT predictions
        
        :param text: Original text
        :param entities: List of dicts with 'start', 'end' and 'entity_type' keys
        :return: Anonymized text
        """
        unmasker = self.supported_unmaskers.get(language, self.unmasker)
        
        result = text
        engine_result = EngineResult()
        for entity in entities:
            start = entity.start
            end = entity.end
            original_text = text[start:end]
            
            # left_context = text[max(0, start-50):start]
            # right_context = text[end:min(len(text), end+50)]
            
            # masked_text = f"{left_context}{self.mask_token}{right_context}"
            masked_text = text[:start] + self.mask_token + text[end:]

            # Get predictions
            predictions = unmasker(masked_text)
            
            replacement = None
            for pred in predictions:
                if pred['token_str'].lower() != original_text.lower():
                    replacement = pred['token_str']
                    break
            
            if not replacement:
                replacement = '[MASKED]'
                
            # Replace in text
            result = result[:start] + replacement + result[end:]

            result_item = OperatorResult(
                start=start,
                end=end,
                entity_type=entity.entity_type,
                text=result,
                operator=f'bert-anonymizer-{language}',
                original_text=text
            )

            engine_result.add_item(result_item)
            engine_result.set_text(result)
            
        return engine_result
