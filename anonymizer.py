"""
Anonymizer
"""
import logging
from presidio_anonymizer import AnonymizerEngine
from presidio_anonymizer.entities import OperatorConfig

class Anonymizer():
    """
    Anonymizer class to anonymize the text based on the NER results
    """

    # Default operator configuration
    operator_config = {
        "DEFAULT": OperatorConfig(operator_name="fake", params={"type": "default"}),
        "PERSON": OperatorConfig(operator_name="fake", params={"type": "name"}),
        "LOCATION": OperatorConfig(operator_name="fake", params={"type": "address"}),
        "ORGANIZATION": OperatorConfig(operator_name="fake", params={"type": "organization"}),
        "CITY": OperatorConfig(operator_name="fake", params={"type": "city"}),
        "COUNTRY": OperatorConfig(operator_name="fake", params={"type": "country"}),
        "EMAIL": OperatorConfig(operator_name="fake", params={"type": "email"}),
        "PHONE_NUMBER": OperatorConfig(operator_name="fake", params={"type": "phone_number"}),
        "DATE_TIME": OperatorConfig(operator_name="fake", params={"type": "date"}),
        "IBAN_CODE" : OperatorConfig(operator_name="fake", params={"type": "iban"}),
        "CREDIT_CARD": OperatorConfig(operator_name="fake", params={"type": "credit_card"}),
        "URL" : OperatorConfig(operator_name="fake", params={"type": "url"}),
        "IP_ADDRESS":  OperatorConfig(operator_name="fake", params={"type" : "ip_address"}),
    }

    def __init__(self):
        self.anonymizer = AnonymizerEngine()
        self.logger = logging.getLogger("presidio-anonymizer-wrapper")
    
    def anonymize(self, text : str, analyzer_results : dict, language : str = None, operators : dict = operator_config) -> str:
        """
        Anonymize the given text based on the NER results
        """

        # Add the language to the operator configuration
        if language is not None:
            for k,v in operators.items():
                v.params['language'] = language
        
        anonymized_results = self.anonymizer.anonymize(
             text=text, 
             analyzer_results=analyzer_results, 
             operators=operators
        )
        return anonymized_results
    
