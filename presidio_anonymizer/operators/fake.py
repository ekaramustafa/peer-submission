"""
Generate fake data based on the given type.
"""

from typing import Dict
from faker import Faker
from presidio_anonymizer.operators import Operator, OperatorType

class Fake(Operator):
    """Generate fake data based on the given type."""
        
    def operate(self, text: str, params: Dict = None) -> str:
        """
        Generate fake data based on the given type.

        :param text: the text to be anonymized
        :param params:
            type: The type of fake data to generate
        :return: the fake data
        """
        if not params or 'type' not in params:
            return text
        
        language_prefix = params.get('language', 'en')
        
        # if language_prefix not in list(Faker.locales): TO-DO
        #     raise ValueError(f"Language {language_prefix} is not supported. Supported languages are {Faker.LOCALES}")
        
        if language_prefix == 'en':
            self.fake = Faker()
        else:
            language = f"{language_prefix}_{language_prefix.upper()}"
            self.fake = Faker(language)
        
        type = params['type'].lower()
        entity_type = params['entity_type'].lower()
        if type == 'name' or entity_type == 'person':
            return self.fake.name()
        elif type == 'address' or entity_type == 'address':
            return self.fake.city() 
        elif type == 'organization' or entity_type == 'organization':
            return self.fake.company()
        elif type == 'city': 
            return self.fake.city() 
        elif type == 'country':
            return self.fake.country()
        elif type == 'location' or entity_type == 'location':
            return self.fake.country()
        elif type == 'email_address' or entity_type == 'email_address':
            return self.fake.email()
        elif type == 'phone_number' or entity_type == 'phone_number':
            return self.fake.phone_number()
        elif type == 'date' or entity_type == 'date_time':
            return self.fake.date()
        elif type == 'iban' or entity_type == 'iban_code':
            return self.fake.iban()
        elif type == 'credit_card' or entity_type == 'credit_card':
            return self.fake.credit_card_number()
        elif type == 'url' or entity_type == 'url':
            return self.fake.url()
        elif type == "ip_address" or entity_type == "ip_address":
            return self.fake.ipv4_public()
        else:
            return text

    def validate(self, params: Dict = None) -> None:
        """
        Validate the parameters for fake data generator.

        :param params:
            type: The type of fake data to generate 
        """
        if not params or 'type' not in params:
            raise ValueError("Invalid parameters for FakeDataGenerator. 'type' is required.")

    def operator_name(self) -> str:
        """Return operator name."""
        return "fake"

    def operator_type(self) -> OperatorType:
        """Return operator type."""
        return OperatorType.Anonymize