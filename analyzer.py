"""
Analyzer class is a wrapper class for Presidio Analyzer Engine.
"""
import os
from dotenv import load_dotenv
from presidio_analyzer import AnalyzerEngine
from presidio_analyzer.nlp_engine import NlpEngineProvider
from typing import List

from utils import detect_language
load_dotenv()
class Analyzer:
    """
    Presidio Custom Analyzer class
    """
    defaut_config_file_path = "NER/cfg/turkish_spacy_config.yml"
    def __init__(self, supported_languages : List[str] = ["en","tr"], config_file_path: str = None):
        """
        Initialize the Analyzer class
        """
        self.supported_languages = supported_languages
        # self.config_file = os.environ.get("DEFAULT_TRANSFORMER_CONFIG_FILE") if config_file_path is None else config_file_path 
        self.config_file = Analyzer.defaut_config_file_path if config_file_path is None else config_file_path 
        self.provider = NlpEngineProvider(conf_file=self.config_file)
        self.nlp_engine = self.provider.create_engine()
        self.analyzer = AnalyzerEngine(
            nlp_engine=self.nlp_engine, 
            supported_languages=self.supported_languages
        )
    
    def analyze(self, text, language):
        """
        Analyze the text with the given language
        """
        if language not in self.supported_languages:
            raise ValueError(f"Language {language} is not supported. Supported languages are {self.supported_languages}")
        results = self.analyzer.analyze(text=text, language=language)

        return results
    
    def analyze_all(self, text):
        """
        Analyze the text with all supported languages
        """
        results = {}
        for language in self.supported_languages:
            results[language] = self.analyze(text, language)
        return results
