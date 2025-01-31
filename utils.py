"""
Utility Functions
"""
from typing import Dict
import spacy
import spacy_fastlang

def detect_language(input_text : str):
    """
    Detect the language of the given text
    """
    nlp = spacy.load("en_core_web_sm")
    nlp.add_pipe("language_detector")
    doc = nlp(input_text)
    return doc._.language


def format_ner_results(ner_results : Dict, text : str):
    """
    Print the NER results in a human-readable format
    """
    res = ""
    for result in ner_results:
        result_dict = result.to_dict()
        start = int(result_dict['start'])
        end = int(result_dict['end'])
        res += f"{result_dict['entity_type']}: {text[start:end]}\n"
    return res

def sample_llm_output(anonymized_text : str):
    """
    Sample the output of the LLM, deterministic for testing purposes
    """
    return "Here is my answer to your prompt : \n\n" + anonymized_text


def read_file(file_path):
    """
    Read the content of the file
    """
    with open(file_path, 'r', encoding='utf-8') as file:
        return file.read()

def write_to_file(file_path, content):
    """
    Write the content to the file
    """
    with open(file_path, 'w', encoding='utf-8') as file:
        file.write(content)