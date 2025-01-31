from analyzer import Analyzer
from anonymizer import Anonymizer
from deanonymizer import Deanonymizer
from transformer_anonymizer import TransformerAnonymizer

from evaluation_utils import calculate_rouge_n, RougeScores, CosineScores, BertDeanonymizer, DeanonymizationResult
from statistics import mean

from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from datasets import load_dataset


analyzer = Analyzer()
anonymizer = Anonymizer()
deanonymizer = Deanonymizer()
transformer_anonymizer = TransformerAnonymizer(model_name="bert-base-uncased")

def cosine_similarity_test(selected_anonymizer, tokens, tags, language="tr"):
    model = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
    results = []
    
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        try:
            sentence = token
            y_hat = analyzer.analyze(sentence, language=language)
            anonymized = selected_anonymizer.anonymize(sentence, y_hat, language=language)
            if anonymized.text is None:
                continue
            original_embedding = model.encode(sentence, convert_to_tensor=True)
            anonymized_embedding = model.encode(anonymized.text, convert_to_tensor=True)
            similarity = cosine_similarity(
                original_embedding.unsqueeze(0).numpy(), 
                anonymized_embedding.unsqueeze(0).numpy()
            )[0][0]

            # if similarity == 0:
                # continue

            results.append(CosineScores(
                original=sentence,
                anonymized=anonymized,
                cosine_similarity=similarity
            ))
            
            if i % 100 == 0:
                print(f"Example {i} successfully processed")
            
        except Exception as e:
            print(f"Error processing example: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")

    print(f"\nTotal results processed: {len(results)}")
    cosine_similarity_avg = mean([r.cosine_similarity for r in results])
    print(f"Average cosine similarity: {cosine_similarity_avg:.3f}")
    return results

def rouge_test(selected_anonymizer, tokens, tags, language="tr"):

    results = []
    for i, (token, tag) in enumerate(zip(tokens, tags)):
        try:
            sentence = token
            
            y_hat = analyzer.analyze(sentence, language=language)
            anonymized = selected_anonymizer.anonymize(sentence, y_hat, language=language)
            if anonymized.text is None:
                continue
            rouge_1 = calculate_rouge_n(sentence, anonymized.text, n=1)
            rouge_2 = calculate_rouge_n(sentence, anonymized.text, n=2)
            rouge_3 = calculate_rouge_n(sentence, anonymized.text, n=3)

            results.append(RougeScores(
                original=sentence,
                anonymized=anonymized,
                rouge_1=rouge_1.fmeasure,
                rouge_2=rouge_2.fmeasure,
                rouge_3=rouge_3.fmeasure
            ))
            if i % 100 == 0:
                print(f"Example {i} successfully processed")
            
        except Exception as e:
            print(f"Error processing example: {str(e)}")
            print(f"Error type: {type(e)}")
            import traceback
            print(f"Full traceback: {traceback.format_exc()}")

    print(f"\nTotal results processed: {len(results)}")

    rouge_1_avg_test = mean([r.rouge_1 for r in results])
    rouge_2_avg_test = mean([r.rouge_2 for r in results])
    rouge_3_avg_test = mean([r.rouge_3 for r in results])

    print(f"Average ROUGE-1: {rouge_1_avg_test:.3f}")
    print(f"Average ROUGE-2: {rouge_2_avg_test:.3f}")
    print(f"Average ROUGE-3: {rouge_3_avg_test:.3f}")
    return results


def deanonymization_attack_test(tokens, anonymizer, bert_deanonymizer, language="en"):
    """
    Test the deanonymization attack on a set of sentences
    """
    results = []
    
    for sentence in tokens:
        try:
            # Get entities using analyzer
            analyzer_results = analyzer.analyze(sentence, language)
            if not analyzer_results:
                continue
                
            # Anonymize the text
            anonymized = anonymizer.anonymize(sentence, analyzer_results, language=language)
            
            # Attempt to deanonymize
            attack_results = bert_deanonymizer.attack(
                original_text=sentence,
                anonymized_text=anonymized.text,
                analyzer_results=analyzer_results,
                language=language
            )
            
            # Calculate success metrics
            total_entities = len(attack_results)
            successful_recoveries = sum(1 for r in attack_results if r.is_correct)
            
            results.append({
                'original': sentence,
                'anonymized': anonymized.text,
                'attack_results': attack_results,
                'success_rate': successful_recoveries / total_entities if total_entities > 0 else 0
            })
            
        except Exception as e:
            print(f"Error processing sentence: {str(e)}")
            continue

    if results:
        avg_success_rate = sum(r['success_rate'] for r in results) / len(results)
        print(f"\nAverage deanonymization success rate: {avg_success_rate:.2%}")
        
        # Entity type breakdown
        success_by_type = {}
        total_by_type = {}
        
        for result in results:
            for attack_result in result['attack_results']:
                entity_type = attack_result.entity_type
                if entity_type not in success_by_type:
                    success_by_type[entity_type] = 0
                    total_by_type[entity_type] = 0
                    
                total_by_type[entity_type] += 1
                if attack_result.is_correct:
                    success_by_type[entity_type] += 1
        
        print("\nSuccess rates by entity type:")
        for entity_type in success_by_type:
            rate = success_by_type[entity_type] / total_by_type[entity_type]
            print(f"{entity_type}: {rate:.2%} ({success_by_type[entity_type]}/{total_by_type[entity_type]})")
    

def main():
    dataset_names = ["conll2003", "turkish-nlp-suite/turkish-wikiNER"]
    # dataset_name = "conll2003"
    dataset_name = "turkish-nlp-suite/turkish-wikiNER"
    anonymizers = [anonymizer, transformer_anonymizer]
      

    dataset = load_dataset(dataset_name, trust_remote_code=True)
    test_dataset = dataset['test']
    tokens = [" ".join(token_list) for token_list in test_dataset["tokens"]]
    tags = None
    language = "en"
    if dataset_name == "turkish-nlp-suite/turkish-wikiNER":
        tags = test_dataset["tags"]
        language = "tr"
    else:
        tags = test_dataset["ner_tags"]

    # rouge_test(transformer_anonymizer, tokens,tags, language=language)
    # cosine_similarity_test(transformer_anonymizer, tokens, tags, language=language)
    # bert_deanonymizer = BertDeanonymizer(top_k=5,use_context=False)
    # deanonymization_attack_test(tokens, anonymizer, bert_deanonymizer, language=language)

    
if __name__ == "__main__":
    main()
