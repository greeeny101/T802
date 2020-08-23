from nltk.translate.bleu_score import SmoothingFunction, sentence_bleu
from nltk.tokenize import TreebankWordTokenizer
import json
from collections import defaultdict
from math import sqrt


def load(filename):

    data = None
    with open(filename, 'rb') as fp:
        data = json.load(fp)
    return data
    

def compare(data):

    for _, sentence in data.items():

        source_sentence = sentence['source_sentence']
        reference_translation = sentence['reference_translation']
        tokenised_reference_translation = [TreebankWordTokenizer().tokenize(reference_translation)]
        sentence['target_bleu_score'] = defaultdict(dict)
        sentence['recommendations'].sort(key=lambda k: k['trained_model_size'], reverse=True)
        for trained_model_recommendations in sentence['recommendations']:
            machine_translated_sentence = trained_model_recommendations['translated_sentence']
            tokenised_machine_translation = TreebankWordTokenizer().tokenize(
                machine_translated_sentence
            )
            try:
                mt_score = sentence_bleu(
                    tokenised_reference_translation, 
                    tokenised_machine_translation, 
                    weights=(1,0,0,0), 
                    smoothing_function=SmoothingFunction().method4
                )
            except Exception:
                mt_score = 0
            trained_model_recommendations['translated_sentence_bleu_score'] = mt_score
            if 'source_bleu_score' not in sentence:
                from_source = trained_model_recommendations['recommenadtions']['from_source']
                score_bleu = []
                score_max = []
                for recommendation in from_source:

                    tokenised_translation = TreebankWordTokenizer().tokenize(
                        recommendation['translation']
                    )
                    try:
                        score = sentence_bleu(
                            tokenised_reference_translation, 
                            tokenised_translation, 
                            weights=(1,0,0,0), 
                            smoothing_function=SmoothingFunction().method4
                        )
                    except Exception:
                        score = 0
                    score_bleu.append(score)
                    score_max.append(score * 100)
                av = sum(score_bleu) / len(score_bleu)
                va = sum([(x-av)*2 for x in score_max])/len(score_max)
                sd = sqrt(va)
                sentence['source_bleu_score'] = {'each': score_bleu, 'mean': av, 'sd': sd/100}

            model_size = trained_model_recommendations['trained_model_size']
            from_target = trained_model_recommendations['recommenadtions']['from_target']

            score_bleu = []
            score_max = []
            for recommendation in from_target:

                tokenised_translation = TreebankWordTokenizer().tokenize(
                    recommendation['translation']
                    )
                try:
                    score = sentence_bleu(
                        tokenised_reference_translation, 
                        tokenised_translation, 
                        weights=(1,0,0,0), 
                        smoothing_function=SmoothingFunction().method4
                    )
                except Exception:
                    score = 0
                score_bleu.append(score)
                score_max.append(score * 100)
            if len(score_bleu):
                av = sum(score_bleu) / len(score_bleu)
                va = sum([(x-av)*2 for x in score_max])/len(score_max)
                sd = sqrt(va)
            else:
                av = 0
                sd = 0
            sentence['target_bleu_score'][model_size] = {
                'each': score_bleu, 
                'mean': av, 
                'sd': sd/100
            }

        print(f'^^^ {model_size} ^^^')
        print('SENTENCE `end -------------------------')


if __name__ == '__main__':

    fn = 'results_es_cosine_sim.json'
    data = load(fn)
    compare(data)
    
    with open('bleu_' + fn, 'w') as f:

        json.dump(data, f)
