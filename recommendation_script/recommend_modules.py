"""
Module to create plugin string matching algorithms
"""

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from concurrent.futures.process import ProcessPoolExecutor
from concurrent.futures import as_completed
from fuzzywuzzy import fuzz, process


def fuzz_check(sentence, data, scorer, limit):
    """
    ProcessPool method
    """

    module = __import__('fuzzywuzzy')
    scorer= getattr(getattr(module, 'fuzz'), scorer)
    results = process.extract(sentence, data, scorer=scorer, limit=limit) 


    return [(b,a) for a,b in results]


class ModuleBase:

    def __init__(self, test_sentences, reference_translations):
        super().__init__()

        self.test_sentences = test_sentences
        self.reference_translations = reference_translations

    def split_dataset(self, dataset):
    
        for i in range(0, len(dataset), self.split_size):
            yield dataset[i:i + self.split_size]

    def extract(self, results_sentence, results_translation, limit=6):

        source_translations = []
        for count, res in enumerate(results_sentence):

            match = res[0]
            result = res[1]
            if count == limit:
                break
            source_idx = self.test_sentences.index(result)
            translation = self.reference_translations[source_idx]
            source_translations.append({
                'match': match, 
                'translation': translation, 
                'reference': result
            })

        target_translations = [] 
        for count, res in enumerate(results_translation):

            match = res[0]
            result = res[1]
            if count == limit:
                break
            translation_idx = self.reference_translations.index(result)
            reference = self.test_sentences[translation_idx]
            target_translations.append({
                'match': match, 
                'translation': result, 
                'reference': reference
            })
        return {'from_source': source_translations, 'from_target': target_translations}

class FuzzyRatio(ModuleBase):

    def __init__(self, source_sentences, target_sentences):
        super().__init__(source_sentences, target_sentences)

        self.pool = ProcessPoolExecutor(10)
        self.split_size = 25000
        self.file_identifier = 'fuzzy_ratio'

    def do(self, sentences, sentence):

        futures = []
        results = []
        for data in self.split_dataset(sentences):
            futures.append(self.pool.submit(fuzz_check, sentence, data, 'ratio', 5))
        
        for future in as_completed(futures):
            if not future.exception():
                results.extend(future.result())
        return results

    def find(self, sentence, translation):

        results_sentence = self.do(self.test_sentences, sentence)
        results_translation = self.do(self.reference_translations, translation)

        results_sentence.sort(key=lambda k: k[0], reverse=True)
        results_translation.sort(key=lambda k: k[0], reverse=True)

        return self.extract(results_sentence, results_translation)



class CosineSimilarity(ModuleBase):

    def __init__(self, source_sentences, target_sentences):
        super().__init__(source_sentences, target_sentences)

        self.vectorizer = TfidfVectorizer()

        # insert placeholders for the test sentences
        self.test_sentences.insert(0, '******TEST SENTENCE *****')
        self.reference_translations.insert(0, '******TEST SENTENCE *****')
        self.file_identifier = 'cosine_sim'


    def sort_results(self, data, source_sentences, target_sentences):
        """
        Sort the result fomr the matrix into similarity desc
        """
        def extract(idx, f):
            return [f, source_sentences[idx], target_sentences[idx]]
            

        unsorted_results = [extract(idx,f) for idx, f in enumerate(data[0])]
        sorted_results = sorted(unsorted_results,key=lambda k: k[0], reverse=True)
        results = []

        last_similarity_match = 0
        for result in sorted_results:

            if result[0] == last_similarity_match:
                continue
            last_similarity_match = result[0]
            results.append(result)
        return results


    def find(self, source_sentence, target_sentence=None):

        # Need to do this twice, on for the source, and one for  the target

        # Inject the current sentece under test into the list of all sentences
        self.test_sentences[0] = source_sentence
        self.reference_translations[0] = target_sentence

        tf_idf_matrix = self.vectorizer.fit_transform(self.test_sentences)
        w = tf_idf_matrix[0:1]
        source_data =  cosine_similarity(tf_idf_matrix[0:1], tf_idf_matrix)

        tf_idf_matrix = self.vectorizer.fit_transform(self.reference_translations)
        w = tf_idf_matrix[0:1]
        target_data =  cosine_similarity(tf_idf_matrix[0:1], tf_idf_matrix)

        # extract the values and sort into similarity desc
        results_sentence = self.sort_results(source_data, self.test_sentences, self.reference_translations)
        results_translation = self.sort_results(target_data, self.reference_translations, self.test_sentences)

        return self.extract(results_sentence, results_translation)
