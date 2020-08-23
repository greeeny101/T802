from fuzzywuzzy import process
from nltk.tokenize import WordPunctTokenizer
import datetime
import os
from collections import defaultdict
import json
import logging
from recommend_modules import FuzzyRatio

logging.getLogger


def fuzz_check(sentence, data, scorer, limit):
    
    # search for top n matches of sentence in dataset

    module = __import__('fuzzywuzzy')
    scorer = getattr(getattr(module, 'fuzz'), scorer)

    results = process.extract(sentence, data, scorer=scorer, limit=limit) 

    return results


class Recommendations:

    def __init__(self, language='en', split_size=25000):
        super().__init__()
        self.language = language
        self.clean_source_sentences = None  # to fuzzy english sentences
        self.clean_target_sentences = None  # to fuzzy translated sentences
        self.recommend_module = None

    def load_dataset(self, source, target):

        with open(source) as source, open(target) as target:

            self.clean_source_sentences = [line.replace('\n', ' ').strip() for line in source.readlines()]
            self.clean_target_sentences = [line.replace('\n', ' ').strip() for line in target.readlines()]

    def tokenise_sentence(self, sentence):

        tokens = WordPunctTokenizer().tokenize(sentence)
        return ' '.join(tokens).replace('\n', '').trim()

    def find(self, source_sentence, target_sentence=None):

        results_source = self.do(self.source_sentences, source_sentence)
        results_target = self.do(self.target_sentences, target_sentence)

        results_source.sort(key=lambda k: k[1], reverse=True)
        results_target.sort(key=lambda k: k[1], reverse=True)

        sources = []
        source_translations = []
        for count, res in enumerate(results_source):

            result = res[0]
            match = res[1]
            if count > 6:
                break
            source_id = self.source_dataset.get(result)
            sources.append(result)
            translation = self.target_dataset.get(source_id)
            source_translations.append((result, translation, match))

        target_translations = [] 
        for count, res in enumerate(results_target):

            result = res[0]
            match = res[1]
            if count > 6:
                break
            source_id = {v:k for k,v in self.target_dataset.items()}.get(result)
            sources.append(result)
            translation = {v:k for k,v in self.source_dataset.items()}.get(source_id)
            target_translations.append((result, translation, match)) 
        return zip(source_translations, target_translations)

    def get_recommendations(self, sentence, translation):

        # source_sentence = self.tokensise_sentence(source_sentence)
        # target_sentence = self.tokensise_sentence(target_sentence)
        start = datetime.datetime.now()
        sentence = sentence.replace('\n', ' ').strip()
        translation = translation.replace('\n', ' ').strip()

        result = self.recommend_module.find(sentence, translation)
        end = datetime.datetime.now()
        print(end - start)
        logging.info(f'Fetched recommendations in {end -  start} seconds')
        return result


def get_size(val):

    size = ''
    lang_split = val.split('.')
    if len(lang_split) == 2:
        tag = lang_split[0]
        try:
            size = int(tag)
        except Exception:
            pass
    return size


def details(file_name):

    def three(val, lang=None, fn=None):
        lang_dir = '_'.join(val[0:2]) if val[1] != 'clean' else val[0]
        language_model = f'{lang_dir}_blm.{lang}'
        # we may have a size here
        size = get_size(val[2])
        return lang_dir, language_model, size

    def four(val, lang=None, fn=None):
        lang_dir = '_'.join(val[0:2])
        language_model = f'{lang_dir}_blm.{lang}'
        size = get_size(val[3])
        return lang_dir, language_model, size

    def default(val, lang=None, fn=None):
        size = ''
        lang_dir = val[0]
        language_model = fn.replace('clean', 'blm')
        return lang_dir, language_model, size

    thing_map = {
        3: three,
        4: four,
        0: default,
    }

    lang_dir = ''
    size=''
    lang = file_name.split('.')[-1:][0]
    base_file_name = file_name.split('.')[0:-1][0]
    file_name_split = file_name.split('_')

    lang_dir, language_model, size = thing_map.get(len(file_name_split), default)(file_name_split, lang, file_name)

    return lang, base_file_name, lang_dir, language_model, size


def create_recommenders(base):

    CLEAN_FILES = f'{base}/clean'
    recommendars = {}
    for dirpath, dirnames, files in os.walk(CLEAN_FILES):
        print(f'Found directory: {dirpath}')
        size = ''
        for file_name in files:
            if 'spanish_ca' in file_name:
                continue
            lang, base_file_name, lang_dir, language_model, size = details(file_name)
            if size or lang == 'en':
                continue
            r = Recommendations(language=lang)
            r.load_dataset(f'{dirpath}/{base_file_name}.en', f'{dirpath}/{file_name}')

            recommendars[lang] = r
    return(recommendars)


class TestData:

    def __init__(self, lang='en'):
        super().__init__()

        self.test_sentences = []
        self.reference_translations = []

    def load(self, source_file, target_file):

        with open(source_file, 'r') as source, open(target_file, 'r') as target:

            self.test_sentences = source.readlines()
            self.reference_translations = target.readlines()

    @staticmethod
    def load_all_test_data(base):

        TEST_FILES = f'{base}/clean' # english sentences

        test_data = {}
        for dirpath, dirnames, files in os.walk(TEST_FILES):
            print(f'Found directory: {dirpath}')
            size = ''
            for file_name in files:
                if 'spanish_ca' in file_name:
                    continue
                lang, base_file_name, lang_dir, language_model, size = details(file_name)
                if size or lang == 'en':
                    continue
                r = TestData(lang=lang)

                r.load(f'{dirpath}/{base_file_name}.en', f'{dirpath}/{file_name}')

                test_data[lang] = r
        return(test_data)


class Translation:

    def __init__(self, lang='en'):
        super().__init__()

        self.lang = lang
        self.trained_model_size = 1000000
        self.bleu = ''
        self.translations = []

    def load(self, bleu_file, translations_file):

        with open(bleu_file, 'r') as bleu, open(translations_file, 'r') as tr:

            self.bleu = bleu.readline()
            self.translations = tr.readlines()


    @staticmethod
    def load_all_translated_data(base):

        import re

        TRANSLATED_FILES = f'{base}/translated' # translated, languages, per trained model
        translated_data = defaultdict(list)
        for dirpath, dirnames, files in os.walk(TRANSLATED_FILES):
            print(f'Found directory: {dirpath}')
            size = ''
            for file_name in files:
                if '_bleu' not in file_name:
                    continue
                lang, _, lang_dir, _, _ = details(file_name)


                t = Translation(lang=lang)
                try:
                    size = int(re.search(r'\d+', lang_dir).group())
                    t.trained_model_size = size
                except AttributeError:
                    pass

                t.load(f'{dirpath}/{file_name}', f'{dirpath}/{file_name.replace("_bleu", "")}')
                translated_data[lang].append(t)
        return(translated_data)


class Processor:

    def __init__(self, lang, recommender, test_data_objects, translated_data, recommend_module):
        super().__init__()

        self.lang = lang
        self.recommender = recommender[lang] # trained_data
        self.test_data = test_data_objects[lang] # cleanded data
        self.translated_data = translated_data[lang] # actual translations of test sentences
        self.recommender.recommend_module = recommend_module(self.recommender.clean_source_sentences, self.recommender.clean_target_sentences)

    def process(self):

        result_data = {}
        for sentence_idx, sentence in enumerate(self.test_data.test_sentences):
            # This count ensures we get the correct sentence/translation pair
            model_data = []
            tokens = WordPunctTokenizer().tokenize(sentence)
            if len(tokens) < 4:
                print(f'skipping because it is too short or too long')
                continue
            print(f'Fetching recommendations for \n{sentence} Ref translation\n{self.test_data.reference_translations[sentence_idx]}')
            for translation_object in self.translated_data:

                translation = translation_object.translations[sentence_idx]
                results = self.recommender.get_recommendations(sentence, translation)
                data = {
                    'trained_model_size': translation_object.trained_model_size,
                    'translated_sentence': translation,
                    'recommendations': results
                }
                model_data.append(data)

            result_data[sentence_idx] = {
                'source_sentence': sentence, 
                'reference_translation': self.test_data.reference_translations[sentence_idx],
                'recommendations': model_data
                }
            if len(result_data) == 100:
                break
        
        with open(f'results_{self.lang}_{self.recommender.recommend_module.file_identifier}.json', 'w') as w:
            w.write(json.dumps(result_data))
        return result_data


if __name__ == '__main__':

    DATA_DIR = 'path/to/data/dir'
    RAW_CORPUS_TRAIN_FILES = f'{DATA_DIR}/train'
    RAW_CORPUS_TEST_FILES = f'{DATA_DIR}/test'

    recommenders = create_recommenders(RAW_CORPUS_TRAIN_FILES)
    # 2. Load the test data
    test_data_objects = TestData.load_all_test_data(RAW_CORPUS_TEST_FILES)

    # 3. get translated data as part of testing models we translated everying from the test source
    translated_data = Translation.load_all_translated_data(RAW_CORPUS_TEST_FILES)

    for l,  td in test_data_objects.items():
        print(len(td.test_sentences), len(td.reference_translations))
        for t in translated_data.get(l):
            print(len(t.translations))
    # 4. loop through languages, create a class and pass in all three dicts
    for language in ['es', 'it', 'de', 'fr']:

        for module in [FuzzyRatio]:
            p = Processor(language, recommenders, test_data_objects, translated_data, recommend_module=module)
        p.process()




