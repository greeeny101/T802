"""
This file python script based upon the EMS experimentaion.perl script provided by Moses that can be found at 
https://github.com/moses-smt/mosesdecoder/blob/master/scripts/ems/fix-info.perl
"""
import os
from subprocess import run
import subprocess
from pathlib import Path
import shutil


WORKING_DIR = '/path_to_working_dir'
  

# moses
MOSES_DIR = '/path_to_/mosesdecoder'

# moses binaries
MOSES_BINARIES = f'{MOSES_DIR}/bin'

# moses scripts
MOSES_SCRIPTS = f'{MOSES_DIR}/scripts'

MULTI_BLEU = f'{MOSES_SCRIPTS}/generic/multi-bleu.perl' 

# directory where GIZA++/MGIZA programs resides
EXTERNAL_BIN_DIR = f'{MOSES_DIR}/tools'

DATA_DIR = '/path_to_data_dir'



# moses decoder
DECODER = f'{MOSES_BINARIES}/moses'

# conversion of rule table into binary on-disk format
TTABLE_BINARISER = f'{MOSES_BINARIES}/CreateOnDiskPt 1 1 4 100 2'

# tokenizers 
INPUT_TOKENISER = f"{MOSES_SCRIPTS}/tokenizer/tokenizer.perl"
OUTPUT_TOKENISER = f"{MOSES_SCRIPTS}/tokenizer/tokenizer.perl"


# truecase training
TRAIN_TRUECASER = f'{MOSES_SCRIPTS}/recaser/train-truecaser.perl'

# truecasers
INPUT_TRUECASER = f'{MOSES_SCRIPTS}/recaser/truecase.perl'
OUTPUT_TRUECASER = f'{MOSES_SCRIPTS}/recaser/truecase.perl'
DETRUECASER = f'{MOSES_SCRIPTS}/recaser/detruecase.perl'

# clean

CLEAN = f'{MOSES_SCRIPTS}/training/clean-corpus-n.perl'

# language models
LANGUAGE_MODEL = f'{MOSES_BINARIES}/lmplz'
LANGUAGE_MODEL_BINARISER = f'{MOSES_BINARIES}/build_binary'


TRAAIN_MODEL = f'{MOSES_SCRIPTS}/training/train-model.perl'
TUNE_MODEL = f'{MOSES_SCRIPTS}/training/mert-moses.pl'

BINARISE_PHRASE_TABLE = f'{MOSES_BINARIES}/processPhraseTableMin'
BINARISE_LEXICAL_TABLE = f'{MOSES_BINARIES}/processLexicalTableMin'

CORES = 8

MAX_SENTENCE_LENGTH = 80

RAW_CORPUS_TRAIN_FILES = (f'{DATA_DIR}/train', 'train')
RAW_CORPUS_TRAIN_TOK_FILES = (f'{DATA_DIR}/train/token', 'tok')
RAW_CORPUS_DEV_FILES = (f'{DATA_DIR}/dev', 'dev')
RAW_CORPUS_TEST_FILES = (f'{DATA_DIR}/test', 'test')

TRAINED_MODEL_DIR = 'trained_model_dir'

def remove_long_lines(base, ext):

    if ext != 'tok':
        return
    for dirpath, dirnames, files in os.walk(base):
        print(f'Found directory: {dirpath}')
        for file_name in files:
            if 'spanish_ca' not in file_name:
                continue
            lang, base_file_name, lang_dir, language_model, size = details(file_name)
            source = []
            target = []
            
            if lang == 'en':
                continue # just process the target ones. It works out the english equivalent
            with open(dirpath + '/' + file_name.replace('es', 'en').replace('train', 'tok'), 'r') as src, open(dirpath + '/' + file_name.replace('train', 'tok'), 'r') as tgt:

                source_lines = src.readlines()
                target_lines = tgt.readlines()

                for idx, s_line in enumerate(source_lines):
                    t_line = target_lines[idx]

                    if len(s_line) >= 80 or len(t_line) >= 80:
                        continue
                    if s_line == '\n' or t_line == '\n':
                        continue
                    source.append(s_line)
                    target.append(t_line)
            
            with open(dirpath + '/' + file_name.replace('es', 'en').replace('train', 'tok'), 'w') as src, open(dirpath + '/' + file_name.replace('train', 'tok'), 'w') as tgt:

                src.writelines(source)
                tgt.writelines(target)


def tokenise(base, ext):

    TOKENISED_FILES = f'{base}/token'
    if ext != 'train':
        return
    for dirpath, dirnames, files in os.walk(base):
        print(f'Found directory: {dirpath}')
        for file_name in files:
            if 'spanish_ca' not in file_name:
                continue
            if ext in file_name:
                outfile = file_name.replace(ext, 'tok')
                lang = file_name.split('.')[-1:][0]

                args = [
                    f'{INPUT_TOKENISER}', '-a', '-l', lang, 
                    '-threads', '8',
                ]
                print(f'processing {file_name}')
                a = subprocess.run(
                    args, 
                    stdin=open(f'{base}/{file_name}', 'r'),
                    stdout=open(f'{TOKENISED_FILES}/{outfile}', 'w'),
                    shell=False
                    )
                print(f'written to {outfile}')


def truecase_train_models(base, ext):

    if ext != 'train':
        return
    TOKENISED_FILES = f'{base}/token'
    TRUECASE_MODEL_FILES = f'{base}/truecase_models'
    for dirpath, dirnames, files in os.walk(TOKENISED_FILES):
        print(f'Found directory: {dirpath}')
        for file_name in files:
            if 'spanish_ca' not in file_name:
                continue
            if 'tok' in file_name:
                outfile = file_name.replace('tok', 'model')
                lang = file_name.split('.')[-1:][0]

                args = [
                    f'{TRAIN_TRUECASER}', 
                    '--model', f'{TRUECASE_MODEL_FILES}/{outfile}',
                    '--corpus', f'{TOKENISED_FILES}/{file_name}',
                ]
                print(f'processing {file_name}')
                print(' '.join(args))
                a = subprocess.run(
                    args, 
                    shell=False
                    )
                print(f'written to {outfile}')


def truecase(base):

    TOKENISED_FILES = f'{base}/token'
    TRUECASE_MODEL_FILES = f'{base}/truecase_models'
    TRUECASE_FILES = f'{base}/truecase'

    TRUECASE_MODEL_FILES = TRUECASE_MODEL_FILES.replace('dev', 'train').replace('test', 'train')
    for dirpath, dirnames, files in os.walk(TOKENISED_FILES):
        print(f'Found directory: {dirpath}')
        for file_name in files:
            if 'spanish_ca' not in file_name:
                continue
            if 'tok' in file_name:
                model = file_name.replace('tok', 'model')
                outfile = file_name.replace('tok', 'true')

                args = [
                    f'{INPUT_TRUECASER}', 
                    '--model', f'{TRUECASE_MODEL_FILES}/{model}'
                ]
                print(f'processing {file_name}')
                print(' '.join(args))
                a = subprocess.run(
                    args, 
                    stdin=open(f'{TOKENISED_FILES}/{file_name}', 'r'),
                    stdout=open(f'{TRUECASE_FILES}/{outfile}', 'w'),
                    shell=False
                    )
                print(f'written to {outfile}')


def clean(base, ext):

    #  ~/mosesdecoder/scripts/training/clean-corpus-n.perl \
    #     ~/corpus/news-commentary-v8.fr-en.true fr en \
    #     ~/corpus/news-commentary-v8.fr-en.clean 1 80

    TRUECASE_FILES = f'{base}/truecase'
    CLEAN_FILES = f'{base}/clean'

    for dirpath, dirnames, files in os.walk(TRUECASE_FILES):
        print(f'Found directory: {dirpath}')
        for file_name in files:
            if 'spanish_ca' not in file_name:
                continue
            lang = file_name.split('.')[-1:][0]
            base_file_name = file_name.split('.')[0:-1][0] # needs the extensions removing as it adds them during the training process

            outfile = base_file_name.replace('true', 'clean')
            args = [
                f'{CLEAN}',
                f'{TRUECASE_FILES}/{base_file_name}', 'en', lang,
                f'{CLEAN_FILES}/{outfile}', '1', '80'
            ]
            print(f'Cleaning {base_file_name}')
            print(' '.join(args))
            a = subprocess.run(
                args, 
                shell=False
                )
            print(f'-----------------------')

def language_models(base, ext):
    """
    Whe processing the training data, create language models for the target languages
    """
    if ext != 'train':
        return

    CLEAN_FILES = f'{base}/clean'
    LANGUAGE_MODEL_FILES = f'{base}/language_models'

    for dirpath, dirnames, files in os.walk(CLEAN_FILES):
        print(f'Found directory: {dirpath}')
        for file_name in files:
            if 'spanish_ca' not in file_name:
                continue
            lang = file_name.split('.')[-1:][0]
            if lang == 'en':
                continue
            if 'clean' in file_name:
                outfile = file_name.replace('clean', 'arpa')
                args = [
                    f'{LANGUAGE_MODEL}', '-o', '3',
                ]
                print(f'processing truecase to language model {file_name}')
                print(' '.join(args))
                a = subprocess.run(
                    args, 
                    stdin=open(f'{CLEAN_FILES}/{file_name}', 'r'),
                    stdout=open(f'{LANGUAGE_MODEL_FILES}/{outfile}', 'w'),
                    shell=False
                    )
                print(f'written to {outfile}')


                infile = f'{LANGUAGE_MODEL_FILES}/{outfile}'
                outfile = outfile.replace('arpa', 'blm')
                base_file_name = file_name.split('.')[0:-1][0]
                print(f'now converting to binary {outfile}')
                args = [
                    f'{LANGUAGE_MODEL_BINARISER}',
                    infile,
                    f'{LANGUAGE_MODEL_FILES}/{outfile}',
                ]
                a = subprocess.run(
                    args, 
                    shell=False
                    )
                print(f'written to {outfile}')


def get_size(val):

    size=''
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


def train(base, ext, size=None):

    #  -root-dir train \
    #  -corpus ~/corpus/news-commentary-v8.fr-en.clean                             \
    #  -f fr -e en -alignment grow-diag-final-and -reordering msd-bidirectional-fe \
    #  -lm 0:3:$HOME/lm/news-commentary-v8.fr-en.blm.en:8                          \
    #  -external-bin-dir ~/mosesdecoder/tools >& training.out &

    if ext != 'train':
        return
    size = size if size else ''
    CLEAN_FILES = f'{base}/clean'
    LANGUAGE_MODEL_FILES = f'{base}/language_models'

    for dirpath, dirnames, files in os.walk(CLEAN_FILES):
        print(f'Found directory: {dirpath}')
        size = ''
        for file_name in files:
            if 'spanish_ca' not in file_name:
                continue
            lang, base_file_name, lang_dir, language_model, size = details(file_name)
            if lang == 'en':
                continue # just process the target ones. It works out the english equivalent

            # Need to update this so it creates the correct root dir, 
            root_dir = f'{TRAINED_MODEL_DIR}/{lang_dir}/train{size}'

            if not Path(root_dir).is_dir():
                Path(root_dir).mkdir(parents=True, exist_ok=True)
            else:
                shutil.rmtree(root_dir) 
                Path(root_dir).mkdir(parents=True, exist_ok=True)
            args = [
                f'{TRAAIN_MODEL}',
                '-root-dir', root_dir,
                '-cores', f'{CORES}',
                '-corpus', f'{CLEAN_FILES}/{base_file_name}',
                '-f', 'en', '-e', f'{lang}',
                '-alignment',  'grow-diag-final-and',  
                '-reordering',  'msd-bidirectional-fe',
                '-lm', f'0:3:{LANGUAGE_MODEL_FILES}/{language_model}:8',
                '-external-bin-dir', f'{EXTERNAL_BIN_DIR}',

            ]
            print(f'Creating training model  {base_file_name}')
            print(' '.join(args))
            a = subprocess.run(
                args, 
                stdout=open(f'{root_dir}/training.out', 'w'), 
                shell=False
                )
            print(f'-----------------------')

def tune(base, ext):

    #  nohup nice ~/mosesdecoder/scripts/training/mert-moses.pl \
    #   ~/corpus/news-test2008.true.fr ~/corpus/news-test2008.true.en \
    #   ~/mosesdecoder/bin/moses train/model/moses.ini --mertdir ~/mosesdecoder/bin/ \
    #   &> mert.out &
    # If you have several cores at your disposal, then it'll be a lot faster to run Moses multi-threaded. Add --decoder-flags="-threads 4" to the last line above in order to run the decoder with 4 threads. With this setting, tuning took about 4 hours for me.

    if ext != 'dev':
        return

    DEV_FILES = f'{base}/clean'
    LANGUAGE_MODEL_FILES = f'{base}/language_models'

    for dirpath, dirnames, files in os.walk(DEV_FILES):
        print(f'Found directory: {dirpath}')
        for file_name in files:
            if 'spanish_ca' not in file_name:
                continue
            lang, base_file_name, lang_dir, language_model, size = details(file_name)
            if lang == 'en':
                continue # just process the target ones. It works out the english equivalent
            
            # Because we use the same set of data, se need to loop through
            # each size here to tune each one
            for size in ['', '100', '1000', '10000', '100000']:
                # Need to update this so it creates the correct root dir, 
                working_dir = f'{TRAINED_MODEL_DIR}/{lang_dir}/mert{size}'
                trained_model_dir = f'{TRAINED_MODEL_DIR}/{lang_dir}/train{size}'

                if not Path(working_dir).is_dir():
                    Path(working_dir).mkdir(parents=True, exist_ok=True)
                else:
                    shutil.rmtree(working_dir) 
                    Path(working_dir).mkdir(parents=True, exist_ok=True)
                args = [
                    f'{TUNE_MODEL}',
                    '--decoder-flags', '-threads 8',
                    '--working-dir', working_dir,
                    f'{DEV_FILES}/{base_file_name}.en',
                    f'{DEV_FILES}/{file_name}',
                    f'{MOSES_BINARIES}/moses',
                    f'{trained_model_dir}/model/moses.ini',
                    '--mertdir',  f'{MOSES_BINARIES}'
                ]
                print(f'tuning model  {base_file_name}')
                print(' '.join(args))
                a = subprocess.run(
                    args, 
                    stdout=open(f'{working_dir}/mert.out', 'w'),
                    shell=False
                    )
                print(f'-----------------------')

def binarise_trained_models(base, ext):

    if ext != 'train':
        return

    
    # Just loop through the clean files and use this as a loop for the mert work updates
    TRAINED_MODELS = f'{base}/clean'

    for dirpath, dirnames, files in os.walk(TRAINED_MODELS):
        print(f'Found directory: {dirpath}')
        for file_name in files:
            if 'spanish_ca' not in file_name:
                continue
            lang = file_name.split('.')[-1:][0]
            base_file_name = file_name.split('.')[0:-1][0] # needs the extensions removing as it adds them during the training process
            if lang == 'en':
                continue  #  just process the target ones
            lang_dir = file_name.split('_')
            if len(lang_dir) == 3:
                lang_dir = '_'.join(lang_dir[0:2])
            else:
                lang_dir = lang_dir[0]
                lang_dir = lang_dir + '_ca'

            for size in ['']:
            # for size in ['', '100', '1000', '10000', '100000']:
                train = f'train{size}'                
                args = [
                    f'{BINARISE_PHRASE_TABLE}',
                    '-in', f'{TRAINED_MODEL_DIR}/{lang_dir}/{train}/model/phrase-table.gz', '-nscores', '4',
                    '-out', f'{TRAINED_MODEL_DIR}/{lang_dir}/{train}/model/phrase_table',
                ]
                print(f'binarising model phrase table - {base_file_name}')
                print(' '.join(args))
                a = subprocess.run(
                    args, 
                    shell=False
                    )
                print(f'-----------------------')
                args = [
                    f'{BINARISE_LEXICAL_TABLE}',
                    '-in', f'{TRAINED_MODEL_DIR}/{lang_dir}/{train}/model/reordering-table.wbe-msd-bidirectional-fe.gz', '-nscores', '4',
                    '-out', f'{TRAINED_MODEL_DIR}/{lang_dir}/{train}/modelreordering-table',
                ]
                print(f'binarising model reordering table - {base_file_name}')
                print(' '.join(args))
                a = subprocess.run(
                    args, 
                    shell=False
                    )
                print(f'-----------------------')

                with open(f'{TRAINED_MODEL_DIR}/{lang_dir}/mert{size}/moses.ini') as e:
                    data = e.read()
                    data.replace('PhraseDictionaryMemory', 'PhraseDictionaryCompact')
                    data.replace(f'{TRAINED_MODEL_DIR}/{lang_dir}/{train}/model/phrase-table.gz', f'{TRAINED_MODEL_DIR}/{lang_dir}/{train}/model/phrase-table.minphr')
                    data.replace(f'{TRAINED_MODEL_DIR}/{lang_dir}/{train}/model/reordering-table.wbe-msd-bidirectional-fe.gz', f'{TRAINED_MODEL_DIR}/{lang_dir}/{train}/model/reordering-table')


def test(base, ext):

    #  nohup nice ~/mosesdecoder/bin/moses            \
    #    -f ~/working/filtered-newstest2011/moses.ini   \
    #    < ~/corpus/newstest2011.true.fr                \
    #    > ~/working/newstest2011.translated.en         \
    #    2> ~/working/newstest2011.out 
    if ext != 'test':
        return
    TEST_DATA = f'{base}/clean'
    TRANSLATED_DATA = f'{base}/translated'

    for dirpath, dirnames, files in os.walk(TEST_DATA):
        print(f'Found directory: {dirpath}')
        for file_name in files:
            if 'spanish_ca' not in file_name:
                continue
            lang, base_file_name, lang_dir, language_model, size = details(file_name)

            if lang != 'en':
                continue
            for size in ['']:
            # for size in ['', '100', '1000', '10000', '100000']:
                working_dir = f'{TRAINED_MODEL_DIR}/{lang_dir}/mert{size}'
                translated_dir = f'{base}/translated/{lang_dir}'
                if not Path(translated_dir).is_dir():
                    Path(translated_dir).mkdir(parents=True, exist_ok=True)
                args = [
                    f'{DECODER}',
                    '-f', f'{working_dir}/moses.ini',
                ]
                print(f'testing - {base_file_name}')
                print(' '.join(args))
                file_name = file_name.replace(f'.{lang}', '.en')
                target_extension = extension(lang_dir)
                try:
                    a = subprocess.run(
                        args, 
                        stdin=open(f'{TEST_DATA}/{file_name}', 'r'),
                        stdout=open(f'{translated_dir}/translated{size}.{target_extension}', 'w'),
                        shell=False
                        )
                except Exception:
                    print(f'error processing {file_name}')
                print(f'-----------------------')


def extension(lang):

    map = {
        'italian': 'it',
        'german': 'de',
        'french': 'fr',
        'spanish_ca': 'es',
        'spanish_la': 'es',
    }

    return map[lang]


def BLEU(base, ext):

    #  ~/mosesdecoder/scripts/generic/multi-bleu.perl \
    #    -lc ~/corpus/newstest2011.true.en              \
    #    < ~/working/newstest2011.translated.en
    
    if ext != 'test':
        return
    TEST_DATA = f'{base}/clean'
    TRANSLATED_DATA = f'{base}/translated'

    for dirpath, dirnames, files in os.walk(TRANSLATED_DATA):
        print(f'Found directory: {dirpath}')
        lang_dir = dirpath.split('/')[-1:][0]
        for file_name in files:
            if 'bleu' in file_name:
                continue
            lang, base_file_name, _, language_model, size = details(file_name)
            
            args = [
                f'{MULTI_BLEU}',
                '-lc', f'{TEST_DATA}/{lang_dir}_clean.{lang}',
            ]
            try:
                a = subprocess.run(
                    args, 
                    stdin=open(f'{TRANSLATED_DATA}/{lang_dir}/{file_name}', 'r'),
                    stdout=open(f'{TRANSLATED_DATA}/{lang_dir}/{base_file_name}_bleu.{lang}', 'w'),
                    shell=False
                    )
            except Exception:
                print(f'error processing {file_name}')
            print(f'-----------------------')


if __name__ == '__main__':

    for base, ext in [RAW_CORPUS_TEST_FILES, RAW_CORPUS_TRAIN_FILES]:

        tokenise(base, ext)
        remove_long_lines(base, ext) # requires tokensied files
        truecase_train_models(base, ext)
        truecase(base)
        
        clean(base, ext)
        language_models(base, ext)
        # The ones abbove can be bypassed once we have cleaned
        # as we have run the clean split function to 
        # make all the other training corpora
        train(base, ext) # PICKS UP THE LANGUAGE MODELS AND TRUECASE FILES
        tune(base, ext)
        binarise_trained_models(base, ext
        test(base, ext)
        BLEU(base, ext)
