
from openpyxl import load_workbook
import re
from typing import List


def to_json(my_path):

    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(my_path) if isfile(join(my_path, f))]

    subs = []

    def load_data():

        for data_file in onlyfiles:

            if not data_file.endswith('.xlsx'):
                continue
            wb = load_workbook(f'{my_path}{data_file}')

            names = wb.get_sheet_names()
            sheet = wb.get_sheet_by_name(name=names[0])
            sub_count = 0

            for row in sheet.rows:

                sub = {'source': row[0].value,  'target': row[1].value}
                r = split_multi(sub)

                sub_count += len(r)
                if r:
                    yield r[0]

    for sub in load_data():

        newline_filter(sub)
        ellipsis_filter(sub)
        name_filter(sub)
        subs.append(sub)

    import random
    random.shuffle(subs)

    s = my_path.split('/')
    ext = s[len(s)-2]
    m = {
        'spanish_castilian': 'es',
        'spanish_latin_america': 'es',
        'german': 'de',
        'italian': 'it',
        'french_parisian': 'fr',
    }

    src_file = f'{my_path}{ext}_corpus.en'
    tgt_file = f'{my_path}{ext}_corpus.{m[ext]}'

    with open(src_file, 'a') as src, open(tgt_file, 'a') as tgt:
        for s,t in subs:
            if not s or not t:
                continue
            src.write(s + '\n')
            tgt.write(t + '\n')


def filter(subs, func):

    filtered = []
    for s in subs:
        filtered.extend(func(s))

    return filtered


def ellipsis_filter(sub):
    """
    Filter out ...
    """

    source = sub[0]
    target = sub[1]

    sub[0] = source.replace('...', '')
    sub[1] = target.replace('...', '')


def newline_filter(sub):
    """
    Filter out \n
    """

    source = sub[0]
    target = sub[1]

    sub[0] = source.replace('\n', ' ')
    sub[1] = target.replace('\n', ' ')


c = re.compile('\(.\)^')


def hu_filter(sub):
    """
    Filter out human utterance.
    Sometimes we  have (SIGHS) in a source dialogue, but they never seem to be translated.

    So lets just remove them
    """
    
    source = sub['source']
    target = sub('target')

    sub['source'] = re.sub(source, c, ' ')
    sub['target'] = re.sub(target, c, ' ')

    return sub


# need to split  combined ones first
c = re.compile('^.*?:') # name: subtitle
b = re.compile('^(\(.*?\))') # (numbering) subtitle
a = re.compile('^(\[.*?\])') # [over phone] subtitle


def name_filter(sub):
    """
    Filter out names.
    Some subs have the name within the dialogue in the format XXXXX: blah 
    They start from the beginning of the source, 
    but We need to get they never seem to 
    follow through to the target
    So lets just remove them
    """
    
    source = sub[0]
    target = sub[1]

    sub[0] = re.sub(c, '',  source)
    sub[1] = re.sub(c, '',  target)

    source = sub[0]
    target = sub[1]

    sub[0] = re.sub(b, '',  source)
    sub[1] = re.sub(b, '', target)

    source = sub[0]
    target = sub[1]

    sub[0] = re.sub(a, '', source)
    sub[1] = re.sub(a, '', target)       


def split_multi(sub):
    """
    Splits multi dialogue subs into their individual dialogues

    Multi line typically start with '-' and also contain a 
    carriage return plus another
    '-' to indicate the start of the second dialogue event

    For this to be accurate we need to find in the source and the target, so if we don't find 
    the same pattern in both we need to discard it

    Takes a sub dict and returns a list of dicts containing zero or more subs 
    """
    source = sub['source']
    target = sub['target']
    split = False
    discard = False
    if source is None or target is None:
        return[]
    subs = [[source, target]]
    if source.startswith('-') and '\n' in source:
        if target.startswith('-') and '\n' in target:
            split = True
        else:
            discard = True
    elif target.startswith('-') and '\n' in target:
        # We've already tested the other case
        discard = True

    if split:
        s = [t.replace('-', '').strip() for t in source.split('\n-')]
        t = [t.replace('-', '').strip() for t in target.split('\n-')]
        subs = [[f[0], f[1]] for f in (zip(s,t))]
    return subs if not discard else []


if __name__ == '__main__':

    paths: List[str] = [
        # 'Paths/to/specific/languages/',
    ]

    for path in paths:
        to_json(path)
