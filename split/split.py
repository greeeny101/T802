
from openpyxl import load_workbook
from typing import List


def to_json(my_path, dev, test_set, train_set_size):
    """
    Split a corpus into test, development and training sets

    Works in reverse by first creating the test set, then the dev set
    and finally the training set
    """

    for path in my_path:

        s = path.split('/')
        ext = s[len(s)-2]
        m = {
            'spanish_castilian': 'es.ca',
            'spanish_latin_america': 'es.la',
            'german': 'ge',
            'italian': 'it',
            'french_parisian': 'fr',
        }
        with open(f'{path}{ext}_corpus.en', 'r') as source, \
            open(f'{path}{ext}_corpus.{m[ext]}', 'r') as target, \
            open(f'{path}{ext}_dev_corpus.en', 'w') as c_s, \
            open(f'{path}{ext}_dev_corpus.{m[ext]}', 'w') as c_t, \
            open(f'{path}{ext}_test_corpus.en', 'w') as d_s, \
            open(f'{path}{ext}_test_corpus.{m[ext]}', 'w') as d_t, \
            open(f'{path}{ext}_train_corpus.en', 'w') as e_s, \
            open(f'{path}{ext}_train_corpus.{m[ext]}', 'w') as e_t:

            count = 0
            count_t = 0
            train_count = 0
            source = []
            target = []
            sentences_used = []
            for z  in zip(source.readlines(), target.readlines()):
                if train_set_size == train_count:
                    continue
                s = z[0]
                t = z[1]
                if count > -1 or count_t > -1:
                    if s.strip() not in sentences_used:
                        source.append(s.strip())
                        target.append(t.strip())
                else:
                    train_count += 1
                    e_s.write(f'{s}\n')
                    e_t.write(f'{t}\n')
                    if train_count >= train_set_size:
                        continue
                if count != -1:
                    count += 1
                if count == -1 and count_t != -1:
                    count_t += 1         
                if count >= dev:
                    print(f'writing dev set {count}')
                    for item in source:
                        sentences_used.append(source)
                        c_s.write(f'{item}\n')
                    for item in target:
                        c_t.write(f'{item}\n')
                    count = -1
                    source = []
                    target = []
                if count_t >= test_set:
                    print(f'writing test set {count_t}')
                    for item in source:
                        sentences_used.append(source)
                        d_s.write(f'{item}\n')
                    for item in target:
                        d_t.write(f'{item}\n')
                    count_t = -1
                    source = []
                    target = []
            print(f'test count for {ext} is {train_count}')


if __name__ == '__main__':

    paths: List[str] = [
        # 'Paths/to/specific/languages/',
    ]

    dev = 10000
    test = 5000
    train = 1000000
    to_json(paths, dev, test, train)
    

