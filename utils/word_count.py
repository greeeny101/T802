"""
Simple script to count words in a corpus
"""

if __name__ == '__main__':
    
    files = [
        'train/clean/spanish_ca_train.en',
    ]

    for fi in files:
        count = 0
        with open(fi ,'r') as f:
            for line in f.readlines():

                c = len(line.split(' '))
                count += c
        print (count)
