import re
import argparse
import pickle
import collections
from tqdm import tqdm


class Tokenizer:
    __slots__ = [
            'lang',
            'to_lower',
            'remove_suffix',
            'replace_digits',
            'removed_char',
            'split_digits',
            't',
            'segmenter'
        ]

    def __init__(self, lang='jp', tokenize=True, to_lower=True, remove_suffix=True, replace_digits=True):
        self.lang = lang
        self.tokenize = tokenize
        self.to_lower = to_lower
        self.remove_suffix = remove_suffix
        self.replace_digits = replace_digits
        self.removed_char = re.compile(r'[.,!?"\'\";:。、]')
        self.split_digits = re.compile(r'\d')

        if self.tokenze:
            if lang == 'jp':
                from janome.tokenizer import Tokenizer
                self.t = Tokenizer()
                self.segmenter = lambda sentence: list(token.surface for token in self.t.tokenize(sentence))

            elif lang == 'ch':
                import jieba
                self.segmenter = lambda sentence: list(jieba.cut(sentence))

            elif lang == 'en':
                import nltk
                self.segmenter = lambda sentence: list(nltk.word_tokenize(sentence))

    def pre_process(self, sentence):
        
        if self.to_lower:
            sentence = sentence.strip().lower()
        
        if self.remove_suffix:
            sentence = self.removed_char.sub('', sentence)
        
        if self.replace_digits:
            sentence = self.split_digits.sub('0', sentence)
        
        return self.segmenter(sentence) if self.tokenize else sentence.split()

def count_lines(path):
    pass

def token2index(tokens, word_ids):
    return [ word_ids[token] if token in word_ids else word_ids['<UNK>'] for token in tokens ]

def index2token(encoded_tokens, word_ids):
    pass

def load_pickle(in_file):
    with open(in_file, 'rb') as f:
        row_data = pickle.load(f)
    return row_data

def save_pickle(in_file, out_file):
    with open(out_file, 'wb') as f:
        pickle.dump(in_file, f, pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument('in_path', type=str,
                        help="input path to corpas")
    parser.add_argument('out_path', type=str,
                        help="output path to result")
    parser.add_argument('--tokenize', action='store_true',
                        help='tokenize in_path file')
    parser.add_argument('--lang', type=str, choices=['jp', 'en', 'ch'],
                        help="language to be processed")
    parser.add_argument('--tolower', action='store_true',
                        help="lower all characters for all sentences.")
    parser.add_argument('--remove_suffix', action='store_true',
                        help="remove all suffix like ?,! for all sentences.")
    parser.add_argument('--replace_digits', action='store_true',
                        help="replace digits to 0 for all sentences.")
    parser.add_argument('--cutoff', type=int, default=5,
                        help="cutoff words less than the number digignated here")
    parser.add_argument('--vocab_size', type=int, default=0,
                        help='vocabrary size')
    args = parser.parse_args()

    tokenizer = Tokenizer(lang=args.lang, tokenize=args.tokenize, to_lower=args.tolower, remove_suffix=args.remove_suffix)

    sentence_idx = 0
    sentences = []

    word_counter = collections.Counter()
    word_ids = collections.Counter({
                                    '<UNK>':0, 
                                    '<SOS>':1,
                                    '<EOS>':2
                                })

    # read files
    f = open(args.in_path, 'r')
    lines = f.readlines()
    
    #tokenize sentences
    for line in tqdm(lines):
        tokens = []
        tokens += ['<SOS>']
        tokens += tokenizer.pre_process(line)
        tokens += ['<EOS>']

        sentences.append({
            'sentence': line.strip(),
            'tokens': tokens,
            'encoded_tokens': tokens,
            'sentence_idx': sentence_idx
        })

        sentence_idx += 1

        #add each word to word_counter
        word_counter.update(tokens)
    
    print("total distinct words:{0}".format(len(word_counter)))
    print('top 30 frequent words:')
    for word, num in word_counter.most_common(30):
        print('{0} - {1}'.format(word, num))

    # delete words less than cutoff
    for word, num in tqdm(word_counter.items()):
        if num > args.cutoff: del word_counter[word]
    
    # pick up words of vocab_size
    if args.vocab_size > len(word_counter):
        word_counter = word_counter.most_common(args.vocab_size)

    for word, num in tqdm(word_counter.items()):
        if word not in word_ids:
            word_ids[word] = len(word_ids)

    print('total distinct words except words less than {0}: {1}'.format(args.cutoff, len(word_ids)))

    #encoding
    for sentence in tqdm(sentences):
        sentence['encoded_tokens'] = token2index(sentence['encoded_tokens'], word_ids)

    output_dataset = {}
    output_dataset['word_ids'] = word_ids
    output_dataset['train'] = sentences
    
    save_pickle(output_dataset, args.out_path)
