import os
import glob
import struct
from tools.tokenizer import Tokenizer, BPETokenizer, CharTokenizer

test_file = '/mnt/home/jonathan/projects/seq2seq.pytorch/README.md'
text = 'machine learning - hello world'

data_dir = "/mnt/home/jonathan/datasets/cnn-dailymail/cnn_dailymail_data/finished_files"
filelist = glob.glob(os.path.join(data_dir, "chunked/train*")) # get the list of datafiles


tokenizer = Tokenizer(vocab_file=os.path.join(data_dir, 'vocab'), max_length=200000)
tokenizer.get_vocab([test_file], from_filenames=True)
tokenized = tokenizer.tokenize(text)
print(tokenized, tokenizer.detokenize(tokenized))

char_tokenizer = CharTokenizer(vocab_file=os.path.join(data_dir, 'vocab'))
char_tokenizer.get_vocab([test_file], from_filenames=True)
tokenized = char_tokenizer.tokenize(text)
print(tokenized, char_tokenizer.detokenize(tokenized))

bpe_tokenizer = BPETokenizer('test_bpe.codes', vocab_file=os.path.join(data_dir, 'vocab'), num_symbols=100)
bpe_tokenizer.learn_bpe(filelist, bin_file=True, from_filenames=True)
bpe_tokenizer.get_vocab(filelist, bin_file=True, from_filenames=True)

tokenized = bpe_tokenizer.tokenize(text)
vocab = bpe_tokenizer.vocab
print(tokenized, bpe_tokenizer.detokenize(tokenized))

