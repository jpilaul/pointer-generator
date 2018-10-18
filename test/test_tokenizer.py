import os
import glob
import struct
import logging
from tools.tokenizer import Tokenizer, BPETokenizer, CharTokenizer

test_file = '/mnt/home/jonathan/projects/seq2seq.pytorch/README.md'
text = 'machine learning - hello world'

data_dir = "/mnt/home/jonathan/datasets/cnn-dailymail/cnn_dailymail_data/finished_files"
filelist = glob.glob(os.path.join(data_dir, "chunked/train*"))  # get the list of datafiles


tokenizer = Tokenizer(vocab_file=os.path.join(data_dir, 'vocab'), max_length=200000)
tokenizer.get_vocab([test_file], from_filenames=True)
tokenized = tokenizer.tokenize(text)
print(tokenized, tokenizer.detokenize(tokenized))

char_tokenizer = CharTokenizer(vocab_file=os.path.join(data_dir, 'vocab'))
char_tokenizer.get_vocab([test_file], from_filenames=True)
tokenized = char_tokenizer.tokenize(text)
print(tokenized, char_tokenizer.detokenize(tokenized))

bpe_codes = '/mnt/home/jonathan/projects/pointer-generator/test/test_bpe.codes'
bpe_tokenizer = BPETokenizer(bpe_codes, vocab_file=os.path.join(data_dir, 'vocab'))
bpe_tokenizer.learn_bpe(filelist, bin_file=True, from_filenames=True)
bpe_tokenizer.get_vocab(filelist, bin_file=True, from_filenames=True)
bpe_tokenizer.save_vocab("/mnt/home/jonathan/datasets/cnn-dailymail/cnn_dailymail_data/finished_files/bpe_vocab")

tokenized = bpe_tokenizer.tokenize(text)
vocab = bpe_tokenizer.vocab
w2i = bpe_tokenizer.word2idx
print(tokenized, bpe_tokenizer.detokenize(tokenized))
