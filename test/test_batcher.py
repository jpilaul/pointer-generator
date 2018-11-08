from tools.data import Vocab
from collections import namedtuple
from tools. batcher import Batcher
from tools.tokenizer import BPETokenizer

vocab_path = "/mnt/home/jonathan/datasets/cnn-dailymail/cnn_dailymail_data/finished_files/bpe_vocab"
data_path = "/mnt/home/jonathan/datasets/cnn-dailymail/cnn_dailymail_data/finished_files/chunked/train_*"

vocab = Vocab(vocab_path, 50000)

hps_dict = {
    "batch_size": 1,
    "max_enc_steps": 400,
    "max_dec_steps": 100,
    "pointer_gen": True,
    "mode": "train",
}

hps = namedtuple("HParams", hps_dict.keys())(**hps_dict)

batcher = Batcher(data_path, vocab, hps, single_pass=False)
batch = batcher.next_batch()
print(batch)

bpe_codes = '/mnt/home/jonathan/projects/pointer-generator/test/test_bpe.codes'
bpe_tokenizer = BPETokenizer(codes_file=bpe_codes, vocab_file=vocab_path)
vocab = bpe_tokenizer.vocab

same_items = {k: batcher._vocab._word_to_id[k] for k in batcher._vocab._word_to_id
                    if k in bpe_tokenizer.word2idx and batcher._vocab._word_to_id[k] == bpe_tokenizer.word2idx[k]}
assert len(same_items) - len(bpe_tokenizer.word2idx) + 4 == 0

# TODO: bpe_tokenizer.word2idx[<s>]=2 and bpe_tokenizer.word2idx[</s>]=3 and these correspond to <start> and <stop>

tokenized = bpe_tokenizer.tokenize(batch.original_abstracts[0])
#print(tokenized)
for idx in tokenized:
    print(batcher._vocab._id_to_word[idx.item()])

print(bpe_tokenizer.detokenize(tokenized))


print("-----------------------------")
print(batch.original_abstracts[0])