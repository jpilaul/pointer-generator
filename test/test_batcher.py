from tools.data import Vocab
from collections import namedtuple
from tools. batcher import Batcher


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