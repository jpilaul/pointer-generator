import sys
import torch
import logging
import numpy as np
import torch.nn as nn
from itertools import chain
from sklearn.utils import shuffle

sys.path.append('../utils')
sys.path.append('../gensen')
try:
    from utils import DataIterator, show_memusage
except:
    from utils.utils import DataIterator, show_memusage
try:
    from gensen import GenSenSingle
except:
    from gensen.gensen import GenSenSingle


test_num_docs = 50


def sub_list_by_length(data, lengths):
    start_idx = 0
    sub_list = []
    for l in lengths:
        sub_list.append(data[start_idx:start_idx+l])
        start_idx = start_idx + l
    return sub_list


def add_tensor_padding(ht_rep, max_sent, rep_dim):
    """ adding padding for documents w/ num sentences < max_sent
    :param ht_rep: list containing each doc's sentence tensors
                   num docs x num sentences x embedding size
    :param max_sent: max num of sentences allowed in each doc
    :return: returns a tensor - batch_size x max_sent x rep_dim
    """
    return [torch.cat((
                ht_rep_i,
                torch.zeros(max_sent - len(ht_rep_i), rep_dim).cuda(0)
            )) for ht_rep_i in ht_rep]


def add_sen_id_padding(ids, max_sent, max_tokens, w2id):
    """ adding padding for documents w/ num sentences < max_sent
    :param ht_rep: list containing each doc's sentence tensors
                   num docs x num sentences x embedding size
    :param max_sent: max num of sentences allowed in each doc
    :return: returns a tensor - batch_size x max_sent x rep_dim
    """
    return [torch.cat((
                docids,
                torch.full(
                    ((max_sent - len(docids), max_tokens)),
                    w2id['<pad>'], dtype=torch.long
                ).cuda()
            )) for docids in ids]


class EncodingIteratorBase(DataIterator):
    """ Base generator class of sentence encodings."""
    def __init__(self, max_sent_length, max_sent_src, max_sent_trg,
                 data_folder, model_folder, pretrain_path, prefix,
                 source_file, target_file, use_gensen_w2i,
                 device_ids=[0], data_parallelize=False,
                 test=False):
        """
        :param max_sent_length: max words in sentence
               gensen_h --> batch size x max_len x rep_size
        :param max_sent_src: number of sentences in source doc
        :param max_sent_trg: number of sentences in target doc
        :param data_folder: data location
        :param model_folder: location of pretrained gensen
        :param pretrain_path: location of pretrained embeddings (e.g. Glove)
        :param prefix: used of the type of gensen ["nli_large"+"bothskip"+"arxiv"]
        :param source_file: name of source file in data_folder
        :param target_file: name of target file in data_folder
        :param use_gensen_w2i: use the word to ids for pretrained gensen
        :param device_ids: used when data_parallelize = True, specify devices to use
        :param data_parallelize:
        :param test:
        """
        self.max_len = max_sent_length    # max words
        self.max_sent_src = max_sent_src  # max sentences src
        self.max_sent_trg = max_sent_trg  # max sentences trg
        self.data_folder = data_folder
        self.source_file = source_file
        self.target_file = target_file
        self.src_data = []
        self.atrg_data = []
        self.data_parallelize = data_parallelize
        self.device_ids = device_ids
        self.test = test

        logging.debug(
            """ max_len: {}, max_sent_src: {}, max_sent_trg: {},
                data_folder: {}, source_file: {}, target_file: {}
            """.format(self.max_len, self.max_sent_src, self.max_sent_trg,
                       self.data_folder, self.source_file, self.target_file))
        self.gensen = GenSenSingle(model_folder=model_folder, filename_prefix=prefix,
                                   pretrained_emb=pretrain_path, cuda=True,
                                   max_sentence_length=max_sent_length,
                                   data_parallelize=data_parallelize,
                                   device_ids=device_ids[::-1])
        self.sen_rep_dim = self.gensen.sen_rep_dim
        self.vocab_size = self.gensen.vocab_size
        self.emb_dim = self.gensen.embedding_dim
        self.vocab_expansion(use_gensen_w2i)

    def vocab_expansion(self, use_gensen_w2i):
        """ Read data from files."""
        if self.test: logging.debug(" Testing with 100 documents")
        files = [self.source_file, self.target_file]
        data = [self.src_data, self.atrg_data]
        maxes_sen = [self.max_sent_src, self.max_sent_trg]

        for file, dt, max_sen in zip(files, data, maxes_sen):
            with open('%s/%s' % (self.data_folder, file), 'r', encoding="utf-8") as source:
                doc = []
                for sentence in source:
                    if doc and sentence.startswith("\n"):
                        if len(doc) > max_sen: doc = doc[0:max_sen]
                        dt.append(doc)
                        doc = []
                    elif sentence.strip(): doc.append(sentence.strip())
                    if self.test and len(dt) > test_num_docs: break
        self.num_docs = len(self.src_data)
        assert self.num_docs == len(self.atrg_data)
        logging.info(" Constructing vocabulary...")

        if use_gensen_w2i:  # if True does not construct a new vocab
            self.word2id = self.gensen.word2id
            self.id2word = self.gensen.id2word
        else:
            self.word2id, self.id2word = self.construct_vocab(
                    list(chain.from_iterable(self.src_data)) +
                    list(chain.from_iterable(self.atrg_data)),
                    self.vocab_size
            )
        self.gensen.vocab_expansion(self.word2id.keys())
        self.vocab_size = self.gensen.vocab_size
        logging.info(" Data has been read")


class EncodingIterator(DataIterator):
    """ Generator class of sentence encodings"""

    def __init__(self, enc_it_base, source_file, target_file, batch_size,
                 add_emb_2in=False, get_all_gensen_h=False):
        """
        :param enc_it_base: Base encoder object :type DataIterator
        :param source_file: name of source file in data_folder
        :param target_file: name of target file in data_folder
        :param batch_size: batch size
        :param add_emb_2in: token embeddings are appended to gensen_h
               gensen_h --> batch size x max_len x (rep_size + emb_size)
        :param get_all_gensen_h: can return gensen_h or gensen_h_t (the last hidden state)
        """
        self.max_len = enc_it_base.max_len
        self.max_sent_src = enc_it_base.max_sent_src
        self.max_sent_trg = enc_it_base.max_sent_trg
        self.sen_rep_dim = enc_it_base.sen_rep_dim
        self.emb_dim = enc_it_base.emb_dim
        self.data_folder = enc_it_base.data_folder
        self.gensen = enc_it_base.gensen
        self.word2id = enc_it_base.word2id
        self.id2word = enc_it_base.id2word
        self.data_parallelize = enc_it_base.data_parallelize
        self.device_ids = enc_it_base.device_ids
        self.test = enc_it_base.test
        self.add_emb_2in = add_emb_2in
        self.get_all_gensen_h = get_all_gensen_h

        if source_file == enc_it_base.source_file:
            self.source_file = enc_it_base.source_file
            self.target_file = enc_it_base.target_file
            self.src_data = enc_it_base.src_data
            self.trg_data = enc_it_base.atrg_data
            self.num_docs = enc_it_base.num_docs
        else:
            self.source_file = source_file
            self.target_file = target_file
            self.src_data = []
            self.trg_data = []
            self.read_data()

        self.shuffle_dataset()
        self.index = 0
        self.batch_size = batch_size

    def read_data(self):
        """ Read data from files."""
        files = [self.source_file, self.target_file]
        data = [self.src_data, self.trg_data]
        maxes_sen = [self.max_sent_src, self.max_sent_trg]

        for file, dt, max_sen in zip(files, data, maxes_sen):
            with open('%s/%s' % (self.data_folder, file), 'r') as source:
                doc = []
                for sentence in source:
                    if doc and sentence.startswith("\n"):
                        if len(doc) > max_sen: doc = doc[0:max_sen]
                        dt.append(doc)
                        doc = []
                    elif sentence.strip():
                        doc.append(sentence.strip())
                    if self.test and len(dt) > test_num_docs: break

        self.num_docs = len(self.src_data)
        logging.info(" Data has been read and has %d docs" %self.num_docs)

    def shuffle_dataset(self):
        """ Shuffle dataset."""
        logging.info(" Dataset has been shuffled")
        self.data_idx = shuffle(range(self.num_docs))

    def reset_index(self, idx=0):
        self.index = idx % self.num_docs

    def next_minibatch(self):
        start = self.index
        end = (self.index + self.batch_size) % self.num_docs
        self.index = end
        return self.get_minibatch(start, end)

    def get_minibatch(self, start, end):
        """ Contruct minibatches for txt and return encoding of each doc sorted by
            sorted by number of sentences of source shuffled docs.
            Returns:
                sorted_lens_idx --> [batch_size] txt indices sorted by src doc size
                txt_src  -->  list of word tokens - not sorted
                sorted_len_src --> tensor with txt lengths sorted by src doc size
                if get_all_gensen_h == True:
                    rep_src  -->  list of size batch_size with tensors sorted by src doc size
                                  [num_sentences x max_sen_len x rep_dim]
                else:
                    rep_src  -->  [batch_size x max_sent_src x rep_dim] sorted by src doc size
        """
        if end < start:
            minibatch_idx = self.data_idx[start:] + self.data_idx[0:end]
        else:
            minibatch_idx = self.data_idx[start:end]
        minibatch_src = list(map(lambda i: self.src_data[i]
            if len(self.src_data[i]) < self.max_sent_src
            else self.src_data[i][0:self.max_sent_src], minibatch_idx))
        minibatch_trg = list(map(lambda i: self.trg_data[i]
            if len(self.trg_data[i]) < self.max_sent_trg
            else self.trg_data[i][0:self.max_sent_trg], minibatch_idx))

        sen_lens_src = [len(doc) for doc in minibatch_src]
        sen_lens_atrg = [len(doc) for doc in minibatch_trg]
        sorted_lens_idx = np.argsort(sen_lens_src)[::-1]  # decreasing order

        with torch.no_grad():
            chained_all_sent = list(chain.from_iterable(minibatch_src)) + list(chain.from_iterable(minibatch_trg))
            all_h_rep, all_ht_rep, all_ids = self.gensen.get_representation(sentences=chained_all_sent,
                                                                   pool="last",
                                                                   return_numpy=False,
                                                                   add_emb_2in=self.add_emb_2in)
            del chained_all_sent
            assert len(all_ht_rep) == sum(sen_lens_src) + sum(sen_lens_atrg)

            logging.debug(" Size of all_ht_rep = [%d, %d]" % (all_ht_rep.size()))
            logging.debug(" Size of all_h_rep = [%d, %d, %d]" % (all_h_rep.size()))

            all_ids = sub_list_by_length(all_ids, sen_lens_src + sen_lens_atrg)
            src_wids = all_ids[0:len(all_h_rep) // 2]
            src_wids = add_sen_id_padding(src_wids, self.max_sent_src, self.max_len, self.word2id)
            del all_ids
            if self.get_all_gensen_h:
                # note 1: get_all_gensen_h is used to re-encode sentence vectors
                # note 2: add_tensor_padding(...) used after the model's sen_encoder
                del all_ht_rep
                all_h_rep = sub_list_by_length(all_h_rep, sen_lens_src + sen_lens_atrg)
                src_h_rep = all_h_rep[0:len(all_h_rep) // 2]
                if self.test:
                    trg_h_rep = all_h_rep[len(all_h_rep)//2:]
                del all_h_rep
                if self.data_parallelize:
                    gpu_count = len(self.device_ids)
                    assert self.batch_size % gpu_count == 0
                    items_per_gpu = self.batch_size // gpu_count
                    # distribute largest sequences first across GPUs
                    sorted_lens_idx = [sorted_lens_idx[g + i * gpu_count]
                                       for g in self.device_ids for i in range(items_per_gpu)]
                    src_h_rep = [src_h_rep[idx].detach()
                                 for i, idx in enumerate(sorted_lens_idx)]
                    src_h_rep = [[h.to(gpu_id) for h in src_h_rep[i:i+items_per_gpu]]
                                 for i, gpu_id in
                                 zip(range(0, self.batch_size-items_per_gpu+1, items_per_gpu), self.device_ids)
                                 ]
                    if self.test:
                        trg_h_rep = [trg_h_rep[idx].detach()
                                     for i, idx in enumerate(sorted_lens_idx)]
                        trg_h_rep = [[h.to(gpu_id) for h in trg_h_rep[i:i+items_per_gpu]]
                                     for i, gpu_id in
                                     zip(range(0, self.batch_size - items_per_gpu + 1, items_per_gpu), self.device_ids)
                                     ]
                else:
                    src_h_rep = [src_h_rep[k] for k in sorted_lens_idx]
                    if self.test:
                        trg_h_rep = [trg_h_rep[k] for k in sorted_lens_idx]
            else:
                del all_h_rep
                all_ht_rep = sub_list_by_length(all_ht_rep.data, sen_lens_src + sen_lens_atrg)
                src_ht_rep = all_ht_rep[0:len(all_ht_rep)//2]
                trg_ht_rep = all_ht_rep[len(all_ht_rep)//2:]
                del all_ht_rep
                src_ht_rep = [src_ht_rep[i] for i in sorted_lens_idx]
                src_ht_rep = add_tensor_padding(src_ht_rep, self.max_sent_src, self.sen_rep_dim)
                src_ht_rep = torch.stack(src_ht_rep)

            sorted_lens_src = torch.LongTensor([sen_lens_src[k] for k in sorted_lens_idx]).cuda()
            src_wids = torch.stack([src_wids[k] for k in sorted_lens_idx]).cuda()
            return {
                'rep_src': src_h_rep if self.get_all_gensen_h else src_ht_rep,
                'rep_trg': None if not self.test else trg_h_rep if self.get_all_gensen_h else trg_ht_rep,
                'txt_src': minibatch_src,
                'txt_trg': minibatch_trg,
                'sorted_lens_idx': sorted_lens_idx,
                'sorted_lens_src': sorted_lens_src,
                'src_wids': src_wids
            }


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
            "--log",
            help="log level to use [debug, info, warning, error, critical]",
            default='DEBUG'
    )
    args = parser.parse_args()
    log_num_lvl = getattr(logging, args.log.upper())
    if not isinstance(log_num_lvl, int):
        raise ValueError('Invalid log level: %s' % log_num_lvl)
    logging.basicConfig(format='%(levelname)s:%(message)s', level=log_num_lvl,
                        handlers=[
                            logging.FileHandler('log_enc_it.log'),
                            logging.StreamHandler()
                        ])
    logging.info(" Testing encoding_iterator.py")


    def get_w2i_seq(txt, word2ids, max_trg_sent_len,
                    max_num_trg_sent, sorted_lens_idx):
        """ Provides a continuous vector of ids for each document
            ONE LINE (sorted by decreasing doc length)
            - method is different then in s2s_entropy_AttBPTT_summ
        """
        one_liners = [" ".join(txt[idx][:max_num_trg_sent])
                      for idx in sorted_lens_idx]  # sorted by source doc length
        txt = [
            ['<s>'] + line.split(" ")[:max_trg_sent_len - 2] + ['</s>']
            for line in one_liners
        ]
        input_trg = torch.LongTensor([
            [word2ids[w] if w in word2ids else word2ids['<unk>'] for w in line[:-1]] +
            [word2ids['<pad>']] * (max_trg_sent_len - len(line))
            for line in txt
        ]).cuda()
        input_trg = input_trg.view(input_trg.size(0), -1)
        output_trg = torch.LongTensor([
            [word2ids[w] if w in word2ids else word2ids['<unk>'] for w in line[1:]] +
            [word2ids['<pad>']] * (max_trg_sent_len - len(line))
            for line in txt
        ]).cuda()
        output_trg = output_trg.view(output_trg.size(0), -1)
        return input_trg, output_trg


    get_all_gensen_h = True
    data_parallelize = False
    device_ids = [0, 1]
    batch_size = 6
    test = True
    enc_iterator_base = EncodingIteratorBase(
            max_sent_length=75, max_sent_src=180,
            max_sent_trg=10, test=test,
            data_folder='/mnt/home/jonathan/datasets/arxiv',
            model_folder='/mnt/home/jonathan/models/GenSen',
            pretrain_path='/mnt/home/jonathan/datasets/embeddings/new_glove.840B.300d.h5',
            prefix='nli_large_bothskip_arxiv',
            source_file="train_source_arxiv.txt",
            target_file="train_target_abs_arxiv.txt",
            data_parallelize=data_parallelize,
            device_ids=device_ids,
            use_gensen_w2i=False
    )
    enc_iterator1 = EncodingIterator(
        source_file="train_source_arxiv.txt",
        target_file="train_target_abs_arxiv.txt",
        batch_size=batch_size, enc_it_base=enc_iterator_base,
        get_all_gensen_h=True
    )
    mini_batch = enc_iterator1.next_minibatch()

    #  Testing text lengths
    if data_parallelize:
        src = list(chain.from_iterable(mini_batch["rep_src"]))
        trg = list(chain.from_iterable(mini_batch["rep_trg"]))
    else:
        src = mini_batch["rep_src"]
        trg = mini_batch["rep_trg"]
    if get_all_gensen_h:
        if data_parallelize:
            logging.debug("source size: %s" % [doc.size(0) for doc in src])
            logging.debug("source gpus: %s" % [doc.get_device() for doc in src])
            logging.debug("target size: %s" % [doc.size(0) for doc in trg])
            logging.debug("target gpus: %s" % [doc.get_device() for doc in trg])
        else:
            logging.debug("target size: %s" % [doc.size(0) for doc in src])
            logging.debug("source size: %s" % [doc.size(0) for doc in trg])

    else:
        logging.debug("source size: %s" % (tuple(src.size()),))
    logging.debug("txt_trg lengths of docs: %s" % [len(doc) for doc in mini_batch["txt_trg"]])
    logging.debug("txt_src lengths of docs: %s" % [len(doc) for doc in mini_batch["txt_src"]])
    sorted_txt_src = [mini_batch["txt_src"][idx] for idx in mini_batch['sorted_lens_idx']]
    sorted_txt_trg = [mini_batch["txt_trg"][idx] for idx in mini_batch['sorted_lens_idx']]
    logging.debug("sorted txt_trg lengths of docs: %s" % [len(txt) for txt in sorted_txt_trg])
    logging.debug("sorted_lens_src: %s" % mini_batch["sorted_lens_src"])

    #  Testing vectors
    all_h_rep, all_ht_rep, _ = enc_iterator1.gensen.get_representation(
        list(chain.from_iterable(sorted_txt_src)) +
        list(chain.from_iterable(sorted_txt_trg)),
        pool="last", return_numpy=False)

    test_iterator_h_src = torch.cat(src)
    test_iterator_h_trg = torch.cat(trg)
    test_iterator_h_all = torch.cat((test_iterator_h_src, test_iterator_h_trg))
    print(torch.equal(all_h_rep, test_iterator_h_all))
    for t1, t2 in zip(all_h_rep, test_iterator_h_all):
        if not torch.equal(t1, t2):
            print(t1)
            print(t2)

    enc_iterator2 = EncodingIterator(
        source_file="train_source_arxiv.txt",
        target_file="train_target_abs_arxiv.txt",
        batch_size=batch_size, enc_it_base=enc_iterator_base,
        get_all_gensen_h=False
    )
    mini_batch = enc_iterator2.next_minibatch()

    sorted_txt_src = [mini_batch["txt_src"][idx] for idx in mini_batch['sorted_lens_idx']]
    src_h_rep, src_ht_rep, src_ids = enc_iterator2.gensen.get_representation(
        list(chain.from_iterable(sorted_txt_src)),
        pool="last", return_numpy=False)
    src_ht_rep = sub_list_by_length(src_ht_rep.data, mini_batch['sorted_lens_src'])
    src_ht_rep = add_tensor_padding(src_ht_rep, 180, 2048)
    src_ht_rep = torch.stack(src_ht_rep)

    src = mini_batch["rep_src"]
    print(torch.allclose(src_ht_rep, src))
    for i, (t1, t2) in enumerate(zip(src_ht_rep, src)):
        if not torch.equal(t1.data, t2.data):
            print(i)
            print(torch.mean(t1))
            print(torch.mean(t2))

    src_ids = sub_list_by_length(src_ids.data, mini_batch['sorted_lens_src'])
    src_ids = add_sen_id_padding(src_ids, 180, 75, enc_iterator2.word2id)
    src_ids = torch.stack(src_ids).cuda()

    ids = mini_batch['src_wids']
    print(torch.all(torch.eq(ids, src_ids)))

    for _ in range(8):
        mini_batch = enc_iterator2.next_minibatch()
        print(mini_batch["rep_src"].size())
        print(mini_batch['src_wids'].size())


