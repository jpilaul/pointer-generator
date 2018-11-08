"""Minibatching utilities."""

import torch
import math
import nltk
import pickle
import codecs
import logging
import gpustat
import operator
import numpy as np
import torch.nn as nn
from sklearn.utils import shuffle
from torch.autograd import Variable
from collections import Counter


def bleu_stats(hypothesis, reference):
    """Compute statistics for BLEU."""
    stats = []
    stats.append(len(hypothesis))
    stats.append(len(reference))
    for n in range(1, 5):
        s_ngrams = Counter(
            [tuple(hypothesis[i:i + n]) for i in range(len(hypothesis) + 1 - n)]
        )
        r_ngrams = Counter(
            [tuple(reference[i:i + n]) for i in range(len(reference) + 1 - n)]
        )
        stats.append(max([sum((s_ngrams & r_ngrams).values()), 0]))
        stats.append(max([len(hypothesis) + 1 - n, 0]))
    return stats

def bleu(stats):
    """ Compute BLEU given n-gram statistics."""
    if len(list(filter(lambda x: x == 0, stats))) > 0:
        return 0
    (c, r) = stats[:2]
    log_bleu_prec = sum(
        [math.log(float(x) / y) for x, y in zip(stats[2::2], stats[3::2])]
    ) / 4.
    return math.exp(min([0, 1 - float(r) / c]) + log_bleu_prec)

def get_bleu(hypotheses, reference):
    """ Get validation BLEU score for dev set."""
    hypotheses = [x.split() for x in hypotheses]
    reference = [x.split() for x in reference]
    stats = np.array([0., 0., 0., 0., 0., 0., 0., 0., 0., 0.])
    for hyp, ref in zip(hypotheses, reference):
        stats += np.array(bleu_stats(hyp, ref))
    return 100 * bleu(stats)


def calc_gradient_penalty(discriminator, real_data, fake_data, gp_lambda):
    """ Calculate GP.
        inputs: discriminator, real_data, fake_data, gp_lambda

    """
    assert real_data.size() == fake_data.size()
    size = [real_data.size(0) if i == 0 else 1 for i, _ in enumerate(real_data.size())]
    size = torch.Size(size)  # size = torch.Size([batch_size, 1, ..., 1])
    alpha = torch.rand(size).cuda()

    interpolates = Variable(
        alpha * real_data + ((1 - alpha) * fake_data),
        requires_grad=True
    )
    disc_interpolates = discriminator(interpolates)
    grad = torch.autograd.grad(
        outputs=disc_interpolates, inputs=interpolates,
        grad_outputs=torch.ones(disc_interpolates.size()).cuda(),
        create_graph=True, retain_graph=True, only_inputs=True
    )[0]
    dim = len(grad.size()) - 1
    gradient_penalty = ((grad.norm(2, dim=dim) - 1) ** 2).mean() * gp_lambda
    return gradient_penalty


def show_memusage(device=0, text=""):
    gpu_stats = gpustat.GPUStatCollection.new_query()
    num_gpus = torch.cuda.device_count()
    for i in range(num_gpus):
        txt = text + "GPU: %d " % i
        text = ""
        item = gpu_stats.jsonify()["gpus"][i]
        logging.info("{}{}/{}".format(txt, item["memory.used"], item["memory.total"]))

def repackage_hidden(h):
    """Wraps hidden states in new Tensors, to detach them from their history."""
    if isinstance(h, torch.Tensor):
        return h.detach()
    else:
        return tuple(repackage_hidden(v) for v in h)

class DataIterator(object):
    """Data Iterator."""

    def _trim_vocab(self, vocab, vocab_size):
        # Discard start, end, pad and unk tokens if already present
        if '<s>' in vocab:
            del vocab['<s>']
        if '<pad>' in vocab:
            del vocab['<pad>']
        if '</s>' in vocab:
            del vocab['</s>']
        if '<unk>' in vocab:
            del vocab['<unk>']

        word2id = {
            '<s>': 0,
            '<pad>': 1,
            '</s>': 2,
            '<unk>': 3,
        }

        id2word = {
            0: '<s>',
            1: '<pad>',
            2: '</s>',
            3: '<unk>',
        }

        sorted_word2id = sorted(
            vocab.items(),
            key=operator.itemgetter(1),
            reverse=True
        )

        if vocab_size != -1:
            sorted_words = [x[0] for x in sorted_word2id[:vocab_size]]
        else:
            sorted_words = [x[0] for x in sorted_word2id]

        for ind, word in enumerate(sorted_words):
            word2id[word] = ind + 4

        for ind, word in enumerate(sorted_words):
            id2word[ind + 4] = word

        return word2id, id2word

    def construct_vocab(self, sentences, vocab_size, lowercase=False):
        """Create vocabulary."""
        vocab = {}
        for sentence in sentences:
            if isinstance(sentence, str):
                if lowercase:
                    sentence = sentence.lower()
                sentence = sentence.split()
            for word in sentence:
                if word not in vocab:
                    vocab[word] = 1
                else:
                    vocab[word] += 1

        word2id, id2word = self._trim_vocab(vocab, vocab_size)
        return word2id, id2word


class SentenceIterator(DataIterator):
    """Iterator over sentences in a large corpus."""

    def __init__(self, file_path, vocab_size, max_length=150):
        """Initialize params."""
        self.file_path = file_path
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.read_data()

    def read_data(self):
        """Read data from files."""
        print('Reading file ...')
        self.lines = [line.strip().lower().split() for line in open(self.file_path, 'r', encoding='utf-8')]
        self.lines = [line for line in self.lines if len(line) < self.max_length]
        print('Building vocabulary ...')
        self.word2id, self.id2word = self.construct_vocab(self.lines, self.vocab_size)

    def shuffle_dataset(self):
        """Shuffle dataset."""
        self.lines = shuffle(self.lines)

    def get_minibatch(self, index, batch_size, lm=False, volatile=False, cuda=True):
        """Prepare minibatch."""
        if isinstance(batch_size, str) and "all" in batch_size:
            lines = [
                ['<s>'] + line[:self.max_length] + ['</s>']
                for line in self.lines
            ]
        else:
            lines = [
                ['<s>'] + line[:self.max_length] + ['</s>']
                for line in self.lines[index: index + batch_size]
            ]

        lens = [len(line) for line in lines]
        sorted_indices = np.argsort(lens)[::-1]
        rev_input = np.argsort(sorted_indices) # keeps original order of lines

        sorted_lines = [lines[idx] for idx in sorted_indices]
        sorted_lens = [len(line) for line in sorted_lines]

        max_len = max(sorted_lens)

        if not lm:
            input_lines = [
                [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in line] +
                [self.word2id['<pad>']] * (max_len - len(line))
                for line in sorted_lines
            ]
            output_lines = None
        else:
            input_lines = [
                [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in line[:-1]] +
                [self.word2id['<pad>']] * (max_len - len(line))
                for line in sorted_lines
            ]
            output_lines = [
                [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in line[1:]] +
                [self.word2id['<pad>']] * (max_len - len(line))
                for line in sorted_lines
            ]

        with torch.no_grad():
            input_lines = torch.LongTensor(input_lines)
            if lm:
                output_lines = torch.LongTensor(output_lines)
            sorted_lens = torch.LongTensor(sorted_lens).squeeze()
            rev_input = torch.LongTensor(rev_input).squeeze()

            return {
                'input': input_lines.cuda() if cuda else input_lines,
                'output': output_lines.cuda() if cuda else output_lines,
                'lens': sorted_lens.cuda() if cuda else sorted_lens,
                'rev_input': rev_input.cuda() if cuda else rev_input
            }



class ParagraphIterator(DataIterator):
    """Iterator over sentences in a large corpus."""

    def __init__(
        self, file_path, vocab_size,
        max_length=150, num_sentences=10
    ):
        """Initialize params."""
        self.file_path = file_path
        self.vocab_size = vocab_size
        self.max_length = max_length
        self.num_sentences = num_sentences
        self.read_data()

    def read_data(self):
        """Read data from files."""
        print('Reading file ...')
        self.lines = [line.strip().lower().split() for line in open(self.file_path, 'r', encoding='utf-8')]
        print('Building vocabulary ...')
        self.word2id, self.id2word = self.construct_vocab(self.lines, self.vocab_size)

    def shuffle_dataset(self):
        """Shuffle dataset."""
        self.lines = shuffle(self.lines)

    def get_minibatch(self, index, batch_size):
        """Prepare minibatch."""
        lines = []
        for i in range(0, batch_size * self.num_sentences, self.num_sentences):
            lines.append(self.lines[i: i + self.num_sentences])


class InterpolationIterator(DataIterator):
    """Iterator over sentences in a large corpus."""

    def __init__(self, file_path, vocab_size):
        """Initialize params."""
        self.file_path = file_path
        self.vocab_size = vocab_size
        self.read_data()

    def read_data(self):
        """Read data from files."""
        print('Reading file ...')
        self.lines = [line.strip().split('\t') for line in open(self.file_path, 'r', encoding='utf-8')]

        print('Building vocabulary ...')
        self.word2id, self.id2word = self.construct_vocab(
            [x[0].split() + x[1].split() for x in self.lines],
            self.vocab_size
        )

    def shuffle_dataset(self):
        """Shuffle dataset."""
        self.lines = shuffle(self.lines)

    def get_minibatch(self, index, batch_size, max_length, lm=False, volatile=False):
        """Prepare minibatch."""
        pass


class STSBenchmarkIterator(DataIterator):
    """Iterator over sentences in a large corpus."""

    def __init__(self, file_path, vocab_size):
        """Initialize params."""
        self.file_path = file_path
        self.vocab_size = vocab_size
        self.read_data()

    def read_data(self):
        """Read data from files."""
        print('Reading file ...')
        self.lines = [line.strip().split('\t') for line in open(self.file_path, 'r', encoding='utf-8')]

        print('Building vocabulary ...')
        self.word2id, self.id2word = self.construct_vocab(
            [x[-1].split() + x[-2].split() for x in self.lines],
            self.vocab_size
        )

    def shuffle_dataset(self):
        """Shuffle dataset."""
        self.lines = shuffle(self.lines)

    def get_minibatch(self, index, batch_size, max_length, lm=False, volatile=False):
        """Prepare minibatch."""
        pass


class NLIIterator(DataIterator):
    """Data iterator for tokenized NLI datasets."""

    def __init__(
        self, train, dev, test,
        vocab_size, lowercase=True, vocab=None
    ):
        """Initialize params."""
        self.train = train
        self.dev = dev
        self.test = test
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.vocab = vocab
        self.train_lines = [line.strip().lower().split('\t') for line in open(self.train) if line.strip().split('\t')[-1] != '-']
        self.dev_lines = [line.strip().lower().split('\t') for line in open(self.dev) if line.strip().split('\t')[-1] != '-']
        self.test_lines = [line.strip().lower().split('\t') for line in open(self.test) if line.strip().split('\t')[-1] != '-']

        self.word2id, self.id2word = self.construct_vocab(
            [x[0] for x in self.train_lines] + [x[1] for x in self.train_lines],
            self.vocab_size, lowercase=self.lowercase
        )

        self.text2label = {
            'entailment': 0,
            'neutral': 1,
            'contradiction': 2
        }

        self.shuffle_dataset()

    def shuffle_dataset(self):
        """Shuffle training data."""
        self.train_lines = shuffle(self.train_lines)

    def get_parallel_minibatch(
        self, index, batch_size, sent_type='train', pad_max=False, max_len=1000
    ):
        """Prepare minibatch."""
        if sent_type == 'train':
            lines = self.train_lines
        elif sent_type == 'dev':
            lines = self.dev_lines
        else:
            lines = self.test_lines

        sent1 = [
            ['<s>'] + line[0].split()[:max_len] + ['</s>']
            for line in lines[index: index + batch_size]
        ]

        sent2 = [
            ['<s>'] + line[1].split()[:max_len] + ['</s>']
            for line in lines[index: index + batch_size]
        ]

        labels = [self.text2label[line[2]] for line in lines[index: index + batch_size]]

        sent1_lens = [len(line) for line in sent1]
        sorted_sent1_indices = np.argsort(sent1_lens)[::-1]
        sorted_sent1_lines = [sent1[idx] for idx in sorted_sent1_indices]
        rev_sent1 = np.argsort(sorted_sent1_indices)

        sent2_lens = [len(line) for line in sent2]
        sorted_sent2_indices = np.argsort(sent2_lens)[::-1]
        sorted_sent2_lines = [sent2[idx] for idx in sorted_sent2_indices]
        rev_sent2 = np.argsort(sorted_sent2_indices)

        sorted_sent1_lens = [len(line) for line in sorted_sent1_lines]
        sorted_sent2_lens = [len(line) for line in sorted_sent2_lines]

        if pad_max:
            max_sent1_len = 128
            max_sent2_len = 128
        else:
            max_sent1_len = max(sorted_sent1_lens)
            max_sent2_len = max(sorted_sent2_lens)

        sent1 = [
            [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in line] +
            [self.word2id['<pad>']] * (max_sent1_len - len(line))
            for line in sorted_sent1_lines
        ]

        sent2 = [
            [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in line] +
            [self.word2id['<pad>']] * (max_sent2_len - len(line))
            for line in sorted_sent2_lines
        ]

        sent1 = Variable(torch.LongTensor(sent1)).cuda()
        sent2 = Variable(torch.LongTensor(sent2)).cuda()
        labels = Variable(torch.LongTensor(labels)).cuda()
        sent1_lens = Variable(torch.LongTensor(sorted_sent1_lens), requires_grad=False).squeeze().cuda()
        sent2_lens = Variable(torch.LongTensor(sorted_sent2_lens), requires_grad=False).squeeze().cuda()
        rev_sent1 = Variable(torch.LongTensor(rev_sent1), requires_grad=False).squeeze().cuda()
        rev_sent2 = Variable(torch.LongTensor(rev_sent2), requires_grad=False).squeeze().cuda()

        return {
            'sent1': sent1,
            'sent2': sent2,
            'sent1_lens': sent1_lens,
            'sent2_lens': sent2_lens,
            'rev_sent1': rev_sent1,
            'rev_sent2': rev_sent2,
            'labels': labels
        }


class AmazonReviewIterator(DataIterator):
    """Data iterator for tokenized NLI datasets."""

    def __init__(
        self, train, vocab_size,
        lowercase=True, vocab=None, max_length=50
    ):
        """Initialize params."""
        self.train = train
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        self.vocab = vocab
        self.max_length = max_length
        print('Reading file ...')
        self.lines = [
            line.strip().lower().split('\t') for line in codecs.open(self.train, encoding='utf-8')
        ]
        print('Trimming lines with length < %d ' % (self.max_length))
        self.lines = [
            line for line in self.lines if len(line[0].split()) < self.max_length
        ]
        print('Constructing vocabulary ...')
        self.word2id, self.id2word = self.construct_vocab(
            [x[0] for x in self.lines],
            self.vocab_size, lowercase=self.lowercase
        )

        self.shuffle_dataset()

    def shuffle_dataset(self):
        """Shuffle training data."""
        self.lines = shuffle(self.lines)

    def get_minibatch(self, index, batch_size, max_length=None, lm=False, volatile=False):
        """Prepare minibatch."""
        max_length = self.max_length if max_length is None else max_length
        lines = [
            ['<s>'] + line[0].split()[:self.max_length] + ['</s>']
            for line in self.lines[index: index + batch_size]
        ]

        lens = [len(line) for line in lines]
        sorted_indices = np.argsort(lens)[::-1]
        rev_input = np.argsort(sorted_indices)

        sorted_lines = [lines[idx] for idx in sorted_indices]
        sorted_lens = [len(line) for line in sorted_lines]

        max_len = max(sorted_lens)

        if not lm:
            input_lines = [
                [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in line] +
                [self.word2id['<pad>']] * (max_len - len(line))
                for line in sorted_lines
            ]
            output_lines = None
        else:
            input_lines = [
                [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in line[:-1]] +
                [self.word2id['<pad>']] * (max_len - len(line))
                for line in sorted_lines
            ]
            output_lines = [
                [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in line[1:]] +
                [self.word2id['<pad>']] * (max_len - len(line))
                for line in sorted_lines
            ]

        with torch.no_grad():
            input_lines = torch.LongTensor(input_lines).cuda()
            if lm:
                output_lines = torch.LongTensor(output_lines).cuda()
            sorted_lens = torch.LongTensor(sorted_lens).squeeze().cuda()
            rev_input = torch.LongTensor(rev_input).squeeze().cuda()

            return {
                'input': input_lines,
                'output': output_lines,
                'lens': sorted_lens,
                'rev_input': rev_input
            }


class COCOCaptioningIterator(DataIterator):
    """Data iterator for tokenized NLI datasets."""

    def __init__(self, pkl_path, vocab_size, lowercase=True):
        """Initialize params."""
        self.pkl_path = pkl_path
        self.vocab_size = vocab_size
        self.lowercase = lowercase
        print('Reading pickle file : %s ...' % (self.pkl_path))
        feat_dict = pickle.load(open(self.pkl_path, 'rb'))

        train_dict = {
            k: v for k, v in feat_dict.items()
            if v['split'] == 'train' or v['split'] == 'restval'
        }
        valid_dict = {
            k: v for k, v in feat_dict.items() if v['split'] == 'val'
        }
        test_dict = {
            k: v for k, v in feat_dict.items() if v['split'] == 'test'
        }

        print('Processing train captions ...')
        self.train_img_feats, self.train_captions = self._process_data(
            train_dict
        )

        print('Processing valid captions ...')
        self.valid_img_feats, self.valid_captions = self._process_data(
            valid_dict
        )

        print('Processing test captions ...')
        self.test_img_feats, self.test_captions = self._process_data(
            test_dict
        )

        print('Constructing vocabulary ...')
        self.word2id, self.id2word = self.construct_vocab(
            self.train_captions, self.vocab_size, lowercase=self.lowercase
        )

        self.shuffle_dataset()

    def _process_data(self, feat_dict):
        img_feats = []
        captions = []
        for k, v in feat_dict.items():
            for sentence in v['sentences']:
                captions.append(' '.join(nltk.word_tokenize(sentence['raw'].lower())))
                img_feats.append(v['img_feats'])
        return img_feats, captions

    def shuffle_dataset(self):
        """Shuffle training data."""
        self.train_captions, self.train_img_feats = shuffle(
            self.train_captions, self.train_img_feats
        )

    def get_minibatch(
        self, index, batch_size, max_length=None,
        lm=False, volatile=False, split='train'
    ):
        """Prepare minibatch."""
        if split == 'train':
            lines = [
                ['<s>'] + line.split() + ['</s>']
                for line in self.train_captions[index: index + batch_size]
            ]
        elif split == 'valid':
            lines = [
                ['<s>'] + line.split() + ['</s>']
                for line in self.valid_captions[index: index + batch_size]
            ]
        elif split == 'test':
            lines = [
                ['<s>'] + line.split() + ['</s>']
                for line in self.test_captions[index: index + batch_size]
            ]

        lens = [len(line) for line in lines]
        max_len = max(lens)

        if not lm:
            input_lines = [
                [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in line] +
                [self.word2id['<pad>']] * (max_len - len(line))
                for line in lines
            ]
            output_lines = None
        else:
            input_lines = [
                [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in line[:-1]] +
                [self.word2id['<pad>']] * (max_len - len(line))
                for line in lines
            ]
            output_lines = [
                [self.word2id[w] if w in self.word2id else self.word2id['<unk>'] for w in line[1:]] +
                [self.word2id['<pad>']] * (max_len - len(line))
                for line in lines
            ]

        input_lines = Variable(
            torch.LongTensor(input_lines), volatile=volatile
        ).cuda()
        if lm:
            output_lines = Variable(
                torch.LongTensor(output_lines), volatile=volatile
            ).cuda()

        return {
            'input': input_lines,
            'output': output_lines
        }
#Conditional batchnorm
class TwoInputModule(nn.Module):
    """Abstract class."""

    def forward(self, input1, input2):
        """Forward method."""
        raise NotImplementedError


class CondBatchNorm(nn.BatchNorm2d, TwoInputModule):
    """Conditional batch norm."""

    def __init__(self, x_dim, z_dim, eps=1e-5, momentum=0.1):
        """Constructor.

        - `x_dim`: dimensionality of x input
        - `z_dim`: dimensionality of z latents
        """
        super(CondBatchNorm, self).__init__(x_dim, eps, momentum, affine=False)
        self.eps = eps
        self.shift_conv = nn.Sequential(
            nn.Conv2d(z_dim, x_dim, kernel_size=1, padding=0, bias=True),
            # nn.ReLU(True)
        )
        self.scale_conv = nn.Sequential(
            nn.Conv2d(z_dim, x_dim, kernel_size=1, padding=0, bias=True),
            # nn.ReLU(True)
        )

    def forward(self, input, noise):
        """Forward method."""
        shift = self.shift_conv.forward(noise)
        scale = self.scale_conv.forward(noise)

        norm_features = super(CondBatchNorm, self).forward(input)
        output = norm_features * scale + shift
        return output


if __name__ == '__main__':
    import sys, time
    import argparse
    sys.path.append('../gensen')
    from gensen import GenSenSingle
    start = time.time()
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--folder_path",
        help="path to model folder",
        default='/data/milatmp1/subramas/models/GenSen'
    )
    parser.add_argument(
        "--prefix",
        help="prefix to model 1",
        default='nli_large_bothskip'  # (leave as is)
    )
    parser.add_argument(
        "--train_filename",
        help="Train source file",
        default='/data/milatmp1/subramas/datasets/MultiSeq2Seq/bookcorpus_12m_train.txt',
    )
    parser.add_argument(
        "--pretrain",
        help="Path to pretrained embeddings",
        default='/data/milatmp1/subramas/embeddings/new_glove.840B.300d.h5',  # (Don't mess with this)
    )
    args = parser.parse_args()

    batch_size = 20000
    hidden_size = 2048
    max_length = 100
    data_file = args.train_filename

    iterator = SentenceIterator(
        data_file,
        vocab_size=80000,
        max_length=max_length
    )
    model = GenSenSingle(
        model_folder=args.folder_path,
        filename_prefix=args.prefix,
        pretrained_emb=args.pretrain,
        cuda=True
    )
    iterator.word2id = model.word2id
    iterator.id2word = model.id2word
    model.vocab_expansion(model.id2word.values())
    sentences = iterator.lines if batch_size is 'all' else iterator.lines[0:batch_size]
    sentences = [' '.join(s[:max_length]) for s in sentences]
    repr_last_h = np.empty((0, hidden_size))
    for mbatch_idx, mbatch in enumerate(range(0, len(sentences), 200)):
        less_sentences = sentences[mbatch: mbatch + 200]
        _, last_h = model.get_representation(
            less_sentences, pool='last', return_numpy=True, tokenize=False
        )
        repr_last_h = np.append(repr_last_h, last_h, axis=0)
    print(repr_last_h.shape)
    iterator.build_kde(repr_last_h=repr_last_h, num_dim_pca=40, grid_search_num=7)
    data_gen = iterator.sample_kde(batch_size=10, cuda=True)
    print(data_gen['input'].shape)
    iterator.save_kde(file_name_kde="kde.sav", file_name_pca="pca.sav")
    iterator.load_kde(file_name_kde="kde.sav", file_name_pca="pca.sav")
    data_gen = iterator.sample_kde(batch_size=10, cuda=False)
    print(data_gen['input'].shape)
    total_time = time.time() - start
    print(total_time)

