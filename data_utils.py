
import gensim, os, re, sys
import numpy as np
import unicodedata
import nltk
import jieba
import jieba.posseg as pseg
import thulac
from pyltp import Segmentor
from pyltp import Postagger
import pyopencc
import synonyms

WORD2VEC = None
__location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

class Vocab(object):
    def __init__(self, filename, init=None, cutoff=None, embedding_files=None):
        self.filename = filename
        self.reserved_vocab_list = []
        self.vocab_list = []
        self.vocab_dict = {}
        if os.path.exists(filename):
            for line in open(filename, 'r'):
                [key, count] = line.strip("\n").split("\t")
                if int(count) < 0:
                    self.reserved_vocab_list.append(key)
                self.vocab_list.append(key)
                self.vocab_dict[key] = [len(self.vocab_dict), int(count)]
        elif init != None:
            for item in init:
                self.reserved_vocab_list.append(item)
                self.vocab_list.append(item)
                self.vocab_dict[item] = [len(self.vocab_dict), -1]
        self.changed = False
        self.cutoff = cutoff if cutoff else sys.maxsize
        if embedding_files != None:
            word2vecs = []
            w2v_sizes = []
            for path in embedding_files.split(','):
                word2vecs.append(FastWord2vec(path))
                w2v_sizes.append(word2vecs[-1].syn0.shape[1])
            self.embedding_init = np.random.normal(0.0, 0.01, (self.size(), sum(w2v_sizes)))
            for idx, word in enumerate(self.vocab_list[:self.size()]):
                ptr = 0
                for i, word2vec in enumerate(word2vecs):
                    try:
                        hit = word2vec[word]
                        self.embedding_init[idx, ptr:ptr+w2v_sizes[i]] = hit
                    except:
                        pass
                    finally:
                        ptr += w2v_sizes[i]
        else:
            self.embedding_init = None

    def idx2key(self, idx):
        """given index return key"""
        if idx >= min(self.cutoff, len(self.vocab_list)):
            return None
        else:
            return self.vocab_list[idx]

    def key2idx(self, key):
        """given key return index"""
        value = self.vocab_dict.get(key)
        if value:
            if value[0] < self.cutoff:
                return value[0]
            else:
                return None
        else:
            return None

    def key2count(self, key):
        """given key return count"""
        value = self.vocab_dict.get(key)
        if value:
            if value[0] < self.cutoff:
                return value[1]
            else:
                return 0
        else:
            return 0

    def size(self):
        """return size of the vocab"""
        return min(self.cutoff, len(self.vocab_list))

    def dump(self):
        """dump the vocab to the file"""
        if self.changed:
            with open(self.filename, 'w') as f:
                for key in self.vocab_list:
                    f.write(key+'\t'+str(self.vocab_dict[key][1])+'\n')

    def update(self, patch):
        """update the vocab"""
        self.changed = True
        for key in patch:
            if self.vocab_dict.has_key(key):
                if self.vocab_dict[key][1] >= 0:
                    self.vocab_dict[key][1] += patch[key]
            else:
                self.vocab_dict[key] = [len(self.vocab_dict), patch[key]]
        self.vocab_list = self.reserved_vocab_list + sorted(
            filter(lambda i: self.vocab_dict[i][1] >= 0, self.vocab_dict),
            key=lambda i: self.vocab_dict[i][1],
            reverse=True)
        for idx, item in enumerate(self.vocab_list):
            self.vocab_dict[item][0] = idx


    def sentence_to_token_ids(self, text):
        """encode a sentence in plain text into a sequence of token ids
        """
        if not type(text) is list:
            text = text.strip()
            text = map(lambda i: i.encode('utf-8'), list(text.decode('utf-8')))
        seq = [self.key2idx(key) for key in text]
        seq = [idx if idx!=None else self.key2idx("_UNK") for idx in seq]
        return seq

    def token_ids_to_sentence(self, token_ids):
        """decode a sequence of token ids to a sentence
        """
        token_ids = map(lambda i: i if self.idx2key(i) else self.key2idx("_UNK"), token_ids)
        text = "".join([self.idx2key(i) if i != self.key2idx("_PAD") else ' ' for i in token_ids])
        return text.strip(' ')

class Dict(object):
    def __init__(self, filename):
        self.filename = filename
        self._dict = {}
        if os.path.exists(filename):
            for line in open(filename, 'r'):
                key, values = line.strip().split('\t')
                self._dict[key] = values
    def lookup(self, key):
        value = self._dict.get(key)
        if value:
            return value
        else:
            return None

class FastWord2vec(object):
    """
    Load word2vec model using gensim, cache the embedding matrix using numpy
    and memory-map it so that future loads are fast.
    """
    def __init__(self, path):
        if not os.path.exists(path + ".npy"):
            # load gensim
            word2vec = gensim.models.KeyedVectors.load_word2vec_format(
                path,
                binary=False)

            # save as numpy
            np.save(path + ".npy", word2vec.syn0)
            # also save the vocab
            with open(path + ".vocab", "wt") as fout:
                for word in word2vec.index2word:
                    fout.write(word.encode('utf-8') + "\n")

        self.syn0 = np.load(path + ".npy", mmap_mode="r")
        self.index2word = [l.strip("\n") for l in open(path + ".vocab", "rt")]
        self.word2index = {word: k for k, word in enumerate(self.index2word)}
        self._word_ending_tables = {}
        self._word_beginning_tables = {}

    def __getitem__(self, key):
        return np.array(self.syn0[self.word2index[key]])

    def __contains__(self, key):
        return key in self.word2index

    def words_ending_in(self, word_ending):
        if len(word_ending) == 0:
            return self.index2word
        self._build_word_ending_table(len(word_ending))
        return self._word_ending_tables[len(word_ending)].get(word_ending, [])

    def _build_word_ending_table(self, length):
        if length not in self._word_ending_tables:
            table = {}
            for word in self.index2word:
                if len(word) >= length:
                    ending = word[-length:]
                    if ending not in table:
                        table[ending] = [word]
                    else:
                        table[ending].append(word)
            self._word_ending_tables[length] = table

    def words_starting_in(self, word_beginning):
        if len(word_beginning) == 0:
            return self.index2word
        self._build_word_beginning_table(len(word_beginning))
        return self._word_beginning_tables[len(word_beginning)].get(word_beginning, [])

    def _build_word_beginning_table(self, length):
        if length not in self._word_beginning_tables:
            table = {}
            for word in get_progress_bar('building prefix lookup ')(self.index2word):
                if len(word) >= length:
                    ending = word[:length]
                    if ending not in table:
                        table[ending] = [word]
                    else:
                        table[ending].append(word)
            self._word_beginning_tables[length] = table

    @staticmethod
    def get(path):
        global WORD2VEC
        if WORD2VEC is None:
            WORD2VEC = FastWord2vec(path)
        return WORD2VEC


class Cutter(object):
    """
    one cutter to include them all!
    """
    def __init__(self, _cutter):

        self._cutter = _cutter
        if _cutter == 'jieba':
            jieba.enable_parallel(64)

        elif _cutter == 'thulac':
            self.thulac = thulac.thulac()
            self.thulac_pos_map = Dict(os.path.join(__location__, 'thulac_pos_map'))

        elif _cutter == 'ltp':
            ltp_path = os.path.join(__location__, "ltp_data_v3.4.0")
            self.ltp_seg = Segmentor()
            self.ltp_seg.load(os.path.join(ltp_path, "cws.model"))
            self.ltp_pos = Postagger()
            self.ltp_pos.load(os.path.join(ltp_path, "pos.model"))
            self.ltp_pos_map = Dict(os.path.join(__location__, 'ltp_pos_map'))

    def cut_words(self, text):
        """
        cut the words

        cutter: jieba|thulac
        """
        if self._cutter == 'jieba':
            s = list(jieba.cut(text))
        elif self._cutter == 'thulac':
            s = self.thulac.cut(text)
            s = map(lambda i: (i[0], self.thulac_pos_map.lookup(i[1])), s)
        elif self._cutter == 'ltp':
            s1 = self.ltp_seg.segment(text)
            s2 = self.ltp_pos.postag(s1)
            s2 = map(lambda i: self.ltp_pos_map.lookup(i), s2)
            s = zip(s1, s2)
        elif self._cutter == 'nltk':
            s = nltk.tokenize.word_tokenize(text)
        return s

class Synonyms(object):
    """
    Get synonyms using various resources!
    """
    def __init__(self):
        with open(os.path.join(__location__, 'cilin.txt'), 'r') as f:
            cilin = map(lambda i: i.strip().split()[1:], f.readlines())
        self.cilin_map = dict(zip(range(len(cilin)), cilin))
        self.cilin_index = {}
        for id, words in self.cilin_map.items():
            for word in words:
                if word in self.cilin_index:
                    self.cilin_index[word].add(id)
                else:
                    self.cilin_index[word] = set([id])

    def get(self, word, source='synonyms'):
        if source == 'synonyms':
            candidates, scores = synonyms.nearby(word)
            if len(candidates) > 0:
                results = zip(candidates, scores)
                results = filter(lambda i: i[1] > 0.5, results)
                results = map(lambda i: i[0].encode('utf-8'), results)
                results.remove(word)
            else:
                results = []
        elif source == 'cilin':
            if word in self.cilin_index:
                results = map(lambda i: self.cilin_map[i], self.cilin_index[word])
                results = reduce(lambda a, b: a+b, results)
                results = list(set(results))
                results.remove(word)
            else:
                results = []
        return results

zht2zhs = pyopencc.OpenCC('zht2zhs.ini').convert
def normalize(text):
    text = unicodedata.normalize('NFKC', text.decode('utf-8')).encode('utf-8')
    text = zht2zhs(text)
    return text

def labels_to_ids_array(labels, vocab):
    """
    args:
        labels: list of labels
        vocab: vocab of characters
    return:
        label_ids: num_labels x char_length
    """
    label_ids_list = []
    for label in labels:
        label_ids = vocab.sentence_to_token_ids(label)
        label_ids_list.append(label_ids)
    max_char_length = max(map(lambda i: len(i), label_ids_list))
    label_ids_list = map(lambda i: i+[0]*(max_char_length-len(i)), label_ids_list)
    return label_ids_list

def words_to_token_ids(words, vocab):
    """
    Turn outputs of posseg into two seq of ids
    """

    seqs = []
    segs = []
    for word in words:
        word_ids = vocab.sentence_to_token_ids(word)
        if len(word_ids) > 100:
            word_ids = [vocab.key2idx("_UNK")]
        elif len(word_ids) == 0:
            continue
        seqs.extend(word_ids)
        segs.extend([1.0]+[0.0]*(len(word_ids)-1))
    if len(segs) > 0:
        segs.append(1.0)
    return seqs, segs

def posseg_to_token_ids(pos_segs, vocab, posseg_vocab):
    """
    Turn outputs of posseg into two seq of ids
    """

    seqs = []
    segs = []
    pos_labels = []
    for token in pos_segs:
        word, tag = token
        if tag == None:
            print(pos_segs)
        word_ids = vocab.sentence_to_token_ids(word)
        if len(word_ids) == 0:
            continue
        seqs.extend(word_ids)
        segs.extend([1.0]+[0.0]*(len(word_ids)-1))
        pos_labels.append(tag)
    if len(segs) > 0:
        segs.append(1.0)
    pos_labels = posseg_vocab.sentence_to_token_ids(pos_labels)
    return seqs, segs, pos_labels
