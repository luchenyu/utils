import gensim, os, re, sys
import numpy as np
import unicodedata
import nltk
from hanziconv import HanziConv

WORD2VEC = None
__location__ = os.path.realpath(
        os.path.join(os.getcwd(), os.path.dirname(__file__)))

class Vocab(gensim.corpora.Dictionary):
    """
    General vocab based on gensim Dictionary
    """
    def __init__(self, filename=None, special_token_dict=None):
        gensim.corpora.Dictionary.__init__(self)
        self.special_token_dict = special_token_dict
        if filename != None:
            other = Vocab.load_from_text(filename)
            for key in self.__dict__:
                if not other.__dict__.get(key) is None:
                    self.__dict__[key] = other.__dict__[key]
        if special_token_dict != None:
            self.patch_with_special_tokens(special_token_dict)

    def idx2doc(self, idxs, unknown_word=''):
        doc = [self.get(i, default=unknown_word) for i in idxs]
        return doc

class AtomicVocab(Vocab):
    """
    Typically vocab of characters
    with special tokens [<PAD>, <SEP>, <UNK>] in the beginning
    """
    def __init__(self, filename=None, special_tokens=[],
                 atomize_method=None, rev_atomize_method=None,
                 embedding_files=None):
        self.pad, self.sep, self.unk = '<PAD>', '<SEP>', '<UNK>'
        special_token_dict = {self.pad: 0, self.sep: 1, self.unk: 2}
        for i, tok in enumerate(special_tokens):
            special_token_dict[tok] = i+3
        Vocab.__init__(self, filename, special_token_dict)
        if atomize_method != None:
            self.atomize_method = atomize_method
        else:
            _sep_ = re.compile(r'\s+')
            def _atomize_method(molecule):
                if molecule == '':
                    toks = []
                elif molecule in self.token2id:
                    toks = [molecule]
                elif not re.fullmatch(_sep_, molecule) is None:
                    toks = [self.sep]
                else:
                    toks = list(molecule)
                return toks
            self.atomize_method = _atomize_method
        if rev_atomize_method != None:
            self.rev_atomize_method = rev_atomize_method
        else:
            _map_ = {self.pad: '', self.sep: '\t', self.unk: '*'}
            def _rev_atomize_method(toks):
                toks = [_map_[tok] if tok in _map_ else tok for tok in toks]
                molecule = ''.join(toks)
                return molecule
            self.rev_atomize_method = _rev_atomize_method
        self.embedding_init = None
        if embedding_files != None:
            self.load_embeddings(embedding_files)

    def load_embeddings(self, embedding_files):
        """
        load embeddings from files
        """
        word2vecs = []
        w2v_sizes = []
        for path in embedding_files.split(','):
            word2vecs.append(FastWord2vec(path))
            w2v_sizes.append(word2vecs[-1].syn0.shape[1])
        self.embedding_init = np.random.normal(0.0, 0.01, (len(self), sum(w2v_sizes)))
        for word, idx in self.token2id.items():
            ptr = 0
            for i, word2vec in enumerate(word2vecs):
                try:
                    hit = word2vec[word]
                    self.embedding_init[idx, ptr:ptr+w2v_sizes[i]] = hit
                except:
                    pass
                finally:
                    ptr += w2v_sizes[i]

    def doc2idx(self, document):
        return Vocab.doc2idx(self, document, unknown_word_index=self.token2id[self.unk])

    def idx2doc(self, idxs):
        return Vocab.idx2doc(self, idxs, unknown_word=self.unk)

    def molecule2idx(self, molecule):
        return self.doc2idx(self.atomize_method(molecule))

    def idx2molecule(self, idxs):
        return self.rev_atomize_method(self.idx2doc(idxs))

class MolecularVocab(Vocab):
    """
    Typically vocab of words
    with special tokens [<PAD>] and optional [<SEP>, <UNK>] in the beginning
    """
    def __init__(self, atomic_vocab, filename=None, special_tokens=[]):
        self.atomic_vocab = atomic_vocab
        self.pad, self.sep, self.unk = '<PAD>', None, None
        special_token_dict = {self.pad: 0}
        for i, tok in enumerate(special_tokens):
            special_token_dict[tok] = i+1
        Vocab.__init__(self, filename, special_token_dict)
        if '<UNK>' in self.values():
            self.unk = '<UNK>'
        if '<SEP>' in self. values():
            self.sep = '<SEP>'
        self.decompose_table = None
        self.update_decompose_table()

    def update_decompose_table(self):
        """
        decompose_table is a list of same-length lists of ids, each list is a tok
        """
        atomic_ids = [self.atomic_vocab.doc2idx([tok]) \
                      if tok in self.special_token_dict else \
                      self.atomic_vocab.molecule2idx(tok) \
                      for tok in self.values()]
        max_length = max([len(i) for i in atomic_ids])
        self.decompose_table = [i+[0]*(max_length-len(i)) for i in atomic_ids]

    def doc2idx(self, document):
        if self.unk is None:
            return Vocab.doc2idx(self, document, unknown_word_index=None)
        else:
            return Vocab.doc2idx(self, document, unknown_word_index=self.token2id[self.unk])

    def idx2doc(self, idxs):
        return Vocab.idx2doc(self, idxs, unknown_word=self.unk)

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
            import jieba
            import jieba.posseg as pseg
            jieba.enable_parallel(64)

        elif _cutter == 'thulac':
            import thulac
            self.thulac = thulac.thulac()
            self.thulac_pos_map = Dict(os.path.join(__location__, 'thulac_pos_map'))

        elif _cutter == 'ltp':
            from pyltp import Segmentor
            from pyltp import Postagger
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
            s = jieba.lcut(text)
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
        import synonyms
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

def normalize(text):
    """
    1. unicodedata normalize
    2. traditional chinese to simplified chinese
    """
    text = unicodedata.normalize('NFKC', text)
    text = HanziConv.toSimplified(text)
    return text

def tokens_to_seqs(tokens, word_vocab):
    """
    Turn list of tokens into seqs of word_ids
    """
    sep = re.compile(r'\s+')
    tokens = [word_vocab.sep if not re.fullmatch(sep, word) is None else word for word in tokens]
    seqs = word_vocab.doc2idx(tokens)
    return seqs

def seqs_to_tokens(seqs, word_vocab):
    """
    Turn seqs of word_ids to list of tokens
    """
    pad_id = word_vocab.token2id[word_vocab.pad]
    seqs = list(filter(lambda i: i != pad_id, seqs))
    tokens = word_vocab.idx2doc(seqs)
    if word_vocab.sep != None:
        tokens = ['\t' if tok == word_vocab.sep else tok for tok in tokens]
    if word_vocab.unk != None:
        tokens = ['*' if tok == word_vocab.unk else tok for tok in tokens]
    return tokens

def tokens_to_seqs_segs(tokens, char_vocab):
    """
    Turn list of tokens into seqs and segs
    args:
        tokens: list of tokens
        char_vocab: vocab of characters
    return:
        seqs: list of char_ids
        segs: list of 1/0 segment flags
    """

    def _process(token):
        char_ids = char_vocab.molecule2idx(token)
        if len(char_ids) > 30:
            char_ids = [char_vocab.token2id[char_vocab.unk]]
        if len(char_ids) == 0:
            return ([], [])
        else:
            return (char_ids, [1.0]+[0.0]*(len(char_ids)-1))
    zipped_list = [_process(token) for token in tokens]
    (seqs, segs) = zip(*zipped_list)
    seqs = [i for w in seqs for i in w]
    segs = [i for w in segs for i in w]
    if len(segs) > 0:
        segs.append(1.0)
    return seqs, segs

def seqs_segs_to_tokens(seqs, segs, char_vocab):
    """
    Turn seqs and segs to list of tokens
    args:
        seqs: list of char_ids
        segs: list of 1/0 segment flags
        char_vocab: vocab of characters
    return:
        tokens: list of tokens
    """
    tokens = []
    char_ids = [seqs[0]]
    for i, seg in enumerate(segs[1:-1]):
        if seg != 0.0:
            tok = char_vocab.idx2molecule(char_ids)
            if tok != '':
                tokens.append(tok)
            char_ids = []
        char_ids.append(seqs[i+1])
    tok = char_vocab.idx2molecule(char_ids)
    if tok != '':
        tokens.append(tok)
    return tokens