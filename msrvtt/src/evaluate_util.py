import os
import re
import subprocess
import threading
import tempfile

from nltk.tree import Tree
from zss import simple_distance, Node
import io

STANFORD_CORENLP = '/DATA-NFS/yuanyitian/stanford-corenlp-full-2018-10-05/'

def enc(s):
    return s.encode('utf-8')


def dec(s):
    return s.decode('utf-8')


def deleaf(parse_string):
    tree = Tree.fromstring(parse_string.strip(), read_leaf=lambda s: "")
    for sub in tree.subtrees():
        for n, child in enumerate(sub):
            if isinstance(child, str):
                continue
            if len(list(child.subtrees(filter=lambda x: x.label() == '-NONE-'))) == len(child.leaves()):
                del sub[n]
    oneline = tree.pformat(margin=10000, parens=[" ( ", " ) "])
    oneline = re.sub(' +', ' ', oneline)
    return oneline


def extract_parses(fname):
    # extract parses from corenlp output
    # based on https://github.com/miyyer/scpn/blob/master/read_paranmt_parses.py
    f = io.open(fname, "r", encoding='utf-8')

    count = 0
    sentences = []
    data = {'tokens': [], 'pos': [], 'parse': '', 'deps': []}
    for idx, line in enumerate(f):
        if idx <= 1:
            continue
        if line.startswith('Sentence #'):
            new_sent = True
            new_pos = False
            new_parse = False
            new_deps = False
            if idx == 2:
                continue

            sentences.append(data)
            count += 1

            data = {'tokens': [], 'pos': [], 'parse': '', 'deps': []}

        # read original sentence
        elif new_sent:
            new_sent = False
            new_pos = True

        elif new_pos and line.startswith("Tokens"):
            continue

        # read POS tags
        elif new_pos and line.startswith('[Text='):
            line = line.strip().split()
            w = line[0].split('[Text=')[-1]
            pos = line[-1].split('PartOfSpeech=')[-1][:-1]
            data['tokens'].append(w)
            data['pos'].append(pos)

        # start reading const parses
        elif (new_pos or new_parse) and len(line.strip()):
            if line.startswith("Constituency parse"):
                continue
            new_pos = False
            new_parse = True
            data['parse'] += ' ' + line.strip()

        # start reading deps
        elif (new_parse and line.strip() == "") or \
                line.startswith("Dependency Parse"):
            new_parse = False
            new_deps = True

        elif new_deps and len(line.strip()):
            line = line.strip()[:-1].split('(', 1)
            rel = line[0]
            x1, x2 = line[1].split(', ')
            x1 = x1.replace("'", "")
            x2 = x2.replace("'", "")
            x1 = int(x1.rsplit('-', 1)[-1])
            x2 = int(x2.rsplit('-', 1)[-1])
            data['deps'].append((rel, x1 - 1, x2 - 1))

        else:
            new_deps = False

    sentences.append(data)

    return sentences


class stanford_parsetree_extractor:
    def __init__(self):
        self.stanford_corenlp_path = os.path.join(STANFORD_CORENLP, "*")
        print("standford corenlp path:", self.stanford_corenlp_path)
        self.output_dir = './tmp/'
        self.cmd = ['java', '-cp', self.stanford_corenlp_path,
                    '-Xmx2G', 'edu.stanford.nlp.pipeline.StanfordCoreNLP',
                    '-annotators', 'tokenize,ssplit,pos,parse',
                    '-ssplit.eolonly', '-outputFormat', 'text',
                    '-outputDirectory', self.output_dir,
                    '-file', None]

    def run(self, file_list):
        all_output = []
        for file in file_list:
            print("parsing file:", file)
            self.cmd[-1] = file
            out = subprocess.call(
                self.cmd,
                cwd=os.path.dirname(os.path.abspath(__file__)),
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE)
            parsed_file = \
                os.path.join(
                    self.output_dir,
                    os.path.split(file)[1] + ".out")
            output = [deleaf(e['parse']).strip() for e in extract_parses(parsed_file)]
        all_output = all_output + output
        return all_output

    def cleanup(self):
        None


def build_tree(s):
    old_t = Tree.fromstring(s)
    new_t = Node("S")

    def create_tree(curr_t, t):
        if t.label() and t.label() != "S":
            new_t = Node(t.label())
            curr_t.addkid(new_t)
        else:
            new_t = curr_t
        for i in t:
            if isinstance(i, Tree):
                create_tree(new_t, i)
    create_tree(new_t, old_t)
    return new_t


def strdist(a, b):
    if a == b:
        return 0
    else:
        return 1


def compute_tree_edit_distance(pred_parse, ref_parse):
    return simple_distance(
        build_tree(ref_parse), build_tree(pred_parse), label_dist=strdist)