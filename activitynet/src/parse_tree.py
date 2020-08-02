import numpy

import sys
reload(sys)
sys.setdefaultencoding('utf-8')

def build_tree(depth, sen):
    assert len(depth) == len(sen)

    if len(depth) == 1:
        parse_tree = sen[0]
    else:
        idx_max = numpy.argmax(depth)
        parse_tree = []
        if len(sen[:idx_max]) > 0:
            tree0 = build_tree(depth[:idx_max], sen[:idx_max])
            parse_tree.append(tree0)
        tree1 = sen[idx_max]
        if len(sen[idx_max + 1:]) > 0:
            tree2 = build_tree(depth[idx_max + 1:], sen[idx_max + 1:])
            tree1 = [tree1, tree2]
        if parse_tree == []:
            parse_tree = tree1
        else:
            parse_tree.append(tree1)
    return parse_tree


def MRG(tr):
    if not isinstance(tr, list):
        return '(' + tr + ')'
        # return tr + ' '
    else:
        s = '('
        for subtr in tr:
            s += MRG(subtr)
        s += ')'
        return s


# parse_tree = build_tree(gates, words)
# print MRG(parse_tree)