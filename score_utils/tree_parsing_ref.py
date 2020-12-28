from nltk.tree import ParentedTree, Tree
import re
from stanfordcorenlp import StanfordCoreNLP

def is_leaf(node):
    if type(node[0]) == str and len(node) != 1:
        print(1)
    return type(node[0]) == str


def get_span_to_node_mapping(tree):
    def dfs(node, span_to_node, node_to_span, idx):
        if is_leaf(node):
            span_to_node[idx] = node
            node_to_span[id(node)] = idx
            return idx + 1
        prev_idx = idx
        for child in node:
            idx = dfs(child, span_to_node, node_to_span, idx)
        span_to_node[(prev_idx, idx-1)] = node
        node_to_span[id(node)] = (prev_idx, idx-1)
        return idx
    span2node, node2span = {}, {}
    dfs(tree, span2node, node2span, 0)
    return span2node, node2span

if __name__ == "__main__":

    tree_string = "(ROOT  (S    (S      (NP (JJ Pairwise) (NN contrast) (NNS tests))      (VP (VBD revealed)        (SBAR (IN that) (, ,)          (S            (SBAR (IN as)              (S                (VP (VBN predicted))))            (, ,)            (NP (NNP DenComp))            (VP (VBD had)              (NP                (NP (JJR higher) (NNS ratings))                (PP (IN of)                  (NP (NN experience))))              (SBAR                (WHADVP (WRB when))                (S                  (VP (VBN represented)                    (PP (IN by)                      (NP                        (NP                          (NP (NN theCEO)                            (PRN (-LRB- -LRB-)                              (S                                (NP (NN M))                                (VP (SYM =)                                  (NP (CD 4.11))))                              (, ,)                              (S                                (NP (NN SD))                                (VP (SYM =)                                  (NP (CD 1.80))))                              (-RRB- -RRB-)))                          (PP (IN versus)                            (NP (DT the) (NN headquarters)                              (PRN (-LRB- -LRB-)                                (NP (NN M) (SYM =) (CD 3.58))                                (, ,)))))                        (NFP |)                        (PRN                          (NP (NN SD) (SYM =) (CD 1.83))                          (-RRB- -RRB-))))))))))))    (, ,)    (S      (NP (NN t) (-LRB- -LRB-) (CD 299) (-RRB- -RRB-))      (VP (SYM =)        (NP (CD 2.29))))    (, ,)    (S      (NP (NN p))      (VP (SYM =)        (NP (CD .023))))    (, ,)    (S      (NP (NN d))      (VP (SYM =)        (NP (CD 0.29))))    (. .)))"
    # print(tree_string)
    tree1 = Tree.fromstring(tree_string)
    tree1.pretty_print()
    tree_string = tree_string.replace("-LRB-", " ").replace("-RRB-", " ")
    tree2 = Tree.fromstring(tree_string)
    tree2.pretty_print()
    # print(tree_string)
    # print(tree_string)

    # tree = ParentedTree.fromstring(tree_string)
    # span2node, node2span = get_span_to_node_mapping(tree)
    # print(span2node)
    # print(node2span)


    # tree_pos = tree.treepositions()
    # phrase_list = set()
    # for cur_pos in tree_pos:
    #     cur_phrase = tree[cur_pos]
    #     if type(cur_phrase) != str:
    #         cur_phrase = " ".join(cur_phrase.leaves())
    #     phrase_list.add(cur_phrase)
    #
    #
    # phrase_list = list(filter(lambda x: len(x) >= 2, phrase_list))
    # phrase_list = sorted(list(phrase_list), key=lambda x: len(x))
    # print(len(phrase_list))
    # for i in phrase_list:
    #     print(i)
