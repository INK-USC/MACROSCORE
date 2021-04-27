from nltk.tree import Tree, ParentedTree
import pandas as pd
from tqdm import tqdm
import os
import jsonlines, json
import stanza
import numpy as np
from stanfordcorenlp import StanfordCoreNLP
import argparse

# Dependency parsing utils
# tokens_list = None
def dfs(source, cur_hop, max_hop, cand_dict, entity_node_span, dep_graph, visited):
    # global tokens_list
    # print("Cur node = ", source, tokens_list[source-1])
    if cur_hop > max_hop:
        return
    visited[source] = True

    if cand_dict.get(cur_hop) is None:
        cand_dict[cur_hop] = set()
    cand_dict[cur_hop].add(source-1)

    for cur_child in dep_graph[source]:
        # If the child node is not in the entity span and is not visited
        if cur_child not in entity_node_span and visited.get(cur_child) is None:
            dfs(cur_child, cur_hop+1, max_hop, cand_dict, entity_node_span, dep_graph, visited)


def getCandidiates_dep_parse(nlp, input_text, entity_mention, entity_span, k_hop=2):
    # global tokens_list
    cand_result = []

    # Parse the input text
    parsed_doc = nlp(input_text)
    cur_sent = parsed_doc.sentences[0]

    # Create a undirected graph using adj list representation
    dep_graph = {}
    tokens_list = []
    for cur_word in cur_sent.words:
        tokens_list.append(cur_word.text)
        u = cur_word.id
        v = cur_word.head

        if dep_graph.get(u) is None:
            dep_graph[u] = []

        if dep_graph.get(v) is None:
            dep_graph[v] = []

        dep_graph[u].append(v)
        dep_graph[v].append(u)
    tokens_list = np.array(tokens_list)

    # Check if the entity mention corresponds to the entity span
    identified_entity = " ".join([tokens_list[x] for x in entity_span])
    if identified_entity != entity_mention:
        raise Exception("Entity span and mention doesn't correlate!")

    # # Print the graph
    # for k in dep_graph.keys():
    #     print("Node = ", k, tokens_list[k-1])
    #     print("Children = ", dep_graph[k], [tokens_list[x-1] for x in dep_graph[k]])

    # Run dfs
    cand_dict = {}
    entity_node_span = [x+1 for x in entity_span]
    for idx in entity_node_span:
        visited = {}
        # print("Start node = ", idx)
        dfs(idx, 0, k_hop, cand_dict, entity_node_span, dep_graph, visited)

    # Get candidate phrase spans
    for cur_hop in range(1, k_hop+1, 1):
        cur_idx_list = list(sorted(cand_dict[cur_hop]))

        # print(cur_hop, cur_idx_list)
        # print(cur_hop, [tokens_list[x] for x in cur_idx_list])

        cur_span = []
        for idx in cur_idx_list:
            # Skip the root node in dep parse (or) skip the phrase containing the entity span
            if idx == -1 or idx in entity_span:
                if len(cur_span) != 0:
                    cand_result.append({
                        "candidate_span": (cur_span[0], cur_span[-1])
                    })
                cur_span = []
                continue

            if len(cur_span) == 0:
                cur_span.append(idx)
            elif idx == cur_span[-1] + 1:
                cur_span.append(idx)
            else:
                cand_result.append({
                    "candidate_span": (cur_span[0], cur_span[-1])
                })
                cur_span = [idx]

        # Accounting to the last span
        if len(cur_span) != 0:
            cand_result.append({
                "candidate_span": (cur_span[0], cur_span[-1])
            })

    return cand_result

def demo_dep_parse(nlp):
    def find_sublist(sub, bigger):
        if not bigger:
            return -1
        if not sub:
            return 0
        first, rest = sub[0], sub[1:]
        pos = 0
        try:
            while True:
                pos = bigger.index(first, pos) + 1
                if not rest or bigger[pos:pos + len(rest)] == rest:
                    return pos - 1
        except ValueError:
            return -1

    # Test 1
    temp_text = "however , the size of the rho estimates indicate that a small amount of total error is being accounted for by level-2 variation in each model , and regression analysis of each threat model using ols produced extremely similar inferences to the multilevel model estimates ."
    temp_entity = "regression analysis"

    # # Test 2
    # temp_text = "in addition , the model found the predicted effect of phi on preference , b = 0.50 , se = 0.15 , z = 3.40 , p = .001 ."
    # temp_entity = "p = .001"

    temp_entity_span_start = find_sublist(temp_entity.split(), temp_text.split())
    temp_entity_span = [temp_entity_span_start+x for x in range(len(temp_entity.split()))]

    temp_candidates = getCandidiates_dep_parse(nlp, temp_text, temp_entity, temp_entity_span)
    print("Input text = ", temp_text)
    print("Input entity mention = ", temp_entity)
    print("Candidates = ", temp_candidates)

# Constituency parsing utils
def is_leaf(node):
    if type(node[0]) == str and len(node) != 1:
        print(1)
    return type(node[0]) == str

def dfs_v2(node, span_to_node, node_to_span, idx):
    if is_leaf(node):
        span_to_node[idx] = node
        node_to_span[id(node)] = idx
        return idx + 1
    prev_idx = idx
    for child in node:
        idx = dfs_v2(child, span_to_node, node_to_span, idx)
    span_to_node[(prev_idx, idx-1)] = node
    node_to_span[id(node)] = (prev_idx, idx-1)
    return idx

def get_span_to_node_mapping(tree):
    span2node, node2span = {}, {}
    dfs_v2(tree, span2node, node2span, 0)
    return span2node, node2span

def getCandidiates_const_parse(nlp, input_text, entity_mention, entity_span):
    cand_result = []

    # Parse the input text
    props = {'annotators': 'tokenize,ssplit,pos,lemma,ner,parse',
             'parse.model': 'edu/stanford/nlp/models/srparser/englishSR.ser.gz',
             'tokenize.whitespace': 'true', 'pipelineLanguage': 'en', 'outputFormat': 'json'}

    parsed_doc = json.loads(nlp.annotate(input_text, properties=props))
    const_parse = parsed_doc["sentences"][0]["parse"]
    const_parse = const_parse.replace("\n", "")
    const_tree = ParentedTree.fromstring(const_parse)
    tokens_list = np.array(list(const_tree.leaves()))

    # Check if the entity mention corresponds to the entity span
    identified_entity = " ".join([tokens_list[x] for x in entity_span])
    identified_entity = identified_entity.replace("-LRB-", "(")
    identified_entity = identified_entity.replace("-RRB-", ")")
    if identified_entity != entity_mention:
        print(identified_entity, entity_mention)
        raise Exception("Entity span and mention doesn't correlate!")

    # Get candidate phrase spans
    span2node, node2span = get_span_to_node_mapping(const_tree)
    cand_spans_old = span2node.keys()

    # Get unique spans
    cand_spans = set()
    for span in cand_spans_old:
        if type(span) is int:
            span = (span, span)
        cand_spans.add(span)
    cand_spans = list(sorted(list(cand_spans)))

    for span in cand_spans:
        cand_result.append({
            "candidate_span": span
        })

    return cand_result

def demo_const_parse(nlp):
    def find_sublist(sub, bigger):
        if not bigger:
            return -1
        if not sub:
            return 0
        first, rest = sub[0], sub[1:]
        pos = 0
        try:
            while True:
                pos = bigger.index(first, pos) + 1
                if not rest or bigger[pos:pos + len(rest)] == rest:
                    return pos - 1
        except ValueError:
            return -1

    # Test 1
    temp_text = "however , the size of the rho estimates indicate that a small amount of total error is being accounted for by level-2 variation in each model , and regression analysis of each threat model using ols produced extremely similar inferences to the multilevel model estimates ."
    temp_entity = "regression analysis"

    # # Test 2
    # temp_text = "in addition , the model found the predicted effect of phi on preference , b = 0.50 , se = 0.15 , z = 3.40 , p = .001 ."
    # temp_entity = "p = .001"

    temp_entity_span_start = find_sublist(temp_entity.split(), temp_text.split())
    temp_entity_span = [temp_entity_span_start+x for x in range(len(temp_entity.split()))]

    temp_candidates = getCandidiates_const_parse(nlp, temp_text, temp_entity, temp_entity_span)
    print("Input text = ", temp_text)
    print("Input entity mention = ", temp_entity)
    print("Candidates = ", temp_candidates)


def filterNERTriggerCandidates(candidates_list, entity_mention_span):
    """
    Remove the candidate spans that contain the target entity mention; These candidates cannot be used as trigger phrases
    for the target entity
    :param candidates_list:
    :param entity_mention_span:
    :return: Filtered candidates_list
    """

    entity_start_idx = entity_mention_span[0]
    entity_end_idx = entity_mention_span[-1]
    candidates_list_filtered = []
    for cur_cand in candidates_list:
        # Span contains the entity mention
        if entity_start_idx >= cur_cand["candidate_span"][0] and entity_end_idx <= cur_cand["candidate_span"][1]:
            continue
        # Span is a subset of the entity mention
        if cur_cand["candidate_span"][0] >= entity_start_idx and cur_cand["candidate_span"][1] <= entity_end_idx:
            continue
        candidates_list_filtered.append(cur_cand)

    return candidates_list_filtered

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_dir", type=str, default="../phrase_level_extraction/hiexpl_soc_ner/data/ner_dataset_tc/",
                        help="NER-TC Data directory (with train.jsonl and test.jsonl files)")
    parser.add_argument("--out_dir", type=str, default="../phrase_level_extraction/hiexpl_soc_ner/data/ner_dataset_tc/depparse_trees/",
                        help="NER Data directory (with train.txt and test.txt files)")
    parser.add_argument("--parse_method", type=str, default="dep_parse",
                        help="Type of parse used to generate candidates. Options: 1. const_parse and 2. dep_parse")
    parser.add_argument("--show_demo", action="store_true", help="Whether to show demo or not?")
    args = parser.parse_known_args()[0]

    base_dir = args.data_dir
    parse_method = args.parse_method
    if parse_method == "dep_parse":
        # Use Stanza
        nlp = stanza.Pipeline(lang="en", processors="tokenize,mwt,pos,lemma,depparse", tokenize_pretokenized=True)

        # Sample demo
        if args.show_demo:
            demo_dep_parse(nlp)

        # Parse all the samples and create the candidate trigger phrases
        for type_path in ["train", "test"]:
            inp_path = os.path.join(base_dir, type_path + ".jsonl")
            out_dir = args.out_dir
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, type_path + ".jsonl")

            result = []
            with open(inp_path, "r") as f:
                data_lines = f.readlines()

            for idx, cur_line in enumerate(tqdm(data_lines, desc="Type path-" + type_path)):
                cur_line = cur_line.strip()
                try:
                    cur_dict = json.loads(cur_line)
                    candidates = getCandidiates_dep_parse(nlp, cur_dict["sentence"], cur_dict["entity"], cur_dict["entity_span"])

                    # Update the dict with tokenized version of the input
                    cur_dict["candidates"] = filterNERTriggerCandidates(candidates, cur_dict["entity_span"])
                    result.append(cur_dict)
                except Exception as e:
                    print(e)
                    pass

            print("Number of valid instances with candidates = ", len(result))
            with jsonlines.open(out_path, "w") as f:
                f.write_all(result)

    elif parse_method == "const_parse":
        # Use StanfordCoreNLP
        nlp = StanfordCoreNLP("stanford-corenlp-4.1.0")

        # Sample demo
        if args.show_demo:
            demo_const_parse(nlp)

        # Parse all the samples and create the candidate trigger phrases
        for type_path in ["train", "test"]:
            inp_path = os.path.join(base_dir, type_path + ".jsonl")
            out_dir = args.out_dir
            if not os.path.exists(out_dir):
                os.makedirs(out_dir)
            out_path = os.path.join(out_dir, type_path + ".jsonl")

            result = []
            with open(inp_path, "r") as f:
                data_lines = f.readlines()

            for idx, cur_line in enumerate(tqdm(data_lines, desc="Type path-" + type_path)):
                cur_line = cur_line.strip()
                try:
                    cur_dict = json.loads(cur_line)
                    cur_text = cur_dict["sentence"]
                    candidates = getCandidiates_const_parse(nlp, cur_dict["sentence"], cur_dict["entity"], cur_dict["entity_span"])
                    cur_dict["candidates"] = filterNERTriggerCandidates(candidates, cur_dict["entity_span"])
                    result.append(cur_dict)
                except Exception as e:
                    print(e)
                    pass

            print("Number of valid instances with candidates = ", len(result))
            with jsonlines.open(out_path, "w") as f:
                f.write_all(result)

        # Close nlp client
        nlp.close()
