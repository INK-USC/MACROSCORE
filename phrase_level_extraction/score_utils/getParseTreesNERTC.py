from nltk.tree import Tree
import pandas as pd
from tqdm import tqdm
import os
import jsonlines, json
import stanza
import numpy as np

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


def getCandidiates(parsed_doc, entity_span, k_hop=2):
    # global tokens_list
    cand_result = []
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

        cur_span = []
        for idx in cur_idx_list:
            # Skip the root node in dep parse (or) skip the phrase containing the entity span
            if idx == -1 or idx in entity_span:
                if len(cur_span) != 0:
                    cand_result.append({
                        "candidate_span": cur_span,
                        "candidate_phrase": " ".join(tokens_list[cur_span]),
                    })
                cur_span = []
                continue

            if len(cur_span) == 0:
                cur_span.append(idx)
            elif idx == cur_span[-1] + 1:
                cur_span.append(idx)
            else:
                cand_result.append({
                    "candidate_span": cur_span,
                    "candidate_phrase": " ".join(tokens_list[cur_span]),
                })
                cur_span = [idx]

        if len(cur_span) != 0:
            cand_result.append({
                "candidate_span": cur_span,
                "candidate_phrase": " ".join(tokens_list[cur_span]),
            })

    return cand_result

def demo(nlp):
    # Test
    temp_text = "however , the size of the rho estimates indicate that a small amount of total error is being accounted for by level-2 variation in each model , and regression analysis of each threat model using ols produced extremely similar inferences to the multilevel model estimates ."
    temp_entity = "regression analysis"

    temp_entity_start = temp_entity.split()[0]
    temp_entity_start_idx = -1
    for idx, cur_token in enumerate(temp_text.split()):
        if cur_token == temp_entity_start:
            temp_entity_start_idx = idx
            break

    temp_entity_span = [temp_entity_start_idx+x for x in range(len(temp_entity.split()))]

    tokens_list = np.array(temp_text.split())
    # print(tokens_list[temp_entity_span])

    parsed_doc = nlp(temp_text)
    candidates = getCandidiates(parsed_doc, temp_entity_span)
    # print(cur_text)
    # print(candidates)

if __name__ == "__main__":
    nlp = stanza.Pipeline(lang="en", processors="tokenize,mwt,pos,lemma,depparse", tokenize_pretokenized=True)
    base_dir = "../hiexpl_soc_ner/data/ner_dataset_tc"

    # demo(nlp)

    for type_path in ["train"]:
        inp_path = os.path.join(base_dir, type_path + ".jsonl")
        out_dir = os.path.join(base_dir, "trees_depparse")
        if not os.path.exists(out_dir):
            os.makedirs(out_dir)
        out_path = os.path.join(out_dir, type_path + ".jsonl")

        result = []
        cnt = 0
        with open(inp_path, "r") as f:
            for cur_line in f:
                try:
                    cur_dict = json.loads(cur_line)
                    cur_text = cur_dict["sentence"]
                    parsed_doc = nlp(cur_text)
                    candidates = getCandidiates(parsed_doc, cur_dict["entity_span"])
                    cur_dict["depparse_candidates"] = candidates
                    result.append(cur_dict)
                except Exception as e:
                    print(e)
                    pass

                cnt += 1
                if cnt % 10 == 0:
                    print("Progress = ", cnt)


        print(len(result))
        with jsonlines.open(out_path, "w") as f:
            f.write_all(result)
