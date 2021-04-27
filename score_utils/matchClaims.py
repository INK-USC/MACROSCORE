import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import json
import os
import rltk, re
from collections import Counter
from tqdm import tqdm
import nltk
import statistics
from unidecode import unidecode

def normalizeText(input_text, remove_spaces=True):
    # Lowercase and retain only alpha-numeric characters and underscore (_)
    result = " ".join(re.sub("\W", " ", input_text.lower()).split())

    # Replace known greek symbols to corresponding english alphabets: (Eg: "ﬁ" to "fi" etc)
    result = unidecode(result)
    result = result.lower()

    # Remove remaining unicode characters (greek symbols)
    result_cleaned = ""
    for char in result:
        if char == " " or char.isnumeric() or (ord(char) >= ord('a') and ord(char) <= ord('z')):
            result_cleaned += char

    # Remove spaces
    if remove_spaces:
        result_cleaned = result_cleaned.replace(" ", "")

    return result_cleaned


def getClaimSentList(inp_claim):
    claim_sent_list = []
    claim_sent_list_normalized = []

    for cur_sent in nltk.sent_tokenize(inp_claim):
        cur_sent = " ".join(nltk.word_tokenize(cur_sent))
        cur_sent_normalized = normalizeText(cur_sent)

        if len(cur_sent_normalized) <= 3:
            continue

        claim_sent_list.append(cur_sent)
        claim_sent_list_normalized.append(cur_sent_normalized)

    return claim_sent_list, claim_sent_list_normalized


def matchClaim(cur_claim_list_normalized, cur_sent_normalized, sim_threshold=0.8):
    is_matched = False
    match_claim_idx = None
    match_sent_start = None
    match_sent_end = None
    for idx, cur_claim in enumerate(cur_claim_list_normalized):
        cur_claim_len = len(cur_claim)

        sent_idx_list = [0]
        cur_sent_normalized_without_spaces = ""
        for i in range(len(cur_sent_normalized)):
            cur_char = cur_sent_normalized[i]
            if cur_char == " ":
                sent_idx_list.append(len(cur_sent_normalized_without_spaces))
                continue

            cur_sent_normalized_without_spaces += cur_char

        for i in sent_idx_list:
            start = i
            end = min(i + cur_claim_len, len(cur_sent_normalized_without_spaces))

            sim_val = rltk.levenshtein_similarity(cur_sent_normalized_without_spaces[start:end], cur_claim)
            if sim_val >= sim_threshold:
                is_matched = True
                match_claim_idx = idx
                match_sent_start = start
                match_sent_end = end
                print("Sim Val = ", sim_val)
                break

    return is_matched, match_claim_idx, match_sent_start, match_sent_end


def computeDistancesToClaims(inp_data, doi_field_name="DOI_CR"):
    result = {}

    # Claim match stats
    cur_cnt = -1
    df_claim_match_data = []
    df_claim_match_cols = ["claim2", "claim3a", "claim3b", "claim4"]
    start_idx = 0
    end_idx = 10

    temp = []
    temp1 = []
    for idx in tqdm(range(len(inp_data))):
        data = inp_data[idx]
        cur_cnt += 1

        if (start_idx != -1 and cur_cnt < start_idx) or (end_idx != -1 and cur_cnt > end_idx):
            continue

        ### Get the paper identifier
        if doi_field_name is not None:
            try:
                paper_id = data[doi_field_name]
            except Exception as e:
                raise Exception("No unique identifier found for paper.")
        else:
            if data.get("DOI_CR") is not None:
                paper_id = data['DOI_CR']
            elif data.get("doi") is not None:
                paper_id = data['doi']
            elif data.get("paper_id") is not None:
                paper_id = data['paper_id']
            else:
                raise Exception("No unique identifier found for paper.")

        ### Get the claims
        if data.get("claim2") is not None and data.get("claim3a") is not None and data.get(
                "claim3b") is not None and data.get("claim4") is not None:
            claim2 = data['claim2']
            claim3a = data['claim3a']
            claim3b = data['claim3b']
            claim4 = data['claim4']
        elif data.get("coded_claim2") is not None and data.get("coded_claim3a") is not None and data.get(
                "coded_claim3b") is not None and data.get("coded_claim4") is not None:
            claim2 = data['coded_claim2']
            claim3a = data['coded_claim3a']
            claim3b = data['coded_claim3b']
            claim4 = data['coded_claim4']
        else:
            print("Paper with id {} doesn't have claims.".format(data["paper_id"]))
            claim2 = None
            claim3a = None
            claim3b = None
            claim4 = None

        ### Get the content
        if data.get("content") is not None:
            content = data['content']
        else:
            print("Paper with id {} doesn't have content attribute. Skipping it.".format(data["paper_id"]))
            continue

        ### Split the content into a list of sentences:
        content_sentences_list = []
        for cur_section in content:
            if cur_section.get("text") is not None:
                cur_text = cur_section["text"]
                cur_sent_list = nltk.sent_tokenize(cur_text)
                cur_sent_list_filtered = []
                for cur_sent in cur_sent_list:
                    cur_sentence_tokens = cur_sent.split()

                    # Filter out sentences with only urls/citations etc.
                    if len(cur_sentence_tokens) >= 7:
                        cur_sent_tokenized = " ".join(nltk.word_tokenize(cur_sent))
                        cur_sent_list_filtered.append(cur_sent_tokenized)

                content_sentences_list += cur_sent_list_filtered

        cur_claim_stats = []
        ### Get the claim indexes in the corpus
        if claim2 is not None:
            claim2_sent_list, claim2_sent_list_normalized = getClaimSentList(claim2)
            cur_claim_stats.append(1)
        else:
            claim2_sent_list_normalized = None
            cur_claim_stats.append(0)

        if claim3a is not None:
            claim3a_sent_list, claim3a_sent_list_normalized = getClaimSentList(claim3a)
            cur_claim_stats.append(1)
        else:
            claim3a_sent_list_normalized = None
            cur_claim_stats.append(0)

        if claim3b is not None:
            claim3b_sent_list, claim3b_sent_list_normalized = getClaimSentList(claim3b)
            cur_claim_stats.append(1)
        else:
            claim3b_sent_list_normalized = None
            cur_claim_stats.append(0)

        if claim4 is not None:
            claim4_sent_list, claim4_sent_list_normalized = getClaimSentList(claim4)
            cur_claim_stats.append(1)
        else:
            claim4_sent_list_normalized = None
            cur_claim_stats.append(0)

        claim2_sents_idx = []
        claim3a_sents_idx = []
        claim3b_sents_idx = []
        claim4_sents_idx = []

        # First try matching the claim using exact string matching:
        for idx in range(len(content_sentences_list)):
            cur_sent = content_sentences_list[idx]
            cur_sent_normalized = normalizeText(cur_sent)

            # Get the claim2 index:
            if claim2 is not None:
                if any([claim_sent in cur_sent_normalized for claim_sent in claim2_sent_list_normalized]):
                    claim2_sents_idx.append(idx)

            # Get the claim3a index:
            if claim3a is not None:
                if any([claim_sent in cur_sent_normalized for claim_sent in claim3a_sent_list_normalized]):
                    claim3a_sents_idx.append(idx)

            # Get the claim3b index:
            if claim3b is not None:
                if any([claim_sent in cur_sent_normalized for claim_sent in claim3b_sent_list_normalized]):
                    claim3b_sents_idx.append(idx)

            # Get the claim4 index:
            if claim4 is not None:
                if any([claim_sent in cur_sent_normalized for claim_sent in claim4_sent_list_normalized]):
                    claim4_sents_idx.append(idx)

        temp.append(len(claim4_sent_list_normalized))
        temp1.append(len(claim4_sents_idx))

        #         if len(claim4_sents_idx) == 0:
        #             print("Not Matched!!!")

        claim_sent_sim_threshold = 0.8
        # Second try matching the claim using moving-window based levenstein similarity:
        for idx in range(len(content_sentences_list)):
            cur_sent = content_sentences_list[idx]
            cur_sent_normalized = normalizeText(cur_sent, remove_spaces=False)

            # Get the claim2 index:
            if len(claim2_sents_idx) == 0 and claim2 is not None:
                is_matched, matched_claim_idx, matched_start, matched_end = matchClaim(claim2_sent_list_normalized,
                                                                                       cur_sent_normalized,
                                                                                       claim_sent_sim_threshold)
                if is_matched:
                    print("Claim2: \nA = ", claim2_sent_list[matched_claim_idx])
                    print("B = ", cur_sent)

                    cur_sent_normalized_without_spaces = cur_sent_normalized.replace(" ", "")
                    print("Anorm = ", claim2_sent_list_normalized[matched_claim_idx])
                    print("Bnorm = ", cur_sent_normalized_without_spaces[matched_start:matched_end], "\n\n")
                    claim2_sents_idx.append(idx)

            # Get the claim3a index:
            if len(claim3a_sents_idx) == 0 and claim3a is not None:
                is_matched, matched_claim_idx, matched_start, matched_end = matchClaim(claim3a_sent_list_normalized,
                                                                                       cur_sent_normalized,
                                                                                       claim_sent_sim_threshold)
                if is_matched:
                    print("Claim3a: \nA = ", claim3a_sent_list[matched_claim_idx])
                    print("B = ", cur_sent)

                    cur_sent_normalized_without_spaces = cur_sent_normalized.replace(" ", "")
                    print("Anorm = ", claim3a_sent_list_normalized[matched_claim_idx])
                    print("Bnorm = ", cur_sent_normalized_without_spaces[matched_start:matched_end], "\n\n")
                    claim3a_sents_idx.append(idx)

            # Get the claim3b index:
            if len(claim3b_sents_idx) == 0 and claim3b is not None:
                is_matched, matched_claim_idx, matched_start, matched_end = matchClaim(claim3b_sent_list_normalized,
                                                                                       cur_sent_normalized,
                                                                                       claim_sent_sim_threshold)
                if is_matched:
                    print("Claim3b: \nA = ", claim3b_sent_list[matched_claim_idx])
                    print("B = ", cur_sent)

                    cur_sent_normalized_without_spaces = cur_sent_normalized.replace(" ", "")
                    print("Anorm = ", claim3b_sent_list_normalized[matched_claim_idx])
                    print("Bnorm = ", cur_sent_normalized_without_spaces[matched_start:matched_end], "\n\n")
                    claim3b_sents_idx.append(idx)

            # Get the claim4 index:
            if len(claim4_sents_idx) == 0 and claim4 is not None:
                is_matched, matched_claim_idx, matched_start, matched_end = matchClaim(claim4_sent_list_normalized,
                                                                                       cur_sent_normalized,
                                                                                       claim_sent_sim_threshold)
                if is_matched:
                    print("Claim4: \nA = ", claim4_sent_list[matched_claim_idx])
                    print("B = ", cur_sent)

                    cur_sent_normalized_without_spaces = cur_sent_normalized.replace(" ", "")
                    print("Anorm = ", claim4_sent_list_normalized[matched_claim_idx])
                    print("Bnorm = ", cur_sent_normalized_without_spaces[matched_start:matched_end], "\n\n")
                    claim4_sents_idx.append(idx)

        cur_dict = dict()
        cur_dict["claim2_sents_idx"] = claim2_sents_idx
        cur_dict["claim3a_sents_idx"] = claim3a_sents_idx
        cur_dict["claim3b_sents_idx"] = claim3b_sents_idx
        cur_dict["claim4_sents_idx"] = claim4_sents_idx
        result[paper_id] = cur_dict

        # Claim match stats
        if len(claim2_sents_idx) > 0:
            cur_claim_stats[0] = 1
        else:
            cur_claim_stats[0] = 0

        if len(claim3a_sents_idx) > 0:
            cur_claim_stats[1] = 1
        else:
            cur_claim_stats[1] = 0

        if len(claim3b_sents_idx) > 0:
            cur_claim_stats[2] = 1
        else:
            cur_claim_stats[2] = 0

        if len(claim4_sents_idx) > 0:
            cur_claim_stats[3] = 1
        else:
            cur_claim_stats[3] = 0

        df_claim_match_data.append(cur_claim_stats)

    # Print claim match stats
    df_claim_match = pd.DataFrame(data=df_claim_match_data, columns=df_claim_match_cols)

    total_num = df_claim_match.shape[0]
    claim2_cnts = dict(df_claim_match["claim2"].value_counts())
    claim3a_cnts = dict(df_claim_match["claim3a"].value_counts())
    claim3b_cnts = dict(df_claim_match["claim3b"].value_counts())
    claim4_cnts = dict(df_claim_match["claim4"].value_counts())

    print("Total Count = ", total_num)
    print("Claim2 match: Count = ", claim2_cnts[1], ", Percent = ", round(claim2_cnts[1] * 100 / total_num, 2), "%")
    print("Claim3a match: Count = ", claim3a_cnts[1], ", Percent = ", round(claim3a_cnts[1] * 100 / total_num, 2), "%")
    print("Claim3b match: Count = ", claim3b_cnts[1], ", Percent = ", round(claim3b_cnts[1] * 100 / total_num, 2), "%")
    print("Claim4 match: Count = ", claim4_cnts[1], ", Percent = ", round(claim4_cnts[1] * 100 / total_num, 2), "%")

    return result, df_claim_match, temp, temp1

if __name__ == "__main__":
    full_data_path = "notebooks/data/ta2_classify_folds/fold_full/data.json"

    with open(full_data_path, "r") as f:
        full_data = json.load(f)

    print(len(full_data))