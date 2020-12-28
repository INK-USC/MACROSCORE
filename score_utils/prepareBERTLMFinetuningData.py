import argparse
import json
import nltk


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument("--data_path", default="../data/paper_data/RPP_scienceparse_classify_data.json", help="Data path")
    parser.add_argument("--output_path", default="../data/bert_data/RPP_BERTData.txt",
                        help="Data path")
    parser.add_argument("--attribute_to_consider", default="content",
                        help="Attribute of the paper to consider for creating the BERT LM fine-tuning corpus")

    args = parser.parse_known_args()[0]
    print(args)

    with open(args.data_path, "r") as f:
        data = json.load(f)

    output_corpus = ""
    for cur_doc in data:
        cur_content_list = cur_doc["content"]
        cur_content_para = ""
        for cur_sec in cur_content_list:
            cur_content_para += cur_sec["text"]

        cur_sent_list = nltk.sent_tokenize(cur_content_para)
        for cur_sent in cur_sent_list:
            cur_sent = cur_sent.replace("\n", " ")
            output_corpus += cur_sent.strip() + "\n"

        output_corpus += "\n"

    with open(args.output_path, "w") as f:
        f.write(output_corpus)