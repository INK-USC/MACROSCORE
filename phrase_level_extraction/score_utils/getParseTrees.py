from nltk.tree import Tree
from stanfordcorenlp import StanfordCoreNLP
import pandas as pd
from tqdm import tqdm

if __name__ == "__main__":
    nlp = StanfordCoreNLP("stanford-corenlp-4.1.0")

    for type_path in ["train", "dev", "full_data"]:
        inp_path = "repr_sentences/{}.csv".format(type_path)
        out_path = "repr_sentences/trees/{}.csv".format(type_path)

        df = pd.read_csv(inp_path)
        tree_string_list = []
        for idx, cur_row in tqdm(df.iterrows()):
            cur_text = cur_row["important_segment"].replace("...", "")
            try:
                cur_tree = nlp.parse(cur_text)
                cur_tree = cur_tree.replace("\n", "")
            except:
                cur_tree = None
            tree_string_list.append(cur_tree)

        df["important_segment_parsed_tree"] = tree_string_list
        df = df[["paper_id", "label", "important_segment", "important_segment_parsed_tree"]]
        df = df[df["important_segment_parsed_tree"].notnull()]

        print(out_path, df.shape)
        df.to_csv(out_path, index=False)

