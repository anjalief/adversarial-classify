# Given a tokenized (training) data set, extact log_odds features
# And add them as a new column. Then reformat data, dropping uneeded
# columns. If log_odds is set to false, just do the reformatting
import argparse
from collections import Counter, defaultdict
from basic_log_odds import write_log_odds
import pandas
import math
import numpy as np

def sigmoid(x):
  return 1 / (1 + math.exp(-x))

def write_csv(df, args):
    if args.odds_column:
        df.to_csv(args.output_file, index=False, header=False, sep="\t",
                  columns=["post_id", "response_text", "op_gender", "log_odds"])
    else:
        df.to_csv(args.output_file, index=False, header=False, sep="\t",
                  columns=["post_id", "response_text", "op_gender"])


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_file")
    parser.add_argument("--add_log_odds", action='store_true')
    parser.add_argument("--odds_column")
    parser.add_argument("--output_file")
    args = parser.parse_args()


    df = pandas.read_csv(args.input_file, sep="\t")

    if not args.odds_column:
        write_csv(df, args)
        return

    header_to_count = Counter()

    header_to_counter = defaultdict(Counter)
    for i, row in df.iterrows():
        words = row["response_text"].split()
        header = row[args.odds_column]
        header_to_counter[header].update(words)
        header_to_count[header] += 1

    # This scores w:[p(w | y1), p(w | y2),...]
    word_to_scores = defaultdict(list)
    header_to_py = []
    for h1,c1 in header_to_counter.items():
        alt_counter = Counter()
        prior = Counter()
        for h2,c2 in header_to_counter.items():
            prior.update(c2)
            if h2 != h1:
                alt_counter.update(c2)
        word_to_score = write_log_odds(c1, alt_counter, prior)
        for w,s in word_to_score.items():
            word_to_scores[w].append(sigmoid(s))
        header_to_py.append(header_to_count[h1] / len(df))

    word_to_scores = {w:np.array(s) for w,s in word_to_scores.items()}

    log_odds = []
    for i,row in df.iterrows():
        words = row["response_text"].split()
        scores = [word_to_scores[w] for w in words]
        pw = np.prod(scores, axis=0) # multiply over words
        final_scores = np.multiply(pw, header_to_py) # multiple by prior (elementwise)

        log_odds.append(" ".join([str(s) for s in final_scores]))
    df["log_odds"] = log_odds

    write_csv(df, args)

if __name__ == "__main__":
    main()
