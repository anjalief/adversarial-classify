from collections import defaultdict

import pickle
import sys
import argparse
import os
import numpy as np
import math

SKIP=set(["?", ".", ":", ","])

def getTopK(word2attention, k):
    retpair = sorted(word2attention.items(), key=lambda x:x[1], reverse=True)[:k]
    retwords = [x for (x,y) in retpair]
    return ", ".join(retwords)

def process_attns(wordattns):
    attns = []
    words = []

    for wordattn in wordattns:
        if len(wordattn.split(":")) > 2:
            continue

        [word, attn] = wordattn.split(":")
        attns.append(float(attn))
        words.append(word)

    # I think this happens if the post only has poorly formatted words with : in them (e.g. URLs)
    if len(attns) == 0:
        return [], []

    attns = np.array(attns)
    mean = np.mean(attns)
    std = np.std(attns)
    attns = (attns-mean)/std
    return words, attns

# for now skip the beginning, just take bigram
def process_bigrams(wordattns):
    attns = []
    words = []

    for i,wordattn1 in enumerate(wordattns):
        if i == len(wordattns) - 1:
            continue

        wordattn2 = wordattns[i+1]

        if len(wordattn1.split(":")) > 2 or len(wordattn2.split(":")) > 2:
            continue

        [word1, attn1] = wordattn1.split(":")
        [word2, attn2] = wordattn2.split(":")
        if word1 in SKIP or word2 in SKIP:
            continue

        attns.append(float(attn1) + float(attn2))
        words.append(word1 + "_" + word2)

    # I think this happens if the post only has poorly formatted words with : in them (e.g. URLs)
    if len(attns) == 0:
        return [], []

    attns = np.array(attns)
    mean = np.mean(attns)
    std = np.std(attns)
    attns = (attns-mean)/std
    return words, attns

# For each post, compute the normalized attn weight for
# each word in the post
# Then, sum over the weights for each word across all posts to get a final
# score for each word
def all_atn_words(args, aggregate=sum):
    print("Printing highest attended to words overall using", str(aggregate))

    word2attentions = defaultdict(list)
    f = open(args.attention_file)
    c, C = 0.0, 0.0
    for l in f:
        items = l.strip().split("\t")
        attentions, label, predicted = items[0], items[1], items[2]
        wordattns = attentions.split()
        C += 1
        if label == predicted:
            c+=1
        attns = []
        words = []

        words, attns = process_attns(wordattns)
        for word, attn in zip(words, attns):
            word2attentions[word].append(attn)

    word2attention = defaultdict(float)

    for w, attns in word2attentions.items():
        word2attention[w] = aggregate(attns)

    f.close()

    print (c/C)
    print (getTopK(word2attention, args.topk))
    print ("")

# Print attn words for each label instead of accross entire corpus
# Optionally only include posts that were labeled correctly
def attn_words_by_label(args, labels, aggregate=sum):
    label2word2attentions = {}
    for l in labels:
        label2word2attentions[l] = defaultdict(list)

    f = open(args.attention_file)

    c, C = 0.0, 0.0
    for l in f:
        items = l.strip().split("\t")
        attentions, label, predicted = items[0], items[1], items[2]
        wordattns = attentions.split()

        C += 1
        if label == predicted:
            c += 1
        else:
            if args.correct_only:
                continue

        if args.bigrams:
            words, attns = process_bigrams(wordattns)
        else:
            words, attns = process_attns(wordattns)
        for word, attn in zip(words, attns):
            label2word2attentions[predicted][word].append(attn)

    label2word2attention = {}
    for l in labels:
        label2word2attention[l] = defaultdict(float)

    for l, word2attentions in label2word2attentions.items():
        for w, attns in word2attentions.items():
            label2word2attention[l][w] = aggregate(attns)

    f.close()

    print (c/C)
    for l in labels:
        print("Attn words for", l)
        print (getTopK(label2word2attention[l], args.topk))

def main():
    parser = argparse.ArgumentParser(description='Remove topical words')
    parser.add_argument('--attention_file', type=str, required=True,
                        help='file containing the word:attention data')
    parser.add_argument('--labels', type=str, required=True,
                        help='file containing the label names')
    parser.add_argument('--topk', type=int, default=10,
                        help='file containing the label names')
    parser.add_argument('--aggregate', type=str, choices = ["sum", "mean"],
                        required=True,
                        help='how to aggregate attention scores accross posts')
    parser.add_argument('--correct_only', action='store_true')
    parser.add_argument('--bigrams', action='store_true')
    args = parser.parse_args()



    labels = []
    with open(args.labels) as f:
        for l in f:
            labels.append(l.strip())

    if args.aggregate == "sum":
        attn_words_by_label(args, labels, sum)
    elif args.aggregate == "mean":
        attn_words_by_label(args, labels, np.mean)

if __name__ == "__main__":
    main()



