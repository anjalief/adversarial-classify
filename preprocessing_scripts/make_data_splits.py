# Filter out short posts and stratify split the remaining
# post to have balanced splits

# python make_data_splits.py "/projects/tir3/users/anjalief/adversarial_gender/rt_gender_fb"

import pandas
import argparse
import glob
from sklearn.model_selection import train_test_split
import os

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--input_files")
    parser.add_argument("--output_dir")
    parser.add_argument("--header_to_split")
    parser.add_argument("--header_to_balance")
    parser.add_argument("--suffix")
    args = parser.parse_args()

    dfs = []
    for filename in glob.iglob(args.input_files):
        df = pandas.read_csv(filename, sep="\t")
        dfs.append(df)
    all_data = pandas.concat(dfs)

    forums = set(all_data[args.header_to_split])

    def short_filter(text):
        return len(str(text).split()) > 4

    print("Before short removal", len(all_data))

    all_data = all_data[all_data["response_text"] .apply(short_filter)]

    print("After short removal", len(all_data))


    trains = []
    tests = []
    valids = []

    for f in forums:
        f_data = all_data[all_data[args.header_to_split] == f]

        try:
            train, test_valid = train_test_split(f_data, test_size=0.2, stratify=f_data[args.header_to_balance])
            test, valid = train_test_split(test_valid, test_size=0.5, stratify=test_valid[args.header_to_balance])
        except ValueError:
            print("Value error on"), f
            train, test_valid = train_test_split(f_data, test_size=0.2)
            test, valid = train_test_split(test_valid, test_size=0.5)

        trains.append(train)
        tests.append(test)
        valids.append(valid)


    def save_files(name, dfs):
        filename = os.path.join(args.output_dir, name + args.suffix + ".txt")
        df = pandas.concat(dfs)
        print("Saving", filename)
        df.to_csv(filename, sep="\t")

    save_files("train.", trains)
    save_files("test.", tests)
    save_files("valid.", valids)


if __name__ == "__main__":
    main()
