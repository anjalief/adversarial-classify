#!/bin/bash
source activate py36

# THESE SET THE DATA SET
SUFFIX="reddit"
ODDS_COL="subreddit"
NUM_TOPICS=98

# SUFFIX="facebook_congress"
# ODDS_COL="op_name"

# These should be the same for all rt_gender data
EXTRA_SUFFIX="withtopics"
RAW_FILE="/projects/tir3/users/anjalief/corpora/rt_gender/${SUFFIX}_responses.csv"
TOKENIZED_FILE="/projects/tir3/users/anjalief/corpora/rt_gender/${SUFFIX}_responses.tok.tsv"
TOP_DIR="/projects/tir3/users/anjalief/adversarial_gender"
DATA_DIR="${TOP_DIR}/rt_gender_${SUFFIX}"

echo ${SUFFIX}
echo ${DATA_DIR}


# Keep a record of what we ran by printing it
run_cmd () {
  echo $cmd
  $cmd
}

##### Tokenize the raw data ######
cd preprocessing_scripts
# cmd=python preprocess_rtgender.py --input_file ${RAW_FILE} --output_file ${TOKENIZED_FILE}  --header_to_tokenize response_text
# run_cmd

##### Make train and test splits #####
# cmd="python make_data_splits.py --input_files ${TOKENIZED_FILE} --header_to_balance op_gender --header_to_split ${ODDS_COL} --output_dir ${DATA_DIR} --suffix ${SUFFIX}"
# run_cmd

##### Add log odds to training data and reformat train and test #####
# cmd="python add_log_odds_feats.py --input_file ${DATA_DIR}/train.${SUFFIX}.txt --output_file ${DATA_DIR}/train.${SUFFIX}.${EXTRA_SUFFIX}.txt --odds_column ${ODDS_COL} --add_log_odds"
# run_cmd

# cmd="python add_log_odds_feats.py --input_file ${DATA_DIR}/test.${SUFFIX}.txt --output_file ${DATA_DIR}/test.${SUFFIX}.txt"
# run_cmd

# cmd="python add_log_odds_feats.py --input_file ${DATA_DIR}/valid.${SUFFIX}.txt --output_file ${DATA_DIR}/valid.${SUFFIX}.txt"
# run_cmd
cd ".."


##### Train models #####
# cmd="python train_ganlike_multiple_decoders.py --data RT_GENDER --base_path ${DATA_DIR} --save_dir ${DATA_DIR} --model RNN --model_name rt_gender_${SUFFIX}_sk.model --gpu 0 --batch_size 64 --suffix .${SUFFIX} --extrasuffix .${EXTRA_SUFFIX} --num_topics ${NUM_TOPICS} --write_attention"
# run_cmd

# cmd="python train.py --data RT_GENDER --base_path ${DATA_DIR} --save_dir ${DATA_DIR} --model RNN --model_name rt_gender_${SUFFIX}_baseline.model --gpu 0 --batch_size 64 --suffix .${SUFFIX} --extrasuffix .${EXTRA_SUFFIX} --write_attention"
# run_cmd

# ##### Write attention scores #####
cd analysis_scripts
DATA_FOLD="test" #"train", "valid", "test"
python ./get_topk_from_attention.py --attention_file "${DATA_DIR}/${DATA_FOLD}.rt_gender_${SUFFIX}_sk.model_attention.txt" --labels "${TOP_DIR}/labels.txt" --topk 100 --correct_only --bigrams --aggregate sum
echo "DONE BIGRAMS SUM ${DATA_FOLD}"

python ./get_topk_from_attention.py --attention_file "${DATA_DIR}/${DATA_FOLD}.rt_gender_${SUFFIX}_baseline.model_attention.txt" --labels "${TOP_DIR}/labels.txt" --topk 100 --correct_only --bigrams --aggregate sum
echo "DONE BASELINE BIGRAMS SUM ${DATA_FOLD}"
echo "#######################################################################################################################"
echo "#######################################################################################################################"

python ./get_topk_from_attention.py --attention_file "${DATA_DIR}/${DATA_FOLD}.rt_gender_${SUFFIX}_sk.model_attention.txt" --labels "${TOP_DIR}/labels.txt" --topk 100 --correct_only --bigrams --aggregate mean
echo "DONE BIGRAMS MEAN ${DATA_FOLD}"

python ./get_topk_from_attention.py --attention_file "${DATA_DIR}/${DATA_FOLD}.rt_gender_${SUFFIX}_baseline.model_attention.txt" --labels "${TOP_DIR}/labels.txt" --topk 100 --correct_only --bigrams --aggregate mean
echo "DONE BASELINE BIGRAMS MEAN ${DATA_FOLD}"
echo "#######################################################################################################################"
echo "#######################################################################################################################"


python ./get_topk_from_attention.py --attention_file "${DATA_DIR}/${DATA_FOLD}.rt_gender_${SUFFIX}_sk.model_attention.txt" --labels "${TOP_DIR}/labels.txt" --topk 100 --correct_only  --aggregate sum
echo "DONE SUM ${DATA_FOLD}"

python ./get_topk_from_attention.py --attention_file "${DATA_DIR}/${DATA_FOLD}.rt_gender_${SUFFIX}_baseline.model_attention.txt" --labels "${TOP_DIR}/labels.txt" --topk 100 --correct_only  --aggregate sum
echo "DONE BASELINE SUM ${DATA_FOLD}"
echo "#######################################################################################################################"
echo "#######################################################################################################################"

python ./get_topk_from_attention.py --attention_file "${DATA_DIR}/${DATA_FOLD}.rt_gender_${SUFFIX}_sk.model_attention.txt" --labels "${TOP_DIR}/labels.txt" --topk 100 --correct_only  --aggregate mean
echo "DONE MEAN ${DATA_FOLD}"

python ./get_topk_from_attention.py --attention_file "${DATA_DIR}/${DATA_FOLD}.rt_gender_${SUFFIX}_baseline.model_attention.txt" --labels "${TOP_DIR}/labels.txt" --topk 100 --correct_only  --aggregate mean
echo "DONE BASELINE MEAN ${DATA_FOLD}"
echo "#######################################################################################################################"
echo "#######################################################################################################################"
