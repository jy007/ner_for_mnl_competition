CURRENT_DIR=`pwd`
export BERT_BASE_DIR=$CURRENT_DIR/bert-base
export DATA_DIR=$CURRENT_DIR/data
export OUTPUR_DIR=$CURRENT_DIR/outputs
TASK_NAME="cner"
#
python run_ner_crf.py \
  --model_type=bert \
  --model_name_or_path=$BERT_BASE_DIR \
  --config_name=$BERT_BASE_DIR/bert_config.json \
  --tokenizer_name=$BERT_BASE_DIR/vocab.txt \
  --task_name=$TASK_NAME \
  --do_train \
  --do_eval \
  --do_lower_case \
  --data_dir=$DATA_DIR/${TASK_NAME}/ \
  --train_max_seq_length=128 \
  --eval_max_seq_length=512 \
  --per_gpu_train_batch_size=24 \
  --per_gpu_eval_batch_size=24 \
  --learning_rate=3e-5 \
  --crf_learning_rate=1e-3 \
  --num_train_epochs=4.0 \
  --logging_steps=448 \
  --save_steps=448 \
  --output_dir=$OUTPUR_DIR/${TASK_NAME}_output/ \
  --overwrite_output_dir \
  --seed=42
