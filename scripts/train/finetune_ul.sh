INPUT_DIR=../../data/tailor/
OUTPUT_DIR=../../models
MODEL_TYPE=t5-base
identifier=unlikelihood

python run_seq2seq_ul.py \
    --use_unlikelihood \
    --model_name_or_path ${MODEL_TYPE} \
    --do_train \
    --task summarization \
    --train_file ${INPUT_DIR}/${identifier}/train.csv \
    --output_dir ${OUTPUT_DIR}/${identifier} \
    --validation_file ${INPUT_DIR}/${identifier}/dev.csv \
    --overwrite_output_dir \
    --per_device_train_batch_size=4 \
    --per_device_eval_batch_size=4 \
    --save_steps 10000 \
    --metric_for_best_model eval_loss \
    --num_train_epochs 10 \
    --predict_with_generate \
    --text_column prompt \
    --summary_column answer \
    --reward_column reward \
    --weight_column weight \
    --dataloader_drop_last
#    --load_best_model_at_end \
#    --use_resampling \
