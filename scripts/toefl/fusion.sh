
# <<"COMMENT"
python3 train.py --do_train \
                 --dataset="toefl_p1" \
                 --label_list="low, medium, high" \
                 --fold_id=1 \
                 --model_type="transformer_sent_rel_fusion" \
                 --model_name_or_path="roberta-base" \
                 --train_batch_size=24 \
                 --learning_rate=0.001 \
                 --dropout=0.1

# COMMENT
