# clinton, enron, yahoo, yelp
# <<"COMMENT"
# for param in "1 8 5e-3 0.1" "2 48 1e-2 0.5" "3 32 5e-3 0.1" "4 32 1e-2 0.5" "5 32 5e-3 0.1" "6 16 1e-3 0.0" "7 32 1e-2 0.1" "8 16 5e-3 0.0" "9 8 1e-2 0.5" "10 8 1e-3 0.1"
# for param in "1 32 1e-2 0.3" "2 48 1e-2 0.0" "3 32 1e-2 0.2" "4 8 5e-3 0.2" "5 16 5e-3 0.2" "6 32 1e-2 0.0" "7 16 1e-2 0.0" "8 16 1e-2 0.0" "9 48 1e-2 0.0" "10 16 1e-2 0.1"
# for param in "1 16 1e-3 0.5" "2 32 1e-3 0.1" "3 8 5e-3 0.1" "4 32 1e-3 0.5" "5 48 1e-3 0.0" "6 48 1e-3 0.5" "7 32 5e-3 0.5" "8 16 1e-3 0.3" "9 8 5e-3 0.3" "10 32 1e-3 0.5"
# for param in "1 48 5e-3 0.5" "2 16 5e-3 0.1" "3 48 1e-3 0.1" "4 8 5e-3 0.2" "5 8 1e-3 0.2" "6 16 1e-2 0.3" "7 8 1e-3 0.5" "8 64 1e-3 0.3" "9 48 5e-3 0.0" "10 48 1e-2 0.5"
# for roberta for param in "1 64 1e-3 0.2" "2 32 1e-3 0.0" "3 128 1e-2 0.1" "4 8 1e-2 0.3" "5 8 1e-3 0.2" "6 48 1e-3 0.1" "7 64 1e-3 0.2" "8 32 1e-3 0.2" "9 32 1e-3 0.2" "10 32 5e-3 0.1"
# for param in "3 16 1e-2 0.1" "5 48 1e-3 0.3" "6 48 1e-3 0.0"
# for param in "3 128 1e-3 0.1" "5 8 1e-3 0.2" "6 48 1e-3 0.1" "7 64 1e-3 0.2"
for param in "3 16 1e-2 0.1" "5 48 1e-3 0.0" "6 48 1e-3 0.0" "7 32 1e-3 0.2"
do
    set -- $param
    echo "fold_id= $1, batch=$2, lr=$3, drop=$4"

    python3 fast/fast_flat_tran.py --do_train \
                                   --dataset="enron" \
                                   --label_list="1, 2, 3" \
                                   --fold_id=$1 \
                                   --model_type="transformer_sent_rel_fusion" \
                                   --model_name_or_path="llama-2-7b" \
                                   --train_batch_size=$2 \
                                   --learning_rate=$3 \
                                   --dropout=$4
done
# COMMENT
