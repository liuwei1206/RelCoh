# <<"COMMENT"
python3 gpt4.py --do_test \
                --dataset="test2" \
                --output_name="gpt4" \
                --fold_id=1
# COMMENT


<<"COMMENT"
for data in clinton enron yahoo yelp
do
    python3 gpt4.py --do_test \
                    --dataset=${data} \
                    --fold_id=1
done
COMMENT


<<"COMMENT"
for data in p1 p2 p3 p4 p5 p6 p7 p8
do
    for idx in 1
    do
        python3 gpt4.py --do_test \
                        --dataset="toefl_${data}" \
                        --fold_id=${idx}
    done
done
COMMENT
