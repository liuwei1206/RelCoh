<<"COMMENT"
python3 pdtb/pdtb_parser.py --dataset="enron" \
                            --fold_id=1 \
                            --label_level=1
COMMENT

<<"COMMENT"
for fold in 5
do
    python3 pdtb/pdtb_parser.py --dataset="toefl_p1" \
                                --fold_id=${fold}
done
COMMENT

<<"COMMENT"
for dataset in clinton yelp enron yahoo
do
    for fold in 1 2 3 4 5 6 7 8 9 10
    do
        python3 pdtb/pdtb_parser.py --dataset=${dataset} \
                                    --fold_id=${fold}
    done 
done
COMMENT

<<"COMMENT"
for dataset in enron
do
    for fold in 6 7 8 9 10
    do
        python3 pdtb/pdtb_parser.py --dataset=${dataset} \
                                    --fold_id=${fold} \
                                    --label_level=1
    done 
done
COMMENT

# <<"COMMENT"
for dataset in p1
do
    for fold in 1 2 3
    do
        python3 pdtb/pdtb_parser.py --dataset=toefl_${dataset} \
                                    --fold_id=${fold} \
                                    --label_level=1
    done 
done
# COMMENT

<<"COMMENT"
for dataset in p5 p6 p7 p8
do
    for fold in 1 2 3 4 5
    do
        python3 pdtb/pdtb_parser.py --dataset=toefl_${dataset} \
                                    --fold_id=${fold}
    done 
done
COMMENT

