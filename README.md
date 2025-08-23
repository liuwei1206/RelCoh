# Discourse Relation-Enhanced Neural Coherence Modeling
Code for the ACL2025 paper "[Discourse Relation-Enhanced Neural Coherence Modeling](https://aclanthology.org/2025.acl-long.236.pdf)".

If any questions, please contact the email: willie1206@163.com

## 1 Requirement
Please refer to the requirements.txt for the environment required for this project.

Then you need to prepare the pdtb parser: download the files for the parser here and put it under the folder: "data/parser".


## 2 GCDC
To run experiments on GCDC, you should:
1. Put the processed data under the folder "data/dataset". You can refer to the preprocessing code in Strusim.
2. Call the script. For example, you can `sh script/gcdc/fusion.sh` to run experiments on gcdc corpus using the fusion model.

## 3 TOEFL
To run experiments on TOEFL, you should:
1. Put the processed data under the folder "data/dataset". You can refer to the preprocessing code in Strusim.
2. Call the script. For example, you can `sh script/toefl/fusion.sh` to run experiments on gcdc corpus using the fusion model.

## 4 Citation
```
@inproceedings{liu-strube-2025-discourse,
    title = "Discourse Relation-Enhanced Neural Coherence Modeling",
    author = "Liu, Wei  and
      Strube, Michael",
    editor = "Che, Wanxiang  and
      Nabende, Joyce  and
      Shutova, Ekaterina  and
      Pilehvar, Mohammad Taher",
    booktitle = "Proceedings of the 63rd Annual Meeting of the Association for Computational Linguistics (Volume 1: Long Papers)",
    month = jul,
    year = "2025",
    address = "Vienna, Austria",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.acl-long.236/",
    doi = "10.18653/v1/2025.acl-long.236",
    pages = "4748--4762",
    ISBN = "979-8-89176-251-0",
    abstract = "Discourse coherence theories posit relations between text spans as a key feature of coherent texts. However, existing work on coherence modeling has paid little attention to discourse relations. In this paper, we provide empirical evidence to demonstrate that relation features are correlated with text coherence. Then, we investigate a novel fusion model that uses position-aware attention and a visible matrix to combine text- and relation-based features for coherence assessment. Experimental results on two benchmarks show that our approaches can significantly improve baselines, demonstrating the importance of relation features for coherence modeling."
}
```