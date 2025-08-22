# author = liuwei
# date = 2024-05-22

import logging
import os
import json
import random
import time
from tqdm import tqdm, trange
import argparse
import numpy as np
from sklearn.metrics import f1_score, accuracy_score, classification_report

import openai

random.seed(106524)

"""Refer to: https://cookbook.openai.com/examples/how_to_format_inputs_to_chatgpt_models"""

def save_res_into_file(file_name, question, response):
    with open(file_name, "w", encoding="utf-8") as f:
        f.write("%s ##:## %s"%(question, response))


def call_chatgpt(prompts, file_prefix, model_type="gpt-4o"):
    """
    Args:
        prompts: all the questions that you want to ask
                if is str, then questions should be splited by ##
                if list, then fine
        model_type: refer to https://platform.openai.com/docs/models/gpt-3-5
    """
    openai.api_key = "xxx"

    assert type(prompts) == list, (type(prompts))
    all_responses = []
    for item in prompts:
        instance_id = item[0]
        question = item[1]
        response = openai.ChatCompletion.create(
            model=model_type,
            messages=[
                {
                    "role": "system",
                    "content": "You are an excellent assistant in assessing the coherence level of a text."
                },
                {
                    "role": "user",
                    "content": question
                }
            ],
            temperature=0, # very important to keep the return answer be stable
        )
        all_responses.append((question, response['choices'][0]['message']['content']))

        file_name = file_prefix.format(str(instance_id))
        save_res_into_file(file_name, question, response['choices'][0]['message']['content'])


def read_data_from_file(file_name):
    all_samples = []
    with open(file_name, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)

                text = sample["text"]
                score = sample["score"]
                all_samples.append((text.strip(), score))

    return all_samples


def get_argparse():
    parser = argparse.ArgumentParser()

    # path
    parser.add_argument("--data_dir", default="data/dataset", type=str)
    parser.add_argument("--output_name", default="gpt4", type=str)
    parser.add_argument("--dataset", default="clinton", type=str, help="clinton, enron...")
    parser.add_argument("--fold_id", default=1, type=int, help="1, 2, 3, 4")

    parser.add_argument("--do_train", default=False, action="store_true")
    parser.add_argument("--do_dev", default=False, action="store_true")
    parser.add_argument("--do_test", default=False, action="store_true")
    parser.add_argument("--do_cal", default=False, action="store_true", help="cal results on test set")

    return parser


def apply_prompt_template3(text):
    high_text = """Actually, it is not my belief that it is better for an educated person to have a broad knowledge of many academic subjects than to specialize in one specific subject. In today's changing society, it is far from enough to be just an \"expert\".\n\tWhen we watch television or read newspaper, we think that to have an expert talking about a subject is reasonable or even that only the expert has the authority to talk about the subject in public. Based on this psychological effect, we tend to support the idea that the society should produce more \"experts\", so that people will get more reliable information from the media and their lives. However, when it comes to education, it will be a bad idea for the students nowadays. For example, I've always wanted to be an simultaneous interpreter. But being an interpreter requires a lot more than the knowledge about translating.  He has to know \"something about everything, and everything about something\". If he has to go to a conference which is dicussing the oil crisis. He has to know the basic information about oil, all kinds of oil in fact. Therefore, a successful professional shouldn't be limited in the arena of his specialized subject. Otherwise, his success wouldn't last very long. For example, if you are an employee working for an insurance company. The basic rule for you to success is that you know how to sell. That is a skill that involves with a lot of other subjects. You'd better know how the hospital works, so that you can sell your insurance to its patiences. Then you need to know what kind of patients can make protential customers and what kind of patients can not (like those who have developed uncurable disease). So, after years of studing in college, if you know nothing other than what insurance is, you are definitely doomed.\n\tSo, in my opinion, being an \"expert\" is not a bad thing, but what we really need in today's challenging world are people who have their specialization and also the broad knowledge about the nature and the society."""
    medium_text = """Some people think that,it would be better to have broad knowledge of different academic subjects because variations of subjects adds to human experience and provides possible solutions in many situations. Others think it would be better to concentrate on one topic and go on. Based on my experience, both opinion are helpful according to my grade.\n When i was in under graduate school, I was fond of reading and  attending different classes which helped me to great extend to pass my exams easily and receive high score.\n On the other hand, when I was in physical therapy school I concentrated on only sports medicine classes which helped me to apply for master degree and find convineant job.\n Finally both thoughts are important depend on needs and goals."""
    low_text = """I agree that it is better to have broad knowledge of many academic subjects than to specialize in one specific subject. The academic subjects are very important to every body and we need to understand many academic subjects and try to apply it in our life to have good knowledge. There are many academic subjects like nutrition and food sciences, chemistry, medicine, biology, pharmacology, histology and management which must be read about it and study it. \n     The cademic subjects will help us to understand our life and promote us to develop our life styles to the best way. We have good chance in this time to reach many academic subjects by internet or library or any academic centres in the world ( universities or colleges ). If we take different academic subjects in this time we can focus on specific subject to be an expert in that subject and to take more experince. Now, the world have fast changes which need different sciences to deal with it. We need many people to take specific academic subjects to solve our problems in our life. We need specific academic centres in many sciences. \n Finally, Each person must be take many academic subjects and try to specified in specific subject."""
    high_text_2 = """Specialization versus generalized knowledge\n\n\nThe question of specializing in one particular topic or becoming a all-rounder with a broad knowledge confronts every student in his beginning of academic career. \n\nIt is difficult to give a clear-cut answer and again the buzzword \"everything depends on the context\" seems to apply here.  There are certain professions which are dealt better with specilization in a very particular topic rather than having general knowledge. Someone who would like to become a dentist does not need to distract himself in other \"superfluous\" topics.  \n\nOn the other hand, the world is becoming more complext and many cross-cutting issues can be only covered by a broad range of knowledge. Just to give an example: the question of legal property of genetically modified genes is something that can not be answered soley from a biological point of view. It requires a more holistic approach by including ethics, law, politics or environmental science. \n\nPersonnally I could benefit from a large scope of academic topics ranging from economics, history to sociology and law. All these courses helped me become a \"Swiss armyknife\" enabling me to understand and deal with various issues in professional life. \n\nThis question of specialization versus generalized knowledge is a personal choice and my personnal positive experience with the latter does not necessarily apply to others. Let us also bear in mind that this question does not stop at the academic level but also continue in professional life. If one has specialized himself in one academic topic it is never too late to broaden himself up in the professional world and vice versa."""
    medium_text_2 = """Knowledge\n \"Its better to have broad knowledge of many academic subjects than to specialize in one specific subject.\" I agree on this statment because knoeledge saves alot of people from a lot of dangerous things.\n\" A little knowledge is dangerous.\" I have faith in this saying lets take the medicines such as pain killers as an example. People with a little knowledge mite think that taking more pain killers will help killing the pain faster, in the other hand well educated people will rialize that taking over dose of pills will have a lot of side affects on the person who takes these pills, thats why a person must have a broad knowledge of many academic subjects. This doesn't mean that the person must be well educated in every thing because the saying says: \" A man of all trades is master of none.\" It's hard to know each detail of every thing in life because this will rouin your focus at the important thing in each subject in life.\nMy advice to every one is to try to educate your self in every thing and this doesn't mean to go through  details in every thing. But each person must have a specipic subject to know every thing about, and try to gain informations from different people you meet in your life  with different education because: \" Christmas comes but once a year.\""""
    low_text_2 = """I think that the following statement \"It is better to have broad knowledge of many academic subjects than to specialize in one specific subject\" because this is a very important functions. \nIf you have broad knowledge of many academic subject you have the possibility to have a panoramic scenary to all over situations you have in your \"hands\".\nWith this conoscence you will be prepared to every situations. You will have a answer for every moment of your professional live.\nI think that many academic subject permetted at all person to be able to take all over profession.\nIn stead  of one specific academic subject do not permetted to be able to take any professional job.\nIn end I think that this selections is prefered for me, but this is not equals for every other person."""
    high_text_3 = """I disagree that it is better to have a broad knowledge of many academic subjects rather than to specialize in one.All of us have heard of the quote \"Jack of all trades,Master of none\".I think I would like to be the master of atleast one subject.\nI believe that unless one has detailed or specialized information regarding a subject,it is difficult to speak challengingly on that subject.To quote myself as an example,I have a broad knowledge of seven languages.I speak English at college,Telugu at home which an Indian regional language and have learnt a smattering of French,German,Hindi(India's national language),Spanish and Italian.But,I am handicapped when I have to converse in atleast  five of the seven languages I learnt which doesn't serve the purpose of learning.\nI also believe that specialization is required in most fields.Some of those fields being engineering,medicine and law.When you have a toothache,you would visit a dentist rather than a general practicioner.A similar logic would drive you to a cardiologist when you have a heart attack and to an opthamologist when one has a problem with one's eyes.Hence,it can be seen that in the above cases,a person with broad knowledge of the ailments would serve no purpose as one requires specialists.\nAnother area where I want to illustrate the use of specialists is engineering.Civil engineers are required for building bridges and computer engineers are required to work on computers.A non-specialized engineer would simply do a bad job or a blotched job.We could also consider the field of law as an example.We approach criminal lawyers for  fighting our cases in case of crimes such as murder,extortion or kidnapping but we would approach a consumer court lawyer for problems such as adulterated food.Thus,we infact need specialists all the time and in all walks of life !\nAnother advantage of being a specialist is that people look up to your opinions much more than if you were un-specialized.We would much rather believe a person who has obtained his doctor of philosophy degree in a particular subject rather than an undergraduate student.Hence,i believe a broad knowlegde of several subjects doesn't really serve many purposes.Infact,it might even be dangerous.Haven't we all heard of the saying \"Half-knowledge is dangerous\"??An intern might suggest a wrong course of treatment to a patient which could have a fatal affect.Such affects are not limited to medicine alone.Thus,i believe in specialization and aim for that kind of approach in all my endeavours.   \n"""
    medium_text_3 = """Recently, people tend to specialize in one specific subject. However I think that broadening knowledge of many academic subjects is helpful. In this essay, I will present some reasons to support my idea. \nFirst of all, i think that recent goal of our study does not want specialized knowledge. Although specialzed knowledge is also needed, broadened ones are really required to us. For example, in the applied mechanics, statistics and management are also needed. If we do not know about them, we can only produce a safe something. However if we know about them, we also make things cost-efficient. By doing so, we can reduce wasted money and provide things on lower price. \n\nIn addition, boadening knowledge of many academic subjects makes us diverse. If we are specialized on particular one, we would have narrow thought about what makes us puzzeld. As we have a variety of knowledge, that gives us diversed view. Thus, various knowledge may result in synergy effect. For instance, since i was studying various subjects like statistics and mechanics, it encourage me to study hard. Because i was boring at studying only mechanics, \nby doing so, i was excited to study mechanis. As a result, i got the good grades on both of them.\n\nTaking all those into consideration, broadening knowledge of various fields makes us more efficent, having widened visioin, and having active motivation. We are living in a world that demands a multi-player.\n\n"""
    low_text_3 = """Is a better  have to many subjets, because is important learning  to the everything topics. The people is not interestin in the live. The life is the reason for woking old the time. They are very special sensitive to the live. The nature has take everything big reason for live. The humans no others tanding to the reason for the live, because is neccesary to  continues. \n\n\nThe live gife to the opportunities the born in day to day and resposabilies to the  action in everything. I nead live to continues express my formuns, to the exprestion, to taking, to woking, to learning and much. The live is the perfect expression to the art in live in the people. Is the reason to coming to earth and the reason to the express. Is important  standing  why live?  Today is day to the change in my live. The possibily and research  is big, but  the reason fith is must important. I am not  said is the perfect moment in my life but this moments most wuanderfull. Comming to new history in  my live, comming to the new reason for victory.\n\n\nThe chage is neu the live is one. They coming to new future are not ordenstanding, other reason. Luck your live is a rich most important for you every time. The live is the complement to excencial and ideal by love your self. Loveling and LNothing is most important what  the reason, the yes accept to the resposabily to the live. \n"""
    prompt = f"Replace the MASK token by selecting only one of the following coherence labels: [low, medium, high].\n" \
             f"Examples:\nText: [{{low_text}}]\nCoherence level: low\n\nText: [{{medium_text}}]\nCoherence level: medium\n\nText: [{{high_text}}]\nCoherence level: high\n\n" \
             f"Text: [{{low_text_2}}]\nCoherence level: low\n\nText: [{{medium_text_2}}]\nCoherence level: medium\n\nText: [{{high_text_2}}]\nCoherence level: high\n\n" \
             f"Text: [{{low_text_3}}]\nCoherence level: low\n\nText: [{{medium_text_3}}]\nCoherence level: medium\n\nText: [{{high_text_3}}]\nCoherence level: high\n\n" \
             f"Text: [{{text}}]\nCoherence level: [MASK]"

    return prompt.format(
        low_text=low_text, medium_text=medium_text, high_text=high_text,
        low_text_2=low_text_2, medium_text_2=medium_text_2, high_text_2=high_text_2,
        low_text_3=low_text_3, medium_text_3=medium_text_3, high_text_3=high_text_3,
        text=text
    )

def apply_prompt_template(text, examples):
    high_text_1 = examples["high_text_1"]
    high_text_2 = examples["high_text_2"]
    high_text_3 = examples["high_text_3"]
    medium_text_1 = examples["medium_text_1"]
    medium_text_2 = examples["medium_text_2"]
    medium_text_3 = examples["medium_text_3"]
    low_text_1 = examples["low_text_1"]
    low_text_2 = examples["low_text_2"]
    low_text_3 = examples["low_text_3"]
    prompt = f"Replace the MASK token by selecting only one of the following coherence labels: [low, medium, high].\n" \
             f"Examples:\nText: [{{low_text_1}}]\nCoherence level: low\n\nText: [{{medium_text_1}}]\nCoherence level: medium\n\nText: [{{high_text_1}}]\nCoherence level: high\n\n" \
             f"Text: [{{low_text_2}}]\nCoherence level: low\n\nText: [{{medium_text_2}}]\nCoherence level: medium\n\nText: [{{high_text_2}}]\nCoherence level: high\n\n" \
             f"Text: [{{low_text_3}}]\nCoherence level: low\n\nText: [{{medium_text_3}}]\nCoherence level: medium\n\nText: [{{high_text_3}}]\nCoherence level: high\n\n" \
             f"Text: [{{text}}]\nCoherence level: [MASK]"

    return prompt.format(
        low_text_1=low_text_1, medium_text_1=medium_text_1, high_text_1=high_text_1,
        low_text_2=low_text_2, medium_text_2=medium_text_2, high_text_2=high_text_2,
        low_text_3=low_text_3, medium_text_3=medium_text_3, high_text_3=high_text_3,
        text=text
    )

def sample_examples(train_data_file):
    high_groups = []
    medium_groups = []
    low_groups = []
    with open(train_data_file, "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                text = sample["text"].strip()
                score = sample["score"]

                if score.lower() == "high" or score.lower() == "3":
                    high_groups.append(text)
                elif score.lower() == "medium" or score.lower() == "2":
                    medium_groups.append(text)
                elif score.lower() == "low" or score.lower() == "1":
                    low_groups.append(text)

    random.shuffle(high_groups)
    random.shuffle(medium_groups)
    random.shuffle(low_groups)

    examples = {}
    examples["high_text_1"] = high_groups[0]
    examples["high_text_2"] = high_groups[1]
    examples["high_text_3"] = high_groups[2]

    examples["medium_text_1"] = medium_groups[0]
    examples["medium_text_2"] = medium_groups[1]
    examples["medium_text_3"] = medium_groups[2]

    examples["low_text_1"] = low_groups[0]
    examples["low_text_2"] = low_groups[1]
    examples["low_text_3"] = low_groups[2]

    return examples

def cal_metric(dataset, fold_id, output_name):
    def extract_label_from_response(file_name):
        with open(file_name, "r", encoding="utf-8") as f:
            lines = f.readlines()
            lines = [line for line in lines if line.strip()]
        text = lines[-1]
        items = text.strip().split("##:## Coherence level:")
        res = items[-1].strip().lower()
        if "low" in res:
            label = "low"
        elif "medium" in res:
            label = "medium"
        elif "high" in res:
            label = "high"
        else:
            print("Error!!!")
            label = "low"
        return label

    data_dir = "data/dataset/{}/{}".format(dataset, fold_id)
    label_map = {
        "1": "low", "2": "medium", "3": "high",
        "low": "low", "medium": "medium", "high": "high",
    }
    label2id = {
        "low": 0, "medium": 1, "high": 2
    }

    gold_labels = []
    total_num = 0
    with open(os.path.join(data_dir, "test.json"), "r", encoding="utf-8") as f:
        lines = f.readlines()
        for line in lines:
            line = line.strip()
            if line:
                sample = json.loads(line)
                score = sample["score"].strip().lower()
                label = label_map[score]
                gold_labels.append(label)
                total_num += 1

    pred_labels = []
    for idx in range(1, total_num+1):
        cur_pred_file = "data/dataset/{}/{}/{}/test_{}.txt".format(dataset, fold_id, output_name, idx)
        label = extract_label_from_response(cur_pred_file)
        pred_labels.append(label)

    assert len(gold_labels) == len(pred_labels), (len(gold_labels), len(pred_labels))
    print(gold_labels[:30])
    print(pred_labels[:100])
    gold_ids = np.array([label2id[label] for label in gold_labels])
    pred_ids = np.array([label2id[label] for label in pred_labels])

    f1 = f1_score(y_true=gold_ids, y_pred=pred_ids, average="macro")
    acc = accuracy_score(y_true=gold_ids, y_pred=pred_ids)
    print(" %s, acc=%.4f, f1=%.4f"%(dataset, acc, f1))


def main():
    args = get_argparse().parse_args()
    print("Training/evaluation parameters %s", args)

    # 1. prepare data
    data_dir = os.path.join(args.data_dir, args.dataset)
    data_dir = os.path.join(data_dir, str(args.fold_id))
    train_file = os.path.join(data_dir, "train.json")
    dev_file = os.path.join(data_dir, "dev.json")
    test_file = os.path.join(data_dir, "test.json")
    output_dir = os.path.join(data_dir, args.output_name)
    os.makedirs(output_dir, exist_ok=True)
    args.output_dir = output_dir
    examples = sample_examples(train_data_file=train_file)

    # 2.recall chatgpt
    if args.do_dev:
        all_dev_samples = read_data_from_file(dev_file)
        file_prefix = os.path.join(args.output_dir, "dev_{}.txt")
        all_dev_samples = tqdm(all_dev_samples)
        for idx, sample in enumerate(all_dev_samples):
            if os.path.exists(file_prefix.format(idx+1)):
                print(" Doc %d has been processed, skip!"%(idx+1))
                continue
            text = sample[0]
            query = apply_prompt_template(text, examples)
            call_chatgpt([(idx+1, query)], file_prefix)

    if args.do_test:
        all_test_samples = read_data_from_file(test_file)
        file_prefix = os.path.join(args.output_dir, "test_{}.txt")
        all_test_samples = tqdm(all_test_samples)
        for idx, sample in enumerate(all_test_samples):
            if os.path.exists(file_prefix.format(idx + 1)):
                print(" Doc %d has been processed, skip!" % (idx + 1))
                continue
            text = sample[0]
            query = apply_prompt_template(text, examples)
            call_chatgpt([(idx + 1, query)], file_prefix)
            time.sleep(2)

    if args.do_cal:
        cal_metric(args.dataset, args.fold_id, args.output_name)

if __name__ == "__main__":
    main()
    # text = "I am not very happy about this because I feel stressful!"
    # output = apply_prompt_template3(text)
    # print(output)
