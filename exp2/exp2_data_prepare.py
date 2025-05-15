
import json
import csv


def prepare_train_data(file_type, ans_type, ori_json_name, new_end=True):
    ans_type_list = ["ds", "lam", "qw", "ori"]
    assert ans_type in ans_type_list
    keys_name_01 = ["deepseekv3_s1_ans", "llama3.2-1B_s1_ans", "qwen2.5-1.5B_s1_ans", "deepseekv3_s1_ans"]
    keys_name_02 = ["deepseekv3_s2_ans", "llama3.2-1B_s2_ans", "qwen2.5-1.5B_s2_ans", "deepseekv3_s2_ans"]
    new_end_flag = "ne" if new_end else "fe"
    out_csv1 = open("{}_{}_ans_{}.csv".format(ans_type, file_type, new_end_flag), mode="w", newline="", encoding="utf-8")
    csv_writer1 = csv.writer(out_csv1)
    csv_writer1.writerow(["sentence1", "sentence2", "score"])
    all_data = json.load(open(ori_json_name, "r", encoding="utf-8"))

    key_01 = keys_name_01[ans_type_list.index(ans_type)] if file_type == "train" else "{}_s1_ans_1".format(ans_type)
    key_02 = keys_name_02[ans_type_list.index(ans_type)] if file_type == "train" else "{}_s2_ans_1".format(ans_type)

    for i, ele in enumerate(all_data.keys()):
        score = float(all_data[ele]["label"]) / 5
        evidence_1 = all_data[ele]["sent1"]
        evidence_2 = all_data[ele]["sent2"]
        question_base = "What is " + all_data[ele]["condition"][:-1] + "? "

        if ans_type != "ori":
            ans1 = all_data[ele][key_01]
            ans2 = all_data[ele][key_02]
        else:
            ans1 = ""
            ans2 = ""

        question_single = 'summarize answer into a word or a phrase that is clear and concise and end with ".".'
        base_sample1 = "evidence: {} question: {} <|end|>".format(evidence_1, question_base)
        base_sample2 = "evidence: {} question: {} <|end|>".format(evidence_2, question_base)


        sample_1 = "evidence: {} question: {} answer: {}".format(evidence_1,
                                                                         question_base + question_single,
                                                                         ans1)

        sample_2 = "evidence: {} question: {} answer: {}".format(evidence_2,
                                                                         question_base + question_single,
                                                                         ans2)
        s_1 = base_sample1 if ans_type == "ori" else sample_1
        s_2 = base_sample2 if ans_type == "ori" else sample_2

        if not new_end:
            s_1 = s_1.replace("<|end|>", "")
            s_2 = s_2.replace("<|end|>", "")

        csv_writer1.writerow([s_1, s_2, score])

for file_type in ["train", "vali"]:
    for ans_type in ["ds", "lam", "qw", "ori"]:
        prepare_train_data(file_type=file_type, ans_type=ans_type,
                           ori_json_name="../{}_data.json".format(file_type), new_end=True)
        prepare_train_data(file_type=file_type, ans_type=ans_type,
                           ori_json_name="../{}_data.json".format(file_type), new_end=False)




