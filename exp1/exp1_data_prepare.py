
import json
import csv

def get_train_vali_data(all_data, data_type, act_type):
    assert act_type in ["ans_single", "ans_condition", "condition_ans", "sent_single",
                        "sent_condition", "condition_sent", "ans_sent_condition", "condition_ans_sent"]
    # 定义 CSV 文件名
    csv_file = "./{}.csv".format(act_type + "_" + data_type)
    out_csv = open(csv_file, mode="w", newline="", encoding="utf-8")
    writer = csv.writer(out_csv)
    writer.writerow(["sentence1", "sentence2", "score"])
    key_01 = "deepseekv3_s1_ans" if data_type == "train" else "ds_s1_ans_1"
    key_02 = "deepseekv3_s2_ans" if data_type == "train" else "ds_s2_ans_1"
    for i, ele in enumerate(all_data.keys()):
        score = float(all_data[ele]["label"]) / 5
        ds_ans1 = all_data[ele][key_01].replace("<|end|>", "")
        ds_ans2 = all_data[ele][key_02].replace("<|end|>", "")
        sent1 = all_data[ele]["sent1"]
        sent1 = sent1 if sent1.endswith(".") else sent1 + "."
        sent2 = all_data[ele]["sent2"]
        sent2 = sent2 if sent2.endswith(".") else sent2 + "."
        if act_type == "ans_single":
            sentence1 = ds_ans1
            sentence2 = ds_ans2
        elif act_type == "ans_condition":
            sentence1 = ds_ans1 + all_data[ele]["condition"]
            sentence2 = ds_ans2 + all_data[ele]["condition"]
        elif act_type == "condition_ans":
            sentence1 = all_data[ele]["condition"] + ds_ans1
            sentence2 = all_data[ele]["condition"] + ds_ans2
        elif act_type == "sent_single":
            sentence1 = sent1
            sentence2 = sent2
        elif act_type == "sent_condition":
            sentence1 = sent1 + all_data[ele]["condition"]
            sentence2 = sent2 + all_data[ele]["condition"]
        elif act_type == "condition_sent":
            sentence1 = all_data[ele]["condition"] + sent1
            sentence2 = all_data[ele]["condition"] + sent2
        elif act_type == "ans_sent_condition":
            sentence1 = ds_ans1 + sent1 + all_data[ele]["condition"]
            sentence2 = ds_ans2 + sent2 + all_data[ele]["condition"]
        elif act_type == "condition_ans_sent":
            sentence1 = all_data[ele]["condition"] + ds_ans1 + sent1
            sentence2 = all_data[ele]["condition"] + ds_ans2 + sent2

        writer.writerow([sentence1, sentence2, score])


for data_type in ["vali", "train"]:
    file_name = "..\{}_data.json".format(data_type)
    all_data = json.load(open(file_name, "r", encoding="utf-8"))
    for act_type in ["ans_single", "ans_condition", "sent_single", "sent_condition", "ans_sent_condition"]:
        get_train_vali_data(all_data, data_type, act_type=act_type)


