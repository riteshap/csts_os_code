from transformers import AutoModelForCausalLM
from peft import PeftModel
from transformers import AutoTokenizer
import pandas as pd
from torch.utils.data import DataLoader, Dataset
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.stats import pearsonr, spearmanr
import configparser
import json
import csv
import numpy as np
from torch.nn import DataParallel
import time
from model_log import WriteLog


class LlamaScorer(nn.Module):
    def __init__(self, tokenizer, base_model_path, peft_path, trans_path, device):
        super().__init__()
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        base_model.resize_token_embeddings(len(tokenizer))
        self.peft_model = PeftModel.from_pretrained(base_model, peft_path).to(device)
        transformer_layer = nn.TransformerEncoderLayer(
            d_model=base_model.config.hidden_size,
            nhead=8
        )
        self.transformer = nn.TransformerEncoder(
            transformer_layer,
            num_layers=1
        ).to(device)
        self.transformer.load_state_dict(torch.load(trans_path))

    def forward(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, score_labels):
        outputs_1 = self.peft_model(input_ids_1, attention_mask=attention_mask_1, return_dict=True,
                                output_hidden_states=True)
        hidden_states_1 = outputs_1.hidden_states
        last_hidden_state_1 = hidden_states_1[-1]
        src_key_padding_mask_1 = attention_mask_1 == 0

        outputs_2 = self.peft_model(input_ids_2, attention_mask=attention_mask_2, return_dict=True,
                                output_hidden_states=True)
        hidden_states_2 = outputs_2.hidden_states
        last_hidden_state_2 = hidden_states_2[-1]
        src_key_padding_mask_2 = attention_mask_2 == 0

        transformed_1 = self.transformer(last_hidden_state_1.permute(1, 0, 2),
                                    src_key_padding_mask=src_key_padding_mask_1)
        transformed_2 = self.transformer(last_hidden_state_2.permute(1, 0, 2),
                                    src_key_padding_mask=src_key_padding_mask_2)

        pooled_output_1 = transformed_1[-1, :, :]
        pooled_output_2 = transformed_2[-1, :, :]
        embeddings1 = F.normalize(pooled_output_1, p=2, dim=-1)
        embeddings2 = F.normalize(pooled_output_2, p=2, dim=-1)

        similarity = F.cosine_similarity(embeddings1, embeddings2)
        loss = F.mse_loss(similarity, score_labels, reduction="none")

        return similarity, loss


class Exp3:
    def __init__(self):
        tok_path = "./tokenizer"
        peft_path = "./peft_adapter"
        trans_path = "./extra_transformer_layer.pth"
        base_model_path = r"Llama-3.2-1B-Instruct"
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.tokenizer = AutoTokenizer.from_pretrained(tok_path)
        self.cusmodel = LlamaScorer(self.tokenizer, base_model_path, peft_path, trans_path, self.device).to(self.device)
        self.cusmodel = nn.DataParallel(self.cusmodel)
        self.cusmodel.eval()
        self.all_vali_data = json.load(open("../vali_data.json", "r", encoding="utf-8"))
        self.exp_log = WriteLog("exp3_log")

    def get_candidate_pair(self, sample_ele: dict, ds_num: int, lam_num: int, qw_num: int):

        score = float(sample_ele["label"]) / 5
        evidence_1 = sample_ele["sent1"]
        evidence_2 = sample_ele["sent2"]
        question_base = "What is " + sample_ele["condition"][:-1] + "? "
        question_single = 'summarize answer into a word or a phrase that is clear and concise and end with ".".'
        pre_sample_1 = "evidence: {} question: {} answer: ".format(evidence_1,
                                                                         question_base + question_single)

        pre_sample_2 = "evidence: {} question: {} answer: ".format(evidence_2,
                                                                         question_base + question_single)

        def add_ans_ele(ans_list, sam_ele, key_w1, key_w2, llm_num):
            if sam_ele[key_w1] == sam_ele[key_w2][0] and llm_num > 1:
                st = 1
                cut_num = 0
            else:
                st = 0
                cut_num = 1
            ans_list.append(sample_ele[key_w1])
            ans_list.extend(sample_ele[key_w2][st : (llm_num - cut_num)])

            return ans_list

        label_list = []
        ans1_list = []
        ans2_list = []
        if ds_num > 0:
            label_list += ["ds1"] * (ds_num > 0) + ["ds_c"] * (ds_num - 1)

            ans1_list = add_ans_ele(ans1_list, sample_ele, "ds_s1_ans_1", "ds_s1_ans_5", ds_num)
            ans2_list = add_ans_ele(ans2_list, sample_ele, "ds_s2_ans_1", "ds_s2_ans_5", ds_num)

        if lam_num > 0:
            label_list += ["lam1"] * (lam_num > 0) + ["lam_c"] * (lam_num - 1)

            ans1_list = add_ans_ele(ans1_list, sample_ele, "lam_s1_ans_1", "lam_s1_ans_5", lam_num)
            ans2_list = add_ans_ele(ans2_list, sample_ele, "lam_s2_ans_1", "lam_s2_ans_5", lam_num)

        if qw_num > 0:
            label_list += ["qw1"] * (qw_num > 0) + ["qw_c"] * (qw_num - 1)

            ans1_list = add_ans_ele(ans1_list, sample_ele, "qw_s1_ans_1", "qw_s1_ans_5", qw_num)
            ans2_list = add_ans_ele(ans2_list, sample_ele, "qw_s2_ans_1", "qw_s2_ans_5", qw_num)

        pair_1_list = []
        pair_2_list = []
        index_list = []
        for idx_1, c_1 in enumerate(ans1_list):
            for idx_2, c_2 in enumerate(ans2_list):
                pair_1_list.append(pre_sample_1 + c_1)
                pair_2_list.append(pre_sample_2 + c_2)
                index_list.append((idx_1, idx_2))

        return index_list, pair_1_list, pair_2_list, score, label_list

    def select_best_pair(self, index_list, pair_1_list, pair_2_list, score):

        batch_size = 20
        num_pairs = len(pair_1_list)

        all_losses = []
        all_similarity = []
        for i in range(0, num_pairs, batch_size):
            batch_text1 = pair_1_list[i:i + batch_size]
            batch_text2 = pair_2_list[i:i + batch_size]

            encoding1 = self.tokenizer(batch_text1, padding="max_length", truncation=True, max_length=128,
                                  return_tensors="pt", padding_side="left")
            encoding2 = self.tokenizer(batch_text2, padding="max_length", truncation=True, max_length=128,
                                  return_tensors="pt", padding_side="left")
            with torch.no_grad():

                input_ids1 = encoding1["input_ids"].to(self.device)
                attention_mask1 = encoding1["attention_mask"].to(self.device)
                input_ids2 = encoding2["input_ids"].to(self.device)
                attention_mask2 = encoding2["attention_mask"].to(self.device)

                score_labels = torch.tensor([score] * len(batch_text1)).to(self.device)
                similarity, loss = self.cusmodel(input_ids1, input_ids2, attention_mask1, attention_mask2, score_labels)
                loss_numpy = loss.detach().cpu().numpy()
                similarity_numpy = similarity.detach().cpu().numpy()
                all_losses.extend(loss_numpy)
                all_similarity.extend(similarity_numpy)

        min_loss = min(all_losses)
        min_loss_index = all_losses.index(min_loss)
        max_similarity = all_similarity[min_loss_index]

        index_1, index_2 = index_list[min_loss_index]
        ret_text1 = pair_1_list[min_loss_index]
        ret_text2 = pair_2_list[min_loss_index]
        return min_loss, max_similarity, ret_text1, ret_text2, index_1, index_2

    def main(self, ds_num: int, lam_num: int, qw_num: int):

        out_csv = open("count_best_{}_{}_{}.csv".format(ds_num, lam_num, qw_num), mode="w", newline="", encoding="utf-8")
        csv_writer = csv.writer(out_csv)
        csv_writer.writerow(["sentence1", "sentence2", "score", "index1", "index2"])

        cur_time = time.time()
        all_loss = 0
        all_similarity = []
        all_label = []
        count_all_num = {"ds1": 0, "ds_c": 0, "lam1": 0, "lam_c": 0, "qw1": 0, "qw_c": 0}
        for index, ele in enumerate(self.all_vali_data.keys()):
            index_list, pair_1_list, pair_2_list, score, label_list = self.get_candidate_pair(self.all_vali_data[ele], ds_num, lam_num, qw_num)

            min_loss, max_similarity, ret_text1, ret_text2, index_1, index_2 = self.select_best_pair(index_list, pair_1_list, pair_2_list, score)

            all_label.append(score)
            all_loss += min_loss
            all_similarity.append(max_similarity)

            count_all_num[label_list[index_1]] += 1
            count_all_num[label_list[index_2]] += 1

            csv_writer.writerow([ret_text1, ret_text2, score, index_1, index_2])

            if index % 100 == 0 and index != 0:
                print("****{}****{}".format(index, time.time() - cur_time))
                cur_time = time.time()
                print("all_mse_loss:{}".format(all_loss / (index + 1)))
                self.exp_log.write_in("all_mse_loss:{}".format(all_loss / (index + 1)))
                print(count_all_num)
                self.exp_log.write_in(count_all_num)
                all_labels_np = np.asarray(all_label)
                all_preds_np = np.asarray(all_similarity)
                pearson_corr, _ = pearsonr(all_preds_np, all_labels_np)
                spearman_corr, _ = spearmanr(all_preds_np, all_labels_np)
                print(f"Pearson correlation: {pearson_corr:.4f}")
                print(f"Spearman correlation: {spearman_corr:.4f}")
                self.exp_log.write_in(f"Pearson correlation: {pearson_corr:.4f}")
                self.exp_log.write_in(f"Spearman correlation: {spearman_corr:.4f}")

        all_labels_np = np.asarray(all_label)
        all_preds_np = np.asarray(all_similarity)
        self.exp_log.write_in("start：{}——{}——{}".format(ds_num, lam_num, qw_num))
        print("all_mse_loss:{}".format(all_loss / len(self.all_vali_data)))
        self.exp_log.write_in("all_mse_loss:{}".format(all_loss / len(self.all_vali_data)))
        print(count_all_num)
        self.exp_log.write_in(count_all_num)
        pearson_corr, _ = pearsonr(all_preds_np, all_labels_np)
        spearman_corr, _ = spearmanr(all_preds_np, all_labels_np)
        print(f"Pearson correlation: {pearson_corr:.4f}")
        print(f"Spearman correlation: {spearman_corr:.4f}")
        self.exp_log.write_in(f"Pearson correlation: {pearson_corr:.4f}")
        self.exp_log.write_in(f"Spearman correlation: {spearman_corr:.4f}")
        self.exp_log.write_in("**************************")
        print("*******************************************")

if __name__ == "__main__":
    cls_exp3 = Exp3()
    # cls_exp3.main(0, 0, 2)
    # cls_exp3.main(0, 0, 3)
    # cls_exp3.main(0, 0, 4)
    # cls_exp3.main(0, 0, 5)
    # cls_exp3.main(2, 2, 2)
    # cls_exp3.main(3, 3, 3)
    # cls_exp3.main(4, 4, 4)
    # cls_exp3.main(5, 5, 5)
    cls_exp3.main(1, 1, 1)



