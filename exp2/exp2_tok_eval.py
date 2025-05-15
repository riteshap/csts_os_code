"""加载lora模型和transformer层验证模型性能"""

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


class RegressionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text1, text2, score = row["sentence1"], row["sentence2"], row["score"]

        encoding1 = self.tokenizer(text1, padding="max_length", truncation=True, max_length=self.max_length,
                                   return_tensors="pt", padding_side="left")
        encoding2 = self.tokenizer(text2, padding="max_length", truncation=True, max_length=self.max_length,
                                   return_tensors="pt", padding_side="left")

        input_ids1 = encoding1["input_ids"].squeeze(0)
        input_ids2 = encoding2["input_ids"].squeeze(0)
        attention_mask1 = encoding1["attention_mask"].squeeze(0)
        attention_mask2 = encoding2["attention_mask"].squeeze(0)
        label = torch.tensor(score, dtype=torch.float)
        ret_dic = {"input_ids1": input_ids1,
                   "input_ids2": input_ids2,
                   "attention_mask1": attention_mask1,
                   "attention_mask2": attention_mask2,
                   "label": label}
        return ret_dic

class CustomModelAvgpool(nn.Module):
    def __init__(self, tokenizer, base_model_path, peft_path, trans_path):
        super(CustomModelAvgpool, self).__init__()
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        if nt_flag:
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

        self.mse_fn = nn.MSELoss()

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
                                    src_key_padding_mask=src_key_padding_mask_1).permute(1, 0, 2)
        transformed_2 = self.transformer(last_hidden_state_2.permute(1, 0, 2),
                                    src_key_padding_mask=src_key_padding_mask_2).permute(1, 0, 2)

        attention_mask_1 = attention_mask_1.unsqueeze(-1)
        attention_mask_2 = attention_mask_2.unsqueeze(-1)

        masked_1 = transformed_1 * attention_mask_1
        masked_2 = transformed_2 * attention_mask_2

        sum_1 = masked_1.sum(dim=1)
        sum_2 = masked_2.sum(dim=1)
        lengths_1 = attention_mask_1.sum(dim=1).clamp(min=1e-8)
        lengths_2 = attention_mask_2.sum(dim=1).clamp(min=1e-8)

        pooled_output_1 = sum_1 / lengths_1
        pooled_output_2 = sum_2 / lengths_2

        embeddings1 = F.normalize(pooled_output_1, p=2, dim=-1)
        embeddings2 = F.normalize(pooled_output_2, p=2, dim=-1)

        similarity = F.cosine_similarity(embeddings1, embeddings2)

        loss = self.mse_fn(similarity, score_labels)

        return similarity, loss


class CustomModelCLS(nn.Module):
    def __init__(self, tokenizer, base_model_path, peft_path, trans_path):
        super(CustomModelCLS, self).__init__()
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        if nt_flag:
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

        self.mse_fn = nn.MSELoss()

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

        pooled_output_1 = transformed_1[0, :, :]
        pooled_output_2 = transformed_2[0, :, :]
        embeddings1 = F.normalize(pooled_output_1, p=2, dim=-1)
        embeddings2 = F.normalize(pooled_output_2, p=2, dim=-1)

        similarity = F.cosine_similarity(embeddings1, embeddings2)

        loss = self.mse_fn(similarity, score_labels)

        return similarity, loss


class CustomModelLast(nn.Module):
    def __init__(self, tokenizer, base_model_path, peft_path, trans_path):
        super(CustomModelLast, self).__init__()
        base_model = AutoModelForCausalLM.from_pretrained(base_model_path)
        if nt_flag:
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

        self.mse_fn = nn.MSELoss()

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

        loss = self.mse_fn(similarity, score_labels)

        return similarity, loss


avg_pool_list = [(0, 0), (1, 0), (2, 0), (3, 0)]
cls_list = [(0, 1), (1, 1), (2, 1), (3, 1)]
last_list = [(0, 2), (1, 2), (2, 2), (3, 2)]
model_list = [CustomModelAvgpool, CustomModelCLS, CustomModelLast]

cur_train = avg_pool_list[0]

vali_file = ["./ds_vali_ans_fe.csv",
              "./lam_vali_ans_fe.csv",
              "./qw_vali_ans_fe.csv",
              "./ori_vali_ans_fe.csv"]

base_model_path = r"Llama-3.2-1B-Instruct"
nt_flag = False
data_path = vali_file[cur_train[0]]
tok_path = "./tokenizer"
peft_path = "./peft_adapter"
trans_path = "./extra_transformer_layer.pth"
tok_model = model_list[cur_train[1]]


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
tokenizer = AutoTokenizer.from_pretrained(tok_path)

eval_df = pd.read_csv(data_path)
print(eval_df.head())
eval_dataset = RegressionDataset(eval_df, tokenizer)
eval_loader = DataLoader(eval_dataset, batch_size=8, shuffle=False)

cus_model = tok_model(tokenizer, base_model_path, peft_path, trans_path)
cus_model.to(device)

cus_model.eval()
with torch.no_grad():
    total_loss = 0
    total_sample = 0
    i = 1
    all_preds = []
    all_labels = []
    for batch in eval_loader:
        input_ids1, input_ids2, attention_mask1, attention_mask2, labels = (batch["input_ids1"].to(device),
                                                                            batch["input_ids2"].to(device),
                                                                            batch["attention_mask1"].to(device),
                                                                            batch["attention_mask2"].to(device),
                                                                            batch["label"].to(device))

        similarity, mse_loss = cus_model(input_ids1, input_ids2, attention_mask1, attention_mask2, labels)

        sample_num = labels.shape[0]
        total_loss += mse_loss * sample_num
        total_sample += sample_num

        all_preds.append(similarity.detach().cpu())
        all_labels.append(labels.detach().cpu())

        i += 1
        if i % 10 == 0:
            print(total_loss)
            print(total_sample)
            print(total_loss / total_sample)

    print("*****")
    print(total_loss)
    print(total_sample)
    print(total_loss / total_sample)
    all_preds_tensor = torch.cat(all_preds, dim=0)
    all_labels_tensor = torch.cat(all_labels, dim=0)
    all_preds_np = all_preds_tensor.numpy()
    all_labels_np = all_labels_tensor.numpy()
    pearson_corr, _ = pearsonr(all_preds_np, all_labels_np)
    spearman_corr, _ = spearmanr(all_preds_np, all_labels_np)
    print(f"Pearson correlation: {pearson_corr:.4f}")
    print(f"Spearman correlation: {spearman_corr:.4f}")



