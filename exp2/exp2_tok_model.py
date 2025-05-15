from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.optim as optim
import torch
import torch.nn.functional as F
import pandas as pd
from tqdm import tqdm
from peft import PeftModel


class CustomModelAvgpool(nn.Module):
    def __init__(self, tokenizer, load_ck=False, frozen_peft=False):
        super(CustomModelAvgpool, self).__init__()
        model_name = llama_path
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        if nt_flag:
            base_model.resize_token_embeddings(len(tokenizer))
        if load_ck:
            self.peft_model = PeftModel.from_pretrained(base_model, load_ck_peft_path)
        else:
            if "DeepSeek" in llama_path:
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
                )
            else:
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    bias="none"
                )
            self.peft_model = get_peft_model(base_model, lora_config)

        if frozen_peft:

            for param in self.peft_model.parameters():
                param.requires_grad = False

        self.peft_model.print_trainable_parameters()
        # for name, param in cus_model.peft_model.named_parameters():
        #     if param.requires_grad is True:
        #         print(name)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.peft_model.config.hidden_size,
            nhead=8
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=1
        )
        if load_ck:
            self.transformer.load_state_dict(torch.load(load_ck_trans_path))

        self.mse_fn = nn.MSELoss()

    def forward(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, score_labels, labels_1, labels_2):

        outputs_1 = self.peft_model(input_ids_1, attention_mask=attention_mask_1, return_dict=True,
                                    output_hidden_states=True, labels=labels_1)
        hidden_states_1 = outputs_1.hidden_states
        last_hidden_state_1 = hidden_states_1[-1]
        src_key_padding_mask_1 = attention_mask_1 == 0

        outputs_2 = self.peft_model(input_ids_2, attention_mask=attention_mask_2, return_dict=True,
                                    output_hidden_states=True, labels=labels_2)
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
        loss_1 = outputs_1.loss
        loss_2 = outputs_2.loss

        loss = self.mse_fn(similarity, score_labels)

        return loss


class CustomModelCLS(nn.Module):
    def __init__(self, tokenizer, load_ck=False, frozen_peft=False):
        super(CustomModelCLS, self).__init__()
        model_name = llama_path
        base_model = AutoModelForCausalLM.from_pretrained(model_name)
        if nt_flag:
            base_model.resize_token_embeddings(len(tokenizer))
        if load_ck:
            self.peft_model = PeftModel.from_pretrained(base_model, load_ck_peft_path)
        else:
            if "DeepSeek" in llama_path:
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    bias="none",
                    task_type=TaskType.CAUSAL_LM,
                    target_modules=["q_proj", "k_proj", "v_proj", "o_proj"]
                )
            else:
                lora_config = LoraConfig(
                    r=8,
                    lora_alpha=16,
                    lora_dropout=0.1,
                    bias="none"
                )
            self.peft_model = get_peft_model(base_model, lora_config)

        if frozen_peft:
            for param in self.peft_model.parameters():
                param.requires_grad = False

        self.peft_model.print_trainable_parameters()
        # for name, param in cus_model.peft_model.named_parameters():
        #     if param.requires_grad is True:
        #         print(name)

        self.transformer_layer = nn.TransformerEncoderLayer(
            d_model=self.peft_model.config.hidden_size,
            nhead=8
        )
        self.transformer = nn.TransformerEncoder(
            self.transformer_layer,
            num_layers=1
        )
        if load_ck:
            self.transformer.load_state_dict(torch.load(load_ck_trans_path))

        self.mse_fn = nn.MSELoss()

    def forward(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, score_labels, labels_1, labels_2):
        outputs_1 = self.peft_model(input_ids_1, attention_mask=attention_mask_1, return_dict=True,
                                    output_hidden_states=True, labels=labels_1)
        hidden_states_1 = outputs_1.hidden_states
        last_hidden_state_1 = hidden_states_1[-1]
        src_key_padding_mask_1 = attention_mask_1 == 0

        outputs_2 = self.peft_model(input_ids_2, attention_mask=attention_mask_2, return_dict=True,
                                    output_hidden_states=True, labels=labels_2)
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
        loss_1 = outputs_1.loss
        loss_2 = outputs_2.loss

        loss = self.mse_fn(similarity, score_labels)

        return loss


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

    def forward(self, input_ids_1, input_ids_2, attention_mask_1, attention_mask_2, score_labels, labels_1, labels_2):
        outputs_1 = self.peft_model(input_ids_1, attention_mask=attention_mask_1, return_dict=True,
                                    output_hidden_states=True, labels=labels_1)
        hidden_states_1 = outputs_1.hidden_states
        last_hidden_state_1 = hidden_states_1[-1]
        src_key_padding_mask_1 = attention_mask_1 == 0

        outputs_2 = self.peft_model(input_ids_2, attention_mask=attention_mask_2, return_dict=True,
                                    output_hidden_states=True, labels=labels_2)
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
        loss_1 = outputs_1.loss
        loss_2 = outputs_2.loss

        loss = self.mse_fn(similarity, score_labels)

        return loss


class RegressionDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        def find_sublist_index(lst: list):
            sublst = [4320, 25]  # answer: 9399
            index = -1
            for i in range(len(lst) - len(sublst) + 1):
                if lst[i:i + len(sublst)] == sublst:
                    index = i
                    break
            return index + 2

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

        answer_start_idx_1 = find_sublist_index(input_ids1.tolist())
        answer_start_idx_2 = find_sublist_index(input_ids2.tolist())
        labels_1 = input_ids1.clone()
        labels_1[:answer_start_idx_1] = -100
        labels_2 = input_ids1.clone()
        labels_2[:answer_start_idx_2] = -100

        label = torch.tensor(score, dtype=torch.float)
        ret_dic = {"input_ids1": input_ids1,
                   "input_ids2": input_ids2,
                   "attention_mask1": attention_mask1,
                   "attention_mask2": attention_mask2,
                   "labels_1": labels_1,
                   "labels_2": labels_2,
                   "label": label}
        return ret_dic


avg_pool_list = [(0, 0), (1, 0), (2, 0), (3, 0)]
cls_list = [(0, 1), (1, 1), (2, 1), (3, 1)]
last_list = [(0, 2), (1, 2), (2, 2), (3, 2)]
model_list = [CustomModelAvgpool, CustomModelCLS, CustomModelLast]

cur_train = avg_pool_list[0]

train_file = ["./ds_train_ans_fe.csv",
              "./lam_train_ans_fe.csv",
              "./qw_train_ans_fe.csv",
              "./ori_train_ans_fe.csv"]

nt_flag = False
llama_path = r"/root/private_data/Llama-3.2-1B-Instruct"
train_df_file = train_file[cur_train[0]]
batch_size = 64
num_epochs = 10
tok_model = model_list[cur_train[1]]

load_ck = False
frozen_peft = False
load_ck_tk_path = "./tokenizer"
load_ck_peft_path = "./peft_adapter"
load_ck_trans_path = "extra_transformer_layer.pth"


def load_model_data(tok_model, load_ck=False, frozen_peft=False, batch=64):
    model_name = llama_path
    # model_name = r"E:\python_code\pre_model\Llama-3.2-1B-Instruct"
    if load_ck:
        tokenizer = AutoTokenizer.from_pretrained(load_ck_tk_path)
    else:
        tokenizer = AutoTokenizer.from_pretrained(model_name)
        if nt_flag:
            tokenizer.add_tokens(["<|end|>"])
    tokenizer.pad_token = tokenizer.eos_token

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cus_model = tok_model(tokenizer, load_ck=load_ck, frozen_peft=frozen_peft)
    cus_model.to(device)
    train_df = pd.read_csv(train_df_file)
    print(train_df.head())
    train_dataset = RegressionDataset(train_df, tokenizer)
    train_loader = DataLoader(train_dataset, batch_size=batch, shuffle=True)

    return tokenizer, cus_model, device, train_loader


tokenizer, cus_model, device, train_loader = load_model_data(tok_model=tok_model, load_ck=load_ck,
                                                             frozen_peft=frozen_peft, batch=batch_size)
optimizer = optim.AdamW(cus_model.parameters(), lr=5e-5)
cus_model.train()
for epoch in range(num_epochs):
    total_loss = 0
    progress_bar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{num_epochs}")
    for batch in progress_bar:
        input_ids1, input_ids2, attention_mask1, attention_mask2, labels_1, labels_2, labels = (
            batch["input_ids1"].to(device),
            batch["input_ids2"].to(device),
            batch["attention_mask1"].to(device),
            batch["attention_mask2"].to(device),
            batch["labels_1"].to(device),
            batch["labels_2"].to(device),
            batch["label"].to(device))

        optimizer.zero_grad()
        loss = cus_model(input_ids1, input_ids2, attention_mask1, attention_mask2, labels, labels_1, labels_2)

        loss.backward()
        optimizer.step()
        total_loss += loss.item()
        progress_bar.set_postfix({"Loss": loss.item()})

    print(f"Epoch {epoch + 1}: Loss = {total_loss / len(train_loader)}")

save_path_0 = "./peft_adapter" if not load_ck else "./peft_adapter_0"
save_path_1 = "./tokenizer" if not load_ck else "./tokenizer_0"

peft_model = cus_model.peft_model

peft_model.save_pretrained(save_path_0)

tokenizer.save_pretrained(save_path_1)

transformer_layer_name = "extra_transformer_layer.pth" if not load_ck else "extra_transformer_layer_new.pth"
extra_transformer_layer = cus_model.transformer
torch.save(extra_transformer_layer.state_dict(), transformer_layer_name)
