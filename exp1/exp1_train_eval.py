
import pandas as pd
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel
from transformers import AutoTokenizer
import torch.optim as optim
from tqdm import tqdm
from scipy.stats import spearmanr, pearsonr


class TextSimilarityDataset(Dataset):
    def __init__(self, dataframe, tokenizer, max_length=128):
        self.dataframe = dataframe
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.dataframe)

    def __getitem__(self, index):
        row = self.dataframe.iloc[index]
        text1, text2, score = row["sentence1"], row["sentence2"], row["score"]

        encoding1 = tokenizer(text1, padding="max_length", truncation=True, max_length=128, return_tensors="pt")
        encoding2 = tokenizer(text2, padding="max_length", truncation=True, max_length=128, return_tensors="pt")

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


class BertSimilarityModel(nn.Module):
    def __init__(self, simcse_path):
        super(BertSimilarityModel, self).__init__()
        self.simcse = AutoModel.from_pretrained(simcse_path)
        self.loss_fn = nn.MSELoss()

    def forward(self, input_ids1, input_ids2, attention_mask1, attention_mask2, labels=None):
        outputs1 = self.simcse(input_ids=input_ids1, attention_mask=attention_mask1)
        outputs2 = self.simcse(input_ids=input_ids2, attention_mask=attention_mask2)

        embeddings1 = outputs1.pooler_output
        embeddings2 = outputs2.pooler_output
        embeddings1 = F.normalize(embeddings1, p=2, dim=1)
        embeddings2 = F.normalize(embeddings2, p=2, dim=1)

        similarity = F.cosine_similarity(embeddings1, embeddings2)

        if labels is not None:
            loss = self.loss_fn(similarity, labels)
            return loss, similarity
        return similarity


all_tpye_list = ["ans_single", "ans_condition", "sent_single", "sent_condition", "ans_sent_condition"]
record = open("out.txt", "w+", encoding="utf-8")
for act_type in all_tpye_list:
    train_df = pd.read_csv("./{}_train.csv".format(act_type))
    vali_df = pd.read_csv("./{}_vali.csv".format(act_type))
    print(train_df.head())

    model_path = r"E:\python_code\pre_model\princeton_nlpsup_simcse_bert_base_uncased"
    tokenizer = AutoTokenizer.from_pretrained(model_path)

    train_dataset = TextSimilarityDataset(train_df, tokenizer)
    val_dataset = TextSimilarityDataset(vali_df, tokenizer)

    train_loader = DataLoader(train_dataset, batch_size=48, shuffle=True)
    val_loader = DataLoader(val_dataset, batch_size=48)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = BertSimilarityModel(model_path).to(device)
    # model = torch.nn.DataParallel(model)
    # model.to(device)

    optimizer = optim.AdamW(model.parameters(), lr=5e-5)
    num_epochs = 5

    # 训练循环
    for epoch in range(num_epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f"Epoch {epoch+1}/{num_epochs}")

        for batch in progress_bar:
            input_ids1, input_ids2, attention_mask1, attention_mask2, labels = (batch["input_ids1"].to(device),
                                                                                batch["input_ids2"].to(device),
                                                                                batch["attention_mask1"].to(device),
                                                                                batch["attention_mask2"].to(device),
                                                                                batch["label"].to(device))

            optimizer.zero_grad()
            loss, _ = model(input_ids1, input_ids2, attention_mask1, attention_mask2, labels)
            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            progress_bar.set_postfix({"Loss": loss.item()})

        print(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader)}")
        record.write(f"Epoch {epoch+1}: Loss = {total_loss / len(train_loader)}" + "\n")

    torch.save(model.state_dict(), "./model_out/{}.pth".format(act_type))
    tokenizer.save_pretrained("./model_out/{}_model".format(act_type))

    def calcute(x, y):
        corr1, p_value1 = spearmanr(x, y)
        corr2, p_value2 = pearsonr(x, y)
        return corr1, p_value1, corr2, p_value2

    model.eval()
    total_loss = 0
    all_sim = []
    all_label = []
    with torch.no_grad():
        for batch in val_loader:
            input_ids1, input_ids2, attention_mask1, attention_mask2, labels = (batch["input_ids1"].to(device),
                                                                                batch["input_ids2"].to(device),
                                                                                batch["attention_mask1"].to(device),
                                                                                batch["attention_mask2"].to(device),
                                                                                batch["label"].to(device))
            loss, similarity = model(input_ids1, input_ids2, attention_mask1, attention_mask2, labels)
            total_loss += loss.item()
            all_label.extend(labels.to("cpu").numpy())
            all_sim.extend(similarity.to("cpu").numpy())

    print(f"Validation Loss: {total_loss / len(val_loader)}")
    corr1, p_value1, corr2, p_value2 = calcute(all_label, all_sim)
    print("spear{}\np_spear{}\npears{}\np_pears{}\n".format(corr1, p_value1, corr2, p_value2))

    record.write(f"Validation Loss: {total_loss / len(val_loader)}" + "\n")
    record.write("spear{}\np_spear{}\npears{}\np_pears{}\n".format(corr1, p_value1, corr2, p_value2))



