import json
from openai import OpenAI
import torch
from transformers import AutoModelForCausalLM
from transformers import AutoTokenizer


def deepseekv3_request(api_key):
    def ds_get(que, api_key):
        client = OpenAI(api_key=api_key, base_url="https://api.deepseek.com")

        response = client.chat.completions.create(
            model="deepseek-chat",
            messages=[
                {"role": "system", "content": "You are a helpful assistant"},
                {"role": "user", "content": "{}".format(que)},
            ],
            stream=False
        )
        ss = response.choices[0].message
        return ss.content

    file_name = r"..\train_data.json"
    all_data = json.load(open(file_name, "r", encoding="utf-8"))
    sent1 = all_data["0"]["sent1"]
    sent2 = all_data["0"]["sent2"]
    condition = all_data["0"]["condition"]
    que = "What is " + condition[:-1] + "?"

    instruction_1 = "Based on the information given, give answers into a word or a phrase that is clear and concise:"
    instruction_5 = "Based on the information given, give 5 possible answers into a word or a phrase:"

    temp1 = "\nSentence: {}\nQuestion: {}\n".format(sent1, que) + instruction_1
    temp2 = "\nSentence: {}\nQuestion: {}\n".format(sent2, que) + instruction_1
    temp1_5 = "\nSentence: {}\nQuestion: {}\n".format(sent1, que) + instruction_5
    temp2_5 = "\nSentence: {}\nQuestion: {}\n".format(sent2, que) + instruction_5

    ans1 = ds_get(temp1, api_key)
    ans2 = ds_get(temp2, api_key)
    ans1_5 = ds_get(temp1_5, api_key)
    ans2_5 = ds_get(temp2_5, api_key)
    return ans1, ans2, ans1_5, ans2_5

def get_generate(base_model, inputs1, eos_token_id):
    outputs1 = base_model.generate(
        input_ids=inputs1['input_ids'],
        attention_mask=inputs1['attention_mask'],
        max_new_tokens=10,
        num_beams=5,
        early_stopping=True,
        eos_token_id=eos_token_id,
    )
    return outputs1

def get_ans(tokenizer, base_model, device):
    eos_token_id = tokenizer.convert_tokens_to_ids(".")

    file_name = r"..\train_data.json"
    all_data = json.load(open(file_name, "r", encoding="utf-8"))
    sent1 = all_data["0"]["sent1"]
    sent2 = all_data["0"]["sent2"]
    condition = all_data["0"]["condition"]
    que = "What is " + condition[:-1] + "?"
    question_single = 'summarize answer into a word or a phrase that is clear and concise and end with ".".'

    req_1 = "evidence: {} question: {} answer: ".format(sent1, que + question_single)
    req_2 = "evidence: {} question: {} answer: ".format(sent2, que + question_single)

    inputs1 = tokenizer(req_1, padding="max_length", truncation=True, max_length=128,
                        return_tensors="pt", padding_side="left").to(device)
    inputs2 = tokenizer(req_2, padding="max_length", truncation=True, max_length=128,
                        return_tensors="pt", padding_side="left").to(device)
    outputs1 = get_generate(base_model, inputs1, eos_token_id)
    outputs2 = get_generate(base_model, inputs2, eos_token_id)

    s_out1 = tokenizer.decode(outputs1[0], skip_special_tokens=True)
    s_out2 = tokenizer.decode(outputs2[0], skip_special_tokens=True)

    return s_out1, s_out2

def lightweight_llm_request(base_path):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = AutoTokenizer.from_pretrained(base_path)
    tokenizer.pad_token = tokenizer.eos_token
    base_model = AutoModelForCausalLM.from_pretrained(base_path).to(device)

    s_out1, s_out2 = get_ans(tokenizer, base_model, device)

    return s_out1, s_out2