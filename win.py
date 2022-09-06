# %%
from transformers import AutoTokenizer, AutoModelForSequenceClassification
from datasets import load_dataset


tokenizer = AutoTokenizer.from_pretrained("prajjwal1/bert-tiny-mnli")
model = AutoModelForSequenceClassification.from_pretrained("prajjwal1/bert-tiny-mnli")


def preprocess_function(examples):
    return tokenizer(examples["full_text"], truncation=True)


test = load_dataset("csv", data_files=["test.csv"])
train = load_dataset("csv", data_files=["train.csv"])

tokenized_train = train.map(preprocess_function, batched=True)

# %%
# Get word count in essays
print(train.map(lambda x: {"length": len(x["full_text"].split())})["train"]["length"])
print(
    max(train.map(lambda x: {"length": len(x["full_text"].split())})["train"]["length"])
)
print(
    min(train.map(lambda x: {"length": len(x["full_text"].split())})["train"]["length"])
)


# Hello
