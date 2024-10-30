'''微调bert模型做情感分类（文本分类）'''
import torch
from datasets import load_dataset
dataset = load_dataset('imdb')#imdb是关于电影评论的数据集 包含十万条评论
print(dataset)
'''标记化数据，使用Transformers中的BertTokenizer对数据进行标记'''
from transformers import BertTokenizer
#文本长度默认，可自行设置
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased',max_length=512)
#将文本转换为模型可以处理的格式，包括填充和截断
def tokenize_function(examples):
    return tokenizer(examples['text'], padding='max_length', truncation=True)

tokenized_datasets = dataset.map(tokenize_function, batched=True)

'''划分训练集和验证集'''
#按照8：2把训练集进一步划分位训练集和测试集
train_testvalid = tokenized_datasets["train"].train_test_split(test_size=0.1, seed=42)
train_dataset = train_testvalid['train']
valid_dataset = train_testvalid['test']
'''为训练集和验证集创建数据加载器'''
from torch.utils.data import DataLoader
#训练数据随机打乱，增加模型泛化能力
train_dataloader = DataLoader(train_dataset, batch_size=16, shuffle=True)
valid_dataloader = DataLoader(valid_dataset, batch_size=16, shuffle=False)

'''加载预训练模型'''
from transformers import BertForSequenceClassification, AdamW
model = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=3)


'''定义训练函数'''
from transformers import Trainer, TrainingArguments
training_args = TrainingArguments(
    output_dir='./results',          # output directory
    num_train_epochs=3,              # total number of training epochs
    evaluation_strategy='epoch',     # evaluation strategy to adopt during training
    per_device_train_batch_size=1,  # batch size per device during training
    per_device_eval_batch_size=1,   # batch size for evaluation
    weight_decay=0.01,               # strength of weight decay
    learning_rate=2e-5,              # learning rate
)
trainer = Trainer(
    model=model,                         # the instantiated 🤗 Transformers model to be traine
    args=training_args,                  # training arguments, defined above
    train_dataset=train_dataset,         # training dataset
    eval_dataset=valid_dataset,          # evaluation dataset

)
trainer.train()
torch.cuda.empty_cache()
