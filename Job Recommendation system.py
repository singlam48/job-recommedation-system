#!/usr/bin/env python
# coding: utf-8

# In[3]:


import pandas as pd
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder


# In[1]:


pip install nltk


# In[4]:


import nltk


# In[3]:


nltk.download()


# In[5]:


import torch


# In[6]:


pip install torchvision


# In[ ]:


get_ipython().system('pip install pillow')


# In[6]:


from torchvision import models,transforms


# In[7]:


import torch.nn as nn


# In[8]:


import torch.optim as optim


# In[9]:


from PIL import Image


# In[10]:


from pickle import dump


# In[11]:


from torch.utils.data import Dataset,DataLoader


# In[12]:


from collections import Counter


# In[13]:


import nltk


# In[11]:


from nltk.corpus import stopwords


# In[14]:


device = torch.device('cpu')


# In[15]:


data = pd.read_csv("data job posts.csv")


# In[16]:


data.head()


# In[17]:


data.shape


# In[18]:


data.RequiredQual


# In[19]:


data.columns


# In[20]:


data.JobDescription


# In[21]:


data.JobRequirment


# In[22]:


df = data[["RequiredQual","JobDescription","JobRequirment","Title"]].dropna()
df


# In[23]:


classes = df['Title'].value_counts()[:20]
keys = classes.keys().to_list()

df = df[df['Title'].isin(keys)]
df['Title'].value_counts()


# In[24]:


def chane_titles(x):
    x = x.strip()
    if x == 'Senior Java Developer':
        return 'Java Developer'
    elif x == 'Senior Software Engineer':
        return 'Software Engineer'
    elif x == 'Senior QA Engineer':
        return 'Software QA Engineer'
    elif x == 'Senior Software Developer':
        return 'Senior Web Developer'
    elif x =='Senior PHP Developer':
        return 'PHP Developer'
    elif x == 'Senior .NET Developer':
        return '.NET Developer'
    elif x == 'Senior Web Developer':
        return 'Web Developer'
    elif x == 'Database Administrator':
        return 'Database Admin/Dev'
    elif x == 'Database Developer':
        return 'Database Admin/Dev'

    else:
        return x
        
    
df['Title'] = df['Title'].apply(chane_titles)
df['Title'].value_counts()


# In[25]:


df["Combined"] = df.RequiredQual + df.JobDescription + df.JobRequirment
df.Combined = df.Combined.apply(lambda x: x.replace("\r\n"," "))
df.head(50)


# In[26]:


df.iloc[0,4]


# In[26]:


df.to_csv("Modified.csv",index=False)


# In[27]:


class Vocabulary(object):
    """Simple vocabulary wrapper."""
    def __init__(self):
        self.word2idx = {}
        self.idx2word = {}
        self.idx = 0

    def add_word(self, word):
        if not word in self.word2idx:
            self.word2idx[word] = self.idx
            self.idx2word[self.idx] = word
            self.idx += 1

    def __call__(self, word):
        if not word in self.word2idx:
            return self.word2idx['<unk>']
        return self.word2idx[word]

    def __len__(self):
        return len(self.word2idx)


# In[28]:


def build_vocab(df, threshold=3):
    """Build a simple vocabulary wrapper."""
    counter = Counter()
    
    for i in range(len(df)):
        caption = df.iloc[i,4]
        tokens = nltk.tokenize.word_tokenize(str(caption))
        counter.update(tokens)

        if (i+1) % 1000 == 0:
                print("[{}/{}] Tokenized the sentences.".format(i+1, len(df)))

    # If the word frequency is less than 'threshold', then the word is discarded.
    words = [word for word, cnt in counter.items() if cnt >= threshold]

    # Create a vocab wrapper and add some special tokens.
    vocab = Vocabulary()
    vocab.add_word('<pad>')
    vocab.add_word('<start>')
    vocab.add_word('<end>')
    vocab.add_word('<unk>')

    # Add the words to the vocabulary.
    for i, word in enumerate(words):
        vocab.add_word(word)
    return vocab


# In[1]:


pip install nltk


# In[5]:


import nltk


# In[29]:


v = build_vocab(df)
dump(v, open('vocab.pkl', 'wb'))
len(v)


# In[35]:


le = LabelEncoder()
df["TitleUse"] = le.fit_transform(df.Title)
df


# In[36]:


df.iloc[:,5].nunique()


# In[37]:


x = torch.Tensor(np.array(df.iloc[:,5]))
x


# In[38]:


class Data(Dataset):
    def __init__(self,df,vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self,idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()
        tokens = nltk.tokenize.word_tokenize(str(df.iloc[idx,4]))
        caption = []
        caption.append(self.vocab('<start>'))
        caption.extend([self.vocab(token) for token in tokens])
        caption.append(self.vocab('<end>'))
        return caption,x[idx]

def collate_fn(data):
    data.sort(key=lambda x: len(x[0]), reverse=True)
    captions,labels = zip(*data)
    lengths = [len(cap) for cap in captions]
    targets = torch.zeros(len(captions), max(lengths)).long()
    for i, cap in enumerate(captions):
        end = lengths[i]
        targets[i, :end] = torch.Tensor(cap[:end])        
    return targets.to(device), labels


# In[39]:


X_train,X_test,y_train,y_test = train_test_split(df.Combined,df.Title,test_size = 0.15,random_state = 0)
train = Data(pd.DataFrame(X_train),v)
test = Data(pd.DataFrame(X_test),v)
dataloaderTrain = DataLoader(train,4,num_workers=0,collate_fn=collate_fn)
dataloaderTest = DataLoader(test,4,num_workers=0,collate_fn=collate_fn)


# In[40]:


for i,j in dataloaderTrain:
    print(i)
    print(j)
    print(i.shape,torch.stack(j).shape)
    break


# In[41]:


class Model(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers):
        super(Model, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True,bidirectional = True, dropout= 0.3)
        self.relu = nn.ReLU()
        self.linear = nn.Linear(hidden_size*2,256)
        self.linear3 = nn.Linear(256,19)
        
    def forward(self,captions):
        embeddings = self.embed(captions)
        hiddens, _ = self.lstm(embeddings)
        x = self.relu(self.linear(hiddens[:,-1,:]))
        outputs = self.linear3(x)
        return outputs


# In[42]:


from torch.nn.utils.rnn import pack_padded_sequence
def train(model,data_loader,data_loaderTest,learning_rate,num_epochs):  

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

    total_step = len(data_loader)
    print("Total steps are: ", total_step)
    for epoch in range(num_epochs):
        running_loss = 0.0
        model.train()
        for i,(captions, lengths) in enumerate(data_loader):
            # Set mini-batch dataset
            captions = captions.to(device)
            # Forward, backward and optimize
            outputs = model(captions)
            loss = criterion(outputs.to(device),torch.stack(lengths).type(torch.LongTensor).to(device))
            model.zero_grad()
            loss.backward()
            optimizer.step()
            running_loss += loss.item()

            if i % 50 == 0:
                print('Epoch [{}/{}], Step [{}/{}],Training Loss: {:.4f}'
                    .format(epoch, num_epochs, i, total_step, loss.item()))
        correct = 0
        total = 0
        model.eval()
        with torch.no_grad():
            for data in data_loader:
                captions, labels = data
                outputs = model(captions)
                _, predicted = torch.max(outputs.data, 1)
                total += torch.stack(labels).type(torch.LongTensor).size(0)
                correct += (predicted == torch.stack(labels).type(torch.LongTensor).to(device)).sum().item()
        print("Loss after epoch {} is {} and Accuracy is {}".format(epoch,running_loss/total_step,100 * correct / total))
        torch.save(model.state_dict(),"epoch"+str(epoch)+".pb")
        
        correct = 0
        total = 0
        with torch.no_grad():
            for data in data_loaderTest:
                captions, labels = data
                outputs = model(captions)
                _, predicted = torch.max(outputs.data, 1)
                total += torch.stack(labels).type(torch.LongTensor).size(0)
                correct += (predicted == torch.stack(labels).type(torch.LongTensor).to(device)).sum().item()
        print("Loss after epoch {} is {} and Accuracy is {}".format(epoch,running_loss,100 * correct / total))


# In[43]:


model = Model(1024,512,len(v),3).to(device)


# In[ ]:


train(model,dataloaderTrain,dataloaderTest,0.001,29)


# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:





# In[ ]:




