#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


# In[2]:


softskillsDataFrame = pd.read_csv("Soft Skills.csv")
display(softskillsDataFrame)


# In[3]:


softskillsDataFrame.drop(['Unnamed: 2', 'Unnamed: 5','Unnamed: 8'], inplace=True, axis=1)
display(softskillsDataFrame)


# In[4]:


softskillsDataFrame.rename(columns = {'Unnamed: 1': 'Dominance', 'Unnamed: 4': 'Extraversion', 'Unnamed: 7': 'Patience', 'Unnamed: 10': "Conscientious"}, inplace=True)
display(softskillsDataFrame)


# In[6]:


softskillsDataFrame = softskillsDataFrame.iloc[0:54, 0:8]
display(softskillsDataFrame)


# In[7]:


softskillDictionary = {}
# count = 0
for (columnName, columnData) in softskillsDataFrame.iteritems():
    boolSeries = pd.notnull(columnData)
    newSeries = columnData[boolSeries]
    for string in newSeries:
        # print (string)
        softskillDictionary[string] = 0
        #count = count + 1
#print (count) is 262
print (softskillDictionary.keys()) #248 count for some reason


# In[8]:


from sentence_transformers import SentenceTransformer
model = SentenceTransformer('bert-base-nli-mean-tokens')
#Encoding:
keyList = list(softskillDictionary.keys())
key_embeddings = model.encode(keyList)
print(key_embeddings.shape)


# In[9]:


candidates = pd.read_csv("./candidate_data.csv")
candidates = candidates.iloc[:,[38,39,42,74]]
candidates = candidates.rename(columns = {list(candidates.columns)[0]: "Strengths",list(candidates.columns)[1]: "Skills",list(candidates.columns)[2]: "Experience",list(candidates.columns)[3]: "Job Details"})
candidates.head()


# In[10]:


candidateBlobs = []
for index, row in candidates.iterrows():
    candidateBlobs.append(row.str.cat())
candidateBlobs[0]


# In[11]:


testCandidate = candidateBlobs[0]


# In[12]:


import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import re
lemmatizer = WordNetLemmatizer()
stop_words = set(stopwords.words('english'))
testCandidate = re.sub(r'[^\w\s]',' ', testCandidate)
word_tokens = word_tokenize(testCandidate)
filteredParagraph = []
for w in word_tokens:
    if w not in stop_words and len(w) > 2: #To remove most of the punctuation and sped words
        filteredParagraph.append(lemmatizer.lemmatize(w))
        #print(w + ": " + lemmatizer.lemmatize(w))
print(filteredParagraph)


# In[21]:


filteredEmbed = model.encode(filteredParagraph)
cosResult = cosine_similarity([key_embeddings[0]], filteredEmbed)
print(cosResult)


# In[29]:


index = 0
for key in keyList:
    value = cosine_similarity([key_embeddings[index]], filteredEmbed)
    softskillDictionary[key] = value[0] #removes list of lists
    index += 1


# In[30]:


print(softskillDictionary[keyList[0]])
print(softskillDictionary[keyList[1]])
print(softskillDictionary[keyList[2]])


# In[31]:


from statistics import mean
avgskillsDict = {}
for key in keyList:
    previousList = softskillDictionary[key]
    avgskillsDict[key] = mean(previousList)


# In[33]:


sorted_dict = {}
sorted_keys = sorted(avgskillsDict, key=avgskillsDict.get)  # [1, 3, 2]

for w in sorted_keys:
    sorted_dict[w] = avgskillsDict[w]

print(sorted_dict) # {1: 1, 3: 4, 2: 9}


# In[46]:


teamLeader = softskillsDataFrame.iloc[:, 0]
boolSeries = pd.notnull(teamLeader)
teamLeader = teamLeader[boolSeries]

teamPlayer = softskillsDataFrame.iloc[:, 1]
boolSeries = pd.notnull(teamPlayer)
teamPlayer = teamPlayer[boolSeries]

thinker = softskillsDataFrame.iloc[:, 2]
boolSeries = pd.notnull(thinker)
thinker = thinker[boolSeries]

talker = softskillsDataFrame.iloc[:, 3]
boolSeries = pd.notnull(talker)
talker = talker[boolSeries]

steady = softskillsDataFrame.iloc[:, 4]
boolSeries = pd.notnull(steady)
steady = steady[boolSeries]

fastpaced = softskillsDataFrame.iloc[:, 5]
boolSeries = pd.notnull(fastpaced)
fastpaced = fastpaced[boolSeries]

detailOriented = softskillsDataFrame.iloc[:, 6]
boolSeries = pd.notnull(detailOriented)
detailOriented = detailOriented[boolSeries]

innovative = softskillsDataFrame.iloc[:, 7]
boolSeries = pd.notnull(innovative)
innovative = innovative[boolSeries]


# In[48]:


teamLeaderSum = 0
for key in teamLeader:
    teamLeaderSum += avgskillsDict[key]
teamLeaderAvg = float(teamLeaderSum / len(teamLeader))
print(teamLeaderAvg)


# In[49]:


teamPlayerSum = 0
for key in teamPlayer:
    teamPlayerSum += avgskillsDict[key]
teamPlayerAvg = float(teamPlayerSum / len(teamPlayer))
print(teamPlayerAvg)


# In[50]:


thinkerValues = [float(avgskillsDict[key]) for key in thinker]
thinkerAvg = mean(thinkerValues)
print(thinkerAvg)


# In[51]:


talkerValues = [float(avgskillsDict[key]) for key in talker]
talkerAvg = mean(talkerValues)
print(talkerAvg)


# In[52]:


steadyValues = [float(avgskillsDict[key]) for key in steady]
steadyAvg = mean(steadyValues)
print(steadyAvg)


# In[53]:


fastValues = [float(avgskillsDict[key]) for key in fastpaced]
fastAvg = mean(fastValues)
print(fastAvg)


# In[54]:


detailValues = [float(avgskillsDict[key]) for key in detailOriented]
detailAvg = mean(detailValues)
print(detailAvg)


# In[55]:


innovativeValues = [float(avgskillsDict[key]) for key in innovative]
innovativeAvg = mean(innovativeValues)
print(innovativeAvg)


# In[56]:


categories = {"Team Leader" : teamLeaderAvg, "Team Player" : teamPlayerAvg, "Thinker" : thinkerAvg, "Talker" : talkerAvg, "Steady" : steadyAvg, "Fast-Paced" : fastAvg, "Detail-Oriented" : detailAvg, "Innovative" : innovativeAvg}
categories


# In[59]:


categoriesSorted = {}
sortedCategoryKeys = sorted(categories, key=categories.get)  # [1, 3, 2]
for w in sortedCategoryKeys:
    categoriesSorted[w] = categories[w]

print(categoriesSorted) # {1: 1, 3: 4, 2: 9}


# In[ ]:




