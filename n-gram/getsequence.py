# n-gram
from sklearn.feature_extraction.text import CountVectorizer
import pandas as pd
import jieba
data = ["他用报话机向上级呼喊：“为了祖国，为了胜利，向我开炮！向我开炮！",
        "记者：你怎么会说出那番话？",
        "韦昌进：我只是觉得，对准我自己打，才有可能把上了我哨位的这些敌人打死，或者打下去。"]

data = [" ".join(jieba.lcut(e)) for e in data] # 分词，并用" "连接

print(data)

vec = CountVectorizer(min_df=1, ngram_range=(1,2))
# ngram_range=(1,1) 表示 unigram, ngram_range=(2,2) 表示 bigram, ngram_range=(3,3) 表示 thirgram

X = vec.fit_transform(data) #文本转矩阵
vec.get_feature_names() # 得到特征
X.toarray()
print(X)

df = pd.DataFrame(X.toarray(), columns=vec.get_feature_names()) # to DataFrame
df.head()
print(df)
