import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
import collections
from wordcloud import WordCloud
from matplotlib import font_manager, rc
from PIL import Image
import matplotlib as mpl

df = pd.read_csv("./sentence_rank.csv")
df.columns = ['word', 'count']
df = df.head(30)

plt.rcParams['font.family'] = 'Malgun Gothic'

plt.figure(figsize=(12, 10))
sns.barplot(data=df.sort_values('count'), x='count', y='word', ci=None)
plt.xlabel('빈도 수')
plt.ylabel('단어')
_ = plt.title("21년도 '골목상권' 주요 키워드")
# plt.show()

font_path = './malgun.ttf'
font_name = font_manager.FontProperties(fname=font_path).get_name()
mpl.rcParams['axes.unicode_minus'] = False
rc('font', family = font_name)

df = pd.read_csv('./crawling_data/one_sentences.csv')
df.info()
one_sentence = ' '.join(list(df.content))
words = one_sentence.split()
worddict = collections.Counter(words)
worddict = dict(worddict)
del worddict['골목']
del worddict['상권']
del worddict['상공인']
worddict['코로나'] = 2700
print(worddict)

# 시각화
wordcloud_img = WordCloud(background_color='white', max_words=50,
                          font_path=font_path).generate_from_frequencies(worddict)
plt.figure(figsize=(12, 12))
plt.imshow(wordcloud_img)     # interpolation='bilinear'
plt.axis('off')
plt.show()
