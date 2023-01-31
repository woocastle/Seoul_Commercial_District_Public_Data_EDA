import pandas as pd
from konlpy.tag import Okt
import re
import collections
import numpy as np

df = pd.read_csv('./crawling_data/one_sentences.csv')
df.info()
one_sentence = ' '.join(list(df.content))
words = one_sentence.split()
worddict = collections.Counter(words)

# words = pd.DataFrame(words)
# words.to_excel('test2.xlsx')

worddict = dict(worddict)
print(worddict)
sr = pd.Series(worddict)
sr2 = sr.sort_values(ascending=False)
print(sr.sort_values(ascending=False).head(99))

df_1 = sr2.to_csv('./sentence_rank.csv')
#########################################################
#
# import pandas as pd
#
# df = pd.read_csv('./crawling_data/one_sentences.csv')
# df.info()
# one_sentence = ' '.join(list(df.content))
# tokens = one_sentence.split()
#
# import gensim
#
# from gensim import corpora
#
#
# dictionary = corpora.Dictionary(tokens) # 토큰 단어와 gensim 내부 아이디 매칭
# dictionary.filter_extremes(no_below=2, no_above=0.5) # 빈도 2이상 포함, 전체 50% 이상 단어 제거
# corpus = [dictionary.doc2bow(token) for token in tokens]



