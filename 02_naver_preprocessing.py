import pandas as pd
from konlpy.tag import Okt
import re
from konlpy.tag import Kkma


df = pd.read_csv('./crawling_data/골목상권.csv')
df['content'] = df['title'] + " " + df['content']
df = df.loc[:, ['content']]
df.info()
print(df.head())


df_stopwords = pd.read_csv('./stopwords.csv', index_col=0)
stopwords = list(df_stopwords['stopword'])
# stopwords = stopwords + ['안나', '제니퍼', '미국', '중국', '영화', '감독', '리뷰', '연출',
#                          '장면', '주인공', '되어다', '출연', '싶다', '올해', '엘사', '아카리']


kkma = Kkma()
print(kkma.tagset)

okt = Okt()
df['clean_content'] = None
count = 0

for idx, content in enumerate(df.content):
    try:
        count += 1
        if count % 10 == 0:
            print('.', end='')
        if count % 100 == 0:
            print()
        content = re.sub('[^가-힣 ]', ' ', content)
        df.loc[idx, 'clean_content'] = content
        token = kkma.pos(content)
        df_token = pd.DataFrame(token, columns=['word', 'class'])
        df_token = df_token[(df_token['class']=='NNG') |    # 보통명사
                            (df_token['class'] == 'NN') |   # 명사
                            (df_token['class'] == 'NNB') |  # 일반 의존 명사
                            (df_token['class'] == 'NNM') |  # 단위 의존 명사
                            (df_token['class'] == 'NNP') |  # 고유명사
                            (df_token['class'] == 'NR') |   # 대명사
                            (df_token['class']=='VV') |     # 동사
                            (df_token['class']=='VA')]      # 형용사
        words = []
        for word in df_token.word:
            if len(word) > 1:
                if word not in list(df_stopwords.stopword):
                    words.append(word)
        cleaned_sentence = ' '.join(words)
        df.loc[idx, 'clean_content'] = cleaned_sentence
    except:
        print('error')

print(df.head(30))
print(df.isnull().sum())
df.dropna(inplace=True)
df.to_csv('./crawling_data/clean_content.csv', index=False)

