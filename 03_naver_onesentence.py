import pandas as pd

df = pd.read_csv('./crawling_data/clean_content.csv')
df.dropna(inplace=True)
df.info()
one_sentences = []
for contents in df['content'].unique():
    temp = df[df['content']==contents]
    if len(temp) > 30:
        temp = temp.iloc[:30, :]
    one_sentence = ' '.join(temp['clean_content'])
    one_sentences.append(one_sentence)
df_one = pd.DataFrame({'content':df['content'].unique(), 'content':one_sentences})
print(df_one.head())
df_one.to_csv('./crawling_data/one_sentences.csv', index=False)

