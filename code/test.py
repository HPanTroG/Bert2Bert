import pandas as pd 

data = pd.read_csv('/home/nguyen/cd_summarization/data/labeled_data/2015_Nepal_Earthquake_en_CF_labeled_data_final2.csv')
print(data.shape)
data = data.sample(1000)
data = data[['tweet_id', 'tweet_text', 'corrected_label', 'informative_content']]
data.columns = ['tweet_id', 'tweet_text', 'tweet_label', 'rationale_label']
data.to_csv('../data/examples.csv', index = False)