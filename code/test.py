import pandas as pd

data = pd.read_csv('../data/nepal.csv')
data  = data[['tweet_id', 'tweet_text', 'corrected_label', 'informative_content']]
data.columns = ['tweet_id', 'tweet_text', 'tweet_label', 'rationale_label']
data.to_csv("../data/nepal.csv", header = None, index=False)
