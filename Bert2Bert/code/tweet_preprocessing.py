from emoji import demojize
from nltk.tokenize import TweetTokenizer
import re

tokenizer = TweetTokenizer()


def normalizeToken(token):
    lowercased_token = token.lower()
    if token.startswith("@"):
        return ""
    elif lowercased_token.startswith("http") or lowercased_token.startswith("www"):
        return ""
    elif len(token) == 1:
        return demojize(token)
    else:
        if token == "’":
            return "'"
        elif token == "…":
            return "..."
        else:
            return token


def normalizeTweet(tweet):
    tweet = ''.join([i if ord(i) < 128 else '' for i in tweet])
    tweet = tweet.replace("\\\\n", " ")
    tweet = tweet.replace("\\n", " ")
    # tweet = tweet.replace("’", "'").replace("…", "...")
    tokens = tokenizer.tokenize(tweet)
    normTweet = " ".join([normalizeToken(token) for token in tokens])

    normTweet = (
        normTweet.replace("cannot ", "can not ")
        .replace("n't ", " n't ")
        .replace("n 't ", " n't ")
        .replace("ca n't", "can't")
        .replace("ai n't", "ain't")
    )
    normTweet = (
        normTweet.replace("'m ", " 'm ")
        .replace("'re ", " 're ")
        .replace("'s ", " 's ")
        .replace("'ll ", " 'll ")
        .replace("'d ", " 'd ")
        .replace("'ve ", " 've ")
    )
    normTweet = (
        normTweet.replace(" p . m .", "  p.m.")
        .replace(" p . m ", " p.m ")
        .replace(" a . m .", " a.m.")
        .replace(" a . m ", " a.m ")
    )
    normTweet = " ".join(normTweet.split())
    normTweet = re.sub("^RT : RT", '', normTweet)
    normTweet = re.sub("^RT ", '', normTweet)
    normTweet = re.sub("^\. ", "", normTweet)
    normTweet = re.sub("(^: :)|(^: )|(:$)", "", normTweet)
    return normTweet.strip()


if __name__ == "__main__":
    print(
        normalizeTweet(
            "RT :   SC has first two presumptive cases of coronavirus, DHEC confirms https://postandcourier.com/health/covid19/sc-has-first-two-presumptive-cases-of-coronavirus-dhec-confirms/article_bddfe4ae-5fd3-11ea-9ce4-5f495366cee6.html?utm_medium=social&utm_source=twitter&utm_campaign=user-share… via @postandcourier"
        )
    )