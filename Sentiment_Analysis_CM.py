import tweepy
from textblob import TextBlob
import csv
import pandas as pd
from nltk.sentiment.vader import SentimentIntensityAnalyzer
import matplotlib.pyplot as plt
from wordcloud import WordCloud, STOPWORDS

# Autenticarse

consumer_key = 'y2bwwF68CCqUwQ86EzmNxIydX'
consumer_secret = '91YknFkTzDvuUwUajrwhFZ8V4ejeCxz2obLs3BTvRbAXOwScrP'

acces_token = '1092419252827627520-1JZC991bhQpPrC4Mw5Ydq51D5F1LsV'
acces_token_secret = 'xOhQqozgs4gdbNUGgbc0JZadljFCSFiYVt65Bs3TnMTY0'

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(acces_token, acces_token_secret)

api = tweepy.API(auth)

# Obtener los tweets
public_tweets = api.search('Venezuela')

# Manejando los datos con pandas
data = pd.DataFrame(data=[tweet.text for tweet in public_tweets], columns=['Tweets'])
print(data.head(2))

#print(public_tweets[0].text)
#analysis = TextBlob(public_tweets[0].text)
#print(analysis.sentiment)
sid = SentimentIntensityAnalyzer()

polaridad = []

for index, row in data.iterrows():
    s = sid.polarity_scores(row["Tweets"])
    polaridad.append(s)

ser =  pd.Series(polaridad)
data['polaridad'] = ser.values

print(data.head(10))

data.to_csv('Sentiment_VZLA')
'''
with open('Sentiment_VZLA.csv', mode='w') as csv_file:
    escritor_tweets = csv.writer(csv_file, delimiter=',')
    escritor_tweets.writerow(['TWEET', 'POLARIDAD', 'SUBJETIVIDAD'])
    for dato in data:
        escritor_tweets.writerow(dato)
'''

# Unir tweets en un solo string
palabras = ' '.join(data['Tweets'])

no_urls_no_tags = " ".join([word for word in palabras.split()
                            if 'http' not in word
                                and not word.startswith('@')
                                and word != 'RT'
                            ])

wordcloud = WordCloud(                   
                      stopwords=STOPWORDS,
                      background_color='black',
                      width=1800,
                      height=1400
                     ).generate(no_urls_no_tags)

plt.imshow(wordcloud)
plt.axis('off')
plt.savefig('Wordcloud_CM.png')
plt.show()
