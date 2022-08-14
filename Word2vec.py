import nltk
from gensim.models import Word2Vec
from nltk.corpus import stopwords
import re 

paragraph = """Steve got the 4 hang of babysitting five kids pretty quickly, 
which was impressive since the kids definitely were not keen on listening to him. 
Steve put a lot of dedication into the responsibilities of watching over 5 the moody pre-teens,
 and he even put his own life on the line to save them from the monsters. Babysitting the strong-willed 
 band of misfits took a firm hand, so Steve was constantly having to keep the kids in line, and that's 
 where the above quote comes in. The fact that Steve said this while a dishtowel was slung over his shoulder,
  made this moment all the more hilarious."""

## data prepressesing
text =  re.sub(r'\[[0-9]*\]'," ",paragraph)
text = re.sub(r'\s+',' ',text)
text = text.lower()
text = re.sub(r'\d+',' ', text)
text = re.sub(r'\s+',' ',text)
# print(text)

## convert them into sentences 
sentences=nltk.sent_tokenize(text)
sentences = [nltk.word_tokenize(x) for x in sentences]

## removing the uneccessary words
for x in range(len(sentences)):
  sentences[x] =  [y for y in sentences[x] if y not in stopwords.words('english')]


## applying word2vec
model=Word2Vec(sentences,min_count=1)
words=list(model.wv.index_to_key)
# print(words)

vector=model.wv['steve']
similar = model.wv.most_similar("steve")
print(similar)
