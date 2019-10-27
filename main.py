from bs4 import BeautifulSoup
import urllib.request
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import nltk.data
from nltk.corpus import stopwords
import string
from nltk.text import TextCollection
from nltk.stem.lancaster import LancasterStemmer
from nltk.stem import WordNetLemmatizer
from wordcloud import WordCloud
import matplotlib.pyplot as plt

# ************************** 1. HTML PARSING **************************

# grab pure text from the website
website = "https://csee.essex.ac.uk/staff/udo/index.html"
#website = "https://ecir2019.org/industry-day/"

response = urllib.request.urlopen(website)
html = response.read()
soup = BeautifulSoup(html, features="lxml")
text = ""
address = ""
# different parsing strategy
if(website == "https://csee.essex.ac.uk/staff/udo/index.html"):
    text = soup.get_text()
    address = "essex/"
elif (website == "https://ecir2019.org/industry-day/"):
    article = soup.article
    text = article.get_text()
    address = "ecir2019/"


file = open(address+"htmlParsing"+".txt", "w")  # create new file if not exist
file.write(text)  # override the file
file.close()

# ************************** 2. Pre-Processing **************************
# sentence splitting, tokenization and normalization

# 2.1 normalize the text to lower case
text = str.lower(text)

# 2.2 split sentences
sent_tokenize_list = sent_tokenize(text)
file = open(address+"sentenceSplit"+".txt", "w")  # create new file if not exist
file.write(str(sent_tokenize_list))  # override the file
file.close()

# 2.3 tokenization of words
alltokens = []
for sent in sent_tokenize_list:
    for word in word_tokenize(sent):
        alltokens.append(word)

file = open(address+"Tokenization"+".txt", "w")  # create new file if not exist
file.write(str(alltokens))  # override the file
file.close()

# 2.4 remove stopwords & punctuation
cleanTokens = []
for token in alltokens:
    if(token not in stopwords.words('english') and token not in string.punctuation):
        cleanTokens.append(token)

file = open(address+"tokenNoStopword"+".txt", "w")  # create new file if not exist
file.write(str(cleanTokens))  # override the file
file.close()

# ************************** 5. stemming/morphological analysis **************************
lancaster_stemmer = LancasterStemmer()
wordnet_lemmatizer = WordNetLemmatizer()
for token in cleanTokens:
    token = wordnet_lemmatizer.lemmatize(lancaster_stemmer.stem(token))

file = open(address+"stemming"+".txt", "w")  # create new file if not exist
file.write(str(cleanTokens))  # override the file
file.close()
# ************************** 3. part of speech tagging **************************
posTagging = nltk.pos_tag(cleanTokens)
file = open(address+"POS"+".txt", "w")  # create new file if not exist
file.write(str(posTagging))  # override the file
file.close()
# ************************** 4. Selecting keywords **************************
# remove stopwords ( see 2.4 )
corpus = TextCollection(cleanTokens)
words = set(cleanTokens)
value = {}
for word in words:
    value[word] = round(corpus.tf_idf(word, cleanTokens) * 100,2)
rank = sorted(value.items(), key=lambda item: item[1] , reverse = True)
file = open(address+"tfidf"+".txt", "w")  # create new file if not exist
file.write(str(rank))  # override the file
file.close()

# Generate word cloud
wordcloud=WordCloud().generate_from_frequencies(value)
plt.imshow(wordcloud, interpolation='bilinear')
plt.axis("off")
# disable drawing
# plt.show()
