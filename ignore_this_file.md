# CE306 ASSIGNMENT 1

## Abstract

This report describes how I perform the pipeline step by step, by using Python and NLTK. My script takes specific website address as input and output various files, including parsed HTML, Pos tagging, split sentences, tokens, tokens without stop words, stemmed tokens and a ranking of keywords based on TF-IDF.

## Step by step description

### The complete system

The complete system I developed could read specific web pages and produce formatted output by one call. I have different parsing strategy for each site, more details in the next step. In this system, every step ends with a file writer to generate the output into a text file, in the corresponding folder. The core system is built by Python and NLTK.

### HTML parsing

Beautiful soup 4 and urlib.request is the best way to extract text from the website. By study the HTML content of provided URLs, I found that Essex URL is a pure HTML document in big table layout, whereas the other URL, the ecir2019 website is a comprehensive website contains not only the article but also some dynamic content such as tweets, I decided to custom parsing strategy for this website, so I use beautiful soup to only read the "\<article\>" tag, this should be the real content that the URL meant to present. Obviously, this approach is only for the assignment not for general purpose.

I create different folders for these 2 URLs and store the same output files under the folders. It's all automatic.

The output file(htmlParsing.txt) is a parsed HTML document, with pure text content, no tags.

### Preprocessing

This step contains some preparation works:

1. Normalize the text to lower case
2. split sentences by using sent_tokenize() method, the output file is sentenceSplit.txt.
3. tokenization of words by using word_tokenize() method, the output file is tokenization.txt
4. remove the stopwords and punctuation, there is a stopword set in NLTK and punctuation set in String class. I iterate the tokens, keep the tokens that not in both sets. The output file is tokenNoStopword.txt
5. stemming and morphological analysis, there are serval stemmer and lemmatizers in NLTK, I choice LancasterStemmer and WordNetLemmatizer for this task, this helps me to reduce similar words. The output file is stemming.txt

By finishing the above steps, I got complete clean tokens of URL for further processing.

### POS tagging

The pos tagging is generated by NLTK pos_tag() method, the output is POS.txt. Some of them are not accurate, ideally, I should train the model with the specific training set.

### Selecting Keywords

This is the final step, the purpose of this step is to rank the tokens and abstract the important concept of a website. I use NLTK tf_idf() method to calculate each word, store the value in a dictionary: { keyword : tfidf value }. The ranking of the keywords is in tfidf.txt.

The original output of each keyword is extremely smaller than lecture slides, the values are like 0.004, 0.009. So I dig into the source code of NLTK library and find something interesting, the tf, idf and tf_idf methods are under TextCollection class, the comment says this class it is a prototype only, will be slow to load. The way that NLTK calculate tf is count(term)/len(text), whereas the lecture notes simply count the terms, that's the major reason that my values are smaller then slides. By searching on Wikipedia, there are multiple ways to calculate tf, I think either of them is correct because the most frequent term always has the largest tf value. NLTK calculate idf with log(), with the base of e, normally people use the base of 10 instead, anyway, these do not affect the overall ranking, the only side effect is my value is very small, so I round the value and increase 100X for better observation.

I also generated 2 word cloud images for the tfidf outputs, by using wordcloud on GitHub. I put them in the folder as well(see figure_1.png and figure_2.png). I study the sample on GitHub then write the code, I commented the show() method to avoid duplicate.

## Conclusion

This assignment is a very good way to understand how raw information could be processed. The pipeline is very flexible, I can customize unique pipeline in different cases. By completing this assignment, I feel I have a much better understanding of the topics of Information retrieval.

I don't have too much possible improvement of this project, maybe I could use sci-kit learn to generate tfidf faster or I can prepare some training set for better POS tagging result.