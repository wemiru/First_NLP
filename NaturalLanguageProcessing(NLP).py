# NLP

import pandas as pd
import numpy as np
import seaborn as sn
import matplotlib.pyplot as plt
import re
import nltk
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer,CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB,MultinomialNB
from sklearn.svm import SVC
from sklearn import metrics

books = pd.read_csv(r'C:\Users\wubne\Desktop\NLP\BooksDataSet.csv')
books.head()

# removing the unnamed : 0

books = books[['book_id', 'book_name', 'genre', 'summary']]
books.head(3)

books.info()

sn.countplot(x = books['genre'],palette='plasma')
plt.xticks(rotation = 45, ha = 'right')

books['summary'].iloc[1]

## cleaning the text data

def cleantext(text):
    
    # removing the "\"
    
    text = re.sub("'\''","",text)
    
    # removing special symbols
    
    text = re.sub("[^a-zA-Z]"," ",text)
    
    # removing the whitespaces
    
    text = ' '.join(text.split())
    
    # convert text to lowercase
    
    text = text.lower()
    
    return text

    
books['summary'] = books['summary'].apply(lambda x:cleantext(x))
books['summary'].iloc[1]

def showmostfrequentwords(text,no_of_words):
    
    allwords = ' '.join([char for char in text])
    allwords = allwords.split()
    fdist = nltk.FreqDist(allwords)
    
    wordsdf = pd.DataFrame({'word':list(fdist.keys()),'count':list(fdist.values())})
    
    df = wordsdf.nlargest(columns="count",n = no_of_words)
    
    plt.figure(figsize=(7,5))
    ax = sn.barplot(data=df,x = 'count',y = 'word')
    ax.set(ylabel = 'Word')
    plt.show()
    
    return wordsdf
    
    
# 25 most frequent words

wordsdf = showmostfrequentwords(books['summary'],25)

wordsdf.sort_values('count',ascending=False).head(10).style.background_gradient(cmap = 'plasma')

# Removing the stopwords

nltk.download('stopwords', download_dir= './')

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# removing the stopwords

nltk.download('stopwords')
from nltk.corpus import stopwords
stop_words = set(stopwords.words('english'))

# removing the stopwords

def removestopwords(text):
    
    removedstopword = [word for word in text.split() if word not in stop_words]
    return ' '.join(removedstopword)

books['summary'] = books['summary'].apply(lambda x:removestopwords(x))
books['summary'].iloc[1]

## Lemmatizing

nltk.download('wordnet')
from nltk.stem import WordNetLemmatizer

lemma=WordNetLemmatizer()

def lematizing(sentence):
    stemSentence = ""
    for word in sentence.split():
        stem = lemma.lemmatize(word)
        stemSentence += stem
        stemSentence += " "
    stemSentence = stemSentence.strip()
    return stemSentence


books['summary'] = books['summary'].apply(lambda x: lematizing(x))


# Stemming

from nltk.stem import PorterStemmer
stemmer = PorterStemmer()

def stemming(sentence):
    
    stemmed_sentence = ""
    for word in sentence.split():
        stem = stemmer.stem(word)
        stemmed_sentence+=stem
        stemmed_sentence+=" "
        
    stemmed_sentence = stemmed_sentence.strip()
    return stemmed_sentence

books['summary'] = books['summary'].apply(lambda text:stemming(text))
books['summary'].iloc[1]

# To visualize frewords

freq_df = showmostfrequentwords(books['summary'],25)

freq_df.sort_values('count',ascending=False).head(10).style.background_gradient(cmap = 'plasma')

# Encoding
books_list = list(books['genre'].unique())
encode = [i for i in range(len(books_list))]
mapper = dict(zip(books_list,encode))
print(mapper)

books['genre'] = books['genre'].map(mapper)
books['genre'].unique()

# Model Building¶
## count vectorizer

count_vec = CountVectorizer(max_df=0.90,min_df=2,
                           max_features=1000,stop_words='english')

bagofword_vec = count_vec.fit_transform(books['summary'])
bagofword_vec

test = books['genre']
X_train, X_test, y_train, y_test = train_test_split(bagofword_vec,test,
                                                    test_size=0.2)
X_train.shape,X_test.shape

svc = SVC()
svc.fit(X_train,y_train)
svccpred = svc.predict(X_test)
print(metrics.accuracy_score(y_test,svccpred))

mb = MultinomialNB()
mb.fit(X_train,y_train)
mbpred = mb.predict(X_test)
print(metrics.accuracy_score(y_test,mbpred))

rf = RandomForestClassifier()
rf.fit(X_train,y_train)
print(metrics.accuracy_score(y_test,rf.predict(X_test)))

# Part 2 Model Building¶
# Changing from Countvectorizer to TFDIF vectorizer

#Labeling each 'genre' with an unique number 

from sklearn.preprocessing import LabelEncoder
le = LabelEncoder()
y=le.fit_transform(books['genre'])

X_train,X_test,y_train,y_test = train_test_split(books['summary']
                                                ,y,test_size=0.2,
                                                random_state=557)

X_train.shape,X_test.shape

#Performing tf-idf 

#Performing tf-idf 

tfidf_vectorizer = TfidfVectorizer(max_df=0.8, max_features=10000)
xtrain_tfidf = tfidf_vectorizer.fit_transform(X_train.values.astype('U'))
xtest_tfidf = tfidf_vectorizer.transform(X_test.values.astype('U'))

svc = SVC()
svc.fit(xtrain_tfidf,y_train)
svccpred = svc.predict(xtest_tfidf)
print(metrics.accuracy_score(y_test,svccpred))

mb = MultinomialNB()
mb.fit(xtrain_tfidf,y_train)
mbpred = mb.predict(xtest_tfidf)
print(metrics.accuracy_score(y_test,mbpred))

# Testing the Model

def test(text,model):
    
    text = cleantext(text)
    text = removestopwords(text)
    text = lematizing(text)
    text = stemming(text)
    
    text_vector = tfidf_vectorizer.transform([text])
    predicted = model.predict(text_vector)
    return predicted



ans = books['summary'].apply(lambda text:test(text,mb))

# printing the 
# print(list(mapper.keys())[list(mapper.values()).index(ans)])

ans

predicted_genres = []
for i in range(len(ans)):
    
    index_val = ans[i][0]
    predicted_genres.append(list(mapper.keys())[list(mapper.values()).index(index_val)])
    
mapper

## mapping the training genre as well

newmap = dict([(value,key) for key,value in mapper.items()])
newmap

print(newmap)

books['Actual Genre'] = books['genre'].map(newmap)
books['Predicted_genre'] = np.array(predicted_genres)
books.head()

books = books[['book_name','summary','Actual Genre','Predicted_genre']]
books

dict(Counter(books['Actual Genre'].values))
dict(Counter(books['Predicted_genre'].values))

# Visualize predicted genre

sn.countplot(x = books['Predicted_genre'])
plt.xticks(rotation = 45)

# Visualize  Actual genre

sn.countplot(x = books['Actual Genre'])
plt.xticks(rotation = 45)

# saving the model

import pickle
file = open('bookgenremodel.pkl','wb')
pickle.dump(mb,file)
file.close()

books['summary'].iloc[1]

tfidf_vectorizer

file = open('tfdifvector.pkl','wb')
pickle.dump(tfidf_vectorizer,file)
file.close()


