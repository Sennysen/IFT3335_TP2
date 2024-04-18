#IFT3335
#TP2
#Classified with SVM with linear kernel, with pretreatment tests

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd

from Preprocessor import Preprocessor

df = pd.read_csv("offenseval-training-v1.tsv", sep="\t")
tweets = df['tweet'].values
label = df['subtask_a'].apply(lambda x: 1 if x == 'OFF' else 0).values

classifier = SVC(kernel='linear')
vec_Count = CountVectorizer()
vec_tfidf = TfidfVectorizer()




# non pretreatement, with count and TFIDF
X = vec_Count.fit_transform(tweets)
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluation
print("Accuracy of SVM linear with count:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



X = vec_tfidf.fit_transform(tweets)
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

# Evaluation
print("Accuracy of SVM linear with TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))









# lemmatization, with count and TFIDF

preprocessor = Preprocessor('remove_stopwords', 'lemmatize')
cleanTweets = Preprocessor.clean(preprocessor, tweets)

X = vec_Count.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of SVM linear with tokenization, stopword removal, lemmatization and count:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of SVM linear with tokenization, stopword removal, lemmatization and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


















# stemming, with count and TFIDF

preprocessor = Preprocessor('remove_stopwords', 'stem')
cleanTweets = Preprocessor.clean(preprocessor, tweets)



X = vec_Count.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of SVM linear with tokenization, stopword removal, stem and count:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of SVM linear with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))













# only stop word removal, with count and TFIDF


preprocessor = Preprocessor('remove_stopwords')
cleanTweets = Preprocessor.clean(preprocessor, tweets)


X = vec_Count.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of SVM linear with tokenization, stopword removal and count:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of SVM linear with tokenization, stopword removal and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))