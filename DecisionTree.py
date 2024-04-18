#IFT3335
#TP2
#Classified with decision tree

from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score, classification_report

import pandas as pd

from Preprocessor import Preprocessor

df = pd.read_csv("offenseval-training-v1.tsv", sep="\t")
tweets = df['tweet'].values
label = df['subtask_a'].apply(lambda x: 1 if x == 'OFF' else 0).values

vec_Count = CountVectorizer()
vec_tfidf = TfidfVectorizer()


#pretraitement
preprocessor = Preprocessor('remove_stopwords', 'stem')
cleanTweets = Preprocessor.clean(preprocessor, tweets)


#basic classifier
classifier = DecisionTreeClassifier()
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of DECISION TREE with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




#Decision tree with max depth
classifier = DecisionTreeClassifier(max_depth=3)
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of DECISION TREE (max depth = 3) with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




#Decision tree with gini
classifier = DecisionTreeClassifier(criterion='gini')
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of DECISION TREE gini criteria with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))


#Decision tree with entropy
classifier = DecisionTreeClassifier(criterion='entropy')
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of DECISION TREE entropy criteria with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))







