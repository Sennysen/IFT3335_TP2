from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report



import pandas as pd

from Preprocessor import Preprocessor

df = pd.read_csv("offenseval-training-v1.tsv", sep="\t")
tweets = df['tweet'].values
label = df['subtask_a'].apply(lambda x: 1 if x == 'OFF' else 0).values

vec_Count = CountVectorizer()
vec_tfidf = TfidfVectorizer()



preprocessor = Preprocessor('remove_stopwords', 'stem')
cleanTweets = Preprocessor.clean(preprocessor, tweets)



classifier = RandomForestClassifier()
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of random forest with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))



classifier = RandomForestClassifier(max_depth=5, random_state=30)
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of random forest (max depth 5) with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))




classifier = RandomForestClassifier(n_estimators=100, random_state=30)
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of random forest (100 trees) with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))





classifier = RandomForestClassifier(n_estimators=50, random_state=30)
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of random forest (50 trees) with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division=0))


classifier = RandomForestClassifier(max_depth=5, n_estimators=100, random_state=30)
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of random forest (100 trees and max_depth = 5) with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred, zero_division="0"))







