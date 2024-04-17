from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB, GaussianNB
from sklearn.model_selection import train_test_split
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

classifier = MultinomialNB()
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of naif Bayesian with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



classifier = GaussianNB()
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets]).toarray()
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of Gaussian naif Bayesian with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))
