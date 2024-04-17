from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
from sklearn.linear_model import Perceptron
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report



import pandas as pd
from sklearn.neural_network import MLPClassifier

from Preprocessor import Preprocessor

df = pd.read_csv("offenseval-training-v1.tsv", sep="\t")
tweets = df['tweet'].values
label = df['subtask_a'].apply(lambda x: 1 if x == 'OFF' else 0).values

vec_Count = CountVectorizer()
vec_tfidf = TfidfVectorizer()



preprocessor = Preprocessor('remove_stopwords', 'stem')
cleanTweets = Preprocessor.clean(preprocessor, tweets)



classifier = Perceptron(random_state=30)
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of Perceptron with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



classifier = MLPClassifier(hidden_layer_sizes=(30, 20), activation='relu', random_state=30)
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of MLP with 2 layers: 30, 20 and relu as activation function  with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))



classifier = MLPClassifier(hidden_layer_sizes=(100, 50, 20), activation='relu', random_state=30)
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of MLP with 3 layers: 100, 50, 20 and relu as activation function with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




classifier = MLPClassifier(hidden_layer_sizes=(500, 100), activation='relu', random_state=30)
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of MLP with 2 layers: 500, 100 and relu as activation function with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))




classifier = MLPClassifier(hidden_layer_sizes=(30, 20), activation='logistic', random_state=30)
X = vec_tfidf.fit_transform([' '.join(inner_list) for inner_list in cleanTweets])
X_train, X_test, y_train, y_test = train_test_split(X, label, test_size=0.3, random_state=30)
classifier.fit(X_train, y_train)
y_pred = classifier.predict(X_test)

print("Accuracy of MLP with 2 layers: 30, 20 and logistic as activation function with tokenization, stopword removal, stem and TFIDF:", accuracy_score(y_test, y_pred))
print("\nClassification Report:")
print(classification_report(y_test, y_pred))











