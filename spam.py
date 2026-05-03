from xml.parsers.expat import model

import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score  
 

spam_data = pd.read_csv('spam.csv', encoding='latin-1')
spam_data = spam_data[['v1', 'v2']]
spam_data.columns = ['label', 'message']

spam_data['label'] = spam_data['label'].map({'ham': 0, 'spam': 1})

X = spam_data['message']
y = spam_data['label']

X_train, X_test, y_train, y_test = train_test_split(
    spam_data['message'],
    spam_data['label'],
    test_size=0.2,
    random_state=42
)
vectorizer = CountVectorizer()

X_train_vectors = vectorizer.fit_transform(X_train)
X_test_vectors = vectorizer.transform(X_test)


spam_model = MultinomialNB()
spam_model.fit(X_train_vectors, y_train)


predictions = spam_model.predict(X_test_vectors)
print("Accuracy:", accuracy_score(y_test, predictions))

mssegw = ["Congratulations! You've won a free ticket to the Bahamas! Call now to claim your prize."]
mssegw_vector = vectorizer.transform(mssegw)

result = spam_model.predict(mssegw_vector)

if result[0] == 1:
    print("this is a Spam Message")
else:
    print("Not a Spam")