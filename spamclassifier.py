import pandas as pa
sms = pa.read_csv('SMSSpam collection/SMSSpamCollection', sep = '\t', names=['label','message'])

import re
import nltk
# nltk.download('stopwords')

from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

wd = WordNetLemmatizer()
corpus = []
for i in range(len(sms)):
    review = re.sub('[^a-zA-Z]',' ', sms['message'][i])
    review = review.lower()
    review = review.split()
    review = [wd.lemmatize(word) for word in review if not word in set(stopwords.words('english'))]
    review = ' '.join(review)
    corpus.append(review)

from sklearn.feature_extraction.text import TfidfVectorizer
cv = TfidfVectorizer(max_features = 3000)
x = cv.fit_transform(corpus).toarray()

y = pa.get_dummies(sms['label'])
y = y.iloc[:,1].values

from sklearn.model_selection import train_test_split
x_train, x_test, y_train, y_test = train_test_split(x,y,test_size = 0.20, random_state = 0)

from sklearn.naive_bayes import MultinomialNB
spam_detect_model = MultinomialNB().fit(x_train, y_train)

y_pred = spam_detect_model.predict(x_test)

from sklearn.metrics import confusion_matrix
confusion_m = confusion_matrix(y_test, y_pred)

from sklearn.metrics import accuracy_score
accuracy = accuracy_score(y_test, y_pred)
print(accuracy)

