import os
import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import classification_report, confusion_matrix
import requests
import tarfile
import matplotlib.pyplot as plt
import seaborn as sns


#nltk
nltk.download('punkt_tab')  
nltk.download('stopwords')  
nltk.download('wordnet')
nltk.download('omw-1.4')


url = 'https://www.cs.cornell.edu/people/pabo/movie-review-data/rt-polaritydata.tar.gz'
file_name = 'rt-polaritydata.tar.gz'


response = requests.get(url)
with open(file_name, 'wb') as file:
    file.write(response.content)

with tarfile.open(file_name, 'r:gz') as tar:
    tar.extractall()

stop_words = set(stopwords.words('english'))


def load_data(file_path):
    with open(file_path, 'r', encoding='ISO-8859-1') as file:
        return file.readlines()

positive_reviews = load_data('rt-polaritydata/rt-polarity.pos')
negative_reviews = load_data('rt-polaritydata/rt-polarity.neg')


def preprocess_text(text):
    tokens = word_tokenize(text.lower())  
    filtered_tokens = [word for word in tokens if word.isalnum() and word not in stop_words]
    return " ".join(filtered_tokens)

positive_reviews = [preprocess_text(review) for review in positive_reviews]
negative_reviews = [preprocess_text(review) for review in negative_reviews]


pos_df = pd.DataFrame({'text': positive_reviews, 'label': 1})
neg_df = pd.DataFrame({'text': negative_reviews, 'label': 0})


df = pd.concat([pos_df, neg_df], ignore_index=True)


train_df = pd.concat([df.iloc[:4000], df.iloc[5000:9000]])
val_df = pd.concat([df.iloc[4000:4500], df.iloc[9000:9500]])
test_df = pd.concat([df.iloc[4500:5331], df.iloc[9500:10331]])

X_train = train_df['text']
y_train = train_df['label']
X_val = val_df['text']
y_val = val_df['label']
X_test = test_df['text']
y_test = test_df['label']


vectorizer = TfidfVectorizer(stop_words='english')
X_train_vec = vectorizer.fit_transform(X_train)
X_val_vec = vectorizer.transform(X_val)
X_test_vec = vectorizer.transform(X_test)

#Naive Bayes
model = MultinomialNB()
model.fit(X_train_vec, y_train)


val_predictions = model.predict(X_val_vec)
test_predictions = model.predict(X_test_vec)


val_report = classification_report(y_val, val_predictions, output_dict=True)
test_report = classification_report(y_test, test_predictions, output_dict=True)

cm = confusion_matrix(y_test, test_predictions)
tn, fp, fn, tp = cm.ravel()

precision = test_report['1']['precision']
recall = test_report['1']['recall']
f1_score = test_report['1']['f1-score']


print(f"True Positives (TP): {tp}")
print(f"True Negatives (TN): {tn}")
print(f"False Positives (FP): {fp}")
print(f"False Negatives (FN): {fn}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1_score:.2f}")
print(f"Test Set Size: {len(X_test)}")

#plot
report_df = pd.DataFrame(classification_report(y_test, test_predictions, output_dict=True)).transpose()

fig, axes = plt.subplots(1, 2, figsize=(18, 7))

sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
            xticklabels=['Negative', 'Positive'],
            yticklabels=['Negative', 'Positive'], ax=axes[0])
axes[0].set_xlabel('Predicted Labels')
axes[0].set_ylabel('True Labels')
axes[0].set_title('Confusion Matrix')

sns.heatmap(report_df.iloc[:-1, :-1].astype(float), annot=True, cmap='Blues', fmt='.2f', ax=axes[1])
axes[1].set_title('Classification Report')
axes[1].set_xticklabels(axes[1].get_xticklabels(), rotation=45)
axes[1].set_yticklabels(axes[1].get_yticklabels(), rotation=0)

plt.tight_layout()
plt.show()
