import pandas as pd
import numpy as np
import re
import string
import joblib
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from imblearn.over_sampling import SMOTE
import nltk
from sklearn.metrics import classification_report
from flask import Flask, request, render_template_string


nltk.download('stopwords')
nltk.download('wordnet')

# Load Dataset
data = pd.read_csv("SpamDetection/SMSSpamCollection", sep='\t', header=None, names=['label', 'message'])
data['label'] = data['label'].map({'ham': 0, 'spam': 1})

# Important spam-related words to reinforce
important_spam_words = {"win", "click", "free", "account", "urgent", "claim", "package", "suspended", "confirm", "bank", "renew", "details", "verify", "security", "payment", "prize", "lottery", "reward", "credit", "fix", "congratulations", "limited", "paypal"}
custom_stopwords = set(stopwords.words('english')) - important_spam_words

def clean_text(text):
    text = text.lower()
    text = re.sub(f"[{string.punctuation}]", "", text)
    text = re.sub(r"\d+", "", text)  # Remove numbers
    words = text.split()
    words = [word for word in words if word not in custom_stopwords]
    lemmatizer = WordNetLemmatizer()
    words = [lemmatizer.lemmatize(word) for word in words if word]  # Ensure non-empty words
    # Reinforce important spam words
    words.extend([word for word in important_spam_words if word in text])
    return " ".join(words)

data['clean_message'] = data['message'].apply(clean_text)

# Train-Test Split
X_train, X_test, y_train, y_test = train_test_split(
    data['clean_message'], data['label'], test_size=0.2, random_state=42, stratify=data['label']
)

# TF-IDF Vectorization with Improved Parameters
vectorizer = TfidfVectorizer(ngram_range=(1, 5), sublinear_tf=True, max_features=20000, min_df=2, max_df=0.9)
X_train_vec = vectorizer.fit_transform(X_train)
X_test_vec = vectorizer.transform(X_test)

# Handle Imbalance using SMOTE
smote = SMOTE(random_state=42)
X_train_resampled, y_train_resampled = smote.fit_resample(X_train_vec, y_train)

# Train Multinomial Naïve Bayes with Adjusted Alpha
model = MultinomialNB(alpha=0.01)
model.fit(X_train_resampled, y_train_resampled)

# Model Evaluation
y_pred = model.predict(X_test_vec)

# Added Classification Report
print("\nClassification Report:\n", classification_report(y_test, y_pred))

# Save Model and Vectorizer
joblib.dump(model, "SpamDetection/spam_classifier.pkl")
joblib.dump(vectorizer, "SpamDetection/vectorizer.pkl")

def predict_spam(user_input):
    """Function to predict if a message is spam or not."""
    model = joblib.load("SpamDetection/spam_classifier.pkl")
    vectorizer = joblib.load("SpamDetection/vectorizer.pkl")
    input_vec = vectorizer.transform([clean_text(user_input)])
    prediction = model.predict(input_vec)
    return "Spam" if prediction[0] == 1 else "Not Spam"

#Example Usage
test_messages = [
    "Congratulations! You've won a free iPhone. Click here to claim.",
    "URGENT: Your account has been compromised. Verify your details now.",
    "Good Day, Emin, are you ready for tonight?",
    "WIN 300$ By Clicking on this link!",
    "Dear Javidan, your job application is under consideration",
    "Hurry up, your NFT has been sold for 100$, click the link to receive it!",
]

print("Example Test Messages: \n")
for msg in test_messages:
    print(f"Message: {msg} => {predict_spam(msg)}")

# Simple Web Interface using Flask
app = Flask(__name__)

html_template = '''
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <title>Spam Detection</title>
    <link rel="stylesheet" href="https://stackpath.bootstrapcdn.com/bootstrap/4.5.2/css/bootstrap.min.css">
    <style>
      body {
          background-color: #f8f9fa;
          padding-top: 50px;
      }
      .container {
          max-width: 600px;
      }
      h1 {
          margin-bottom: 30px;
      }
      textarea {
          resize: none;
      }
      .prediction {
          margin-top: 20px;
      }
    </style>
</head>
<body>
    <div class="container">
      <h1 class="text-center">Spam Detection</h1>
      <form method="post" action="/">
          <div class="form-group">
              <textarea class="form-control" name="message" rows="5" placeholder="Enter your message here..."></textarea>
          </div>
          <button type="submit" class="btn btn-primary btn-block">Predict</button>
      </form>
      {% if prediction %}
      <div class="alert alert-info prediction" role="alert">
         <strong>Prediction:</strong> {{ prediction }}
      </div>
      {% endif %}
    </div>
</body>
</html>
'''

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction = None
    if request.method == 'POST':
        message = request.form.get('message')
        if message:
            prediction = predict_spam(message)
    return render_template_string(html_template, prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
