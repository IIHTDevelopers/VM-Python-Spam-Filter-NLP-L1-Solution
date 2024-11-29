import pandas as pd
import re
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import accuracy_score, confusion_matrix

# Step 1: Load Dataset
def load_dataset():
    data = {
        "Text": [
            "Free entry in a contest to win $1000!",
            "Hey, can we meet for lunch tomorrow?",
            "Congratulations! You've been selected for a prize.",
            "Your account needs immediate verification.",
            "Call me back when you are free.",
            "Exclusive deal just for you, click the link!",
            "Are we still on for dinner tonight?",
            "Win a brand new car by entering this contest now!",
            "Hello, hope you are doing well.",
            "Urgent: Update your details to avoid suspension.",
        ],
        "Label": ["spam", "ham", "spam", "spam", "ham", "spam", "ham", "spam", "ham", "spam"],
    }
    df = pd.DataFrame(data)
    return df

# Step 2: Preprocess Text
def preprocess_text(text):
    text = re.sub(r"[^a-zA-Z\s]", "", text)  # Remove non-alphabetic characters
    text = text.lower()  # Convert to lowercase
    return text

def preprocess_dataset(df):
    df["Processed_Text"] = df["Text"].apply(preprocess_text)
    return df

# Step 3: Split Dataset
def split_dataset(df):
    X = df["Processed_Text"]
    y = df["Label"]
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)
    return X_train, X_test, y_train, y_test

# Step 4: Feature Extraction
def extract_features(X_train, X_test):
    vectorizer = CountVectorizer()
    X_train_vectors = vectorizer.fit_transform(X_train)
    X_test_vectors = vectorizer.transform(X_test)
    return X_train_vectors, X_test_vectors, vectorizer

# Step 5: Train Model
def train_model(X_train_vectors, y_train):
    model = MultinomialNB()
    model.fit(X_train_vectors, y_train)
    return model

# Step 6: Evaluate Model
def evaluate_model(model, X_test_vectors, y_test):
    y_pred = model.predict(X_test_vectors)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    correct_predictions = cm[0, 0] + cm[1, 1]
    incorrect_predictions = cm[0, 1] + cm[1, 0]
    return accuracy, correct_predictions, incorrect_predictions

# Analytical Functions
def get_total_messages(df):
    return len(df)

def get_spam_count(df):
    return len(df[df["Label"] == "spam"])

def get_ham_count(df):
    return len(df[df["Label"] == "ham"])

def get_spam_ratio(df):
    spam_count = get_spam_count(df)
    total_messages = get_total_messages(df)
    return (spam_count / total_messages) * 100

def get_ham_ratio(df):
    ham_count = get_ham_count(df)
    total_messages = get_total_messages(df)
    return (ham_count / total_messages) * 100

# Main Workflow
if __name__ == "__main__":
    # Load and preprocess dataset
    df = load_dataset()
    df = preprocess_dataset(df)

    # Analytical Question 1: Total number of messages
    total_messages = get_total_messages(df)
    print(f"1. Total number of messages: {total_messages}")

    # Analytical Question 2: Number of spam messages
    spam_count = get_spam_count(df)
    print(f"2. Number of spam messages: {spam_count}")

    # Analytical Question 3: Number of ham messages
    ham_count = get_ham_count(df)
    print(f"3. Number of ham messages: {ham_count}")

    # Analytical Question 4: Spam message ratio (percentage)
    spam_ratio = get_spam_ratio(df)
    print(f"4. Spam message ratio (%): {spam_ratio:.2f}")

    # Analytical Question 5: Ham message ratio (percentage)
    ham_ratio = get_ham_ratio(df)
    print(f"5. Ham message ratio (%): {ham_ratio:.2f}")

    # Split dataset and extract features
    X_train, X_test, y_train, y_test = split_dataset(df)
    X_train_vectors, X_test_vectors, vectorizer = extract_features(X_train, X_test)

    # Train and evaluate model
    model = train_model(X_train_vectors, y_train)
    accuracy, correct_predictions, incorrect_predictions = evaluate_model(model, X_test_vectors, y_test)

    # Analytical Question 6: Accuracy of the classifier
    print(f"6. Accuracy of the classifier: {accuracy:.2f}")

    # Analytical Question 7: Number of correctly classified messages
    print(f"7. Number of correctly classified messages: {correct_predictions}")

    # Analytical Question 8: Number of incorrectly classified messages
    print(f"8. Number of incorrectly classified messages: {incorrect_predictions}")
