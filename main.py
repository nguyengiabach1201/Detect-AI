# Load the human-written and machine-generated text data
data = pd.read_csv("/kaggle/input/llm-detect-ai-generated-text/train_essays.csv")

# Extract linguistic features from the text
vectorizer = TfidfVectorizer()
features = vectorizer.fit_transform(data["text"])

# Label the data as human-written or machine-generated
labels = data["prompt_id"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2)

# Train an SVM model to classify the text
model = SVC()
model.fit(X_train, y_train)

# Evaluate the model's performance
predictions = model.predict(X_test)
accuracy = accuracy_score(y_test, predictions)
print("Accuracy:", accuracy)

# Create a function to classify new text
def classify_text(text):
    features_ = vectorizer.transform([text])
    prediction = model.predict(features_)
    return prediction[0]

# Classify a sample text as human-written or machine-generated
text = "CCC ddd eee."
prediction = classify_text(text)
print("Prediction:", prediction)
