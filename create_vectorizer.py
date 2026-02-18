import pickle
from sklearn.feature_extraction.text import TfidfVectorizer

# Load the saved model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

# Load the saved vectorizer
with open('vectorizer.pkl', 'rb') as vectorizer_file:
    vectorizer = pickle.load(vectorizer_file)

# New text input that you want to predict
new_text = ["This is a sample news headline"]

# Transform the new text into features using the vectorizer
input_features = vectorizer.transform(new_text)

# Check the shape of the transformed features to make sure they have the correct number of features
print(f"Shape of input features: {input_features.shape}")

# Make a prediction using the model
prediction = model.predict(input_features)

# Output the prediction
print("Prediction:", prediction)
