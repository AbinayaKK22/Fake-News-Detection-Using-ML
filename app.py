import joblib
from flask import Flask, render_template, request

model = joblib.load('random_forest_model.pkl')
vectorizer = joblib.load('vectorizer.pkl')

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def predict():
    prediction = None
    if request.method == 'POST':
        
        news_text = request.form['news']
        if news_text:
            
            news_vectorized = vectorizer.transform([news_text])
            
            prediction = model.predict(news_vectorized)
            
            prediction = "Fake News" if prediction == 0 else "Real News"
        else:
            prediction = "No news text provided. Please enter some text."

    return render_template('index.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
