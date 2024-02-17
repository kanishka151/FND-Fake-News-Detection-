import pickle
from flask import Flask, request, render_template

app = Flask(__name__)

#Loading saved model
model = pickle.load(open('pac.pkl', 'rb'))
tfidf = pickle.load(open('tfidf_vectorizer.pkl','rb'))


@app.route('/')
def home():
	return render_template('index.html')


@app.route('/detect',methods=['POST'])
def predict():
    if request.method == 'POST':
         
    	news = request.form['news']
    	data = [news]
    	vect = tfidf.transform(data)
         
		#Predicting true/false
    	my_prediction = model.predict(vect)
    	return render_template('result.html', result=my_prediction)

if __name__ == '__main__':
    app.run(debug=True)
