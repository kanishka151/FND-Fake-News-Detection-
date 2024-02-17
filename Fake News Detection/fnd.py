import pandas as pd 
import pickle


#TQDM
from tqdm import tqdm 

#REGTEX
import re 

#NLTK
import nltk 
nltk.download('punkt') 
nltk.download('stopwords') 
from nltk.corpus import stopwords 
from nltk.tokenize import word_tokenize 
from nltk.stem.porter import PorterStemmer 



#SKLEARN 
from sklearn.model_selection import train_test_split 
from sklearn.linear_model import LogisticRegression 
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_extraction.text import CountVectorizer 
#Visualizing data from csv file using dataframe of pandas
data = pd.read_csv("C:\\Users\\smyl2\\Downloads\\News.csv") 
data = data.drop(["title", "subject","date"], axis = 1)
data.isnull().sum()
data=data.dropna()

# Randomizing and Reindexing 
data = data.sample(frac=1) 
data.reset_index(inplace=True) 
data.drop(["index"], axis=1, inplace=True) 

#***************************Dataframe ready*************************

#Preprocessing function
def preprocess_text(text_data): 
	preprocessed_text = [] 
	
	for sentence in tqdm(text_data): 

		#FOR 209 line error
		if not isinstance(sentence, str):
			sentence=str(sentence)

		#keeps the spaces
		sentence = re.sub(r'[^\w\s]', '', sentence) 

		preprocessed_text.append(' '.join(token.lower() for token in str(sentence).split() if token not in stopwords.words('english'))) 

	return preprocessed_text




#Top 20 frequent words
def get_top_n_words(corpus, n=None): 
	vec = CountVectorizer().fit(corpus) 
	bag_of_words = vec.transform(corpus) 
	sum_words = bag_of_words.sum(axis=0) 
	words_freq = [(word, sum_words[0, idx]) 
				for word, idx in vec.vocabulary_.items()] 
	words_freq = sorted(words_freq, key=lambda x: x[1], 
						reverse=True) 
	return words_freq[:n] 


common_words = get_top_n_words(data['text'], 20) 




#***********************Preprocessing done*******************************





#Splitting data into training and testing
x_train, x_test, y_train, y_test = train_test_split(data['text'], data['class'])


#Vectorizing the training data 
vectorization = TfidfVectorizer() 
x_train = vectorization.fit_transform(x_train) 
x_test = vectorization.transform(x_test)

#Changing to numpy for resizing
x_train=x_train.to_numpy()

#Fitting model on trained data
model = LogisticRegression() 
model.fit(x_train.reshape(-1,1), y_train) 

#Cross verification of Model on previous Data 
model.predict(x_train)
model.predict(x_test)

#Saving this trained model in fnd_model.
pickle.dump(model, open('pac.pkl', 'wb'))









