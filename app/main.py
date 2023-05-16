from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
import re
from sklearn.feature_extraction.text import TfidfVectorizer
import pickle
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import numpy as np


class code(BaseModel):
    codereview : str

app = FastAPI()


@app.post('/review')
def get_code(code : code):
    def concat_reviews(my_string):
        review_list = my_string.splitlines()[0:10]
        while '' in review_list:
            review_list.remove('')
        review_list = ' '.join(review_list)
        return review_list


    stopwords_list = set(stopwords.words("english"))

    # We remove negation words in list of stopwords
    no_stopwords = ["not","don't",'aren','don','ain',"aren't", 'couldn', "couldn't", 'didn', "didn't", 'doesn', "doesn't", 'hadn', "hadn't", 'hasn', "hasn't", 'haven', "haven't", 'isn', "isn't",
                'ma', 'mightn', "mightn't", 'mustn', "mustn't", 'needn', "needn't", 'shan', "shan't", 'shouldn', "shouldn't", 'wasn', "wasn't", 'weren', "weren't",
                "won't", 'wouldn', "wouldn't"]
    for no_stopword in no_stopwords:
        stopwords_list.remove(no_stopword)

    lemmatizer = WordNetLemmatizer()

    # function that receive a list of words and do lemmatization:
    def lemma_stem_text(words_list):
        # Lemmatizer
        text = [lemmatizer.lemmatize(token.lower()) for token in words_list]# eighties->eight or messages->message or drugs->drug
        text = [lemmatizer.lemmatize(token.lower(), "v") for token in text]# going-> go or started->start or watching->watch
        return text

    re_negation = re.compile("n't ")

    def negation_abbreviated_to_standard(sent):
        sent = re_negation.sub(" not ", sent)
        return sent


    def review_to_words(raw_review):
        
        
        # 2. Transform abbreviated negations to the standard form.
        review_text = negation_abbreviated_to_standard(raw_review)
        
        # 3. Remove non-letters and non-numbers   
        letters_numbers_only = re.sub("[^a-zA-Z_0-9]", " ", review_text) 
        
        # 4. Convert to lower case and split into individual words (tokenization)
        words = np.char.lower(letters_numbers_only.split())                             
        
        # 5. Remove stop words
        meaningful_words = [w for w in words if not w in stopwords_list]   
        
        # 6. Apply lemmatization function
        lemma_words = lemma_stem_text(meaningful_words)
        
        # 7. Join the words back into one string separated by space, and return the result.
        return( " ".join(lemma_words))  

    
    review = code.codereview
    
    def preprocess_text(review):
        review = concat_reviews(review)
        cleaned_review = review_to_words(review)
        return cleaned_review


    cleaned_review = preprocess_text(code.codereview)
    vectorizer = pickle.load(open('/code/app/tfidf.pkl','rb'))

    feature_vector_review = vectorizer.transform([cleaned_review])
    feature_vector_review.shape

    model = pickle.load(open('/code/app/model.pkl','rb'))
    predicted = model.predict(feature_vector_review)

    if predicted == 0 :
        print('Negative')
        return False
    else:
        print('Postive')
        return True

