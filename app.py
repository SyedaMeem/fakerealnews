##--------Loading all libraries--------------------------------
from flask import Flask,render_template,url_for,request,jsonify,redirect,session
import re
import pickle
#pip install numpy==1.19.3
import numpy as np
import json
import pandas as pd
#pip install tensorflow==2.3.0
import tensorflow as tf

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.models import load_model

import string
import re
import nltk
nltk.download('stopwords')
nltk.download('punkt')
nltk.download('wordnet')

from newspaper import Article
import requests
import urllib

from nltk.tokenize import sent_tokenize
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
wordnet=WordNetLemmatizer()

from bltk.langtools.banglachars import (vowels,vowel_signs,consonants,digits,operators,punctuations,others)
from bltk.langtools import Tokenizer
tokenizerb = Tokenizer()


stopbn=['অতএব', 'অথচ', 'অথবা', 'অনুযায়ী', 'অনেক', 'অনেকে', 'অনেকেই', 'অন্তত', 'অন্য', 'অবধি', 'অবশ্য', 'অর্থাত', 'আই', 'আগামী', 'আগে', 'আগেই', 'আছে', 'আজ', 'আদ্যভাগে', 'আপনার', 'আপনি', 'আবার', 'আমরা', 'আমাকে', 'আমাদের', 'আমার', 'আমি', 'আর', 'আরও', 'ই', 'ইত্যাদি', 'ইহা', 'উচিত', 'উত্তর', 'উনি', 'উপর', 'উপরে', 'এ', 'এঁদের', 'এঁরা', 'এই', 'একই', 'একটি', 'একবার', 'একে', 'এক্', 'এখন', 'এখনও', 'এখানে', 'এখানেই', 'এটা', 'এটাই', 'এটি', 'এত', 'এতটাই', 'এতে', 'এদের', 'এব', 'এবং', 'এবার', 'এমন', 'এমনকী', 'এমনি', 'এর', 'এরা', 'এল', 'এস', 'এসে', 'ঐ', 'ও', 'ওঁদের', 'ওঁর', 'ওঁরা', 'ওই', 'ওকে', 'ওখানে', 'ওদের', 'ওর', 'ওরা', 'কখনও', 'কত', 'কবে', 'কমনে', 'কয়েক', 'কয়েকটি', 'করছে', 'করছেন', 'করতে', 'করবে', 'করবেন', 'করলে', 'করলেন', 'করা', 'করাই', 'করায়', 'করার', 'করি', 'করিতে', 'করিয়া', 'করিয়ে', 'করে', 'করেই', 'করেছিলেন', 'করেছে', 'করেছেন', 'করেন', 'কাউকে', 'কাছ', 'কাছে', 'কাজ', 'কাজে', 'কারও', 'কারণ', 'কি', 'কিংবা', 'কিছু', 'কিছুই', 'কিন্তু', 'কী', 'কে', 'কেউ', 'কেউই', 'কেখা', 'কেন', 'কোটি', 'কোন', 'কোনও', 'কোনো', 'ক্ষেত্রে', 'কয়েক', 'খুব', 'গিয়ে', 'গিয়েছে', 'গিয়ে', 'গুলি', 'গেছে', 'গেল', 'গেলে', 'গোটা', 'চলে', 'চান', 'চায়', 'চার', 'চালু', 'চেয়ে', 'চেষ্টা', 'ছাড়া', 'ছাড়াও', 'ছিল', 'ছিলেন', 'জন', 'জনকে', 'জনের', 'জন্য', 'জন্যওজে', 'জানতে', 'জানা', 'জানানো', 'জানায়', 'জানিয়ে', 'জানিয়েছে', 'জে', 'জ্নজন', 'টি', 'ঠিক', 'তখন', 'তত', 'তথা', 'তবু', 'তবে', 'তা', 'তাঁকে', 'তাঁদের', 'তাঁর', 'তাঁরা', 'তাঁাহারা', 'তাই', 'তাও', 'তাকে', 'তাতে', 'তাদের', 'তার', 'তারপর', 'তারা', 'তারৈ', 'তাহলে', 'তাহা', 'তাহাতে', 'তাহার', 'তিনঐ', 'তিনি', 'তিনিও', 'তুমি', 'তুলে', 'তেমন', 'তো', 'তোমার', 'থাকবে', 'থাকবেন', 'থাকা', 'থাকায়', 'থাকে', 'থাকেন', 'থেকে', 'থেকেই', 'থেকেও', 'দিকে', 'দিতে', 'দিন', 'দিয়ে', 'দিয়েছে', 'দিয়েছেন', 'দিলেন', 'দু', 'দুই', 'দুটি', 'দুটো', 'দেওয়া', 'দেওয়ার', 'দেওয়া', 'দেখতে', 'দেখা', 'দেখে', 'দেন', 'দেয়', 'দ্বারা', 'ধরা', 'ধরে', 'ধামার', 'নতুন', 'নয়', 'না', 'নাই', 'নাকি', 'নাগাদ', 'নানা', 'নিজে', 'নিজেই', 'নিজেদের', 'নিজের', 'নিতে', 'নিয়ে', 'নিয়ে', 'নেই', 'নেওয়া', 'নেওয়ার', 'নেওয়া', 'নয়', 'পক্ষে', 'পর', 'পরে', 'পরেই', 'পরেও', 'পর্যন্ত', 'পাওয়া', 'পাচ', 'পারি', 'পারে', 'পারেন', 'পি', 'পেয়ে', 'পেয়্র্', 'প্রতি', 'প্রথম', 'প্রভৃতি', 'প্রযন্ত', 'প্রাথমিক', 'প্রায়', 'প্রায়', 'ফলে', 'ফিরে', 'ফের', 'বক্তব্য', 'বদলে', 'বন', 'বরং', 'বলতে', 'বলল', 'বললেন', 'বলা', 'বলে', 'বলেছেন', 'বলেন', 'বসে', 'বহু', 'বা', 'বাদে', 'বার', 'বি', 'বিনা', 'বিভিন্ন', 'বিশেষ', 'বিষয়টি', 'বেশ', 'বেশি', 'ব্যবহার', 'ব্যাপারে', 'ভাবে', 'ভাবেই', 'মতো', 'মতোই', 'মধ্যভাগে', 'মধ্যে', 'মধ্যেই', 'মধ্যেও', 'মনে', 'মাত্র', 'মাধ্যমে', 'মোট', 'মোটেই', 'যখন', 'যত', 'যতটা', 'যথেষ্ট', 'যদি', 'যদিও', 'যা', 'যাঁর', 'যাঁরা', 'যাওয়া', 'যাওয়ার', 'যাওয়া', 'যাকে', 'যাচ্ছে', 'যাতে', 'যাদের', 'যান', 'যাবে', 'যায়', 'যার', 'যারা', 'যিনি', 'যে', 'যেখানে', 'যেতে', 'যেন', 'যেমন', 'র', 'রকম', 'রয়েছে', 'রাখা', 'রেখে', 'লক্ষ', 'শুধু', 'শুরু', 'সঙ্গে', 'সঙ্গেও', 'সব', 'সবার', 'সমস্ত', 'সম্প্রতি', 'সহ', 'সহিত', 'সাধারণ', 'সামনে', 'সি', 'সুতরাং', 'সে', 'সেই', 'সেখান', 'সেখানে', 'সেটা', 'সেটাই', 'সেটাও', 'সেটি', 'স্পষ্ট', 'স্বয়ং', 'হইতে', 'হইবে', 'হইয়া', 'হওয়া', 'হওয়ায়', 'হওয়ার', 'হচ্ছে', 'হত', 'হতে', 'হতেই', 'হন', 'হবে', 'হবেন', 'হয়', 'হয়তো', 'হয়নি', 'হয়ে', 'হয়েই', 'হয়েছিল', 'হয়েছে', 'হয়েছেন', 'হল', 'হলে', 'হলেই', 'হলেও', 'হলো', 'হাজার', 'হিসাবে', 'হৈলে', 'হোক', 'হয়']

from langdetect import detect

import uuid


#----------All feature engineering tasks----------------------------------------

def en_processing(sent):
    corpus1=[]
    txt = re.sub('[^a-zA-Z]', ' ',sent)
    txt = re.sub(r'((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)',' ',txt)
    txt = re.sub('[0-9]+', '', txt)
    txt  = "".join([char for char in txt if char not in string.punctuation])
    txt = ''.join([i for i in txt if not i.isdigit()])
    txt = txt.lower()
    txt = txt.split()

    txt = [wordnet.lemmatize(word) for word in txt if (word not in set(stopwords.words('english')))]

    txt = ' '.join(txt)
    corpus1.append(txt)
    return corpus1

def bn_processing(sent):
    corpus2=[]
    txt = re.sub('[a-zA-Z]+', ' ',sent)
    txt = re.sub(r'((\S+)?(http(s)?)(\S+))|((\S+)?(www)(\S+))|((\S+)?(\@)(\S+)?)',' ',txt)
    txt = re.sub('[0-9]+', '', txt)
    txt = "".join([char for char in txt if char not in string.punctuation])
    txt = "".join([char for char in txt if char not in digits])
    txt = "".join([char for char in txt if char not in operators])
    txt = "".join([char for char in txt if char not in punctuations])

    txt = ''.join([i for i in txt if not i.isdigit()])
    txt = tokenizerb.word_tokenizer(txt)
    txt = [word for word in txt if word not in stopbn]      
    #txt = [stemmer.stem(word) for word in txt if (word not in remove_stopwords(txt,level='moderate'))]

    txt = ' '.join(txt)
    corpus2.append(txt)
    return corpus2

##-----------Connect with mongocloud----------------------------------

from flask_pymongo import pymongo
try:
   CONNECTION_STRING = "mongodb+srv://hearsay:b7o0yttE4gqZLL03@cluster0.hvhca.mongodb.net/<dbname>?retryWrites=true&w=majority"
   client = pymongo.MongoClient(CONNECTION_STRING) 
   db = client.get_database('user_db')
   records = pymongo.collection.Collection(db, 'user_collection')
   print("Successfully connect with database.")
except Exception:
    print("Unable to connect database!")

#-----------Loading model & vocabulary files----------------------------------

print("Loading model...")
model=load_model('fakeh5model.h5',compile=False)
print("Model loaded successfully.")
#model= tf.keras.models.load_model('fake3blstm_bn.h5')
#model._make_predict_function()  # https://github.com/keras-team/keras/issues/6462
print("Loading vocabulary...")
with open('trained_tokenizer.pkl',  'rb') as f:
    tokenizers = pickle.load(f)
print("Vocabulary loaded successfully.")


#---------------FLASK----------------------------------------------------

app = Flask(__name__)

@app.route('/')
@app.route('/index')
def index():
    return render_template("index.html")

@app.route('/predict',methods=['POST'])
def predict():
     if request.method == 'POST':
         url=request.form['url']
         article = Article(url)
         article.download()
         article.parse() # Parse the article
         title=article.title #To extract article’s title
         tx=article.text#To extract article’s text
         if len(tx)==0:
             response = requests.get(url)
             data=response.text
             tx = bn_processing(data)
             tx=str(tx[0])
         sent = title+" "+tx  
         lang=detect(sent)
         if lang=='bn':
             crp=bn_processing(sent)
             encoded_docs = tokenizers.texts_to_sequences(np.array(crp))
             embedded_docs1=pad_sequences(encoded_docs,padding='pre',maxlen=218,truncating='pre')
         elif lang=='en':
             crp=en_processing(sent)
             encoded_docs = tokenizers.texts_to_sequences(np.array(crp))
             embedded_docs1=pad_sequences(encoded_docs,padding='pre',maxlen=218,truncating='pre')                                 
         else:
             raise Exception("cause of the language problem")
            
         predtion = model.predict(embedded_docs1)
         prediction = predtion[0][0]
            
         if(prediction>0.50):
             pred1='Fake'
             a=prediction*100
             ac = float("{:.2f}".format(a))
         else:
             pred1='Real'
             a=prediction*100
             ac = float("{:.2f}".format(a))
                
         mt=int(ac)
         metr=100-mt
            
        ##Insert data to mongodb cloud
         usr={"_id":uuid.uuid4().hex,
             "Link": url,
             "Headline":title,
             "Body":tx,
             "Prediction_fake":ac,
             "Result":pred1}
         records.insert_one(usr)
     return render_template("predict.html",pred=pred1,mtr=metr,acc=prediction,accper=ac)

##About page
@app.route('/about')
def about():
    return render_template("about.html")

if __name__ == "__main__":
   app.run(debug = True)

