import random
import os
from flask import Flask, request, jsonify
import matplotlib.pyplot as plt
import librosa
import numpy as np
from keras.models import Sequential, Model, model_from_json
import keras 
import pickle
import os
import pandas as pd
import sys
import warnings
import librosa.display
import IPython.display as ipd



# instantiate flask app
app = Flask(__name__)

def predictOut(file_path):
       

        json_file = open('./model/model_json.json', 'r')
        loaded_model_json = json_file.read()
        json_file.close()
        loaded_model = model_from_json(loaded_model_json)

# load weights into new model
        loaded_model.load_weights("./model/Emotion_Model.h5")
        print("Loaded model from disk")

# the optimiser
        opt = keras.optimizers.RMSprop(lr=0.00001, decay=1e-6)
        loaded_model.compile(loss='categorical_crossentropy', optimizer=opt, metrics=['accuracy'])
        print("Loaded model from disk - 1")
        #preprocessing
        newdf = preprocess(file_path)
        print("Loaded model from disk - preprocessing")
        # Apply predictions
        newdf= np.expand_dims(newdf, axis=2)
        newpred = loaded_model.predict(newdf, 
                         batch_size=16, 
                         verbose=1)
        print("Loaded model from disk - before labels")
        filename = './labels'
        infile = open(filename,'rb')
        lb = pickle.load(infile)
        infile.close()
        print("Loaded model from disk - labels")
        # Get the final predicted label
        final = newpred.argmax(axis=1)
        final = final.astype(int).flatten()
        final = (lb.inverse_transform((final)))
        print("Loaded model from disk - return", final)
        return final


def preprocess(file_path):
       
        # Lets transform the dataset so we can apply the predictions
        X, sample_rate = librosa.load(file_path
                              ,res_type='kaiser_fast'
                              ,duration=2.5
                              ,sr=44100
                              ,offset=0.5
                             )

        sample_rate = np.array(sample_rate)
        mfccs = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=13),axis=0)
        newdf = pd.DataFrame(data=mfccs).T
        return newdf

@app.route("/predict", methods=["POST"])
def predict():
	

	# get file from POST request and save it
	audio_file = request.files["file"]
	file_name = str(random.randint(0, 100000))
	audio_file.save(file_name)

	# instantiate keyword spotting service singleton and get prediction
	# kss = Keyword_Spotting_Service()
	predicted_keyword = predictOut(file_name)

	# we don't need the audio file any more - let's delete it!
	os.remove(file_name)

	# send back result as a json file
	result = {"keyword": predicted_keyword[0]}
	return jsonify(result)


if __name__ == "__main__":
    app.run(debug=False)