import face_recognition #module to recognize face
import cv2 # opencv module
import speech_recognition as sr #module to interact with the device via speech
import pyaudio #needed for speech_recognition module above, big pain to install the library
import pyttsx3 #if text to speech capabilities are needed
cv2.destroyAllWindows() #in case any windows are already open
import numpy as np

#this opens a sample image of the owner to recognize and interact with him/her later on
original_image = face_recognition.load_image_file('/home/pi/1.jpg')
original_image = cv2.resize(original_image, (1000, 1000)).reshape(1000, 1000, 3)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

#encodings of the face stored as an array
original_encodings = face_recognition.face_encodings(original_image)[0]

from tensorflow import keras

#warnings are a real bother XD
from warnings import filterwarnings as fw
fw('ignore')

model = keras.models.load_model('chat.h5')

import json
import random

#for language processing
import nltk
from nltk import word_tokenize

#used lemmatization here, go ahead if you want any other
from nltk.stem import WordNetLemmatizer as WNL
lemmatizer = WNL()

#to automate lights
from gpiozero import LED
led = LED(27)
led.on()

#intents.json is the main file
with open('intents.json', 'r') as f:
    intents = json.load(f)

#all words
with open('words.json', 'r') as f:
    words = json.load(f)

with open('classes.json', 'r') as f:
    classes = json.load(f)

#cascade classifier
face_classifier = cv2.CascadeClassifier('/home/pi/Downloads/haarcascade_frontalface_default.xml')

#returns face seen in the haar cascade classifier with a box around it
def maker(gray, frame):
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
    
    return frame, faces

#returns result and distance if the face is that of the owner.
def recognize_face(frame):
    faces = face_classifier.detectMultiScale(frame, 1.3, 5)
    
    if faces == ():
        return None, None
    
    else:
        new_encodings = face_recognition.face_encodings(frame)[0]
        result = face_recognition.compare_faces([original_encodings], new_encodings)
        dis = face_recognition.face_distance([original_encodings], new_encodings)
        return result, dis

#to word tokenize the sentence
def tokenize(sentence):
    return word_tokenize(sentence)

#to clean the sentence, that is to word tokenize it and lemmatize the words herein
def clean_up_sentence(sentence):
    sentence_words = tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#yeah, I have used the bag-of-words technique here. Bit primitive, but works well!
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#function to predict the class of the sentence
def predict_class(sentence):
    bow = bag_of_words(sentence)
    res = model.predict(np.array([bow]))[0]
    Error_threshold = 0.25
    results = [[i, r] for i, r in enumerate(res) if r > Error_threshold]
    
    results.sort(key = lambda x: x[1], reverse = True)
    return_list = []
    
    for r in results:
        return_list.append({'intent':classes[r[0]], 'probability': str(r[1])})
        
    return return_list

#this is the main function to get response from the bot
def get_response(intents_list, intents_json):
    tag = intents_list[0]['intent']
    list_of_intents = intents_json['intents']
    if tag == 'goodbye':
        result = random.choice(intents['intents'][1]['responses'])
    else:
        if tag == 'instruction_on' or tag=='instruction_off':
            if tag == 'instruction_on':
                led.off()
                result = 'LED turned on.'
            else:
                led.on()
                result = 'LED turned off.'
        else:
            for i in list_of_intents:
                if i['tag'] == tag:
                    result = random.choice(i['responses'])
    return tag, result

#this is an optional function in case voice is to be given to the bot
def talk(command):
    engine = pyttsx3.init()
    engine.say(command)
    engine.runAndWait()

video_capture = cv2.VideoCapture(0)
flag = 0

#main loop to start the camera. Press 'p' to recognize the face.
while True:
    
    showing, image =  video_capture.read()
    # To keep the video capture on since this was giving an error
    # earlier, becuase video_capture was already intialized.
    if showing == False:
        continue
    
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    canvas, faces = maker(gray, image)
    
    canvas = cv2.resize(canvas, (500, 500)).reshape(500, 500, 3)
    cv2.imshow('Hello.', canvas)
    
    #result, distance = recognize_face(frame)
    
    
    if cv2.waitKey(1) & 0xFF == ord('p'):
        
        frame = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        result, dis = recognize_face(frame)
        
        if result == None and dis == None:
            print("No face detected.")
            break
        
        if result[0] == True:
            print(f'Yashwardhan: {result[0]}, distance: {dis[0]:.4f}')
            print('Hello <Owner>.') #You can change the <owner> to the name of any person you want.
            flag = 1
            break
        
        elif result[0] == False:
            print("You are not Yashwardhan.")
            break
        
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
video_capture.release()
listener = sr.Recognizer()

#if the owner was recognized, the chatbot gets activated!
if flag==1:
    
    bot = str(input('Enter bot name please: '))
    
    while True:
        try:
            with sr.Microphone() as source:
                voice = listener.listen(source)
                command = listener.recognize_google(voice)
                ints = predict_class(command)
                tag, res = get_response(ints, intents)
                print(f"You: {command}")
                talk(res)
                if tag == 'goodbye':
                    print(f'{bot}: {res}')
                    break
                else:
                    print(f'{bot}: {res}')
        except sr.UnknownValueError:
            print(f"{bot}: Please speak again.")
        
# well, if the owner's not in the frame, it's a goodbye XD
elif flag==0:
    print('Goodbye, have a nice day.')
