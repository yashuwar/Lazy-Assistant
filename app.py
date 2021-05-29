import face_recognition
import cv2
import numpy as np
from warnings import filterwarnings as fw
fw('ignore')

original_image = face_recognition.load_image_file('1.jpg')
original_image = cv2.resize(original_image, (1000, 1000)).reshape(1000, 1000, 3)
original_image = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

from tensorflow import keras
model = keras.models.load_model('chat.h5')

import json
import random

from nltk import word_tokenize

from nltk.stem import WordNetLemmatizer as WNL
lemmatizer = WNL()

# nltk.download('punkt'); nltk.download('wordnet') - use this in case the punkt and wordnet data hasn't been downloaded yet.

from gpiozero import LED
# gpiozero is a python3 library to interact with the gpio pins present in the pi.
led = LED(27)
# for some reason, led.on() is equivalent to turning off the led connected to the gpio and led.off() is equivalent to turning it on XP.
led.on()

with open('intents.json', 'r') as f:
    intents = json.load(f)

with open('words.json', 'r') as f:
    words = json.load(f)

with open('classes.json', 'r') as f:
    classes = json.load(f)

face_classifier = cv2.CascadeClassifier('haarcascade_frontalface_default.xml')

#this function makes the red box you see when the video is turned on.
def maker(gray, frame):
    faces = face_classifier.detectMultiScale(gray, 1.3, 5)
    for (x,y,w,h) in faces:
        cv2.rectangle(frame, (x,y), (x+w,y+h), (0,0,255), 2)
    
    return frame, faces

# below function recognizes if the owner is present in the frame or not.
original_encodings = face_recognition.face_encodings(original_image)[0]

def recognize_face(frame):
    faces = face_classifier.detectMultiScale(frame, 1.3, 5)
    
    if faces == ():
        return None, None
    
    else:
        new_encodings = face_recognition.face_encodings(frame)[0]
        result = face_recognition.compare_faces([original_encodings], new_encodings)
        dis = face_recognition.face_distance([original_encodings], new_encodings)
        return result, dis

#function to word tokenize sentences.
def tokenize(sentence):
    return word_tokenize(sentence)

#function to tokenize and lemmatize the provided sentence.
def clean_up_sentence(sentence):
    sentence_words = tokenize(sentence)
    sentence_words = [lemmatizer.lemmatize(word) for word in sentence_words]
    return sentence_words

#function to return the bag of words which we have used in this project insted of tfidf, word2vec or any other technique
def bag_of_words(sentence):
    sentence_words = clean_up_sentence(sentence)
    bag = [0] * len(words)
    for w in sentence_words:
        for i, word in enumerate(words):
            if word == w:
                bag[i] = 1
    return np.array(bag)

#function to predict the class and sort out the classes on basis of preference.
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

#function to get response of the chatbot.
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

video_capture = cv2.VideoCapture(0)
#we set a flag that decides if the chatbot needs to be activated. flag==0 means no activation is needed and flag==1 means chatbot needs to be activated.    
flag = 0

#this is the main piece of code, which displays the main functioning.

while True:
    
    showing, image =  video_capture.read()
    # To keep the video capture on since this was giving an error
    # earlier, becuase video_capture was already intialized even though the code wasn't run properly -_-.
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
            cv2.imwrite('/home/pi/download.jpg', image)
            print('Hello Yashwardhan.')
            flag = 1
            break
        
        elif result[0] == False:
            print("You are not Yashwardhan.")
            break
        
    # Doesn't work on my device if I press 'q', I don't know why yet XD.
    elif cv2.waitKey(1) & 0xFF == ord('q'):
        break
    
cv2.destroyAllWindows()
video_capture.release()

if flag==1:
    
    # Yes, we can name the bot XD.
    bot = str(input('Enter bot name please: '))
    
    while True:
        message = input('You: ')
        ints = predict_class(message)
        tag, res = get_response(ints, intents)
        if tag == 'goodbye':
            print(f'{bot}: {res}')
            break
        else:
            print(f'{bot}: {res}')
            
elif flag==0:
    print('Goodbye, have a nice day.')
