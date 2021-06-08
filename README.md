---
# Lazy-Assistant
This is a simple project that I initiated back February 2021, which was initiated with the aim of learning something new and applying the knowledge I have gained to automation tasks (yes, I am a lazy guy who loves automating things XD). The automation, till now, is only restricted to a simple toggling off and on of an LED bulb with a raspberry pi using a chatbot interface. Originally, the chatbot was developed with the help of pytorch, but since pytorch was not working on my raspberry pi, I migrated to keras. 

What can this app do?
* It can indentify if you are it's owner.
* It can interact with you as a chatbot.
* It can help you automate your tasks.

Sounds simple doesn't it? It is! The development wasn't XP.

Anyway, I will try to give you a brief description of the files included this respository below.

---
## Files in this respository:
* README.md - (oops, you're reading the README.md file right now XD.)
* app.py - this is the main python source code which displays the core app functioning. It captures and shows the face of the viewer in a red rectangle using the 'haar cascade classifier'. When key 'p' is pressed, it passes the image to a function (function recognize_face) which tells if the user is the owner or not (which is me in this case). The image that is captured is compared with an image of the owner which has already been fed into the code. In case there is no face present in the image capture, the app simply quits saying there is no face in the image. If, however, the owner is present in the image, a chatbot starts, which can help automate certain tasks. If there is a face present in the image captured and it is not of the owner, it simply wishes the person well and quits.
* chat.h5 - this is the model hdf5 file that was prepared with the help of keras and the 'intents.json' file (for the chatbot).
* classes.json - a list of all the tags present in the 'intents.json' file.
* words.json - contains a list of all the unique words present in the patterns that the chatbot expects.
* haarcascade_frontal_default.xml - the haar cascade classifier file taken from the OpenCV GitHub repository.

I will share the source code for model development soon, since the code is not yet presentable.

---
