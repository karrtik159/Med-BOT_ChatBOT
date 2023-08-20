import nltk
from nltk.stem import WordNetLemmatizer
lemmatizer = WordNetLemmatizer()
import pickle
import numpy as np

from tensorflow.keras.models import load_model
model = load_model('best_model.h5')
import json
import random
intents = json.loads(open('intents.json', encoding="utf8").read())
words = pickle.load(open('words.pkl','rb'))
classes = pickle.load(open('classes.pkl','rb'))

def clean_up_sentence(sentence):
    # tokenize the pattern - split words into array
    sentence_words = nltk.word_tokenize(sentence)
    # stem each word - create short form for word
    sentence_words = [lemmatizer.lemmatize(word.lower()) for word in sentence_words]
    return sentence_words

# return bag of words array: 0 or 1 for each word in the bag that exists in the sentence

def bow(sentence, words, show_details=True):
    # tokenize the pattern
    sentence_words = clean_up_sentence(sentence)
    # bag of words - matrix of N words, vocabulary matrix
    bag = [0]*len(words)  
    for s in sentence_words:
        for i,w in enumerate(words):
            if w == s: 
                # assign 1 if current word is in the vocabulary position
                bag[i] = 1
                if show_details:
                    print ("found in bag: %s" % w)
    return(np.array(bag))

def predict_class(sentence, model):
    # filter out predictions below a threshold
    p = bow(sentence, words,show_details=False)
    res = model.predict(np.array([p]))[0]
    ERROR_THRESHOLD = 0.25
    results = [[i,r] for i,r in enumerate(res) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append({"intent": classes[r[0]], "probability": str(r[1])})
    return return_list

def getResponse(ints, intents_json):
    tag = ints[0]['intent']
    list_of_intents = intents_json['intents']
    for i in list_of_intents:
        if(i['tag']== tag):
            result = random.choice(i['responses'])
            break
    return result

def chatbot_response(msg):
    ints = predict_class(msg, model)
    res = getResponse(ints, intents)
    return res


import tkinter
from tkinter import *
from PIL import Image,ImageTk

def send():
    msg = EntryBox.get("1.0", 'end-1c').strip()
    EntryBox.delete("0.0", END)

    if msg != '':
        ChatLog.config(state=NORMAL)
        ChatLog.insert(END, "You: " + msg + '\n\n')
        ChatLog.config(foreground="#442265", font=("Verdana", 12))

        res = chatbot_response(msg)
        ChatLog.insert(END, "Bot: " + res + '\n\n')

        ChatLog.config(state=NORMAL)
        ChatLog.yview(END)

def close_app():
    base.destroy()

# Create the main GUI window
base = Tk()
base.title("Med-Care ChatBot")
base.geometry("400x600")
base.resizable(width=FALSE, height=FALSE)

images=[]

# Define a function to make the transparent rectangle
def create_rectangle(x,y,a,b,**options):
   if 'alpha' in options:
      # Calculate the alpha transparency for every color(RGB)
      alpha = int(options.pop('alpha') * 255)
      # Use the fill variable to fill the shape with transparent color
      fill = options.pop('fill')
      fill = base.winfo_rgb(fill) + (alpha,)
      image = Image.new('RGBA', (a-x, b-y), fill)
      images.append(ImageTk.PhotoImage(image))
      chatlog_canvas.create_image(x, y, image=images[-1], anchor='nw')
      chatlog_canvas.create_rectangle(x, y,a,b, **options)
      



bg_image = Image.open("Med_BOT.png")  # Replace with your image path
bg_image = bg_image.resize((400, 600), Image.ANTIALIAS)
background = ImageTk.PhotoImage(bg_image)
background_label = Label(base, image=background)
background_label.place(x=0, y=0, relwidth=1, relheight=1)

# Create a Canvas widget for ChatLog with a background image
chatlog_canvas = Canvas(base, bd=0, highlightthickness=0)
chatlog_canvas.place(x=6, y=6, height=400, width=370)

# Create a translucent background for Canvas
create_rectangle(0, 0, 370, 400, fill="blue", alpha=.3)
create_rectangle(0, 0, 370, 400, fill="red", alpha=.1)

# chatlog_canvas.pack()
# Create ChatLog Text widget inside Canvas
ChatLog = Text(chatlog_canvas, wrap=WORD, font="Arial")
ChatLog.pack(fill=BOTH, expand=True)
chatlog_canvas.create_window(0, 0, anchor=NW, window=ChatLog)

# Create scrollbar for Canvas
scrollbar = Scrollbar(chatlog_canvas, command=chatlog_canvas.yview)
scrollbar.pack(side=RIGHT, fill=Y)
chatlog_canvas.config(yscrollcommand=scrollbar.set)

# Create scrollbar for Canvas
scrollbar = Scrollbar(chatlog_canvas, command=ChatLog.yview)
scrollbar.pack(side=RIGHT, fill=Y)
ChatLog.config(yscrollcommand=scrollbar.set)

# Create Button to send message
SendButton = Button(base, font=("Verdana", 12, 'bold'), text="Send", width="12", height=2,
                    bd=1, bg="#32de97", activebackground="#3c9d9b", fg='#ffffff',
                    command=send)

# Create the box to enter message
EntryBox = Text(base, bd=2, bg="light cyan", width="29", height="5", font="Arial")
EntryBox.bind("<Return>", lambda event=None: send())

# Create "End Session" button
EndButton = Button(base, text='End Session', bg="red", fg="white", command=close_app)



# Place all components on the screen
scrollbar.place(x=376, y=6, height=470)
ChatLog.place(x=6, y=6, height=470, width=370)
EntryBox.place(x=128, y=485, height=80, width=250)
SendButton.place(x=6, y=485, height=70)
EndButton.place(x=320, y=565, width=70)

# Load the model, intents, etc. (your code here)

base.mainloop()
