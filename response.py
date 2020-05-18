import nltk
from nltk.stem.lancaster import LancasterStemmer
from add_utils import clean_up_sentence, lemmatize_word, text_to_speech
import numpy
import tflearn
import tensorflow
import pickle
import json
import random
from spellchecker import SpellChecker
spell = SpellChecker()
ERROR_THRESHOLD = 0.60
stemmer = LancasterStemmer()
incomplete_info = ["Sorry! I didn't quite get you there.", "Sorry! I am unable to understand you.", "Sorry! Please try again."]

context = {}

fileset = ['departments.json', 'central_facilities.json', 'admin.json', 'intents.json']
# with open("intents.json") as file:
#     data = json.load(file)



dataset = []
for file in fileset:
    with open(file) as inputFile:
        dataset.append(json.load(inputFile))

# with open("intents.json") as file:
#     intents = json.load(file)

with open("data.pickle", "rb") as f:
        words, labels, training, output = pickle.load(f)
# print("All words:")
# print(words)

tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 64, activation = 'relu')
net = tflearn.fully_connected(net, 64, activation = 'relu')
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)

model.load("model.tflearn")


def bag_of_words(s, words):
    bag = [0 for _ in range(len(words))]

    s_words = nltk.word_tokenize(s)
    # s_words = [stemmer.stem(word.lower()) for word in s_words]
    s_words = [lemmatize_word(word.lower()) for word in s_words]

    s_words = clean_up_sentence(s_words)

    for se in s_words:
        for i, w in enumerate(words):
            if w == se:
                bag[i] = 1
            
    return numpy.array(bag)

def classify(sentence):
    results = model.predict([bag_of_words(sentence, words)])[0]
    # filter out predictions below a threshold
    results = [[i,r] for i,r in enumerate(results) if r>ERROR_THRESHOLD]
    # sort by strength of probability
    results.sort(key=lambda x: x[1], reverse=True)
    return_list = []
    for r in results:
        return_list.append((labels[r[0]], r[1]))
    # return tuple of intent and probability
    # print("Results are: ", return_list)
    return return_list

def response(sentence, userID='123', show_details=False):
    results = classify(sentence)
    # if we have a classification then find the matching intent tag
    if results:
        # loop as long as there are matches to process
        while results:
            # for i in intents['intents']:
            for data in dataset:
                for i in data['intents']:
                    # find a tag matching the first result
                    if i['tag'] == results[0][0]:
                        # set context for this intent if necessary
                        if 'context_set' in i:
                            # print('True')
                            if show_details: print ('context:', i['context_set'])
                            context[userID] = i['context_set']

                        # check if this intent is contextual and applies to this user's conversation
                        if not   'context_filter' in i or \
                            (userID in context and 'context_filter' in i and i['context_filter'] == context[userID]):
                            if show_details: print ('tag:', i['tag'])
                            # print("Here1:",('context_filter' in i))
                            if(('context_filter' in i)):
                                print("Here2:",(i['context_filter'] == context[userID]))
                            # a random response from the intent
                            reply = random.choice(i['responses'])
                            return reply


            results.pop(0)
    else:
        return random.choice(incomplete_info)

sp_words=["mnit"]
ans=[]
def chat(userID, show_details):
    print("Start talking with the bot (type quit to stop)!")
    while True:
        inp = input("You: ")
        tmp = spell.split_words(inp)
        ans=[]
        for mis in tmp:
        	hmm=mis
        	if mis not in sp_words:
        		# print(mis)
        		hmm=spell.correction(mis)
        	ans.append(hmm)
        query = (' '.join(x for x in ans))
        inp = query
        if inp.lower() == "quit":
            break
        reply = response(inp, userID, show_details)
        print(reply)

        text_to_speech(reply)

chat('123', False)

# def chatbot_response(msg):
# 	reply = response(msg,'123',False)
# 	print(reply)
# 	# text_to_speech(reply)
# 	return reply

# #Creating GUI with tkinter
# import tkinter
# from tkinter import *

# res = ''
# def send():
#     msg = EntryBox.get("1.0",'end-1c').strip()
#     EntryBox.delete("0.0",END)

#     if msg != '':
#         ChatLog.config(state=NORMAL)
#         ChatLog.insert(END, "You: " + msg + '\n\n')
#         ChatLog.config(foreground="#442265", font=("Verdana", 12 ))
    
#         res = chatbot_response(msg)
#         ChatLog.insert(END, "Bot: " + res + '\n\n')
#         ChatLog.config(state=DISABLED)
#         ChatLog.yview(END)
        


# base = Tk()
# base.title("Hello")
# base.geometry("400x500")
# base.resizable(width=FALSE, height=FALSE)

# #Create Chat window
# ChatLog = Text(base, bd=0, bg="white", height="8", width="50", font="Arial",)

# ChatLog.config(state=DISABLED)

# #Bind scrollbar to Chat window
# scrollbar = Scrollbar(base, command=ChatLog.yview, cursor="heart")
# ChatLog['yscrollcommand'] = scrollbar.set

# #Create Button to send message
# SendButton = Button(base, font=("Verdana",12,'bold'), text="Send", width="12", height=5,
#                     bd=0, bg="#32de97", activebackground="#3c9d9b",fg='#ffffff',
#                     command= send )

# #Create the box to enter message
# EntryBox = Text(base, bd=0, bg="white",width="29", height="5", font="Arial")
# #EntryBox.bind("<Return>", send)


# #Place all components on the screen
# scrollbar.place(x=376,y=6, height=386)
# ChatLog.place(x=6,y=6, height=386, width=370)
# EntryBox.place(x=128, y=401, height=90, width=265)
# SendButton.place(x=6, y=401, height=90)

# base.mainloop()
