import nltk
from add_utils import clean_up_sentence, lemmatize_word
from nltk.stem.lancaster import LancasterStemmer
import time
import matplotlib.pyplot as plt
stemmer = LancasterStemmer()

import numpy
import tflearn
import tensorflow
import json
import pickle

fileset = ['departments.json' ,'central_facilities.json', 'admin.json', 'intents.json','placement.json']
# with open("intents.json") as file:
#     data = json.load(file)
# ,, 'intents.json'
dataset = []
for file in fileset:
    with open(file) as inputFile:
        dataset.append(json.load(inputFile))




words = []
labels = []
docs_x = []
docs_y = []
for data in dataset:
    for intent in data["intents"]:
        for pattern in intent["patterns"]:
            wrds = nltk.word_tokenize(pattern)
            words.extend(wrds)
            docs_x.append(wrds)
            docs_y.append(intent["tag"])

        if intent["tag"] not in labels:
            labels.append(intent["tag"])

# words = [stemmer.stem(w.lower()) for w in words if w != "?"]
words = [lemmatize_word(word.lower()) for word in words]
words = clean_up_sentence(words)
words = sorted(list(set(words)))

labels = sorted(labels)

training = []
output = []

out_empty = [0 for _ in range(len(labels))]

for x, doc in enumerate(docs_x):
    bag = []

    # wrds = [stemmer.stem(w.lower()) for w in doc]
    wrds = [lemmatize_word(word.lower()) for word in doc]

    wrds = clean_up_sentence(wrds)

    for w in words:
        if w in wrds:
            bag.append(1)
        else:
            bag.append(0)

    output_row = out_empty[:]
    output_row[labels.index(docs_y[x])] = 1

    training.append(bag)
    output.append(output_row)


training = numpy.array(training)
output = numpy.array(output)

with open("data.pickle", "wb") as f:
        pickle.dump((words, labels, training, output), f)


tensorflow.reset_default_graph()

net = tflearn.input_data(shape=[None, len(training[0])])
net = tflearn.fully_connected(net, 64, activation = 'relu')
net = tflearn.fully_connected(net, 64, activation = 'relu')
net = tflearn.fully_connected(net, len(output[0]), activation="softmax")
net = tflearn.regression(net)

model = tflearn.DNN(net)
start_time = time.time()
print(model.fit(training, output, n_epoch=200, batch_size=32, show_metric=True))
end_time = time.time()
print("The training took {:d} seconds".format(int(end_time-start_time)))
model.save("model.tflearn")

# print(history)
# loss = history['loss']
# acc = history['acc']
# epochs = range(1,199)
# plt.plot(epochs, acc, 'g', label='Training Accuracy')
# plt.plot(epochs, loss, 'b', label='validation loss')
# plt.title('Training and Validation loss')
# plt.xlabel('Epochs')
# plt.ylabel('Loss')
# plt.legend()
# plt.show()