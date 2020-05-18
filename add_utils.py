from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
import pyttsx3
def clean_up_sentence(word_tokens):
    ignore_words = set(stopwords.words('english'))
    ignore_words.update(',', '.', '(', ')', '?')
    filtered_sentence = [w for w in word_tokens if not w in ignore_words]
    return filtered_sentence

def lemmatize_word(word):
    lemmatizer = WordNetLemmatizer()
    return lemmatizer.lemmatize(word)

def text_to_speech(sentence):
	engine = pyttsx3.init()
	engine.say(sentence)
	engine.runAndWait()