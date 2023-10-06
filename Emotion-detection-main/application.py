import pickle
from tkinter import *
import string
from num2words import num2words
import unidecode
import re
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk.tokenize import RegexpTokenizer
import nltk
nltk.download('stopwords')
from nltk.corpus import stopwords
stop = stopwords.words('english')
from nltk import tokenize





path='edm.pkl'
model = pickle.load(open(path, 'rb'))
path1='tfidf.pkl'
vector=pickle.load(open(path1,'rb'))

short_form_list = open('slang.txt', 'r')
chat_words_str = short_form_list.read()

chat_words_map_dict = []
chat_words_list = []
for line in chat_words_str.split("\n"):
    if line != " ":
        s=line.split('=')
        chat_words_map_dict.append(s)
        chat_words_list.append(s[0])


root = Tk()
root.geometry("300x300")
root.title(" Q&A ")


def convert_lowercase(sent):
    sent=sent.lower()
    return sent

def remove_punctuation(sen):
    for punctuations in string.punctuation:
        sen=sen.replace(punctuations,'')
    return sen

def num_to_words(text):
    # splitting text into words with space
    after_spliting = text.split()

    for index in range(len(after_spliting)):
        if after_spliting[index].isdigit():
            after_spliting[index] = num2words(after_spliting[index])

    # joining list into string with space
    numbers_to_words = ' '.join(after_spliting)
    return numbers_to_words

def to_ascii(x):
    x = unidecode.unidecode(x)
    return x

def short_to_original(text):

    new_text = []
    for w in text.split():
        if w.upper() in chat_words_list:
            for i in chat_words_map_dict:
                if i[0]==w.upper():
                    new_text.append(i[1])
        else:
            new_text.append(w)
    return " ".join(new_text)


def remove_emojis(text):

    emoji_pattern = re.compile("["
                               u"\U0001F600-\U0001F64F"  # emoticons
                               u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                               u"\U0001F680-\U0001F6FF"  # transport & map symbols
                               u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
                               u"\U00002500-\U00002BEF"  # chinese char
                               u"\U00002702-\U000027B0"
                               u"\U00002702-\U000027B0"
                               u"\U000024C2-\U0001F251"
                               u"\U0001f926-\U0001f937"
                               u"\U00010000-\U0010ffff"
                               u"\u2640-\u2642"
                               u"\u2600-\u2B55"
                               u"\u200d"
                               u"\u23cf"
                               u"\u23e9"
                               u"\u231a"
                               u"\ufe0f"  # dingbats
                               u"\u3030"
                               "]+", flags=re.UNICODE)

    without_emoji = emoji_pattern.sub(r'',text)
    return without_emoji

lemma = WordNetLemmatizer()
def lemmatization(text):

    # word tokenization
    tokens = word_tokenize(text)

    for index in range(len(tokens)):
        # lemma word
        lemma_word = lemma.lemmatize(tokens[index])
        tokens[index] = lemma_word

    return ' '.join(tokens)




def Take_input():
    sen = inputtxt.get("1.0", "end-1c")
    sen = convert_lowercase(sen)
    sen = remove_punctuation(sen)
    #sen = num2words(sen)
    sen = remove_punctuation(sen)
    sen = to_ascii(sen)
    sen = short_to_original(sen)
    sen = remove_emojis(sen)
    #sen = sen.apply(lambda x: ' '.join([word for word in x.split() if word not in (stop)]))
    sen = lemmatization(sen)
    print(sen)
    tokens_without_sw = [' '.join(word for word in sen if not word in stopwords.words())]
    vect = vector.transform(tokens_without_sw).toarray()
    print(vect)
    my_prediction = model.predict(vect)
    val=my_prediction[0]
    dic={1:'joy',2:'fear',3:'anger',4:'sadness'}
    Output.insert(END, dic[val])










l = Label(text = "Enter the Sentence to Detect Emotion ")
inputtxt = Text(root, height = 10,width = 25,bg = "light yellow")
  
Output = Text(root, height = 5, width = 25, bg = "light cyan")
  
Display = Button(root, height = 2,width = 20, text ="Show",command = lambda:Take_input())
  
l.pack()
inputtxt.pack()
Display.pack()
Output.pack()
  
mainloop()