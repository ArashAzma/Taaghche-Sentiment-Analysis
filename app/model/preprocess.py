import re
from hazm import *
import os

stopword_files = ['verbal.txt', 'nonverbal.txt', 'chars.txt']
stopwords = []

for file in stopword_files:
    stopword_path = os.path.join('app/stopwords', file)
    with open(stopword_path, encoding='utf-8') as f:
        stopwords += f.read().split('\n')

stopwords = set(stopwords)

normalizer = Normalizer()

def normal(text):
    text=str(text)
    text = normalizer.character_refinement(text)
    text = normalizer.punctuation_spacing(text)
    text = normalizer.affix_spacing(text)
    text = normalizer.normalize(text)
    return text

# normalizer = Normalizer(correct_spacing=True, remove_diacritics=True, remove_specials_chars=True, unicodes_replacement=True)
lemmatizer = Lemmatizer()
stemmer = Stemmer()

def remove_stopwords(text):
    text=str(text)
    filtered_tokens = [token for token in text.split() if token not in stopwords]
    filtered_text = ' '.join(filtered_tokens)
    return filtered_text

def remove_emoji(text): 
    # Define a regex pattern to match various emojis and special characters
    emoji_pattern = re.compile("["
                    u"\U0001F600-\U0001F64F"  # emoticons
                    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
                    u"\U0001F680-\U0001F6FF"  # transport & map symbols
                    u"\U0001F1E0-\U0001F1FF"  # flags
                    u"\U00002702-\U000027B0"  # dingbats
                    u"\U000024C2-\U0001F251"  # enclosed characters
                    u"\U0001f926-\U0001f937"  # supplemental symbols and pictographs
                    u'\U00010000-\U0010ffff'  # supplementary private use area-A
                    u"\u200d"                 # zero-width joiner
                    u"\u200c"                 # zero-width non-joiner
                    u"\u2640-\u2642"          # gender symbols
                    u"\u2600-\u2B55"          # miscellaneous symbols
                    u"\u23cf"                 # eject symbol
                    u"\u23e9"                 # fast forward symbol
                    u"\u231a"                 # watch
                    u"\u3030"                 # wavy dash
                    u"\ufe0f"                 # variation selector-16
        "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r' ', text)

def remove_halfspace(text): 
    emoji_pattern = re.compile("["                
                u"\u200c"              
    "]+", flags=re.UNICODE)
    
    return emoji_pattern.sub(r' ', text) 

def remove_link(text): 
    return re.sub(r'http[s]?://(?:[a-zA-Z]|[0-9]|[$-_@.&+]|[!*\(\),]|(?:%[0-9a-fA-F][0-9a-fA-F]))+', '', str(text))

def remove_picUrl(text):
    return re.sub(r'pic.twitter.com/[\w]*',"", str(text))

def remove_rt(text):
    z = lambda text: re.compile('\#').sub('', re.compile('RT @').sub('@', str(text), count=1).strip())
    return z(text)

def remove_hashtag(text):
    return re.sub(r"#[^\s]+", '', str(text))

def remove_mention(text):
    return re.sub(r"@[^\s]+", '', str(text))

def remove_email(text): 
    return re.sub(r'\S+@\S+', '', str(text))

def remove_numbers(text): 
    return re.sub(r'^\d+\s|\s\d+\s|\s\d+$', ' ', str(text))

def remove_html(text):
    html_pattern = re.compile('<.*?>')
    return html_pattern.sub(r'', str(text))

def remove_quote(text): 
    return  str(text).replace("'","")

def remove_chars(text): 
    return  re.sub(r'[$+&+;+]|[><!+،:’,\(\).+]|[-+]|[…]|[\[\]»«//]|[\\]|[#+]|[_+]|[—+]|[*+]|[؟+]|[?+]|[""]', ' ', str(text))

def remove_englishword(text): 
    return re.sub(r'[A-Za-z]+', '', str(text))

def remove_extraspaces(text):
    return re.sub(r' +', ' ', text)

def remove_extranewlines(text):
    return re.sub(r'\n\n+', '\n\n', text)

def lemmatizer_text(text):
    words = []
    for word in text.split():
        words.append(lemmatizer.lemmatize(word))
    return ' '.join(words)

def stemmer_text(text):
    words = []
    for word in text.split():
        words.append(stemmer.stem(word))
    return ' '.join(words)

def normalizer_text(text):
    text = normal(text)
    text = stemmer_text(text)
    text = lemmatizer_text(text)
    return text

def preprocess(text):
    text = remove_link(text)
    text = remove_picUrl(text)
    text = remove_englishword(text)
    text = normalizer_text(text)
    text = remove_stopwords(text)
    text = remove_emoji(text)
    text = remove_rt(text)
    text = remove_mention(text)
    text = remove_emoji(text)
    text = remove_hashtag(text)   
    text = remove_email(text) 
    text = remove_html(text) 
    text = remove_chars(text)
    text = remove_numbers(text)
    text = remove_quote(text)
    text = remove_extraspaces(text)
    text = remove_extranewlines(text)
    text = remove_halfspace(text) 
    text = remove_stopwords(text)
    return text

preprocess('سلام')