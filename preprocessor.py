def remove_number(doc):
    import re

    return re.sub(r'\d+', '', doc)


def remove_spacing(doc):
    import re

    return re.sub(r'\s+', ' ', doc)


def remove_hyperlink(doc):
    import re

    return re.sub(r"http\S+", "", doc)


def remove_tag(doc):
    import re

    return re.sub(r"<[^>]+>", "", doc)


def remove_punctuation(doc):
    import string

    return doc.translate(str.maketrans(dict.fromkeys(string.punctuation)))


def stemmizer_words(words):
    from nltk import PorterStemmer

    stemmizer = PorterStemmer()

    return [stemmizer.stem(w) for w in words]


def lower_words(words):

    return [w.lower() for w in words if w.isalpha()]


def remove_stopwords(words):
    from nltk.corpus import stopwords

    return [w for w in words if w not in stopwords.words('english')]


def lemmatizer_words(words):
    from nltk import WordNetLemmatizer

    lemmatizer = WordNetLemmatizer()

    return [lemmatizer.lemmatize(w) for w in words]


def preprocessing(data):
    from nltk import word_tokenize

    tokens = word_tokenize(data)
    tokens = remove_stopwords(tokens)
    tokens = lower_words(tokens)
    tokens = lemmatizer_words(tokens)

    return tokens


