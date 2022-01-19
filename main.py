import utils
import preprocessor
import pandas as pd
import sklearn as sk
import nltk
from collections import Counter
from random import shuffle

HAM_MAIL_PATH = 'dataset/ham'
SPAM_MAIL_DIR = 'dataset/spam'
MODEL_SAVEFILE = 'NaiveBayesModel.pickle'

def train_model(save_to_file=True):
    print('Reading dataset')
    ham_mails = utils.read_mails(HAM_MAIL_PATH)
    spam_mails = utils.read_mails(SPAM_MAIL_DIR)
    mails = ham_mails + spam_mails
    labels = ['ham'] * len(ham_mails) + ['spam'] * len(spam_mails)
    raw_df = pd.DataFrame({'email': mails, 'label': labels})

    print('Preprocessing')
    processed_df = pd.DataFrame()
    processed_df['email'] = [preprocessor.preprocessing(email) for email in raw_df.email]
    label_encoder = sk.preprocessing.LabelEncoder()
    processed_df['label'] = label_encoder.fit_transform(raw_df.label)

    X, y = processed_df.email, processed_df.label
    X_featurized = [Counter(i) for i in X]

    processed_data = [(X_featurized[i], y[i]) for i in range(len(X))]
    shuffle(processed_data)

    train_data = processed_data[:int(len(processed_data) * 0.7)]
    test_data = processed_data[int(len(processed_data) * 0.7):]

    print('Training')
    Naive_Bayes_model = nltk.classify.NaiveBayesClassifier.train(train_data)

    if save_to_file:
        utils.save_to_file(MODEL_SAVEFILE, Naive_Bayes_model)
        utils.save_to_file('processed_data.pickle', processed_data)

    return Naive_Bayes_model, test_data

def classify_mail(mail, preprocess=False, model=''):
    if not model:
        model = utils.load_from_file(MODEL_SAVEFILE)

    if preprocess:
        x = preprocessor.preprocessing(mail)

    x_featurized = Counter(x)

    return model.classify(x_featurized)

def analysis(test_data, model):
    y_pred = []
    y_test = []
    match = 0
    for data, label in test_data:
        y_pred.append(model.classify(data))
        y_test.append(label)
        if y_pred[-1] == label:
            match += 1

    accuracy = sk.metrics.accuracy_score(y_test, y_pred)
    print('Accuracy: ', accuracy)
    confusion_matrix = sk.metrics.confusion_matrix(y_test, y_pred)
    print('Confusion Matrix: \n', confusion_matrix)
    print(sk.metrics.classification_report(y_test, y_pred))
    return 0


def main():

    model = utils.load_from_file(MODEL_SAVEFILE)
    data = utils.load_from_file('processed_data.pickle')
    test_data = data[int(len(data) * 0.7):]
    analysis(test_data, model)

    return

if __name__ == "__main__":
    main()

