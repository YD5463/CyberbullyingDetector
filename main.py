import nltk
import preprocess
# import models
import exploration
from sklearn.metrics import classification_report

# nltk.download('words')
# nltk.download('wordnet')
# nltk.download('stopwords')


def main():
    x_train, y_train, x_test, y_test = preprocess.preprocess(use_cache=True)
    print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    # exploration.explore()
    # logistic = models.LogisticRegression(x_train,y_train)
    # predictions = logistic.predict(x_test)
    # print(classification_report(y_test, predictions))


if __name__ == '__main__':
    main()
