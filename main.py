import nltk
import preprocess
import models
import exploration
from sklearn.metrics import classification_report

# nltk.download('words')
# nltk.download('wordnet')
# nltk.download('stopwords')


def main():
    x_train, y_train, x_test, y_test = preprocess.preprocess(use_cache=True)
    print(x_train.shape,y_train.shape, x_test.shape, y_test.shape)
    # exploration.explore()
    logistic = models.LogisticRegression(x_train,y_train,num_iter=500)
    predictions = logistic.predict(x_test)
    print(classification_report(y_test, predictions))
    mlp = models.MLP(x_train,y_train,[5,5,5])
    mlp_predictions = mlp.predict(x_test)
    print(classification_report(y_test, mlp_predictions))


if __name__ == '__main__':
    main()
