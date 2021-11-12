import nltk
import preprocess
import models
import exploration
from sklearn.metrics import classification_report


# nltk.download('words')
# nltk.download('wordnet')
# nltk.download('stopwords')


def main():
    x_train, y_train, x_test, y_test, df = preprocess.preprocess(use_cache=True,debug=True)
    # df = df.sample(n=5000, random_state=2)
    # df.to_csv("./Data/cleaned-debug.csv")
    # exploration.explore()
    print(df.head())
    # logistic = models.LogisticRegression(x_train,y_train,num_iter=500)
    # predictions = logistic.predict(x_test)
    # print(classification_report(y_test, predictions,zero_division=0))
    mlp = models.MLP(x_train, y_train, [5, 5, 5])
    mlp_predictions = mlp.predict(x_test)
    print(classification_report(y_test, mlp_predictions,zero_division=0))


if __name__ == '__main__':
    main()
