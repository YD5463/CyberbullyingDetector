import nltk
import preprocess
import models
import exploration
from sklearn.metrics import classification_report

# nltk.download('words')
# nltk.download('wordnet')
# nltk.download('stopwords')


def main():
    x_train, y_train, x_test, y_test, df = preprocess.preprocess(use_cache=True, debug=False)
    print(df.head())
    # logistic = models.LogisticRegression(x_train,y_train,epoch=30)
    # predictions = logistic.predict(x_test)
    # with open("report1.txt","w") as r1:
    #     report = classification_report(y_test, predictions,zero_division=0)
    #     print(report)
    #     r1.write(report)
    df = df.drop("Text",axis=1)

    mlp = models.MLP(x_train, y_train, [500, 400, 350,200,100],epoch=100,learning_rate=0.001)
    mlp_predictions = mlp.predict(x_test)
    with open("report2.txt","w") as r1:
        report = classification_report(y_test, mlp_predictions, zero_division=0)
        print(report)
        r1.write(report)


if __name__ == '__main__':
    main()
