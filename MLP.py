from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, confusion_matrix
import pandas


df = pandas.read_csv("Data/CSV/fulltrain.csv")
y = df.iloc[:, 0]
x = df.iloc[:, 1:5001]

x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=27)

clf = MLPClassifier(hidden_layer_sizes=(5000, 2500, 1250, 500, 11),
                    activation='relu',
                    learning_rate='adaptive',
                    shuffle=True,
                    max_iter=400,
                    batch_size=30,
                    alpha=0.001,
                    solver='sgd',
                    verbose=True,
                    momentum=0.9)

clf.fit(x_train, y_train)
y_pred = clf.predict(x_test)
print(accuracy_score(y_test, y_pred))
print(confusion_matrix(y_test, y_pred))