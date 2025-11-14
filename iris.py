from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import ConfusionMatrixDisplay
import matplotlib.pyplot as plt

iris = load_iris()
X = iris.data
y = iris.target

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = SVC()
model.fit(X_train, y_train)

disp = ConfusionMatrixDisplay.from_estimator(model, X_test, y_test)
plt.title("Iris Classification Confusion Matrix")

plt.savefig("iris_confusion.png")
plt.show()
