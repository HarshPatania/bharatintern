import numpy as np
import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential
from keras.layers import Dense, Flatten
from keras.utils import to_categorical


(X_train, y_train), (X_test, y_test) = mnist.load_data()


X_train = X_train / 255.0
X_test = X_test / 255.0
y_train = to_categorical(y_train)
y_test = to_categorical(y_test)


model = Sequential()
model.add(Flatten(input_shape=(28, 28)))
model.add(Dense(128, activation='relu'))
model.add(Dense(10, activation='softmax'))


model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])


model.fit(X_train, y_train, epochs=10, batch_size=32, verbose=1)


loss, accuracy = model.evaluate(X_test, y_test)
print('Test Loss:', loss)
print('Test Accuracy:', accuracy)


predictions = model.predict(X_test)
predicted_labels = np.argmax(predictions, axis=1)


fig, axes = plt.subplots(5, 5, figsize=(10, 10))
axes = axes.ravel()

for i in np.arange(0, 25):
    axes[i].imshow(X_test[i], cmap='gray')
    axes[i].set_title("Predicted: %d\nTrue: %d" % (predicted_labels[i], np.argmax(y_test[i])))
    axes[i].axis('off')

plt.subplots_adjust(hspace=0.5)
plt.show()
