import pandas as pd

train_df = pd.read_csv("sign_mnist_train.csv")
test_df = pd.read_csv("sign_mnist_test.csv")

print(train_df.shape)
train_df.head()

import numpy as np
from sklearn.preprocessing import LabelBinarizer

# Separate features and labels
X_train = train_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_train = train_df['label']

X_test = test_df.iloc[:, 1:].values.reshape(-1, 28, 28, 1) / 255.0
y_test = test_df['label']

# One-hot encode labels
lb = LabelBinarizer()
y_train = lb.fit_transform(y_train)
y_test = lb.transform(y_test)

import tensorflow as tf
from tensorflow.keras import layers, models

model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
    layers.MaxPooling2D(2,2),
    
    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D(2,2),
    
    layers.Flatten(),
    layers.Dense(128, activation='relu'),
    layers.Dense(24, activation='softmax')  # 25 labels (A–Y without J)
])

model.compile(optimizer='adam',
              loss='categorical_crossentropy',
              metrics=['accuracy'])

model.summary()

history = model.fit(X_train, y_train, epochs=4, validation_data=(X_test, y_test))

model.save("sign_language_model.h5")

hist_df = pd.DataFrame(history.history)
hist_df.to_csv("training_history.csv", index=False)

model.evaluate(X_test, y_test)

import matplotlib.pyplot as plt

plt.imshow(X_test[0].reshape(28, 28), cmap='gray')
prediction = model.predict(np.expand_dims(X_test[0], axis=0))
print("Predicted Letter:", chr(lb.inverse_transform(prediction)[0] + 65))

