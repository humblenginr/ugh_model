import keras
from flask import Flask, request

# -----------------------------------------------

# import pandas as pd
# import numpy as np
# import tensorflow as tf
# import keras
# from keras.models import Sequential
# from keras.layers import Dense
# from keras.optimizers import Adam
# from sklearn.model_selection import train_test_split
# from sklearn.metrics import mean_squared_error
# import matplotlib.pyplot as plt

# file_path = './dataset.csv'
# df = pd.read_csv(file_path)

# X = df.iloc[:, 3:12]
# y = df.iloc[:, 2]

# X.shape

# plt.scatter(df.iloc[:,11], y)

# X.shape

# X_t, X_test, y_t, y_test = train_test_split(X, y, test_size=0.25, random_state=42)
# X_train, X_val, y_train, y_val = train_test_split(X_t, y_t, test_size=0.25, random_state=42)

# # Define the neural network model for regression
# model = Sequential()
# model.add(Dense(64, input_dim=9, activation='tanh'))  # Input layer with 1 feature
# model.add(Dense(32, activation='relu'))               # Hidden layer with 32 units
# model.add(Dense(1, activation='linear'))              # Output layer with 1 unit for regression

# # Compile the model
# model.compile(optimizer=tf.keras.optimizers.legacy.Adam(learning_rate=0.0001, amsgrad=True), loss='mean_squared_error')

# # Train the model
# history = model.fit(X_train, y_train, epochs=1500, batch_size=3, validation_data=(X_val, y_val), verbose=1)

# # Evaluate the model on the test set
# y_pred = model.predict(X_test)

# # Plot training history
# plt.plot(history.history['loss'], label='Training Loss')
# plt.plot(history.history['val_loss'], label='Validation Loss')
# plt.xlabel('Epochs')
# plt.ylabel('Mean Squared Error')
# plt.legend()
# plt.show()

# # Calculate Mean Squared Error
# mse = mean_squared_error(y_test, y_pred)
# print("Mean Squared Error on Test Set:", mse)

# Y_pred = model.predict(X_test)

# # serialize model to JSON
# model_json = model.to_json()
# with open("model.json", "w") as json_file:
#     json_file.write(model_json)
# # serialize weights to HDF5
# model.save_weights("model.h5")
# print("Saved model to disk")

# model.save('model2.keras')

# -----------------------------------------------------------------

app = Flask(__name__, static_folder='./build', static_url_path='/')


@app.route('/')
def index():
    return app.send_static_file('index.html')


@app.route('/predict', methods=['POST'])
def main_page():
    if request.method == 'POST':
        json_data = request.get_json()['data']
        fin_list = []
        for s in json_data:
            fin_list.append(float(s))

        model = keras.models.load_model('./model2.keras')
        res = model.predict([fin_list])
    return str(res[0][0])
