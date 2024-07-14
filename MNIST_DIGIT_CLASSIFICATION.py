import pandas
import numpy
import matplotlib.pyplot as plt
import matplotlib.image as mping
from PIL import Image
import seaborn as sns
import os
import cv2
import tensorflow as tf
tf.random.set_seed(3)
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.datasets import mnist

from tensorflow.math import confusion_matrix

"""so when we import mnit dataset we need not t convert it to numpy array we get numpy array ie already processed data also we get """

"""creating four arrays which are given to us"""
(X_train,Y_train),(X_test,Y_test)=mnist.load_data()
print(type(X_train))


#shape of numpy array
print((X_train.shape,Y_train.shape,X_test.shape,Y_test.shape))

#printint the 11 th image or in 10 th index
# print(X_train[0])
print(X_train.shape)#28X28
plt.imshow(X_train[0])#for creating visualisation using pyplot
#the image is grayscale but we seeeingit coloured so we can use,cmap="gray"

plt.show()

#printing the lable for 50 index image
print(Y_train[00])
print(type(Y_train))#it is also numpy
print(Y_train)


#image lables
print(Y_train.shape,Y_test.shape)

print(numpy.unique(Y_test))
print(numpy.unique(Y_train))
print(numpy.unique(X_test))

print(numpy.unique(X_train))
#scaling the values to lower level
X_train=X_train/255
X_test=X_test/255
print(numpy.unique(X_test))
print(numpy.unique(X_test)) 


"""BUILDING NEURAL NETWORK"""
#SETTING UP THE LAYERS OF NEURAL NETWORK
model=keras.Sequential([keras.layers.Flatten(input_shape=(28,28)),keras.layers.Dense(50,activation='relu'), keras.layers.Dense(50,activation='relu'),keras.layers.Dense(10,activation='sigmoid')])

#compiling the neural network using model
model.compile(optimizer='adam',loss='sparse_categorical_crossentropy',metrics=['accuracy'])

#training the neurall network

model.fit(X_train,Y_train,epochs=10)

#training data accuracy is 99.0%

# #ACCURACY ON TEST DATA

#loss,accuracy=model.evaluate(X_test,Y_test)
#print(accuracy)
#96.6%
"""important"""
#first data or image in the x_test array
plt.imshow(X_test[0])
plt.show()

print(Y_test[0])

Y_predictions=model.predict(X_test)
print(Y_predictions.shape)

#the above line of code gives(10000,10)
"""explaination: as we know we have 10 k images in testing data and y test is basically the the actual value ehich the model needs to predict on x test labels and also 10 represents digits from 0 to 9 """

print(Y_predictions[0])
label_prediction=numpy.argmax(Y_predictions[0])
"""telling me the max value which is actually the value jo humme prediction k baad chaia"""
print(label_prediction)
# 7 is the output


#now here i am going to create the label values from the prediction probabilities similarly as done above but want to this for all input features
label_pred_for_input_features=[numpy.argmax(i) for i in Y_predictions]
print(label_pred_for_input_features)

#y_test are the actual label values and label_pred_for_input_features are the predicted ones

"""making the confusion matrix"""
conf_mat=confusion_matrix(Y_test,label_pred_for_input_features)
print(conf_mat)


"""last part of the model in which i am going to make a predictive model or in other terms a model to which i will be providing the image and need to tell me that which calue it is"""

#saving images in png

save_dir="mnist_png"
os.makedirs(save_dir,exist_ok=True)
for i in range(len(X_train)):
    image = X_train[i]
    label = Y_train[i]

    # Save the image as a PNG file
    file_name = os.path.join(save_dir, f"mnist_{i}_label_{label}.png")
    plt.imsave(file_name, image, cmap='gray')

print("PNG images saved successfully.")
"""so through the above snippet the size is 28X28 and also in grayscale so no need to that task"""

""" doing the scaling as did it for training data but i have already done it """



#reshaped_img=numpy.reshape(input_img_resized,[1,28,28])#telling i am going to pwhy redict label for 1 image
# Read the image
image_to_be_predicted_path = input("Enter the path of the image to be predicted: ")
image_to_be_predicted = cv2.imread(image_to_be_predicted_path, cv2.IMREAD_GRAYSCALE)


resized_image = cv2.resize(image_to_be_predicted, (28, 28))


# Display the image
cv2.imshow("Image to be Predicted", resized_image)
cv2.waitKey(0)
cv2.destroyAllWindows()

scaled_img=resized_image/255


# Predict the label for the image
input_prediction = model.predict(scaled_img.reshape(1, 28, 28))

# Get the predicted label
predicted_label = numpy.argmax(input_prediction)

print("Predicted Label:", predicted_label)
