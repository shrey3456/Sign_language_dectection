from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Convolution2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization, Input
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix

# Initializing the CNN with Input layer
classifier = Sequential()

# Step 1 - Convolution Layer with Input layer
classifier.add(Input(shape=(64, 64, 3)))  # Explicit Input layer
classifier.add(Convolution2D(32, (3, 3), activation='relu', padding='same'))  # Added padding='same'
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 2 - Second Convolution Layer
classifier.add(Convolution2D(64, (3, 3), activation='relu', padding='same'))  # Added padding='same'
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 3 - Third Convolution Layer
classifier.add(Convolution2D(128, (3, 3), activation='relu', padding='same'))  # Added padding='same'
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 4 - Fourth Convolution Layer (New layer added for better performance)
classifier.add(Convolution2D(256, (3, 3), activation='relu', padding='same'))  # Added padding='same'
classifier.add(BatchNormalization())
classifier.add(MaxPooling2D(pool_size=(2, 2)))

# Step 5 - Flattening
classifier.add(Flatten())

# Step 6 - Full Connection
classifier.add(Dense(512, activation='relu'))  # Increased the number of neurons
classifier.add(Dropout(0.5))

# Output Layer
classifier.add(Dense(27, activation='softmax'))

# Compiling The CNN with adjusted learning rate
classifier.compile(
    optimizer='adam',  # Use the optimizer from training
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Part 2 - Fitting the CNN to the images
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.1,
    zoom_range=0.1,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    horizontal_flip=True
)

test_datagen = ImageDataGenerator(rescale=1./255)

# Loading datasets
training_set = train_datagen.flow_from_directory(
    'mydata/training_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

test_set = test_datagen.flow_from_directory(
    'mydata/test_set',
    target_size=(64, 64),
    batch_size=32,
    class_mode='categorical'
)

steps_per_epoch = training_set.samples
validation_steps = test_set.samples

early_stopping = EarlyStopping(
    monitor='val_loss', 
    patience=5, 
    restore_best_weights=True
)

# Fitting the model
model = classifier.fit(
    training_set,
    steps_per_epoch=steps_per_epoch,
    epochs=25,
    validation_data=test_set,
    validation_steps=validation_steps,
    callbacks=[early_stopping]
)

# Saving the model in the updated format
classifier.save('Trained_model.keras')

# Visualizing the training history
print(model.history.keys())

# Summarize history for accuracy
plt.plot(model.history['accuracy'])
plt.plot(model.history['val_accuracy'])
plt.title('Model Accuracy')
plt.ylabel('Accuracy')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Summarize history for loss
plt.plot(model.history['loss'])
plt.plot(model.history['val_loss'])
plt.title('Model Loss')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend(['Train', 'Test'], loc='upper left')
plt.show()

# Load the trained model
model = load_model('Trained_model.keras')  # Use the updated filename

# Recompile the model
model.compile(
    optimizer='adam',  # Use the optimizer from training
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Evaluate the model on the test set
loss, accuracy = model.evaluate(test_set)
print(f"Validation Accuracy: {accuracy * 100:.2f}%")

# Test on a batch of data
X_test, y_true = next(test_set)  # Load a batch of test data
y_pred = model.predict(X_test)
y_pred_classes = y_pred.argmax(axis=1)  # Convert predictions to class labels

# Calculate metrics
accuracy = accuracy_score(y_true.argmax(axis=1), y_pred_classes)  # Correcting the y_true format
print(f"Test Accuracy: {accuracy * 100:.2f}%")

# Confusion matrix
cm = confusion_matrix(y_true.argmax(axis=1), y_pred_classes)
print("Confusion Matrix:")
print(cm)
