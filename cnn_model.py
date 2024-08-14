import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Input, Conv2D, MaxPooling2D
from tensorflow.keras.utils import to_categorical
import logging

# Configure logging
logging.basicConfig(level=logging.DEBUG)
logger = logging.getLogger(__name__)

def main():
    logger.debug('Starting the MNIST deep learning pipeline')

    # Load the MNIST dataset
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    # Reshape the data to add the channel dimension (for CNN)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)

    # Normalize the data
    x_train, x_test = x_train / 255.0, x_test / 255.0
    logger.debug('Data normalized')

    # Convert labels to categorical
    y_train, y_test = to_categorical(y_train), to_categorical(y_test)
    logger.debug('Labels converted to categorical')

    # Create a dataset from the numpy arrays and shuffle/batch/prefetch for efficiency
    train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
    train_dataset = train_dataset.shuffle(buffer_size=1024).batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test))
    test_dataset = test_dataset.batch(32).prefetch(buffer_size=tf.data.experimental.AUTOTUNE)

    # Build the model with convolutional layers
    model = Sequential([
        Input(shape=(28, 28, 1)),
        Conv2D(32, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dense(10, activation='softmax')
    ])
    logger.debug('Model built')

    # Compile the model
    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    logger.debug('Model compiled')

    # Add a callback to log during training
    class LoggingCallback(tf.keras.callbacks.Callback):
        def on_epoch_end(self, epoch, logs=None):
            logger.debug(f'Epoch {epoch + 1} - loss: {logs["loss"]}, accuracy: {logs["accuracy"]}')

    # Train the model using the dataset API
    model.fit(train_dataset, epochs=10, validation_data=test_dataset, callbacks=[LoggingCallback()])
    logger.debug('Model training complete')

    # Evaluate the model
    test_loss, test_acc = model.evaluate(test_dataset)
    logger.debug(f'Test accuracy: {test_acc}')
    print(f'\nTest accuracy: {test_acc}')

if __name__ == "__main__":
    # Enable multithreading for data loading and model training
    tf.config.threading.set_intra_op_parallelism_threads(4)  # Adjust number of threads for your system
    tf.config.threading.set_inter_op_parallelism_threads(4)
    main()
