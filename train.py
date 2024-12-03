import kagglehub
import os, sys, numpy as np, cv2, tensorflow as tf
from glob import glob
from tensorflow.keras.callbacks import ModelCheckpoint, CSVLogger, ReduceLROnPlateau, EarlyStopping, TensorBoard
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from unet import build_unet
from metrics import dice_loss, dice_coef

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

height = 256
width = 256

# Default Parameters
seed = 0
batch_size = 10
num_epochs = 1

data_path = kagglehub.dataset_download("nikhilroxtomar/brain-tumor-segmentation")

# Creates a directory if directory exists
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Loads the dataset, and splits the dataset into three sets: training set, testing set, and validation set
def load_dataset(path, seed, split=0.2):
    images = sorted(glob(os.path.join(path, "images", "*.png")))
    masks = sorted(glob(os.path.join(path, "masks", "*.png")))

    split_size = int(len(images) * split)

    X_train, X_valid = train_test_split(images, test_size=split_size, random_state=seed)
    y_train, y_valid = train_test_split(masks, test_size=split_size, random_state=seed)
    
    X_train, X_test = train_test_split(X_train, test_size=split_size, random_state=seed)
    y_train, y_test = train_test_split(y_train, test_size=split_size, random_state=seed)

    return (X_train, y_train), (X_valid, y_valid), (X_test, y_test)

# Reads the image and returns the image as list
def read_image(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_COLOR)
    x = cv2.resize(x, (width, height))
    x = x / 255.0
    x = x.astype(np.float32)
    return x

# Reads the mask and returns the mask as list
def read_mask(path):
    path = path.decode()
    x = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
    x = cv2.resize(x, (width, height))
    x = x / 255.0
    x = x.astype(np.float32)
    x = np.expand_dims(x, axis=-1)
    return x

# Transform the features of the image and masks
def tf_parse(x, y):
    def _parse(x, y):
        x = read_image(x)
        y = read_mask(y)
        return x, y
    
    x, y = tf.numpy_function(_parse, [x, y], [tf.float32, tf.float32])
    x.set_shape([height, width, 3])
    y.set_shape([height, width, 1])
    return x, y

# Takes dataset and batchs in order for training
def tf_dataset(x, y, batch=2):
    dataset = tf.data.Dataset.from_tensor_slices((x , y))
    dataset = dataset.map(tf_parse)
    dataset = dataset.batch(batch)
    dataset = dataset.prefetch(10)
    return dataset

# Generate the model
def generate_model():
    np.random.seed(seed)
    tf.random.set_seed(seed)

    learning_rate = 1e-4
    
    model_path = os.path.join("files", "model.keras")
    csv_path = os.path.join("files", "log.csv")
    
    data_path = kagglehub.dataset_download("nikhilroxtomar/brain-tumor-segmentation")
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_dataset(data_path, seed)
    print("Location where the dataset is downloaded: ", data_path)

    # Create the training and validation set for training the model
    train_dataset = tf_dataset(X_train, y_train, batch=batch_size)
    valid_dataset = tf_dataset(X_valid, y_valid, batch=batch_size)

    # Build the UNET model
    model = build_unet((height, width, 3))
    model.compile(loss=dice_loss, optimizer=Adam(learning_rate), metrics=[dice_coef])

    # Callbacks to prevent overfitting and regulate training
    callbacks = [
        ModelCheckpoint(model_path, verbose=1, save_best_only=True),
        ReduceLROnPlateau(monitor='val_loss', factor=0.1, patience=5, min_lr=1e-7, verbose=1),
        CSVLogger(csv_path),
        EarlyStopping(monitor='val_loss', patience=20, restore_best_weights=False),
    ]

    # Apply model parameters
    model.fit(
        train_dataset,
        epochs=num_epochs,
        validation_data=valid_dataset,
        callbacks=callbacks
    )

# Writes a txt file called memory.txt: This file contains information for the seed, and number of epochs
def write_memory(seed, epoch):
    filePath = os.path.join("files", "memory.txt")
    f = open(filePath,'w')
    f.write(str(seed)+'\n'+str(epoch))
    f.close()

# Run this function when train.py is run from the command line.
if __name__ == "__main__":

    create_dir("files")

    if len(sys.argv) > 1:
        seed = sys.argv[1]
    if len(sys.argv) > 2:
        batch_size = sys.argv[2]
    if len(sys.argv) > 3:
        num_epochs = sys.argv[3]
    # User Input Implementation on running train.py
    if len(sys.argv) == 1:
        while True:
            seed = input("Insert the seed, this determines the train, test, and validation split of the dataset (Leave blank for default: 0): ")
            try:
                if seed == "":  
                    seed = 0
                    break
                seed = int(seed)
                break
            except ValueError:
                print("Error: Seed needs to be an integer.")
                continue
        while True:
            num_epochs = input("Insert the number of epochs, this determines how many epochs the training will go through. More epochs takes longer, but improves accuracy (Leave blank for default: 1): ")
            try:
                if num_epochs == "":
                    num_epochs = 1
                    break
                num_epochs = int(num_epochs)
                break
            except ValueError:
                print("Error: Number of Epochs needs to be an integer.")
                continue
        while True:
            batch_size = input("Insert the batch size, this determines how many batches the training will go through at once. WARNING: Depending on your computer's specs, and whether or not tensorflow is using the GPU or CPU, it is ill advised to set batch size any higher than 16. (Leave blank for default: 8): ")
            try:
                if batch_size == "":
                    batch_size = 8
                    break
                batch_size = int(batch_size)
                break
            except ValueError:
                print("Error: Batch size needs to be an integer.")
                continue
        # Writes the seed and number of epochs to memory.txt
        write_memory(seed, num_epochs)
    # Generate model and begin training
    generate_model()
   

    