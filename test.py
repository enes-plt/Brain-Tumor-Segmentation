import os, numpy as np, pandas as pd, cv2, tensorflow as tf, warnings

from tqdm import tqdm
from tensorflow.keras.utils import CustomObjectScope
from sklearn.metrics import f1_score, jaccard_score, precision_score, recall_score
from metrics import dice_loss, dice_coef
from train import load_dataset, data_path

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "2"

warnings.filterwarnings("ignore")

seed = 0

height = 256
width = 256

# Creates a directory if it doesn't exist
def create_dir(path):
    if not os.path.exists(path):
        os.makedirs(path)

# Reads the memory file and returns the lines as output
def read_memory():
    filePath = os.path.join("files", "memory.txt")
    f = open(filePath, 'r')
    return f.readlines()

# Saves the final image and writes image to results folder
def save_results(image, mask, y_pred, save_image_path):
    mask = np.expand_dims(mask, axis=-1)
    mask = np.concatenate([mask, mask, mask], axis=-1)

    y_pred = np.expand_dims(y_pred, axis=-1)
    y_pred = np.concatenate([y_pred, y_pred, y_pred], axis=-1)
    y_pred = y_pred * 255

    line = np.ones((height, 10, 3)) * 255

    cat_images = np.concatenate([image, line, mask, line, y_pred], axis=1)
    cv2.imwrite(save_image_path, cat_images)

# Calculates the Mean Squared Error (MSE) for each image between the mask and prediction
def calcMSE(mask, pred):
    error = np.sum((mask.astype('float') - pred.astype('float')) ** 2)
    error /= float(mask.shape[0] * mask.shape[1])

    return error

# Runs on program start
if __name__ == "__main__":
    # Read the memory for the seed and number of epochs
    memory = read_memory()
    seed = int(memory[0])
    epochs = int(memory[1])
    # Apply the seed for the random number generator
    np.random.seed(seed)
    tf.random.set_seed(seed)

    create_dir("results")

    # Load the model
    with CustomObjectScope({"dice_coef": dice_coef, "dice_loss": dice_loss}):
        model = tf.keras.models.load_model(os.path.join("files", "model.keras"))
    # Load the dataset
    (X_train, y_train), (X_valid, y_valid), (X_test, y_test) = load_dataset(data_path, seed)

    print("Dataset Location: "+data_path)
    

    SCORE = []
    for x, y in tqdm(zip(X_test, y_test), total=len(y_test)):
        name = x.split("\\")[-1]
        # Read Image
        image = cv2.imread(x, cv2.IMREAD_COLOR)
        image = cv2.resize(image, (width, height))
        x = image / 255.0
        x = np.expand_dims(x, axis=0)
        # Read Mask
        mask = cv2.imread(y, cv2.IMREAD_GRAYSCALE)
        mask = cv2.resize(mask, (width, height))
        # Predict Segmentation
        y_pred = model.predict(x, verbose=0)[0]
        y_pred = np.squeeze(y_pred, axis=-1)
        y_pred = y_pred >= 0.5
        y_pred = y_pred.astype(np.int32)

        # Write text on Images
        image = cv2.putText(image, text="Seed: "+str(seed)+" Num_Epochs: "+str(epochs),org=(10,30),fontFace=3, fontScale=.5,color=(255,255,255),thickness=1)
        image = cv2.putText(image, text="Image from Dataset",org=(10,240),fontFace=3, fontScale=.5,color=(255,255,255),thickness=1)
        maskCP = cv2.putText(mask.copy(), text="Mask from Dataset",org=(10,240),fontFace=3, fontScale=.5,color=(255,255,255),thickness=1)
        y_predCP = cv2.putText(y_pred.copy(), text="Predicted Segmentation",org=(10,240),fontFace=3, fontScale=.5,color=(255,255,255),thickness=1)
        y_predCP = cv2.putText(y_predCP, text="MSE: "+str(calcMSE(mask,y_pred)),org=(10,30),fontFace=3, fontScale=.5,color=(255,255,255),thickness=1)

        # Save image and write to the results folder
        save_image_path = os.path.join("results", name)
        save_results(image, maskCP, y_predCP, save_image_path)
        
        # Flatten the mask and prediction for calculating metrics
        mask = mask/255.0
        mask = (mask > 0.5).astype(np.int32).flatten()
        y_pred = y_pred.flatten()

        # Calculate metrics
        f1_value = f1_score(mask, y_pred, labels=[0,1],average="binary")
        jaccard_value = jaccard_score(mask, y_pred, labels=[0,1], average="binary")
        precision_value = precision_score(mask, y_pred, labels=[0,1],average="binary")
        recall_value = recall_score(mask, y_pred, labels=[0,1], average="binary")
        SCORE.append([name, f1_value, jaccard_value, precision_value, recall_value])

    # Print the accuracy scores
    score = [s[1:] for s in SCORE]
    score = np.mean(score, axis=0)
    print("Average Mean Accuracy Scores")
    print(f"F1 Score: {score[0]:0.5f}")
    print(f"Jaccard Score: {score[1]:0.5f}")
    print(f"Precision Score: {score[2]:0.5f}")
    print(f"Recall Score: {score[3]:0.5f}")

    # Write scores to a CSV file
    scores = pd.DataFrame(SCORE, columns=["Image", "F1", "Jaccard","Precision","Recall"])
    scores.to_csv("files/scores.csv")