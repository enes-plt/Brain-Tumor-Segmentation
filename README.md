## Brain Tumor Segmentation
As a team, Designed a deep learning - computer vision model to segment medical images (MRI, CT, X-ray), anatomical structures (bones, organs, tumors) to classify diseases, and leveraging advanced frameworks for image processing and data analysis.

## Creating a Model
1. Run train.py first, and input the necessary information.
2. Seed refers to the RNG of the program.
3. Epochs are the runs of the training program. More Epochs mean better accuracy, but takes much longer (my computer ran 1 epoch per hour)
4. Number of Batches refer to the processing speed. A higher batch size means the program will run each epoch faster. However it will require much more out of your computer. Unless you run tensorflow using a video card, I do not recommend a batch size larger than 16.
5. Once training is finished, a files folder containing files called model.keras and memory.txt will be created. This is the training data and a memory text file with the seed and epoch number. Do Not Edit or Remove memory.txt or else you will either get an error, or the test will be inaccurate to the training set.

## Testing the Model
1. Run test.py
2. This will create a results directory. This will contain all images resulted from testing the model.
3. Each image has four portions, each will be directed labelled on the image:
   - the original image of the brain scan.
   - the mask (the segmentation of the tumor)
   - the generated prediction
   - the MSE (Mean Squared Error)
4. test.py will now output four accuracy mean scores in the command prompt:
   - Precision: How many correct drawn pixels vs how many incorrect drawn pixels
   - Recall: How many correct pixels vs how many pixels missed
   - F1: Average of the previous two
   - Jaccard: How similar the prediction and mask are, basically Accuracy.
5. Individual accuracy scores will also be written to a CSV file: scores.csv

## Video Demostration
For the sake of video demoing, please use the link below to download both the model.keras, and memory.txt file I have generated so you don't have to wait several hours for your own.
Download the files folder.

The settings I used:
  Seed: 42
  Num_Epochs: 3

https://github.com/enes-plt/Brain-Tumor-Segmentation/assets/54ad0b0b-2cfa-4e04-8977-50615183436d
