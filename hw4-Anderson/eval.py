import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Define parameters
imgSize = 128  # Ensure this matches the training image size
imageExtensions = ('.jpg', '.jpeg', '.png')

def loadImagesFromDirectory(directory, label):
    images, labels = [], []
    for imgFile in os.listdir(directory):
        if imgFile.endswith(imageExtensions):
            imgPath = os.path.join(directory, imgFile)
            img = tf.keras.preprocessing.image.load_img(imgPath, target_size=(imgSize, imgSize))
            imgArray = tf.keras.preprocessing.image.img_to_array(img)
            images.append(imgArray)
            labels.append(label)
    return images, labels

def loadData(dataDir):
    posImages, posLabels = loadImagesFromDirectory(os.path.join(dataDir, 'positive'), 1)
    negImages, negLabels = loadImagesFromDirectory(os.path.join(dataDir, 'negative'), 0)
    
    images = np.array(posImages + negImages) / 255.0  # Normalize to [0,1]
    labels = np.array(posLabels + negLabels)
    
    return images, labels

def evaluateModel(model, images, labels, outputFile):
    print("Evaluating model...")
    loss, accuracy = model.evaluate(images, labels, verbose=1)
    predictions = model.predict(images)
    predClasses = (predictions > 0.5).astype("int32").flatten()
    
    precision = precision_score(labels, predClasses)
    recall = recall_score(labels, predClasses)
    confMatrix = confusion_matrix(labels, predClasses)

    # Format the results
    results = f"""
    Test Results:
    -------------------
    Number of test images: {len(images)}
    Positive images: {np.sum(labels)}
    Negative images: {len(labels) - np.sum(labels)}
    Test Loss: {loss:.4f}
    Test Accuracy: {accuracy:.4f}
    Precision: {precision:.4f}
    Recall: {recall:.4f}
    Confusion Matrix:
    {confMatrix}
    """

    print(results)  # Display results in console
    with open(outputFile, "w") as file:
        file.write(results)  # Save results to file

    print(f"Results saved to {outputFile}")

def displayIndividualPredictions(images, labels, predClasses, outputFile):
    with open(outputFile, "a") as file:
        file.write("\nIndividual Predictions:\n")
        for i, (trueLabel, predLabel) in enumerate(zip(labels, predClasses)):
            trueClass = "Positive" if trueLabel == 1 else "Negative"
            predClass = "Positive" if predLabel == 1 else "Negative"
            result = "Correct" if trueLabel == predLabel else "Wrong"
            file.write(f"Image {i+1}: True: {trueClass}, Predicted: {predClass}, {result}\n")

    print(f"Individual predictions saved to {outputFile}")

def main():
    scriptDir = os.path.dirname(__file__)
    dataDir = os.path.join(scriptDir, 'Dataset')
    modelPath = os.path.join(scriptDir, 'Output', 'model.h5')
    outputFile = os.path.join(scriptDir, 'Output', 'model_eval.txt')

    print("Loading model...")
    model = load_model(modelPath)

    print("Loading data...")
    images, labels = loadData(dataDir)

    evaluateModel(model, images, labels, outputFile)
    
    predClasses = (model.predict(images) > 0.5).astype("int32").flatten()
    displayIndividualPredictions(images, labels, predClasses, outputFile)

if __name__ == "__main__":
    main()
