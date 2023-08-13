# Run this file from CMD/Terminal
# Example Command: python3 compute-accuracy.py test.sparql trans_test.sparql

import sys

def compute_accuracy(target_test, target_pred):
    # Open the test dataset human translation file and detokenize the references
    refs = []

    with open(target_test) as test:
        for line in test:  # Iterate through each line in the test file
            line = line.strip()  # Remove leading and trailing whitespace
            refs.append(line)  # Add the line to the list of reference translations

    print("Reference 1st sentence:", refs[0])  # Print the first sentence of reference translations

    # Open the translation file by the NMT model and detokenize the predictions
    preds = []

    with open(target_pred) as pred:
        for line in pred:  # Iterate through each line in the prediction file
            line = line.strip()  # Remove leading and trailing whitespace
            preds.append(line)  # Add the line to the list of predicted translations

    print("MTed 1st sentence:", preds[0])  # Print the first sentence of predicted translations

    # Compute accuracy
    total_words = 0  # Initialize a counter for total words
    correct_words = 0  # Initialize a counter for correct words

    for ref, pred in zip(refs, preds):  # Iterate through paired reference and predicted translations
        ref_words = ref.split()  # Split the reference translation into words
        pred_words = pred.split()  # Split the predicted translation into words

        total_words += len(ref_words)  # Increment total word count with the number of words in the reference translation

        for r, p in zip(ref_words, pred_words):  # Iterate through paired reference and predicted words
            if r == p:  # Check if the reference word matches the predicted word
                correct_words += 1  # Increment the correct word count

    accuracy = correct_words / total_words  # Calculate the accuracy as the ratio of correct words to total words
    return accuracy

if __name__ == "__main__":
    target_test = sys.argv[1]  # Get the reference translation file path from command-line arguments
    target_pred = sys.argv[2]  # Get the predicted translation file path from command-line arguments

    accuracy = compute_accuracy(target_test, target_pred)  # Calculate the accuracy using the provided function
    print("Accuracy: ", accuracy)  # Print the calculated accuracy