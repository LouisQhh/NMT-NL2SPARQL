# Run this file from CMD/Terminal
# Example Command: python3 compute-rouge-l.py test.sparql trans_test.sparql

import sys
from rouge import Rouge

target_test = sys.argv[1]  # Get the reference translation file path from command-line arguments
target_pred = sys.argv[2]  # Get the predicted translation file path from command-line arguments

# Open the test dataset human translation file and detokenize the references
refs = []

with open(target_test) as test:
    for line in test: 
        line = line.strip()  # Remove leading and trailing whitespace
        refs.append(line)

print("Reference 1st sentence:", refs[0])  # Print the first sentence of reference translations

# Open the translation file by the NMT model and detokenize the predictions
preds = []

with open(target_pred) as pred:  
    for line in pred: 
        line = line.strip()  # Remove leading and trailing whitespace
        preds.append(line)

print("MTed 1st sentence:", preds[0])  # Print the first sentence of predicted translations

# Compute Rouge-L scores using the Rouge library
rouge = Rouge()
scores = rouge.get_scores(preds, refs, avg=True)
print("Rouge-L: ", scores['rouge-l']['f'])  # Print the average F1 score of Rouge-L