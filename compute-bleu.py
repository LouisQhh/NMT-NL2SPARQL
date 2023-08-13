# Run this file from CMD/Terminal
# Example Command: python3 compute-bleu.py test.sparql trans_test.sparql

import sys
import sacrebleu

target_test = sys.argv[1]  # Get the reference translation file path from command-line arguments
target_pred = sys.argv[2]  # Get the predicted translation file path from command-line arguments

# Open the test dataset human translation file and detokenize the references
refs = []

with open(target_test) as test:
    for line in test: 
        line = line.strip()  # Remove leading and trailing whitespace
        refs.append(line)

print("Reference 1st sentence:", refs[0])  # Print the first sentence of reference translations

refs = [refs]  # Convert the list of references to a list of list(s) as required by sacreBLEU

# Open the translation file by the NMT model and detokenize the predictions
preds = []

with open(target_pred) as pred:  
    for line in pred: 
        line = line.strip()  # Remove leading and trailing whitespace
        preds.append(line)

print("MTed 1st sentence:", preds[0])  # Print the first sentence of predicted translations

# Calculate and print the BLEU score
bleu = sacrebleu.corpus_bleu(preds, refs)  # Compute the BLEU score using sacreBLEU
print("BLEU: ", bleu.score)  # Print the computed BLEU score
