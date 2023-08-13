# The loaded fasttext pre-trained word vectors are in vec format, but the fasttext unsupervised training is in bin format, so a conversion is needed
# The following code is officially recommended by fasttext:

# Importing necessary modules and libraries
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from __future__ import unicode_literals
from __future__ import division, absolute_import, print_function

from fasttext import load_model  # Importing the load_model function from fasttext module
import argparse  # Importing the argparse module for command-line argument parsing
import errno  # Importing the errno module for error handling

if __name__ == "__main__":
    # Creating an argument parser
    parser = argparse.ArgumentParser(
        description=("Print fasttext .vec file to stdout from .bin file")
    )
    # Adding an argument for the model file
    parser.add_argument(
        "model",
        help="Model to use",
    )
    # Parsing the command-line arguments
    args = parser.parse_args()

    # Loading the fasttext model from the provided .bin file
    f = load_model(args.model)

    # Getting the words from the model
    words = f.get_words()

    # Printing the number of words and the dimension of the vectors
    print(str(len(words)) + " " + str(f.get_dimension()))

    # Looping over each word
    for w in words:
        # Getting the vector representation of the word
        v = f.get_word_vector(w)

        # Converting the vector to a string
        vstr = ""
        for vi in v:
            vstr += " " + str(vi)

        try:
            # Printing the word and its vector representation
            print(w + vstr)
        except IOError as e:
            # Handling the IOError (EPIPE)
            if e.errno == errno.EPIPE:
                pass

# To convert .bin file to .vec file, run the following command in the terminal:
# python bin_to_vec.py unsupervised_data.bin > unsupervised_data.vec
