# NMT-NL2SPARQL (Neural Machine Translation of Natural Language to SPARQL)
This repository contains the source code and instructions for training and evaluating different neural machine translation models to convert natural language (NL) queries into SPARQL queries. The project focuses on enhancing the understanding and conversion of NL queries into structured queries using knowledge graphs.

## Table of Contents
- [Project Description](#project-description)
- [Data Files](#data-files)
- [Experiment Setup Guide](#experiment-setup-guide)
- [Evaluation](#evaluation)
- [Acknowledgments](#acknowledgments)

## Project Description
This project aims to build and evaluate neural machine translation models for translating natural language queries into SPARQL queries. It leverages various techniques, including training with FastText, using GloVe embeddings, and incorporating Knowledge Graph Vectors. Different models are trained and compared for their performance in NL to SPARQL translation.

## Data Files
### DBpedia Dataset
[dbpedia.zip](https://drive.google.com/file/d/1HPrvqDJElp2EuZHG7MEtnpvWn-IFWBUj/view?usp=sharing)
(includes: labels_en.ttl, mappingbased_objects_en.ttl, instance_types_en.ttl)

### Processed DBpedia Dataset
[transformed_dbpedia.zip](https://drive.google.com/file/d/1XcgDake1m6ZCiGHK3qnz_0pBvyMBY0GF/view?usp=sharing)
(includes: tans_labels.ttl, trans_objects.ttl, trans_types.ttl)
 
[all.ttl.zip](https://drive.google.com/file/d/1ii3Hn0YmqVInpp6sdD3EaJDvjB0JY4ZG/view?usp=sharing)
(includes: all.ttl)

### Knowledge Graph Vectors Obtained by Training with fastText
[embedding.bin](https://drive.google.com/file/d/1gX0KIX4TSVaJp_92CNt_kC9b1Hcn3oJX/view?usp=sharing)

### Processed Knowledge Graph Vectors
[embedding.vec](https://drive.google.com/file/d/1EIyhNCC0q5bxKAuPG5JYWA12yKeCicgd/view?usp=sharing)

### GloVe
[glove.6B.zip](https://drive.google.com/file/d/1Ux_dHXe1w6q_c_3jd27wgBy_-zcko_8c/view?usp=sharing)
(includes: glove.6B.50d.txt, glove.6B.100d.txt, glove.6B.200d.txt, glove.6B.300d.txt)

### Processed Corpus
A corpus of natural languages that have been preprocessed and segmented into SPARQL queries. train.en, train.sparql are training sets for model training, dev.en, dev.sparql are development sets for model training, and test.en, test.sparql are test sets for model evaluation.
[monument.zip](https://drive.google.com/file/d/1ZL7bN8cA6UPJDv_1lS8Ea6B69pNoQvrh/view?usp=sharing
)
(includes: train.en, train.sparql, dev.en, dev.sparql, test.en, test.sparql)

[monument50.zip](https://drive.google.com/file/d/1fxntOeor_EDd43WK5G3jxfPiJ_M3HjTR/view?usp=sharing) 
(includes: train.en, train.sparql, dev.en, dev.sparql, test.en, test.sparql)

[monument80.zip](https://drive.google.com/file/d/19mVaQB9gxzXbybsaDDeKiSE2e_6OnWTa/view?usp=sharing)
(includes: train.en, train.sparql, dev.en, dev.sparql, test.en, test.sparql)

[lcquad.zip](https://drive.google.com/file/d/1ClwuQ0iOynYZQE7bVfgUepHYFphJkk7F/view?usp=sharing
)
(includes: train.en, train.sparql, dev.en, dev.sparql, test.en, test.sparql)

## Experiment Setup Guide
### Data Preprocessing
Step 1: Download the dbpedia.zip file and extract the files labels_en.ttl, mappingbased_objects_en.ttl, instance_types_en.ttl.

Step 2: Run the code in transform_data_of_dbpedia.ipynb to preprocess the data in labels_en.ttl, mappingbased_objects_en.ttl, instance_types_en.ttl to get the processed files: tans_ labels.ttl, trans_objects.ttl, trans_types.ttl.

Step 3: Run the code in transform_data_of_dbpedia.ipynb to integrate the tans_ labels.ttl, trans_objects.ttl, trans_types.ttl files into the file format all.ttl used to run the fastText model.

### Training FastText Model
Step 1: Since this training process requires high CPU and running memory, the most convenient way is to use the TPU provided in Google colab for training.

Step 2: Run the code of dbpedia_embedding.ipynb to configure the environment.

Step 3: Run the code of dbpedia_embedding.ipynb to train the processed DBpedia knowledge base data file all.ttl using the fastText model.

Step 4: Run the bin_to_vec.py file on the embedding.bin file obtained from the model training to obtain the knowledge graph vector embedding.vec.

### GloVe
Step 1: Download glove.6B.zip.

Step 2: Extract glove.6B.zip.

### Get the Processed Corpus
Step 1: Download monument.zip, monument50.zip, monument80.zip, lcquad.zip respectively.

Step 2: Extract the zip files monument.zip, monument50.zip, monument80.zip, lcquad.zip.

### Training Transformer Model
Step 1: Since the training process of this model is more demanding on GPU, the most convenient way is to use the GPU provided in Google colab to speed up the training process.

Step 2: Run the code in transformer-mon.ipynb, transformer-mon50.ipynb, transformer-mon80.ipynb, transformer-lcquad.ipynb in the models_training folder respectively for the environment configuration and training of the transformer model (hyperparameters are all set).

### Training GVT Model
Step 1: Since the training process of this model is more demanding on the GPU, the most convenient way to accelerate the training process is to use the GPU provided in Google colab.

Step 2: Copy the glove.6B.300d.txt file into the colab environment.

Step 3: Run the code in glove-mon.ipynb, glove-mon50.ipynb, glove-mon80.ipynb, and glove-lcquad.ipynb in the models_training folder for environment configuration and GVT model training, respectively (hyperparameters are all set).

### Training KGET Model
Step 1: Since the training process of this model is more demanding on GPU, the most convenient way is to use the GPU provided in Google colab to accelerate the training process.

Step 2: Copy the embedding.vec file into the colab environment.

Step 3: Run the code in kge-mon.ipynb, kge-mon50.ipynb, kge-mon80.ipynb, and kge-lcquad.ipynb in the models_training folder to configure the environment and train the KGET model, respectively (the hyper-parameters are all set).

## Evaluation
Step 1: Translate test.en from all the corpora using the trained model files to get the trans_test.sparql file.

Step 2: Execute compute-accuracy.py, compute-bleu.py, compute-rouge-l.py on trans_test.sparql and the translated test.sparql file to get the accuracy, BLEU, and Rouge-L scores of the model test.

## Acknowledgments
This project and report would not have been possible without the help of my supervisor (Dr. Albert Meroño-Peñuela) and those great open source projects. I send my sincere thanks to all those who have dedicated themselves to this topic and hope that we will be able to show its value to the world in the near future.

-[OpenNMT-py](https://github.com/OpenNMT/OpenNMT-py)

-[fastText](https://github.com/facebookresearch/fastText)

-[GloVe](https://nlp.stanford.edu/projects/glove/)

-[Neural SPARQL Machines](https://github.com/LiberAI/NSpM)

-[LC-QuAD](https://figshare.com/projects/LC-QuAD/21812)

