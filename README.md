# Classification of Salmon Anomaly Detection (SAD) Transcript Output with Bidirectional Long Short-Term Memory (BiLSTM)
---

# Abstract 

Salmon Anomaly Detection (SAD) method for detecting expression anomalies in transcript data can be computationally expensive on large datasets, especially since the rate at which biological data is produced is ever-increasing. Therefore, it would be beneficial to distill SAD into a deep learning framework that might generalize better on new data.This work acts as a first step in building a stand-alone deep learning network to replace SAD and is essentially a sanity check for SAD output to determine if the results can be duplicated via a simplistic deep learning approach. In this project, I investigate the implementation of the Bidirectional Long Short-Term Memory model to perform binary classification on SAD Human body Map ERR030875 transcript output to classify it as ‘normal’ or ‘abnormal’ on the basis of expected  and observed coverage distribution. The BiLSTM model achieves a test accuracy of 97.9%. I also propose alternative heuristic approaches to integrate outlier transcripts in the classification task to improve overall model performance and scale to more datasets from the Salmon Anomaly Detection paper.

# Introduction 

Several methods have been developed in the past to estimate RNA-seq transcript abundance. This has been especially useful in understanding certain diseases, their underlying mechanisms, and subtypes. However, these suffered from poor scalability to the ever-increasing flow of ‘Big Data’ as well as being computationally expensive. Salmon [1] was developed to fill the gaps in NGS transcript abundance analysis and performed better than its predecessors in terms of speed and accuracy. Salmon works on the basis of : A mapping model and a bi-phasic protocol that considers expression levels, model parameters, and improves expression estimates.

Next, Salmon Anomaly Detection(SAD)[2] was developed to identify the shortcomings of the Salmon approach. The SAD method is able to identify expression anomalies which refer to transcript regions where expected and observed coverage distribution is different. It calculates the significance of the differential expression to determine if the transcript is normal or anomalous. If the anomaly metric, which looks at regions of deviation, is significant, then the transcript is anomalous, else, it’s normal. SAD also provides a methodology to correct these anomalous expression results. Therefore, it finds the cause of divergence, between adjustable and non-adjustable anomalies, as either an incomplete reference transcriptome or algorithmic issues. This is done via linear programming. Again, the anomaly metric is calculated. If it’s not significant then the transcript is adjustable anomalous, if it is, then the transcript is non-adjustable anomalous.

The SAD method can be computationally expensive on large datasets, especially since the rate at which biological data is produced is ever-increasing. Therefore, it would be beneficial to distill SAD into a deep learning framework that might better generalize on new data. This project acts as an important first step in building a stand-alone deep learning network to replace SAD and essentially works as a sanity check for SAD output to assess if the results can be replicated via a deep learning approach. In this project, I investigate the implementation of the Bidirectional LSTM model to classify SAD transcript output as ‘normal’ or ‘abnormal’ on the basis of expected and observed coverage distribution values. Long short term memory(LSTM) based models are especially useful for anomaly detection and label classification tasks for long sequences as they ‘remember’ data for a longer time period. Bidirectional LSTMs, in particular, are more powerful types of LSTMs that process data in the backward and forward direction and produce superior results that traditional LSTMs

# Data
## dataset 
SAD Human Body Map ERR030875 transcript output data was used as input for the BiLSTM model. 150240 total transcripts, with 96042 ‘normal’ transcripts and 54199 ‘abnormal’ transcripts were extracted. This data included transcript name, expected coverage vector, observed coverage vector, and label for each transcript 
<img width="659" alt="transcript" src="https://user-images.githubusercontent.com/77410526/119906482-4efb1780-bf1c-11eb-9d67-48feef5ba2a8.png">
SAD output for transcript ENST00000002829.7
<img width="470" alt="SAD" src="https://user-images.githubusercontent.com/77410526/119906518-5ae6d980-bf1c-11eb-9416-6f496f54754c.png">
Adjustable Anomaly in Human Body Map dataset(

Here, ‘Abnormal’ (with label 1) refers to transcripts with anomalous read coverage, i.e, where the observed coverage distribution deviates significantly in a region compared to the expected coverage distribution. ‘Normal’ transcripts (with label 0) have no such significant difference between expected and observed coverage distribution. This deviation may be the results of a missing transcript or the algorithm not working/optimizing properly etc.  (Figure 2). Salmon outputs the expected and observed distribution [1,2].



## Pre-Processing
Data from the SAD Human Body Map ERR030875 transcript output was pre-processed and combined to include all transcript names, expected coverage, observed coverage, and labels in one dataframe. All transcripts were labeled as being either ‘normal’ or ‘abnormal’. The ‘abnormal’ label is inclusive of both adjustable anomalous and non-adjustable anomalous transcripts. 

The coverage lengths of each transcript were inspected. Expected and observed coverage lengths were equal for a transcript. However, there were a few outlier transcripts with very long and very short expected/observed coverage lengths, ranging from 31 to 205,012  (Figure 3).  These coverage lengths have to be used further as inputs into the deep learning BiLSTM model -- this makes the training process computationally very expensive as there is only 1 GPU(Savanna server) and CPU resources are quickly exhausted. Transcripts with length > 5000 were filtered out which makes the modeling process easier while still preserving >90% total transcript in the dataset, i.e, 141504 transcripts.

<img width="464" alt="filtering" src="https://user-images.githubusercontent.com/77410526/119906688-adc09100-bf1c-11eb-9ec6-3ba526142208.png">
Graph of total transcript vs. coverage lengths.The black plot line shows 5000 coverage length across the 150240 total transcripts

# Methdology
Recurrent neural networks(RNN)[9] are a type of neural network used in the study of sequential data in the fields of speech recognition and speech synthesis etc., wherein the output of one layer forms the input of the next layer and internal memory is used for analysis. It suffers from the problem of vanishing gradient seen in case of long inputs when the gradient for updating model parameters keeps decreasing so learning is compromised.
LSTMs[3,4,5] are a type of RNN which were developed as a solution. Its architecture consists of three main layers: Input layer, hidden layer, and output layer. A layer in LSTM is composed of blocs/neurons, and each block has three gates: input, output, and forget gates . Input gate allows for updates to cell state, output gate controls next hidden state while forget gate dumps information about sequences that is not important. Cell state carries information about data through the network along with the help of sigmoid and tanh activation functions. Further, input data can be reshaped to be in a 3D format, i.e,  (samples/batch size, time step, features)

<img width="488" alt="LSTMbiLSTM" src="https://user-images.githubusercontent.com/77410526/119906757-cb8df600-bf1c-11eb-878d-7b3e8c22fac1.png">
LSTM and BiLSTM Architecture(

BiLSTM[7] is a type of LSTM model that involves duplicating the first recurrent layer in the network so that there are now two layers next to each other.The first layer takes in the original sequence while the second layer takes the reverse sequence as input(Figure 4). It performs better than unidirectional LSTM which only ‘remembers’ information of the past while bidirectional BiLSTMs processes the data from two directions.


## Architecture

A sequential BiLSTM model was implemented with a hidden layer of 100 neurons and an output layer that makes a single value prediction with sigmoid activation (Figure 5).

<img width="455" alt="summary" src="https://user-images.githubusercontent.com/77410526/119906873-17409f80-bf1d-11eb-8dfc-b3a858643f4f.png">
Model Summary

## Design of Experiment

Masking is a feature of Keras which allows us to ignore certain timesteps that may be missing. Sequence data oftentimes can have variable length but these were made equal to ensure the timestep is constant for inputting sequences into the BiLSTM model. This is called Padding, and it is a type of masking. It usually involves adding zeros to the ends of shorter sequences or to the beginning of sequences. In this project, padding has been done to the ends of expected and observed coverage sequences with length < maximum length. All transcripts now have a maximum coverage length of 5000. An alternative approach can be to input sequences in batches of similar coverage lengths.  

Train, validation, and test sets were created with 70\%, 20\%, and 10 \% split after padding (Figure 6), i.e, 99053 training samples, 28301 validation samples, and 14150 test samples. Input was reshaped into a 3D format for inputting into sequential model i.e, Total samples= 141504,Time steps =5000, Features=2

Random.seed was fixed for Numpy and TensorFlow before training the BiLSTM model to reduce randomness, initialize weights uniformly and allow for reproducibility of results

<img width="357" alt="workflow" src="https://user-images.githubusercontent.com/77410526/119906922-32abaa80-bf1d-11eb-9fd4-1429ede38513.png">
Schematic representation of the Design of Experiment 

## Hyperparameters
here are several hyperparameters to be considered while training a deep learning model including epochs, batch size, optimizer, loss, learning rate, etc. These must be tuned to achieve optimal results. Batch size refers to dividing up a dataset into smaller sizes before feeding into the model. After one batch is completed the internal model parameters are updated. Epochs refer to the number of times the algorithm ‘sees’ the training data. Internal model parameters are updated after each epoch, and one epoch can include one or more batches. Optimizers are used for the purpose of minimizing the model’s error rate at various learning rates while Loss Functions ascertain model loss while optimizing the model to keep it at a minimum in the subsequent iteration.
The batch size was set as 20, and the model was trained for 30 epochs. Binary Cross-Entropy was set up as the loss function and is a sigmoid activation along with Cross-Entropy loss.
ADAM [6] is an algorithm that was developed by combining the properties of RMSProp and AdaGrad optimizers. It performs gradient-based optimization by looking at parameters individually and calculating learning rates. ADAM optimizer defaults to a learning rate of 0.001. A small learning rate leads to slower training time as minute changes are made to weights in each iteration. Conversely, a large learning rate leads to faster training as it makes bigger changes in the weights but may get stuck at a local optimum.

BiLSTM model was implemented in Python language using Keras library with Tensorflow. One GPU was used on the Savanna server to execute code.

# Model evaluation Criterion
The model was evaluated on the basis of accuracy and model loss:
\[ Accuracy=(TP+TN)/(TP+TN+FP+FN)\] 		

Where, TP= True Positives, TN=True Negatives, FP=False Positives, FN=False Negatives

Graphs for model accuracy vs Loss for training and validation set over 30 epochs were also plotted 


# Results
Here are 2 graphs for model accuracy (Figure 7) and model loss (Figure 8) on training and validation set and model loss and accuracy metrics on the test set 

<img width="383" alt="accuracy" src="https://user-images.githubusercontent.com/77410526/119907035-6b4b8400-bf1d-11eb-863c-11e30196378e.png">
Model Accuracy vs. Epochs

<img width="365" alt="loss" src="https://user-images.githubusercontent.com/77410526/119907040-6f77a180-bf1d-11eb-8579-3b40dd612194.png">
Model Loss vs. Epochs

Test loss is 0.0838 while test accuracy is 0.979.


# Discussion

This model architecture allows for promising test set results with substantial convergence on the training and validation sets which may be uncommon with biological data, but that may be attributed to the use of BiLSTMs. Initially trained an LSTM and BiLSTM model on a small subset of the data (i.e, 500 transcripts) with 100 neurons and the biLSTM showed better results comparatively (Accuracy > 0.7). The outlier transcripts with very large coverage lengths have been omitted from the dataset which may be making the classification task easier.

Further application studies may be carried out here to verify the veracity of these results by applying the model to larger datasets with all transcripts along with some kind of cross-validation.


# Future work

## Method for classifying transcripts with large timestep

In the case of outlier transcripts with very large coverage lengths/timesteps -- the timestep can be split into pieces or “sub-timesteps” of fixed maximum length. In this project, the maximum length has been fixed at 5000. Trailing segments of timesteps/coverage values of length < 5000 can be padded. This would ensure uniform timesteps while inputting into the BiLSTM model. The label can be cloned for each of these sub-timesteps so the original timestep and sub-timesteps of one transcript have the same labels (Figure 9).
<img width="666" alt="Heuristic" src="https://user-images.githubusercontent.com/77410526/119908660-1a3d8f00-bf21-11eb-9542-defb9e55c9f5.png">
Representation of splitting timestep into sub-timesteps for a transcript ENST00000004531.14 with label 1

During testing with the model, each of the sub-timesteps will have a predicted label on which a majority vote can be applied to get a final predicted label for that transcript. 

## Alternative Heuristic Approach

Another way to split the timestep into chunks of 5000 timesteps would be to do it in regular intervals. Starting from ith position in (expected coverage, observed coverage) or timestep, we can pick 5000 length chunks. Next, we can start from the (i+1)th position(offset by 1) in (expected coverage, observed coverage) or timestep and similarly pick 5000 length chunks. The process can be repeated until the entire length of the transcript is traversed. Trailing segments of timesteps/coverage values of length < 5000 can be padded. This method allows us to look at all regions of the transcript and may be more representative of the SAD approach (Figure 10). 
<img width="653" alt="altheuristic" src="https://user-images.githubusercontent.com/77410526/119908710-39d4b780-bf21-11eb-8c31-abf02d4cb678.png">

Representation of 5000 length sub-timesteps picked starting from ith position(A)Sub-timestep picked starting from 1st position (B)Sub-timestep picked starting from 2nd position(C)Sub-timestep picked starting from 3rd position

Another option may be to have chunks of 1 timestep with shuffling to allow for maximum randomisation. 

The model may be applied on more datasets outputted from SAD like Human Body Map and GEUVADIS. The BiLSTM architecture may be further improved and extended with more hidden layers and increased complexity as needed while inputting larger and more complex datasets. Better modeling results may be obtained with more computational resources like GPUs and computing clusters. Finally,  a future goal is to build a stand-alone deep learning network from scratch that may replace the Salmon Anomaly Detection(SAD) method in detecting expression anomalies in RNA-seq transcript data.

# Conclusion
In this project, I performed binary classification on SAD transcript output to label transcripts as ‘normal’ and ‘abnormal’ using a Bidirectional LSTM model and achieved a test accuracy of 97.9 \% using the Human Body Map ERR030875 dataset.

# Acknowledgments
- Dr. Carl Kingford( Research Advisor) ckingsf@andrew.cmu.edu
- Hongyu ZHeng hongyuz1@andrew.cmu.edu
- Quang Minh Hoang qhoang@andrew.cmu.edu

# references

[1] Rob Patro, Geet Duggal, Michael I Love, Rafael A Irizarry ,Carl Kingsford . Salmon provides fast and bias-aware quantification of transcript expression. Nat Methods 14, 417–419 (2017)
[2] Cong Ma, Carl Kingsford. Detecting, Categorizing, and Correcting Coverage Anomalies of RNA-Seq Quantification.Cell Systems,9(6):589-599.e7(2019)
[3] Rasmus S. Andersen, Abdolrahman Peimankar, Sadasivan Puthusserypady.A deep learning approach for real-time detection of atrial fibrillation. Expert Systems with Applications,10.1016/j.eswa.(2018)
[4] Meriem Zerkouk,Belkacem Chikhaoui.Long Short Term Memory Based Model for Abnormal Behavior Prediction in Elderly Persons.Lecture Notes in Computer Science, vol 11862.(2019) [5]Hochreiter, S., Schmidhuber, J.: LSTM can solve hard long time lag problems . Neural Comput. 9, 1–32 (1997)
[6]Diederik P. Kingma, Jimmy Ba.Adam. A Method for Stochastic Optimization.arXiv:1412.6980 (2014)
[7] Mike Schuster , Kuldip K. Paliwal . Bidirectional Recurrent Neural Networks. IEEE Transaction on Signal Processing, VOL. 45, NO. 11, (1997)
[8]Arvind Mohan , Datta V. Gaitonde. A Deep Learning based Approach to Reduced Order Modeling for Turbulent Flow Control using LSTM Neural Networks .arXiv:1804:09269(2018)
 [9]Zachary C. Lipton, John Berkowitz, Charles Elkan. A Critical Review of Recurrent Neural Networks for Sequence Learning.arXiv:1506.00019(2019)







