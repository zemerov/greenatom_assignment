# Assignment for greenatom internship

The original dataset ai.stanford.edu/~amaas/data/sentiment/

## Overview
The project consists of two parts. Neural Network classifier 
(sources in /classifier) and web-server on django.

File presentation.ipynb allows you to run all experiment on google colab.



## Classification results
I used 2 popular Neural net architectures: GRU and Conv net.

Nets predict review score from 1 to 10.

Both models trained on 3 epochs with Adam optimizer and negative log likehood loss.

There are results of several experiments. Although models predict 
score from 1 to 10, in the original task reviews with score more than 7 considered positive 
and less than 5 considered as negative. All columns except Real accuracy calculated according 
to this rule.  

### GRU

| Model            | Real accuracy  |Binary Accuracy | F1    | Precision | Recall |
| ---------------- |-------------- :|:--------------:| -----:|-----------|--------|
| GRU + dropout    | 0.15           | 0.63           | 0.51  | 0.75      | 0.38   |
| GRU              | 0.11           | 0.57           | 0.41  | 0.64      | 0.31   |

![GRU train](/img/gru.png)
### Neural net with conv layers

| Model                | Real accuracy  |Binary Accuracy | F1    | Precision | Recall |
| ----------------     |-------------- :|:--------------:| -----:|-----------|--------|
| Big ConvNet + dropout| 0.183          | 0.74           | 0.71  | 0.82      | 0.62   |
| ConvNet              | 0.171          | 0.72           | 0.69  | 0.76      | 0.63   |

![GRU train](/img/cnn.png)

 As we can see ConvNet has a better score. Thus, it is out choice for website.
 
 All trained model parameters here: drive.google.com/drive/folders/1S7h-KGgstiYbPRJSlNR39ulGSEAArl_w?usp=sharing