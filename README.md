Kaggle Abstraction and Reasoning Challenge -- Decision tree based part of 8-th place solution
=============================================================================================

This repository contains part of 8-th place submission.
We provide a framework for convenient experiments with machine learning approaches for
[Kaggle Abstraction and Reasoning Challenge](https://www.kaggle.com/c/abstraction-and-reasoning-challenge)

Solution Description
--------------------

Refer to [this kaggle post](https://www.kaggle.com/c/abstraction-and-reasoning-challenge/discussion/154436) for a solution description.

A [kaggle notebook](https://www.kaggle.com/msypetkowski/8-tasks-with-decision-trees-from-8-th-solution) made from this code gets 0.92 LB.

Setup
-----

This repository contains `Dockerfile` that covers all required dependencies.


Running
-------

To train logistic regression model on evaluation set, run:
```
./main.py -s evaluation --train-confidence <output_confidence_model_path>
```

For fast experiments with confidence model, it is recommended to first generate pickled raw results:
```
./main.py -s evaluation --pickle-results <output_pickle_path>
```

Using pickled raw results, to train logistic regression model on evaluation set, run:
```
./main.py -s evaluation --train-confidence <output_confidence_model_path> --picke-results <pickle_path_to_load>
```

Results
-------

Here, we show results on training and evaluation sets.
First, we run:
```
./main.py -t 12 --train-confidence confidence_train_final.pickle --pickle-results train_final.pickle -s training
./main.py -t 12  --load-confidence confidence_train_final.pickle --pickle-results evalu_final.pickle -s evaluation
./main.py -t 12 --train-confidence confidence_eval_final.pickle  --pickle-results evalu_final.pickle -s evaluation
./main.py -t 12  --load-confidence confidence_eval_final.pickle --pickle-results train_final.pickle -s training
```

Results summary of 2-nd command (confidence model trained on training set, inference on evaluation set):
```
Accuracy            0.18615751789976134 (78 / 419)
Accuracy-top1       0.15513126491646778 (65 / 419)
Accuracy-top3       0.18138424821002386 (76 / 419)
Task-accuracy       0.185 (74 / 400)
Task-accuracy-top1  0.155 (62 / 400)
Task-accuracy-top3  0.18 (72 / 400)
```

Results summary of 4-th command (confidence model trained on evaluation set, inference on training set):
```
Accuracy            0.3557692307692308 (148 / 416)
Accuracy-top1       0.30288461538461536 (126 / 416)
Accuracy-top3       0.3269230769230769 (136 / 416)
Task-accuracy       0.3475 (139 / 400)
Task-accuracy-top1  0.2925 (117 / 400)
Task-accuracy-top3  0.3175 (127 / 400)
```

Here, `Accuracy` means accuracy assuming that we take into consideration all predictions (from all configurations) as answer attempts.


Visualization
-------
We provide a convenient script for visualization:
```
./plot_pickled_results.py -i <pickle_path_to_load>
```
