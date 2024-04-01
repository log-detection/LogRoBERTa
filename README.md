# LogRoBERTa: An Innovative Model for Detecting Hidden Anomalies in Long and Complex Log Sequences

This paper introduces an innovative log detection model named ***LogRoBERTa***. Leveraging the RoBERTa model for pre-training, LogRoBERTa captures contextual information from software logs, comprehends complex log structures, and employs Attention-based Bi-LSTM for log anomaly detection.

![alt](img/overview.jpg)

## Datasets

- LogRoBERTa and other benchmark models are implemented on [HDFS](https://github.com/logpai/loghub/tree/master/HDFS) and [BGL](https://github.com/logpai/loghub/tree/master/BGL) datasets

- Please note that due to the large size of the datasets, we have not included the original files in the GitHub repository. You can download the corresponding dataset files by clicking on the provided link.

## Experiment

- You can refer to the code structure of our LogRoBERTa model in the files *`LogRoBERTa_HDFS.py`* and *`LogRoBERTa_BGL.py`*. *`LogRoBERTa_HDFS.py`* is designed for the HDFS dataset, while *`LogRoBERTa_BGL.py`* is designed for the BGL dataset.

- In the paper's Section ***V. EXPERIENCES -> C. RQ3: Comparative Enhancement Effects of Each Module in LogRoBERTa***, you will find our specific configurations and experimental results for our comparative experiments located in the **`Module`** folder.

#### TABLE IV: Settings of the Three Modules.
| Model       | Module 1 | Module 2 | Module 3  |
|:-----------:|:--------:|:--------:|:---------:|
| LogRoBERTa  | RoBERTa  | Bi-LSTM  | Attention |
| Comparison 1| BERT     | Bi-LSTM  | Attention |
| Comparison 2| -        | Bi-LSTM  | Attention |
| Comparison 3| RoBERTa  | LSTM     | Attention |
| Comparison 4| RoBERTa  | Bi-LSTM  | -         |

