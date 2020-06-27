<div align="center">
    <a href="https://github.com/SuperBruceJia/EEG-DL"> <img width="500px" src="https://github.com/SuperBruceJia/EEG-DL/raw/master/Logo.png"></a> 
</div>

---

# Welcome to EEG Deep Learning Library

**EEG-DL** is a Deep Learning (DL) library written by [TensorFlow](https://www.tensorflow.org) for EEG Tasks (Signals) Classification. It provides the latest DL algorithms and keeps updated. 

[![License](https://img.shields.io/badge/license-MIT-blue.svg)](https://github.com/SuperBruceJia/EEG-DL/blob/master/LICENSE)
[![Python 3](https://img.shields.io/badge/Python-3.x-green.svg)](https://www.anaconda.com/)
[![TensorFlow 1.13.1](https://img.shields.io/badge/TensorFlow-1.13.1-red.svg)](https://www.tensorflow.org/install)

## Documentation
**The supported models** include

| No.   | Model                                                  | Example         |
| :----:| :----:                                                 | :----:          |
| 1     | Deep Neural Networks                                   | [DNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/DNN.py) |
| 2     | Convolutional Neural Networks [[Paper]](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta) [[Tutorial]](https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow)| [CNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/CNN.py) |
| 3     | Deep Residual Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/1512.03385) | [ResNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/ResCNN.py) |
| 4     | Thin Residual Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/1902.10107) | [Thin ResNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/Thin_ResNet.py) |
| 5     | Densely Connected Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/1608.06993) | [DenseNet](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/DenseCNN.py) |
| 6     | Fully Convolutional Neural Networks [[Paper]](https://www.cv-foundation.org/openaccess/content_cvpr_2015/papers/Long_Fully_Convolutional_Networks_2015_CVPR_paper.pdf) | [FCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/Fully_Conv_CNN.py) |
| 7     | One Shot Learning with Siamese Networks [[Paper]](https://www.cs.cmu.edu/~rsalakhu/papers/oneshot1.pdf) [[Tutorial]](https://towardsdatascience.com/one-shot-learning-with-siamese-networks-using-keras-17f34e75bb3d) | [Siamese Networks](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/Siamese_Network.py) |
| 8     | Graph Convolutional Neural Networks [[Paper]](https://arxiv.org/abs/2006.08924) [[Presentation]](https://drive.google.com/file/d/1ecMbtZV2eH14sRAqWIIf1iRvDAC7DMDs/view?usp=sharing) [[Tutorial]](https://github.com/SuperBruceJia/eeg-gcns-net) | [GCN / Graph CNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/lib_for_GCN/GCN_Model.py) |
| 9     | Deep Residual Graph Convolutional Neural Networks      | [ResGCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/lib_for_GCN/ResGCN_Model.py) |
| 10    | Densely Connected Graph Convolutional Neural Networks  | [DenseGCN](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/lib_for_GCN/DenseGCN_Model.py) |
| 11    | Recurrent Neural Networks [[Paper]](https://arxiv.org/abs/2005.00777) | [RNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/RNN.py) |
| 12    | Attention-based Recurrent Neural Networks [[Paper]](https://arxiv.org/abs/2005.00777) | [RNN with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/RNN_with_Attention.py) |
| 13    | Bidirectional Recurrent Neural Networks [[Paper]](https://arxiv.org/abs/2005.00777) | [BiRNN](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/BiRNN.py) |
| 14    | Attention-based Bidirectional Recurrent Neural Networks [[Paper]](https://arxiv.org/abs/2005.00777) | [BiRNN with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/BiRNN_with_Attention.py) |
| 15    | Long-short Term Memory [[Paper]](https://arxiv.org/abs/2005.00777) | [LSTM](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/LSTM.py) |
| 16    | Attention-based Long-short Term Memory [[Paper]](https://arxiv.org/abs/2005.00777) | [LSTM with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/LSTM_with_Attention.py) |
| 17    | Bidirectional Long-short Term Memory [[Paper]](https://arxiv.org/abs/2005.00777) | [BiLSTM](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/BiLSTM.py) |
| 18    | Attention-based Bidirectional Long-short Term Memory [[Paper]](https://arxiv.org/abs/2005.00777) | [BiLSTM with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/BiLSTM_with_Attention.py) |
| 19    | Gated Recurrent Unit [[Paper]](https://arxiv.org/abs/2005.00777) | [GRU](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/GRU.py) |
| 20    | Attention-based Gated Recurrent Unit [[Paper]](https://arxiv.org/abs/2005.00777) | [GRU with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/GRU_with_Attention.py) |
| 21    | Bidirectional Gated Recurrent Unit [[Paper]](https://arxiv.org/abs/2005.00777) | [BiGRU](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/BiGRU.py) |
| 22    | Attention-based Bidirectional Gated Recurrent Unit [[Paper]](https://arxiv.org/abs/2005.00777) | [BiGRU with Attention](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Network/BiGRU_with_Attention.py) |

**One EEG Motor Imagery (MI) benchmark** is currently supported. Other benchmarks in the field of EEG can be found [here](https://github.com/meagmohit/EEG-Datasets).

| No.     | Dataset                                                                          |
| :----:  | :----:                                                                           |
| 1       | [EEG Motor Movement/Imagery Dataset](https://archive.physionet.org/pn4/eegmmidb/) [[Tutorial]](https://github.com/SuperBruceJia/EEG-Motor-Imagery-Classification-CNNs-TensorFlow)|

**The evaluation criteria** consists of

| Evaluation Metrics 					                                       |
| :----:                                                                    |
| Accuracy / Precision / Recall / F1 Score / Kappa Coefficient              |
| Receiver Operating Characteristic (ROC) Curve / Area under the Curve (AUC)|
| Confusion Matrix                                                          |
| Paired-wise t-test (via R language [[Tutorial]](https://www.analyticsvidhya.com/blog/2019/05/statistics-t-test-introduction-r-implementation/))                                                           |

The evaluation metrics are mainly supported for **four-class classification**. If you wish to switch to two-class or three-class classification, please modify [this file](https://github.com/SuperBruceJia/EEG-DL/blob/master/DL_Models/Evaluation_Metrics/Metrics.py) to adapt to your personal Dataset classes. Meanwhile, the details about the evaluation metrics can be found in [this paper](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta).

## Usage Demo: EEG Motor Movement/Imagery Dataset

1. Download the [EEG Motor Movement/Imagery Dataset](https://archive.physionet.org/pn4/eegmmidb/) via [this script](https://github.com/SuperBruceJia/EEG-DL/blob/master/Download_Raw_EEG_Data/EEG_Motor_Movement_Imagery_Dataset/MIND_Get_EDF.py).
2. Read the .edf files (One of the raw EEG signals formats) and save them into Matlab .m files via [this script](https://github.com/SuperBruceJia/EEG-DL/blob/master/Download_Raw_EEG_Data/EEG_Motor_Movement_Imagery_Dataset/Extract-Raw-Data-Into-Matlab-Files.py). FYI, this script must be executed under the **Python 2 environment (Python 2.7 is recommended)** due to some Python 2 syntax. If using Python 3 environment to run the file, there might be no error, but the labels of EEG tasks would be totally messed up.
3. Preprocessed the Dataset via the Matlab and save the data into the Excel files (training_set, training_label, test_set, and test_label) via [these scripts](https://github.com/SuperBruceJia/EEG-DL/tree/master/Preprocess_EEG_Data) with regards to different models. FYI, every lines of the Excel file is a sample, and the columns can be regarded as features, e.g., 4096 columns mean 64 channels X 64 time points. Later, the models will reshape 4096 columns into a Matrix with the shape 64 channels X 64 time points. You should can change the number of columns to fit your own needs, e.g., the real dimension of your own Dataset.
4. Train and test deep learning models for EEG signals / tasks classification via [the EEG-DL library](https://github.com/SuperBruceJia/EEG-DL/tree/master/DL_Models), which provides multiple SOTA DL models.
5. Read evaluation criterias (through iterations) via the [Tensorboard](https://www.tensorflow.org/tensorboard). You can follow [this tutorial](https://www.guru99.com/tensorboard-tutorial.html).  Then you can save the criterias into Excel .csv files.
6. Finally, draw beautiful paper photograph using Matlab or Python. Please follow [these scripts](https://github.com/SuperBruceJia/EEG-DL/tree/master/Draw_Photos).

P.S. I have tested all the files (Python and Matlab) under the macOS. Be advised that for some Matlab files, several Matlab functions are different between Windows Operating System (OS) and macOS. For example, I used "readmatrix" function to read CSV files in the MacOS. However, I have to use “csvread” function in the Windows because there was no such "readmatrix" Matlab function in the Windows.
If you have met similar problems, I recommend you to Google or Baidu them. You can definitely work them out.

## Citation

If you find our library useful, please considering citing our paper in your publications.
We provide a BibTeX entry below.

```
@article{hou2019novel,  
    year = 2020,  
    month = {feb},  
    publisher = {IOP Publishing},  
    volume = {17},  
    number = {1},  
    pages = {016048},  
    author = {Yimin Hou and Lu Zhou and Shuyue Jia and Xiangmin Lun},  
    title = {A novel approach of decoding {EEG} four-class motor imagery tasks via scout {ESI} and {CNN}},  
    journal = {Journal of Neural Engineering}  
}

@inproceedings{Lun2020GCNs,
    title={GCNs-Net: A Graph Convolutional Neural Network Approach for Decoding Time-resolved EEG Motor Imagery Signals},
    author={Xiangmin Lun and Shuyue Jia and Yimin Hou and Yan Shi and Yang Li and Hanrui Yang and Shu Zhang and Jinglei Lv},
    year={2020}
}

@inproceedings{Hou2020DeepFM,
    title={Deep Feature Mining via Attention-based BiLSTM-GCN for Human Motor Imagery Recognition},
    author={Yimin Hou and Shuyue Jia and Shu Zhang and Xiangmin Lun and Yan Shi and Yang Li and Hanrui Yang and Rui Zeng},
    year={2020}
}
```

Our papers can be downloaded from:
1. [A Novel Approach of Decoding EEG Four-class Motor Imagery Tasks via Scout ESI and CNN](https://iopscience.iop.org/article/10.1088/1741-2552/ab4af6/meta)<br>

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture2.png" alt="Project2">
</div>

---

2. [GCNs-Net: A Graph Convolutional Neural Network Approach for Decoding Time-resolved EEG Motor Imagery Signals](https://arxiv.org/abs/2006.08924) [[Presentation]](https://drive.google.com/file/d/1ecMbtZV2eH14sRAqWIIf1iRvDAC7DMDs/view?usp=sharing) <br>

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture1.png" alt="Project2">
</div>

---

2. [Deep Feature Mining via Attention-based BiLSTM-GCN for Human Motor Imagery Recognition](https://arxiv.org/abs/2005.00777)

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture4.png" alt="Project2">
</div>

<div>
    <div style="text-align:center">
    <img width=99%device-width src="https://github.com/SuperBruceJia/SuperBruceJia.github.io/raw/master/imgs/Picture5.png" alt="Project2">
</div>

## Contribution

We always welcome contributions to help make EEG-DL Library better. If you would like to contribute or have any question, please don't hesitate to <a href="http://shuyuej.com/">contact me</a>, and my email is <a href="shuyuej@ieee.org">shuyuej@ieee.org</a>.

## Organizations
The library was created and open-sourced by Shuyue Jia @ Human Sensor Laboratory, School of Automation Engineering, Northeast Electric Power University, Jilin, China.<br>
<a href="http://www.neepu.edu.cn/"> <img width="150" height="150" src="https://github.com/SuperBruceJia/EEG-DL/raw/master/NEEPU.png"></a>