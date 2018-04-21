# Kaggle Facial Keypoints Detection with Keras

<!-- TOC -->

- [Kaggle Facial Keypoints Detection with Keras](#kaggle-facial-keypoints-detection-with-keras)
    - [Memo 简介](#memo-简介)
    - [Dataset 数据集](#dataset-数据集)
    - [Contents 内容](#contents-内容)

<!-- /TOC -->

## Memo 简介

This is a step by step replica of [Kaggle Facial Keypoints DetectionをKerasで実装する][m1] in Japanese, many thanks to [Shinya Yuki][m2].

The original [Daniel Nouri's implementation][m3] was built on [Lasagne][m4], a library for an obsolete deep learning framework [Theano][m5].

这是我学习[Kaggle面部标识点识别][m1]博文的代码复现，非常感谢原作者[真也雪][m2]。感谢原作者[Daniel Nouri][m3]基于[Lasagne][m4]（一个[Theano][m5]的高级框架）的[博文][m3]。

[m1]:https://elix-tech.github.io/ja/2016/06/02/kaggle-facial-keypoints-ja.html
[m2]:https://twitter.com/shinyaelix
[m3]:http://danielnouri.org/notes/2014/12/17/using-convolutional-neural-nets-to-detect-facial-keypoints-tutorial/
[m4]:https://github.com/benanne/Lasagne
[m5]:http://deeplearning.net/software/theano/

## Dataset 数据集

[Kaggle][d1] hosts facial keypoints detection competition and all you need is [here][d2].

文中用到的 [Kaggle][d1] 数据集请[点击这里下载][d2]。

[d1]:https://www.kaggle.com
[d2]:https://www.kaggle.com/c/facial-keypoints-detection/data

## Contents 内容

[In the Jupyter Notebook (click here to access)][c1], two neural network models are implemented step by step, a single hidden layer model and a CNN model, and they are named to 1 and 2 in annotation respectively in following histograms.

[点击这里打开Jupyter Notebook][c4]，可以看到单隐层网络模型和CNN模型部署的详细步骤，下图中分别标为1、2.

Loss:
![loss-compare][c2]

Prediction
![pred-compare][c3]

[c1]:FacialKeypoints.ipynb
[c2]:img/hist-compare.png
[c3]:img/pred-compare.png
[c4]:FacialKeypointsCN.ipynb