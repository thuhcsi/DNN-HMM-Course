# Task-3 : DNN-HMM training on TIMIT

## Preliminary
You must finish [Task-2](https://github.com/thuhcsi/DNN-HMM-Course/tree/main/T2-GMM-HMM) first to get a well-trained GMM-HMM model, which we will use for generating frame-level alignments.

First start a terminal and navigate to timit folder:
```sh
cd ~/kaldi/egs/timit/s5
```

Clone this repo to current folder:
```sh
git clone https://github.com/thuhcsi/DNN-HMM-Course.git
```

Navigate to DNN-HMM-Course:
```sh
cd DNN-HMM-Course
```

Install required packages
```sh
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt --user
```

## DNN Training

NOTE: by correctly running the baseline script, you can get basic score (16 points) of this task.

First navigate to DNN-HMM-Course/T3-DNN-HMM:
```sh
cd ~/kaldi/egs/timit/s5/DNN-HMM-Course/T3-DNN-HMM
```
Similar to Task-2, you can see a script `run.sh` which starts a full ASR experiment and performs data preparation, training, evaluation, forward, and decoding steps.

The script `run.sh` progressively creates the following files in the output directory `exp`:
* data/ali\_\*.ark: alignment files (binary format)
* data/ali\_\*.txt: alignment files (utf-8 text format)
* data/mfcc\_apply\_cmvn\_context\_\*.ark: MFCC features with CMVN and context
* data/mfcc\_apply\_cmvn\_context\_\*.scp: index of MFCC features
* data/phone\_counts: the number of occurrences of each state
* data/text\_\*: directory containing ground truth transcripts

The script `dataloader.py` will load features and alignments for training and `train.py` will save the trained model and write log to `exp/$MODEL`.

Let's run `run.sh` stage by stage.

---
---

Run stage-0: generate alignments for dnn training
```sh
./run.sh --stage 0 --stop-stage 0 --kaldi-root ~/kaldi
```

Answer Questions:
- Q1 (1 point): Explain what is force alignment. docs: [link](http://www.voxforge.org/home/docs/faq/faq/what-is-forced-alignment)

<details>
<summary>PROGRESS BAR</summary>

```sh

Total number of context-dependent phones: xxxx
============================================================================
                   Generate Alignments for DNN Training
============================================================================
......
Done align train set, alignment file: exp/data/ali_train.txt
......
Done align dev set, alignment file: exp/data/ali_dev.txt
......
Done align test set, alignment file: exp/data/ali_test.txt

```
</details>

---
---


Run stage-1: extract features for dnn training
```sh
./run.sh --stage 1 --stop-stage 1 --kaldi-root ~/kaldi
```

Answer Questions:
- Q2 (1 point): What is the dimention of transformed MFCC features ?

<details>
<summary>PROGRESS BAR</summary>

```sh
============================================================================
                    Extract Features for DNN Training
============================================================================
......
Done apply cepstral mean and variance normalization (CMVN)
......
Done splice features with x left context and x right context
......
Dimention of transformed MFCC feature is xxx

```
</details>

---
---

Run stage-2: dnn training and decoding
```sh
./run.sh --stage 2 --stop-stage 2 --kaldi-root ~/kaldi
```

Answer questions:
- Q3 (2 points): Explain why we need to subtract prior in `train.py:line114-line115`

<details>
<summary>PROGRESS BAR</summary>

```sh
============================================================================
                        DNN Training & Decoding
============================================================================
......
10/21/2020 03:49:41-***** Running DNN *****
10/21/2020 03:49:41-  total parameters: 0.564328 M
10/21/2020 03:49:43-Train Epoch 0 | Step 1 | Loss 7.683 | Acc 0.04%
10/21/2020 03:49:43-Train Epoch 0 | Step 2 | Loss 7.552 | Acc 0.08%
10/21/2020 03:49:44-Train Epoch 0 | Step 3 | Loss 7.436 | Acc 1.62%
10/21/2020 03:49:44-Train Epoch 0 | Step 4 | Loss 7.326 | Acc 4.82%
10/21/2020 03:49:45-Train Epoch 0 | Step 5 | Loss 7.215 | Acc 5.59%
......
10/21/2020 04:07:08-Train Epoch 29 | Step 1736 | Loss 2.555 | Acc 36.06%
10/21/2020 04:07:08-Train Epoch 29 | Step 1737 | Loss 2.388 | Acc 38.48%
10/21/2020 04:07:09-Train Epoch 29 | Step 1738 | Loss 2.508 | Acc 35.64%
10/21/2020 04:07:10-Train Epoch 29 | Step 1739 | Loss 2.445 | Acc 37.68%
10/21/2020 04:07:10-Train Epoch 29 | Step 1740 | Loss 2.274 | Acc 40.48%
10/21/2020 04:07:10-Evaluation DEV
10/21/2020 04:10:50-Epoch 29 DEV Loss 2.694 Acc 33.90% WER 25.1 | 400 15057 | 78.6 16.0 5.4 3.7 25.1 99.8 | -0.129 | /opt/kaldi/egs/timit/s5/DNN-HMM-Course/T3-DNN-HMM/exp/DenseModel/decode_dev/score_5/ctm_39phn.filt.sys
10/21/2020 04:10:50-Evaluation TEST
10/21/2020 04:12:36-Epoch 29 TEST Loss 2.711 Acc 33.37% WER 26.0 | 192 7215 | 77.6 16.5 5.9 3.6 26.0 100.0 | -0.104 | /opt/kaldi/egs/timit/s5/DNN-HMM-Course/T3-DNN-HMM/exp/DenseModel/decode_test/score_5/ctm_39phn.filt.sys
10/21/2020 04:12:36-***** Done DNN Training *****
```
</details>

---
---

The achieved WER(%) is 25.1%(dev)/26.0%(test). The model we used is `DenseModel` in `model/dense.py`. This model is actually a toy model thus the performance is not really good. However, we will use it as our baseline since it is suitable to train on Desktop CPUs (i.e., time overhead is about 30min on Intel i5-8400).


## Tuning your networks
There are many ways to improve results, students who get better(lower) WERs will get extra points:
* Customize your own model in `model/mymodel.py` to get better WER, i.e., deeper model / CNN model / RNN model / Attention Mode.
* Tricks for training neural networks, i.e., label smoothing / data augmentation. You need to implement those tricks by yourself if you want to use them.
* Try different parameters in `run.sh`, i.e., try longer epochs or set leftcontext=3 && rightcontext=3, this will provide useful information for frame classification. You can also increase those values to see if the result decreases and explain why.
* Try to extract different features, i.e., MFCC / FBANK / FMLLR / I-VECTOR. Generally speaking, the more features that are fused, the better results you can get.

