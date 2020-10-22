# Task-3 : DNN-HMM training on TIMIT

## Preliminary
You must finish [Task-2](https://github.com/thuhcsi/DNN-HMM-Course/tree/main/T2-GMM-HMM) first to get a well-trained GMM-HMM model, which we will use for generating frame-level alignments.

If you have done Task-2, you can check and view your GMM-HMM model by the following command:
```sh
# *** In terminal shell(macos/linux) or ubuntu 20.04 shell(windows10) ***
# navigate to timit folder
cd ~/kaldi/egs/timit/s5
. ./path.sh
# wirte model information to model.txt
gmm-copy --binary=false exp/tri3/final.mdl model.txt
```
Explanations of each part in `model.txt` can be found [here]()

Besides, you can also visualize your decoding graph (WFST) / MFCC features ...  :
```sh
# *** In terminal shell(macos/linux) or ubuntu 20.04 shell(windows10) ***
cd ~/kaldi/egs/timit/s5
. ./path.sh
# write graph information to HCLG.txt
fstprint exp/tri3/graph/HCLG.fst HCLG.txt
# write mfcc features to raw_mfcc_test.1.txt
copy-feats ark:mfcc/raw_mfcc_test.1.ark ark,t:raw_mfcc_test.1.txt
```

Next, clone this repo and install required packages:
```sh
# *** In terminal shell(macos/linux) or ubuntu 20.04 shell(windows10) ***
# we will clone this repo to directory ~/kaldi/egs/timit/s5
# NOTE: if you encounter `temporary failure resolving xxx` while installing pkgs
#       follow [this link](https://gist.github.com/coltenkrauter/608cfe02319ce60facd76373249b8ca6) to fix wsl2 dns problem
cd ~/kaldi/egs/timit/s5
git clone https://github.com/thuhcsi/DNN-HMM-Course.git
cd DNN-HMM-Course
# install required packages
pip3 install -i https://pypi.tuna.tsinghua.edu.cn/simple -r requirements.txt
```

## Step-1
Run the DNN training:
```sh
# *** In terminal shell(macos/linux) or ubuntu 20.04 shell(windows10) ***
cd ~/kaldi/egs/timit/s5/DNN-HMM-Course/T3-DNN-HMM
./run.sh --stage 0 --kaldi-root ~/kaldi
```
This script starts a full ASR experiment and performs data preparation, training, evaluation, forward, and decoding steps. A progress bar shows the evolution of all the aforementioned phases.


```sh
Total number of context-dependent phones: xxxx
============================================================================
                   Generate Alignments for DNN Training
============================================================================
......
Done align test set, alignment file: exp/data/ali_test.txt
============================================================================
                    Extract Features for DNN Training
============================================================================
......
Done apply cepstral mean and variance normalization (CMVN)
......
Done splice features with x left context and x right context
......
Dimention of transformed MFCC feature is xxx
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

The script `run.sh` progressively creates the following files in the output directory `$PWD/exp`:
* data/ali\_\*.ark: ......(WIP)
* data/ali\_\*.txt: ......(WIP)
* data/mfcc\_apply\_cmvn\_context\_\*.ark: ......(WIP)
* data/mfcc\_apply\_cmvn\_context\_\*.scp: ......(WIP)
* data/phone\_counts: ......(WIP)
* data/text\_\*: ......(WIP)

and the script `train.py` will save the trained model and write log to `$PWD/exp/$MODEL`.

The achieved WER(%) is 25.1%(dev)/26.0%(test). The model we used is `DenseModel` in `model/dense.py`. This model is actually a toy model thus the performance is not really good. However, we will use it as our baseline since it is suitable to train on Desktop CPUs (i.e., time overhead is about 30min on Intel i5-8400).

**By correctly running the baseline scripts, you can get the basic score for this task.**


## Step-2
There are many ways to improve results, students who get better(lower) WERs will get extra points:
* Customize your own model in `model/mymodel.py` to get better WER, i.e., deeper model / CNN model / RNN model / Attention Mode.
* Tricks for training neural networks, i.e., label smoothing / data augmentation. You need to implement those tricks by yourself if you want to use them.
* Try different parameters in `run.sh`, i.e., try longer epochs or set leftcontext=3 && rightcontext=3, this will provide useful information for frame classification. You can also increase those values to see if the result decreases and explain why.
* Try to extract different features, i.e., MFCC / FBANK / FMLLR / I-VECTOR. Generally speaking, the more features that are fused, the better results you can get.

NOTE: for feature extraction, we will show some examples here (WIP)

## Step-3
Answer questions
- Q1:
- Q2:
