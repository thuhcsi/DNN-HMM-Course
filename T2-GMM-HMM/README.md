# Task-2: GMM-HMM training on TIMIT

## Preliminary



- windows10
  - [Install WSL and Ubuntu 20.04](https://docs.microsoft.com/en-us/windows/wsl/install-win10), note that in step-6 you must select ubuntu 20.04 LTS (16.04 and 18.04 are not supported right now)

	<details>
	<summary>OPTIONAL</summary>

	WSL is installed on the system disk by default. If the remaining space of your system disk is less than 5G, you can move wsl to other disks. To do this, first download [LxRunOffline](https://github.com/DDoSolitary/LxRunOffline/releases/download/v3.5.0/LxRunOffline-v3.5.0-msvc.zip), copy `LxRunOffline.exe` and `LxRunOfflineShellExt.dll` to `C:\Windows\System32`, then open **PowerShell** and run the following commands:
	
	first shut down your wsl:
	```sh
	wsl --shutdown
	```
	then check names of installed wsl, in our case, the default name should be `Ubuntu-20.04`:
	```sh
	LxRunOffline list
	```
	move wsl(Ubuntu-20.04) to D:\Linux\Ubuntu-20.04:
	```sh
	LxRunOffline m -n Ubuntu-20.04 -d D:\Linux\Ubuntu-20.04
	```
	</details>

  - Start **ubuntu 20.04** and run the following commands

	```sh
	sudo apt-get install gfortran
	sudo apt-get install python3-pip
	wget 219.223.184.252/file/ubuntu20.04.kaldi.tar.gz
	tar xzf ubuntu20.04.kaldi.tar.gz
	```

	<details>
	<summary>OPTIONAL</summary>

	if you encounter `temporary failure resolving xxx` while installing pkgs,
	follow [this link](https://gist.github.com/coltenkrauter/608cfe02319ce60facd76373249b8ca6) to fix wsl2 dns problem.
	if apt-get is too slow,
	follow [this link](https://blog.csdn.net/xiangxianghehe/article/details/105688062) to change apt sources
	
	</details>



- macos [Working In Progress]
  - install [homebrew](https://mirrors.tuna.tsinghua.edu.cn/help/homebrew/)
  - start terminal and run the following commands
	```sh
	brew install gcc coreutils
	wget 219.223.184.252/file/macos.kaldi.tar.gz
	tar xzf macos.kaldi.tar.gz
	```


- linux(ubuntu 20.04)
  - start terminal and run the following commands
	```sh
	sudo apt-get install gfortran
	sudo apt-get install python3-pip
	wget 219.223.184.252/file/ubuntu20.04.kaldi.tar.gz
	tar xzf ubuntu20.04.kaldi.tar.gz
	```

## GMM-HMM Training

NOTE: by correctly running the baseline script, you can get basic score (16 points) of this task.

First start a terminal and navigate to timit folder
```sh
cd ~/kaldi/egs/timit/s5
```

In this folder, you can see `TIMIT` dataset with a directory structure like this:

```sh
TIMIT
├── DOC
├── README.DOC
├── TEST
└── TRAIN
```

You can also see `run.sh` in this folder, this script starts a full GMM-HMM experiment and performs data preparation, training, evaluation, forward, and decoding steps. A progress bar shows the evolution of all the aforementioned phases.

Here we will run this script stage by stage.

---
---

Run stage-0: format data and prepare lexicon:
```sh
./run.sh --stage 0 --stop-stage 0
```
Answer questions:
- Q1 (3 points): Look at the directory `data/train`, describe what is contained in files `text`, `wav.scp` and `utt2spk` respectively (Hint: all those files can be seen as key-value dicts). Here are some docs you can refer to : [link](http://kaldi-asr.org/doc/data_prep.html#data_prep_data)
- Q2 (3 points): Look at the file `data/lang/topo`, which contains two kinds of HMM topology, draw them using circles and arrows like [this](http://img.jeepxie.net/upload/c/42/c429ab922d03742c061f94d27c4b1915.png). You may notice that the HMM topology of a special phoneme is different from other phonemes. Use `data/lang/phones.txt` to map and find the name of the special phoneme. Here are some docs you can refer to: [link](http://kaldi-asr.org/doc/data_prep.html#data_prep_lang_contents), [link](http://kaldi-asr.org/doc/hmm.html)
- Q3 (2 points): You can change `num-sil-states` and `num-nonsil-states` for command `utils/prepare_lang.sh` in `run.sh`, then run this stage again and draw new topologies from `data/lang/topo`. (Note that set them to default values(sil=3 and nonsil=3) and rerun this stage before proceeding to the next stage since other values may affect the performance)

<details>
<summary>PROGRESS BAR</summary>

```
============================================================================
                Data & Lexicon & Language Preparation
============================================================================
......
Data preparation succeeded
......
Dictionary & language model preparation succeeded
......
Checking xxx
......
Succeeded in formatting data.
```
</details>

---
---

Run stage-1: extract MFCC and Cepstral Mean and Variance Normalization (CMVN):
```sh
./run.sh --stage 1 --stop-stage 1
```

Answer questions:
- Q4 (7 points): Describe the process of calculating MFCC, docs: [link](http://kaldi-asr.org/doc/feat.html#feat_mfcc)
- Q5 (1 point): Run following commands (Note: `ark` means archive file with binary format, `ark,t` means archive file with utf-8 text format), check file `raw_mfcc_test.1.txt` and answer the dimention of MFCC features. (Hint: count the number of columns)
	```sh
	source ./path.sh
	copy-feats ark:mfcc/raw_mfcc_test.1.ark ark,t:raw_mfcc_test.1.txt
	```
- Q6 (1 point): Look at the directory `data/train`, describe what is contained in file `feats.scp`, docs: [link](http://kaldi-asr.org/doc/io.html#io_sec_scp)
- Q7 (1 point): Describe the role of script `utils/split_scp.pl` in `steps/make_mfcc.sh:line133`. (Hint: type `wc mfcc/raw_mfcc_train.1.scp` and `wc data/train/feats.scp`, check the value of `feats_nj` in `run.sh` and compare the outputs of the above commands)

<details>
<summary>PROGRESS BAR</summary>

```sh
============================================================================
         MFCC Feature Extration & CMVN for Training and Test set
============================================================================
......
steps/make_mfcc.sh: Succeeded creating MFCC features for train
......
steps/compute_cmvn_stats.sh data/train exp/make_mfcc/train mfcc
......
Succeeded creating CMVN stats for train.
......
```
</details>

---
---

Run stage-2: train MonoPhone system:
```sh
./run.sh --stage 2 --stop-stage 2
```


Answer questions:
- Q8 (1 point): How many MonoPhones do we use ? (Hint: check `exp/mono/phones.txt`, the MonoPhones we mean here do not contain eps, #0 and #1)
- Q9 (1 point): If we choose 3-state HMM to model all those MonoPhones, how many states will we have ? (Hint: run the following commands and count the number of Triples in `final.mono.mdl.txt`)
	```sh
	source ./path.sh
	gmm-copy --binary=false exp/mono/final.mdl final.mono.mdl.txt
	```

<details>
<summary>PROGRESS BAR</summary>

```sh
============================================================================
                     MonoPhone Training & Decoding
============================================================================
......
steps/train_mono.sh: Initializing monophone system.
steps/train_mono.sh: Compiling training graphs
steps/train_mono.sh: Aligning data equally (pass 0)
steps/train_mono.sh: Pass 1
steps/train_mono.sh: Aligning data
steps/train_mono.sh: Pass 2
steps/train_mono.sh: Aligning data
steps/train_mono.sh: Pass 3
......
steps/train_mono.sh: Done training monophone system in exp/mono
......
steps/decode.sh --nj 5 --cmd run.pl --mem 4G exp/mono/graph data/dev exp/mono/decode_dev
......
steps/decode.sh --nj 5 --cmd run.pl --mem 4G exp/mono/graph data/test exp/mono/decode_tsalsest
......
```
</details>

---
---

Run stage-3: train TriPhone system:
```sh
./run.sh --stage 3 --stop-stage 3
```


Answer questions:
- Q10 (2 points): According to the number of MonoPhones in the previous question, how many candidates are there for TriPhones and TriPhones' HMM states (suppose each HMM has 3 states) ?
- Q11 (2 points): Refer to Q9, use the correct command to view the TriPhone system (`exp/tri3/final.mdl`) and count the actual number of HMM states. You may find that the actual number of HMM states is much smaller than the theoretical value calculated in Q10, can you explain why ? (Hint: PPT5:page38~page42)

<details>
<summary>PROGRESS BAR</summary>

```sh
============================================================================
           tri1 : Deltas + Delta-Deltas Training & Decoding
============================================================================
......
steps/train_deltas.sh: accumulating tree stats
steps/train_deltas.sh: getting questions for tree-building, via clustering
steps/train_deltas.sh: building the tree
steps/train_deltas.sh: converting alignments from exp/mono_ali to use current tree
steps/train_deltas.sh: compiling graphs of transcripts
steps/train_deltas.sh: training pass 1
steps/train_deltas.sh: training pass 2
steps/train_deltas.sh: training pass 3
......
steps/decode.sh --nj 5 --cmd run.pl --mem 4G exp/tri1/graph data/dev exp/tri1/decode_dev
......
steps/decode.sh --nj 5 --cmd run.pl --mem 4G exp/tri1/graph data/test exp/tri1/decode_test
......
```
</details>

---
---

Run stage-4/5/6: adapt TriPhone system with [LDA](https://www.cnblogs.com/pinard/p/6244265.html), [MLLT](http://kaldi-asr.org/doc/transform.html#transform_mllt), [SAT](http://jcip.cipsc.org.cn/CN/Y2004/V18/I3/62) and [FMLLR](https://blog.csdn.net/xmdxcsj/article/details/78512645), show final results:
```sh
./run.sh --stage 4 --stop-stage 6
```

NOTE: [LDA](https://www.cnblogs.com/pinard/p/6244265.html), [MLLT](http://kaldi-asr.org/doc/transform.html#transform_mllt), [SAT](http://jcip.cipsc.org.cn/CN/Y2004/V18/I3/62) and [FMLLR](https://blog.csdn.net/xmdxcsj/article/details/78512645) are adaptation techniques for better training TriPhone system, they are not the focus of this experiment and the training pipeline is very similiar to stage-3 thus you don't neet to dig into stage-4/5/6. We keep these stages mainly to help [TASK-3](https://github.com/thuhcsi/DNN-HMM-Course/tree/main/T3-DNN-HMM) get better alignments. As for students who are interested in [LDA](https://www.cnblogs.com/pinard/p/6244265.html), [MLLT](http://kaldi-asr.org/doc/transform.html#transform_mllt), [SAT](http://jcip.cipsc.org.cn/CN/Y2004/V18/I3/62) and [FMLLR](https://blog.csdn.net/xmdxcsj/article/details/78512645), you can refer to the link for more explanations.

<details>
<summary>PROGRESS BAR</summary>

```sh
============================================================================
                 tri2 : LDA + MLLT Training & Decoding
============================================================================
......
steps/train_lda_mllt.sh: Accumulating LDA statistics.
steps/train_lda_mllt.sh: Accumulating tree stats
steps/train_lda_mllt.sh: Getting questions for tree clustering.
steps/train_lda_mllt.sh: Building the tree
steps/train_lda_mllt.sh: Initializing the model
steps/train_lda_mllt.sh: Converting alignments from exp/tri1_ali to use current tree
steps/train_lda_mllt.sh: Compiling graphs of transcripts
Training pass 1
Training pass 2
steps/train_lda_mllt.sh: Estimating MLLT
Training pass 3
Training pass 4
steps/train_lda_mllt.sh: Estimating MLLT
Training pass 5
Training pass 6
......
steps/decode.sh --nj 5 --cmd run.pl --mem 4G exp/tri2/graph data/dev exp/tri2/decode_dev
......
steps/decode.sh --nj 5 --cmd run.pl --mem 4G exp/tri2/graph data/test exp/tri2/decode_test
......
============================================================================
              tri3 : LDA + MLLT + SAT Training & Decoding
============================================================================
......
steps/train_sat.sh: feature type is lda
steps/train_sat.sh: obtaining initial fMLLR transforms since not present in exp/tri2_ali
steps/train_sat.sh: Accumulating tree stats
steps/train_sat.sh: Getting questions for tree clustering.
steps/train_sat.sh: Building the tree
steps/train_sat.sh: Initializing the model
steps/train_sat.sh: Converting alignments from exp/tri2_ali to use current tree
steps/train_sat.sh: Compiling graphs of transcripts
Pass 1
Pass 2
Estimating fMLLR transforms
Pass 3
Pass 4
Estimating fMLLR transforms
Pass 5
Pass 6
......
steps/decode_fmllr.sh --nj 5 --cmd run.pl --mem 4G exp/tri3/graph data/dev exp/tri3/decode_dev
......
steps/decode_fmllr.sh --nj 5 --cmd run.pl --mem 4G exp/tri3/graph data/test exp/tri3/decode_test
......
============================================================================
               DNN Hybrid Training & Decoding (deprecated)
============================================================================
Gmm-Hmm training has been done via the above command lines.
For Dnn training, we will use pytorch instead of kaldi.
============================================================================
                    Getting Results [see RESULTS file]
============================================================================
%WER 31.6 | 400 15057 | 71.9 19.2 8.8 3.5 31.6 100.0 | -0.481 | exp/mono/decode_dev/score_5/ctm_39phn.filt.sys
%WER 24.8 | 400 15057 | 79.2 15.6 5.2 3.9 24.8 100.0 | -0.153 | exp/tri1/decode_dev/score_10/ctm_39phn.filt.sys
%WER 22.7 | 400 15057 | 81.0 14.2 4.8 3.7 22.7 99.5 | -0.294 | exp/tri2/decode_dev/score_10/ctm_39phn.filt.sys
%WER 20.4 | 400 15057 | 82.7 12.6 4.6 3.1 20.4 99.8 | -0.611 | exp/tri3/decode_dev/score_10/ctm_39phn.filt.sys
%WER 23.3 | 400 15057 | 80.7 14.7 4.5 4.0 23.3 99.8 | -0.409 | exp/tri3/decode_dev.si/score_8/ctm_39phn.filt.sys
%WER 31.7 | 192 7215 | 71.6 19.0 9.4 3.3 31.7 100.0 | -0.450 | exp/mono/decode_test/score_5/ctm_39phn.filt.sys
%WER 26.3 | 192 7215 | 77.6 16.9 5.5 4.0 26.3 100.0 | -0.134 | exp/tri1/decode_test/score_10/ctm_39phn.filt.sys
%WER 23.7 | 192 7215 | 79.8 14.9 5.3 3.5 23.7 99.5 | -0.301 | exp/tri2/decode_test/score_10/ctm_39phn.filt.sys
%WER 22.3 | 192 7215 | 80.9 14.0 5.1 3.2 22.3 99.5 | -0.564 | exp/tri3/decode_test/score_10/ctm_39phn.filt.sys
%WER 24.7 | 192 7215 | 78.5 15.7 5.8 3.2 24.7 99.5 | -0.229 | exp/tri3/decode_test.si/score_10/ctm_39phn.filt.sys
============================================================================
Finished successfully on Wed Oct 21 03:06:29 UTC 2020
============================================================================
```
</details>
