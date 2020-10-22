# Task-2: GMM-HMM training on TIMIT

## Preliminary

- windows10
  - [Install WSL2](https://docs.microsoft.com/en-us/windows/wsl/install-win10), note that in step-6 you must install ubuntu 20.04 LTS
  - [**OPTIONAL**] WSL2 is installed on the system disk by default. If the remaining space of your system disk is less than 5G, you can move wsl to other disks. To do this, first download [LxRunOffline](https://github.com/DDoSolitary/LxRunOffline/releases/download/v3.5.0/LxRunOffline-v3.5.0-msvc.zip), copy `LxRunOffline.exe` and `LxRunOfflineShellExt.dll` to `C:\Windows\System32`, then open **PowerShell** and run the following commands
	```sh
	# ****** in powershell ******
	# stop wsl
	wsl --shutdown
	# list names of all installed wsl
	LxRunOffline list
	# move wsl(Ubuntu-20.04) to D:\Linux\Ubuntu-20.04
	LxRunOffline m -n Ubuntu-20.04 -d D:\Linux\Ubuntu-20.04
	```
  - Start **ubuntu 20.04** and run the following commands
	```sh
	# ****** in ubuntu 20.04 shell ******
	# NOTE: if you encounter `temporary failure resolving xxx` while installing pkgs
	#       follow [this link](https://gist.github.com/coltenkrauter/608cfe02319ce60facd76373249b8ca6) to fix wsl2 dns problem
	#       if apt-get is too slow
	#       follow [this link](https://blog.csdn.net/xiangxianghehe/article/details/105688062) to change apt sources
	sudo add-apt-repository ppa:jonathonf/gcc-7.1
	sudo apt-get install gfortran-7
	sudo apt-get install python3-pip
	wget 219.223.184.252/file/ubuntu20.04.kaldi.tar.gz
	tar -xzf ubuntu20.04.kaldi.tar.gz
	```

- macos
  - start terminal and run the following commands
	```sh
	# ****** in terminal shell ******
	WIP
	```


- linux(ubuntu 20.04)
  - start terminal and run the following commands
	```sh
	# ****** in terminal shell ******
	sudo add-apt-repository ppa:jonathonf/gcc-7.1
	sudo apt-get install gfortran-7
	sudo apt-get install python3-pip
	wget 219.223.184.252/file/ubuntu20.04.kaldi.tar.gz
	tar -xzf ubuntu20.04.kaldi.tar.gz
	```

## Step-1
Run the GMM-HMM training.

```sh
# ****** in terminal shell(macos/linux) or ubuntu 20.04 shell(windows10)******
# navigate to timit folder
cd ~/kaldi/egs/timit/s5
# run the script
./run.sh
```
This script starts a full GMM-HMM experiment and performs data preparation, training, evaluation, forward, and decoding steps. A progress bar shows the evolution of all the aforementioned phases.
```sh
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
steps/decode.sh --nj 5 --cmd run.pl --mem 4G exp/mono/graph data/test exp/mono/decode_test
......
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

## Step-2
Answer questions:
- Q1:
- Q2:
