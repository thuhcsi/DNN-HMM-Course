#!/bin/bash
# Xingchen Song @ 2020-10-14

set -e

## Begin configuration section
stage=0                 # start from 0 if you need to start from data preparation
stop_stage=100          # stop stage
kaldi_root=`realpath ../../../../../`   # kaldi installation directory
epochs=30               # total training epochs
lr="1e-3"               # learning rate
batch_size=64          # batch size
output_dir=exp          # output directory
left_context=0          # frame context
right_context=0         # frame context
clustered_states_num=           # total number of context-dependent phones, automatically set by this script
feature_dim=            # feature dimention, automatically set by this script

. kaldi_decoding_script/parse_options.sh || exit 1;


# check kaldi
kaldi_root=`realpath $kaldi_root`
echo 'kaldi root: ' $kaldi_root
if [ -d "$kaldi_root/egs/timit/s5/exp/tri3" ]; then
  export KALDI_ROOT=$kaldi_root
  [ -f $KALDI_ROOT/egs/timit/s5/env.sh ] && . $KALDI_ROOT/egs/timit/s5/env.sh
  # add soft link
  ln -sf $kaldi_root/egs/timit/s5/utils utils
  ln -sf $kaldi_root/egs/timit/s5/steps steps
  # add kaldi to system path
  export PATH=$KALDI_ROOT/egs/timit/s5/utils:$KALDI_ROOT/egs/timit/s5:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$PWD:$PATH
  export LC_ALL=C
else
  echo " Kaldi not Installed or Task-2(GMM-HMM Training for TIMIT) not Finished, exit run.sh now..."
  exit 1;
fi


mkdir -p $output_dir
mkdir -p $output_dir/data
timit=$KALDI_ROOT/egs/timit/s5
clustered_states_num=`hmm-info $timit/exp/tri3/final.mdl | awk '/pdfs/{print $4}'`
echo "Total number of clustered states: $states_num"


if [ ${stage} -le 0 ] && [ ${stop_stage} -ge 0 ]; then
  echo ============================================================================
  echo "                   Generate Alignments for DNN Training                   "
  echo ============================================================================

  # align train set
  $timit/steps/align_fmllr.sh --nj 30 --cmd "$timit/utils/run.pl --mem 4G" \
                              $timit/data/train $timit/data/lang \
                              $timit/exp/tri3 $timit/exp/tri3_ali_train

  # unzip and merge into one file (ali_train.ark)
  ali-to-pdf $timit/exp/tri3_ali_train/final.mdl \
      ark:"gunzip -c $timit/exp/tri3_ali_train/ali.*.gz|" ark:$output_dir/data/ali_train.ark

  # change format of archive file from `binary` to `utf-8 text`, `ali_train.txt` is readable for human
  # while `ali_train.ark` is not
  copy-int-vector ark:$output_dir/data/ali_train.ark ark,t:$output_dir/data/ali_train.txt

  echo "Done align train set, alignment file: $output_dir/data/ali_train.txt"

  # count the number of occurrences of each state
  analyze-counts --print-args=False --verbose=0 --binary=false --counts-dim=$clustered_states_num \
                  ark:$output_dir/data/ali_train.ark $output_dir/data/phone_counts

  # copy ground truth
  mkdir -p $output_dir/data/text_train
  cp $timit/data/train/text $output_dir/data/text_train/
  cp $timit/data/train/*m $output_dir/data/text_train/

  # align dev and test set
  for x in dev test; do
    $timit/steps/align_fmllr.sh $timit/data/$x $timit/data/lang \
                                $timit/exp/tri3 $timit/exp/tri3_ali_$x

    # unzip and merge into one file
    ali-to-pdf $timit/exp/tri3_ali_$x/final.mdl \
        ark:"gunzip -c $timit/exp/tri3_ali_$x/ali.*.gz|" ark:$output_dir/data/ali_$x.ark

    # change format of archive file from `binary` to `utf-8 text`
    copy-int-vector ark:$output_dir/data/ali_$x.ark ark,t:$output_dir/data/ali_$x.txt

    echo "Done align $x set, alignment file: $output_dir/data/ali_$x.txt"

    # copy ground truth
    mkdir -p $output_dir/data/text_$x
    cp $timit/data/$x/text $output_dir/data/text_$x/
    cp $timit/data/$x/*m $output_dir/data/text_$x/
  done
fi


if [ -d "utils" ]; then
  # we will not need those files in following stages
  # so simply delete the soft links we made
  rm -rf utils
  rm -rf steps
fi


if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo ============================================================================
  echo "                    Extract Features for DNN Training                     "
  echo ============================================================================

  # the original MFCC features we created in Task-2(GMM-HMM) is not suitble for DNN training,
  # now we will do some transformation/enhancement on top of the original MFCC features.
  for x in train dev test; do
    # 1: apply cepstral mean and variance normalization (CMVN) for MFCC
    # 2: add delta and delta-delta features for MFCC
    apply-cmvn --utt2spk=ark:$timit/data/$x/utt2spk scp:$timit/data/$x/cmvn.scp scp:$timit/data/$x/feats.scp ark:- | add-deltas ark:- ark:$output_dir/data/mfcc_apply_cmvn_$x.ark
    echo "Done apply cepstral mean and variance normalization (CMVN)"

    # splice features with left and right context
    splice-feats --left-context=$left_context --right-context=$right_context ark:$output_dir/data/mfcc_apply_cmvn_$x.ark ark,scp:$output_dir/data/mfcc_apply_cmvn_context_$x.ark,$output_dir/data/mfcc_apply_cmvn_context_$x.scp
    echo "Done splice features with $left_context left context and $right_context right context"

    # what we need is actually mfcc_apply_cmvn_context_*.ark
    # delete this file (mfcc_apply_cmvn_*.ark) to save memory
    rm -f $output_dir/data/mfcc_apply_cmvn_$x.ark
  done

fi


# note that after transformation/enhancement, the dimention of MFCC features has been changed.
feature_dim=`feat-to-dim "ark:$output_dir/data/mfcc_apply_cmvn_context_train.ark" - 2>/dev/null`
echo "Dimention of transformed MFCC feature is $feature_dim"


if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo ============================================================================
  echo "                        DNN Training & Decoding                           "
  echo ============================================================================

  # For simplicity, the decoding process is also included in train.py
  python3 train.py \
      --epochs $epochs \
      --lr $lr \
      --batch-size $batch_size \
      --feature-dim $feature_dim \
      --cdphones-num $clustered_states_num \
      --output-dir $output_dir \
      --prior-file $output_dir/data/phone_counts \
      --align-file $output_dir/data/ali_train.ark \
      --feats-file $output_dir/data/mfcc_apply_cmvn_context_train.scp \
      --dev-align-file $output_dir/data/ali_dev.ark \
      --dev-feats-file $output_dir/data/mfcc_apply_cmvn_context_dev.scp \
      --test-align-file $output_dir/data/ali_test.ark \
      --test-feats-file $output_dir/data/mfcc_apply_cmvn_context_test.scp \
      --graph-dir $timit/exp/tri3/graph \
      --gmmhmm $timit/exp/tri3/final.mdl \
      --kaldi-root $kaldi_root
fi


if [ -d "kaldi_decoding_script/utils" ]; then
  # we will not need those files in following stages
  # so simply delete the soft links we made
  rm -rf kaldi_decoding_script/utils
  rm -rf kaldi_decoding_script/steps
  rm -rf kaldi_decoding_script/local
fi
