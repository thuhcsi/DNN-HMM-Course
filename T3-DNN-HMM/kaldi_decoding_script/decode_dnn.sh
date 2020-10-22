#!/bin/bash
# Copyright 2013    Yajie Miao    Carnegie Mellon University
# Apache 2.0

# Decode the DNN model. The [srcdir] in this script should be the same as dir in
# build_nnet_pfile.sh. Also, the DNN model has been trained and put in srcdir.
# All these steps will be done automatically if you run the recipe file run-dnn.sh

# Modified 2020 Xingchen Song

[ -f path.sh ] && . ./path.sh

## Begin configuration section
cmd=utils/run.pl
acwt=0.2
num_threads=1
kaldi_root=/opt/kaldi
scoring_opts="--min-lmwt 1 --max-lmwt 10"
beam=13.0
latbeam=8.0
max_active=7000
min_active=200
max_mem=50000000

#echo "$0 $@"  # Print the command line for logging

. ./parse_options.sh || exit 1;

if [ $# != 4 ]; then
   echo "Wrong #arguments ($#, expected 4)"
   echo "Usage: decode_dnn.sh [options] <graph-dir> <data-dir> <gmm-hmm> <logprobs-ark>"
   echo " e.g.: decode_dnn.sh --acwt 0.1 exp/tri3/graph data/test exp/tri3_ali/final.mdl testset-logprobs.ark"
   echo "main options (for others, see top of script file)"
   echo "  --cmd <cmd>                              # command to run in parallel with"
   echo "  --acwt <acoustic-weight>                 # default 0.1 ... used to get posteriors"
   echo "  --num-threads <n>                        # number of threads to use, default 4."
   echo "  --scoring-script <scoring-script>        # local/score.sh"
   echo "  --scoring-opts <opts>                    # options to local/score.sh"
   echo "  --beam <beam>                            # decoding beam.  Larger->slower, more accurate."
   echo "                                           # (float, default = 16)"
   echo "  --latbeam <latbeam>                      # lattice generation beam.  Larger->slower, and deeper lattices."
   echo "                                           # (float, default = 10)"
   echo "  --max-active <maxact>                    # decoder max active states.  Larger->slower; more accurate."
   echo "                                           # (int, default = 2147483647)"
   echo "  --min-active <minact>                    # decoder minimum #active states. (int, default = 200)"
   echo "  --max-mem <maxmem>                       # maximum approximate memory usage in determinization."
   echo "                                           # (int, default = 50000000, real usage might be many times this)."
   echo "  --kaldi-root <kaldiroot>                 # directory of kaldi."
   exit 1;
fi

# check kaldi
if [ -d "$kaldi_root" ]; then
  export KALDI_ROOT=$kaldi_root
  [ -f $KALDI_ROOT/egs/timit/s5/env.sh ] && . $KALDI_ROOT/egs/timit/s5/env.sh
  # add soft link
  ln -sf $kaldi_root/egs/timit/s5/utils utils
  ln -sf $kaldi_root/egs/timit/s5/steps steps
  ln -sf $kaldi_root/egs/timit/s5/local local
  # add kaldi to system path
  export PATH=$kaldi_root/egs/timit/s5/utils:$kaldi_root/egs/timit/s5:$KALDI_ROOT/src/bin:$KALDI_ROOT/tools/openfst/bin:$KALDI_ROOT/tools/irstlm/bin/:$KALDI_ROOT/src/fstbin/:$KALDI_ROOT/src/gmmbin/:$KALDI_ROOT/src/featbin/:$KALDI_ROOT/src/lm/:$KALDI_ROOT/src/sgmmbin/:$KALDI_ROOT/src/sgmm2bin/:$KALDI_ROOT/src/fgmmbin/:$KALDI_ROOT/src/latbin/:$KALDI_ROOT/src/nnetbin:$KALDI_ROOT/src/nnet2bin/:$KALDI_ROOT/src/kwsbin:$PWD:$PATH
  export LC_ALL=C
else
  echo " Kaldi not found, please make sure kaldi_root is set properly, exit decoding now..."
  exit 1;
fi

graph_dir=$1         # /opt/kaldi/egs/timit/s5/exp/tri3/graph
data_dir=$2          # /opt/kaldi/egs/timit/s5/data/test
gmmhmm=$3            # /opt/kaldi/egs/timit/s5/exp/tri3/final.mdl
logprobs=$4          # *.ark

dir=`echo $logprobs | sed 's:/$::g'`  # remove any trailing slash.
decode_dir=`dirname $dir`;            # assume model directory one level up from decoding directory.
echo $num_threads > $decode_dir/num_jobs
thread_string=
[ $num_threads -gt 1 ] && thread_string="-parallel --num-threads=$num_threads"

# Some checks.
for f in $graph_dir/HCLG.fst $data_dir/text; do
  [ ! -f $f ] && echo "$0: no such file $f" && exit 1;
done

# Begin decoding
ck_data=$logprobs
finalfeats="ark,s,cs: cat $ck_data |"
latgen-faster-mapped$thread_string --min-active=$min_active --max-active=$max_active --max-mem=$max_mem --beam=$beam --lattice-beam=$latbeam --acoustic-scale=$acwt --allow-partial=true --word-symbol-table=$graph_dir/words.txt $gmmhmm $graph_dir/HCLG.fst "$finalfeats" "ark:|gzip -c > $decode_dir/lat.1.gz" &> $decode_dir/decode.log &
wait

# Copy the source model in order for scoring
cp $gmmhmm $decode_dir/../

# Begin scoring
scoring_script=$kaldi_root/egs/timit/s5/local/score.sh
[ ! -x $scoring_script ] && \
  echo "$0: not scoring because local/score.sh does not exist or not executable." && exit 1;
$scoring_script $scoring_opts $data_dir $graph_dir $decode_dir

# Show wer
grep Sum $decode_dir/score_*/*.sys 2>/dev/null | $KALDI_ROOT/egs/timit/s5/utils/best_wer.sh;

exit 0;
