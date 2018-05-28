#!/bin/bash
#SBATCH --partition=gpu --gres=gpu:1 --time=1:00:00 --output=decode.out --error=decode.err
#SBATCH --mem=10GB
#SBATCH -c 6

export PYTHONPATH=$PYTHONPATH:/home/lsong10/ws/exp.graph_to_seq/neural-graph-to-seq-mp

python src_s2s/NP2P_beam_decoder.py --model_prefix logs_s2s/NP2P.$1 \
        --in_path data/test.json \
        --out_path logs_s2s/test.s2s.$1\.tok \
        --mode beam

