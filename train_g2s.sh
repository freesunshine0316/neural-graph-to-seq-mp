#!/bin/bash
#SBATCH -J g_2m -C K80 --partition=gpu --gres=gpu:1 --time=5-00:00:00 --output=train.out --error=train.err
#SBATCH --mem=80GB
#SBATCH -c 5

export PYTHONPATH=$PYTHONPATH:/home/lsong10/ws/exp.graph_to_seq/neural-graph-to-seq-mp

python src_g2s/G2S_trainer.py --config_path config_g2s.json

