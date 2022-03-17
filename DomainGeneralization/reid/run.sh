#!/bin/bash

DATA=/path-to-data

# market1501 <-> grid
for SEED in $(seq 1 3)
do
    ### with random shuffle mix.
    python main.py \
    --config-file cfgs/cfg_osnet.yaml \
    -s market1501 \
    -t grid \
    --root ${DATA} \
    model.name osnet_x1_0_efdmix23_a0d1 \
    data.save_dir EFDMix/osnet_x1_0_efdmix23_a0d1/market2grid

    python main.py \
    --config-file cfgs/cfg_osnet.yaml \
    -s grid \
    -t market1501 \
    --root ${DATA} \
    model.name osnet_x1_0_efdmix23_a0d1 \
    data.save_dir EFDMix/osnet_x1_0_efdmix23_a0d1/grid2market
done
