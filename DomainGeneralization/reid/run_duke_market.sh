#!/bin/bash

DATA=/path-to-data

# market1501 <-> grid
for SEED in $(seq 1 3)
do
    ### with domain label mix.
    python main.py \
    --config-file cfgs/cfg_osnet_domprior.yaml \
    -s market1501 \
    -t dukemtmcreid \
    --root ${DATA} \
    model.name osnet_x1_0_efdmix23_a0d1_domprior \
    data.save_dir EFDMix/osnet_x1_0_efdmix23_a0d1_domprior/market2duke

    python main.py \
    --config-file cfgs/cfg_osnet_domprior.yaml \
    -s dukemtmcreid \
    -t market1501 \
    --root ${DATA} \
    model.name osnet_x1_0_efdmix23_a0d1_domprior \
    data.save_dir EFDMix/osnet_x1_0_efdmix23_a0d1_domprior/duke2market
done


### with random shuffle,
#python main.py \
#--config-file cfgs/cfg_osnet.yaml \
#-s market1501 \
#-t dukemtmcreid \
#--root ${DATA} \
#model.name osnet_x1_0_efdmix23_a0d1 \
#data.save_dir EFDMix/osnet_x1_0_efdmix23_a0d1/market2duke
#
#python main.py \
#--config-file cfgs/cfg_osnet.yaml \
#-s dukemtmcreid \
#-t market1501 \
#--root ${DATA} \
#model.name osnet_x1_0_efdmix23_a0d1 \
#data.save_dir EFDMix/osnet_x1_0_efdmix23_a0d1/duke2market
