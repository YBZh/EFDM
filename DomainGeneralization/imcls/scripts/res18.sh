#!/bin/bash

cd ..

DATA=~/data/mixstyle
DASSL=~/syn_project/mixstyle-release-master/Dassl.pytorch

D1=art_painting
D2=cartoon
D3=photo
D4=sketch

################### leave one domain out setting
DATASET=pacs
TRAINER=Vanilla2
NET=resnet18_efdmix_l123
MIX=crossdomain

#
for SEED in $(seq 1 5)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi


        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${S1} ${S2} ${S3}  \
        --target-domains ${T} \
        --dataset-config-file ${DASSL}/configs/datasets/dg/${DATASET}.yaml \
        --config-file configs/trainers/mixstyle/${DATASET}_${MIX}.yaml \
        --output-dir EFDMix/${DATASET}/${TRAINER}/${NET}_nodetach/${MIX}/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET}
    done
done

############## single source generalization setting.
DATASET=pacs
TRAINER=Vanilla2
NET=resnet18_efdmix_l123
MIX=random

for SEED in $(seq 1 5)
do
    for SETUP in $(seq 1 4)
    do
        if [ ${SETUP} == 1 ]; then
            S1=${D2}
            S2=${D3}
            S3=${D4}
            T=${D1}
        elif [ ${SETUP} == 2 ]; then
            S1=${D1}
            S2=${D3}
            S3=${D4}
            T=${D2}
        elif [ ${SETUP} == 3 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D4}
            T=${D3}
        elif [ ${SETUP} == 4 ]; then
            S1=${D1}
            S2=${D2}
            S3=${D3}
            T=${D4}
        fi

        python train.py \
        --root ${DATA} \
        --seed ${SEED} \
        --trainer ${TRAINER} \
        --source-domains ${T}  \
        --target-domains ${S1} ${S2} ${S3} \
        --dataset-config-file ${DASSL}/configs/datasets/dg/${DATASET}.yaml \
        --config-file configs/trainers/mixstyle/${DATASET}_${MIX}.yaml \
        --output-dir EFDMix/${DATASET}/${TRAINER}_singles/${NET}_nodetach/${MIX}/${T}/seed${SEED} \
        MODEL.BACKBONE.NAME ${NET}

    done
done