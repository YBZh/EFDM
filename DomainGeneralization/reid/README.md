# EFDMix on cross-dataset person re-identification

## How to install

This code is based on [Torchreid](https://arxiv.org/abs/1910.10093). Please follow the instructions at https://github.com/KaiyangZhou/deep-person-reid#installation to install `torchreid`.

## How to run

The running commands are provided in `run.sh`. You need to activate the `torchreid` environment using `conda activate torchreid` before running the code. See https://github.com/KaiyangZhou/deep-person-reid#a-unified-interface for more details on how to train and test a model.

Many works conduct cross-domain reid between dukemtmcreid and market1501 datasets. 
However, since the use of DukeMTMC violates CVPR ethical policy, we use the datasets of market1501 and grid, as shown in `run.sh`.
Additionaly, we provide the running commands with dukemtmcreid and market1501 datasets in `run_duke_market.sh`. The corresponding resuls are illustrated as follows:

| MarKet1501->Duke       | mAP  | R1   | R5   | R10  |
|------------------------|------|------|------|------|
| OSNet                  | 27.9 | 48.2 | 62.3 | 68.0 |
| EFDMix w/ domain label | 29.9 | 50.8 | 65.0 | 70.3 |
|                        |      |      |      |      |
| Duke->MarKet1501       | mAP  | R1   | R5   | R10  |
| OSNet                  | 25.0 | 52.8 | 70.5 | 77.5 |
| EFDMix w/ domain label | 29.3 | 59.5 | 76.5 | 82.5 |

## How to extract results from train.log-xxx.
When we finish the model training, there are logs of `train.log-xx` under the save_path.
You can get the mean and standard deivation of results with the following command. 
```python
    python parse_test_res.py ./save_path/osnet_x1_0_efdmix23_a0d1/ --multi-exp
```

## How to cite

If you find this code useful to your research, please cite the following papers.

```
@inproceedings{zhang2021exact,
  title={Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization},
  author={Zhang, Yabin and Li, Minghan and Li, Ruihuang and Jia, Kui and Zhang, Lei},
  booktitle={CVPR},
  year={2022}
}

@article{torchreid,
  title={Torchreid: A Library for Deep Learning Person Re-Identification in Pytorch},
  author={Zhou, Kaiyang and Xiang, Tao},
  journal={arXiv preprint arXiv:1910.10093},
  year={2019}
}

@inproceedings{zhou2019osnet,
  title={Omni-Scale Feature Learning for Person Re-Identification},
  author={Zhou, Kaiyang and Yang, Yongxin and Cavallaro, Andrea and Xiang, Tao},
  booktitle={ICCV},
  year={2019}


```