# EFDMix on image classification across domains

## How to install

This code is based on [Dassl.pytorch](https://github.com/KaiyangZhou/Dassl.pytorch). Please follow the instructions at https://github.com/KaiyangZhou/Dassl.pytorch#installation to install `dassl`.

## How to run

Please follow the steps below before running the script

- modify `DATA` and `DASSL` in `*.sh` based on the paths on your computer
- activate the `dassl` environment via `conda activate dassl`
- `cd` to `scripts/`


### Domain Generalization
```bash
# You can change the `DATASET`, `NET`, `MIX` to conduct experiments with other dataset, network, and mix_type, respectively.
# PACS | EFDMix based on res18
bash res18.sh 

# PACS | EFDMix based on res50
bash res50.sh 
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

@article{zhou2020domain,
  title={Domain Adaptive Ensemble Learning},
  author={Zhou, Kaiyang and Yang, Yongxin and Qiao, Yu and Xiang, Tao},
  journal={arXiv preprint arXiv:2003.07325},
  year={2020}
}

```