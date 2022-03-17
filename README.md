# EFDM
The official codes of our CVPR2022 paper: [Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization](https://arxiv.org/abs/2203.07740)

**One Sentence Summary:** EFDM outperforms AdaIN, which only matches first and second order statistics, by implicitly matching high orders statistics in an efficient manner. 

**A brief introduction:**
Many real-world tasks (e.g., Arbitrary Style Transfer and Domain Generalizaiton) can be cast as a feature distribution matching problem.
With the assumption of Gaussian feature distribution, conventional feature distribution matching methods usually match the mean and standard deviation of features. 
However, the feature distributions of real-world data are usually much more complicated than Gaussian, which cannot be accurately matched by using only the first-order and second-order statistics, while it is computationally prohibitive to use high-order statistics for distribution matching. 
In this work, we, for the first time to our best knowledge, propose to perform Exact Feature Distribution Matching (EFDM) by exactly matching the empirical Cumulative Distribution Functions (eCDFs) of image features, which could be implemented by applying the Exact Histogram Matching (EHM) in the image feature space.
Particularly, a fast EHM algorithm, named Sort-Matching, is employed to perform EFDM in a plug-and-play manner with minimal cost.
Below we show a brief implementation of it in PyTorch:
```python
import torch
def exact_feature_distribution_matching(content, style):
    assert (content.size() == style.size()) ## content and style features should share the same shape
    B, C, W, H = content.size(0), content.size(1), content.size(2), content.size(3)
    _, index_content = torch.sort(content.view(B,C,-1))  ## sort content feature
    value_style, _ = torch.sort(style.view(B,C,-1))      ## sort style feature
    inverse_index = index_content.argsort(-1)
    transferred_content = content.view(B,C,-1) + value_style.gather(-1, inverse_index) - content.view(B,C,-1).detach()
    return transferred_content.view(B, C, W, H)
```

In our paper, we have demonstrated the effectiveness of EFDMix on three tasks: arbitrary style transfer, 
cross-domain image classification, and cross-domain person re-identification. The source code for reproducing all experiments can be found in `EFDM/ArbitraryStyleTransfer`, `EFDM/DomainGeneralization/imcls`, and `EFDM/DomainGeneralization/reid`, respectively.

The review and supplementary material are given in the `Review_CVPR22.pdf` and `Supplementary_Material.pdf`, respectively.

To cite EFDM in your publications, please use the following bibtex entry:
```
@inproceedings{zhang2021exact,
  title={Exact Feature Distribution Matching for Arbitrary Style Transfer and Domain Generalization},
  author={Zhang, Yabin and Li, Minghan and Li, Ruihuang and Jia, Kui and Zhang, Lei},
  booktitle={CVPR},
  year={2022}
}
```
