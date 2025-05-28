# [ _TPAMI2025_ ] Effective and Efficient Multi-Task Learning Frameworks for Dense Scene Understanding 🌇🌆🏙️

##  :scroll: Introduction

This repository presents two complementary multi-task learning frameworks for dense scene understanding: BridgeNet, which achieves comprehensive feature interactions via bridge features across tasks, and HiTTs, which discovers effective supervision from partially annotated data using hierarchical task tokens. Task including:

| Task                         |  NYUD-v2  | PASCAL-Context | 
|------------------------------|:---------:|:--------------:|
| Semantic Segmentation        | ✔️        |      ✔️       |
| Depth Estimation             | ✔️        |       -       |
| Surface Normal Estimation    | ✔️        |      ✔️       |
| Human Parts Segmentation     |  -        |      ✔️       |
| Saliency Estimation          |  -        |      ✔️       |
| Edge Detection               | ✔️        |      ✔️       |




Please check the following papers for details:


### 🌉 [BridgeNet: Comprehensive and Effective Feature Interactions via Bridge Feature for Multi-task Dense Predictions](https://ieeexplore.ieee.org/abstract/document/10870147?casa_token=VmHCD37qqAYAAAAA:etniTucun-iYgFaVamMJO-BhbNxYIJ61PIzpAp9wBJ-KJnxdja1x58mFktAMZoWJAoKi32Pz-Tc)

> [Jingdong Zhang](https://evergreen0929.github.io/), [Jiayuan Fan](https://scholar.google.com/citations?hl=zh-CN&user=gsLd2ccAAAAJ), [Peng Ye](https://scholar.google.com/citations?user=UEZZP5QAAAAJ&hl=en&oi=sra), [Bo Zhang](https://openreview.net/profile?id=~Bo_Zhang17), [Hancheng Ye](https://openreview.net/profile?id=~Hancheng_Ye1), [Baopu Li](https://scholar.google.com/citations?user=OOY-4CwAAAAJ&hl=en), [Yancheng Cai](https://caiyancheng.github.io/) and [Tao Chen](https://eetchen.github.io/)
> 
> TPAMI 2025
<p align="center">
  <img alt="img-name" src="BridgeNet/assets/teaser.png" width="900">
</p>
<p align="center">
  <img alt="img-name" src="BridgeNet/assets/compare_show.png" width="900">
</p>


### 🏷️ [Multi-Task Label Discovery via Hierarchical Task Tokens for Partially Annotated Dense Predictions](https://arxiv.org/abs/2411.18823)

> [Jingdong Zhang](https://evergreen0929.github.io/), [Hanrong Ye](https://sites.google.com/site/yhrspace/), [Xin Li](https://people.tamu.edu/~xinli/), [Wenping Wang](https://scholar.google.com/citations?user=28shvv0AAAAJ&hl=en) and [Dan Xu](https://www.danxurgb.net/)
> 
> Arxiv

<p align="center">
  <img alt="img-name" src="HiTTs/assets/vis_label_init.png" width="900">
</p>

# Cite
BibTex:
```
@article{zhang2025bridgenet,
  title={BridgeNet: Comprehensive and Effective Feature Interactions via Bridge Feature for Multi-Task Dense Predictions},
  author={Zhang, Jingdong and Fan, Jiayuan and Ye, Peng and Zhang, Bo and Ye, Hancheng and Li, Baopu and Cai, Yancheng and Chen, Tao},
  journal={IEEE Transactions on Pattern Analysis and Machine Intelligence},
  year={2025},
  publisher={IEEE}
}
@article{zhang2024multi,
  title={Multi-task label discovery via hierarchical task tokens for partially annotated dense predictions},
  author={Zhang, Jingdong and Ye, Hanrong and Li, Xin and Wang, Wenping and Xu, Dan},
  journal={arXiv preprint arXiv:2411.18823},
  year={2024}
}
```
Please do consider :star2: star our project to share with your community if you find this repository helpful!

# Contact
Please contact [Jingdong Zhang](https://evergreen0929.github.io/) if any questions.

# Related Project
