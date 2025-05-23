# Concepts from Neurons: Building Interpretable Medical Image Diagnostic Models by Dissecting Opaque Neural Networks

Implementation for IPMI 2025 paper Concepts from Neurons: Building Interpretable Medical Image Diagnostic Models by Dissecting Opaque Neural Networks by [Shizhan Gong](peterant330.github.io), Huayu Wang, Xiaofan Zhang, and [Qi Dou](https://www.cse.cuhk.edu.hk/~qdou/).

![Alt text](assets/framework.png?raw=true "Title")
<p align="center"> **Figure** Overview of the proposed framework. </p>

## Sample Results
![Alt text](assets/results.png?raw=true "Title")
<p align="center"> **Figure** Examples of explanations provided by our method. For each input image, we show top-4 concepts and their contributions to the logits of the correct labels. We also present the corresponding reports of Harvard-FairVLMed and MIMIC-CXR for references. Some descriptions of the normal findings are omitted.  </p>

## Setup

## Dataset
We use three datasets to evaluate our method:

- **HAM10000:** The dataset can be accessed via this [link](https://www.kaggle.com/datasets/kmader/skin-cancer-mnist-ham10000).
- **Harvard-FairVLMed:** The dataset can be accessed via this [link](https://github.com/Harvard-Ophthalmology-AI-Lab/FairCLIP?tab=readme-ov-file).
- **MIMIC-CXR:** The dataset can be accessed via this [link](https://github.com/MIT-LCP/mimic-cxr). The original dataset is of extremely large size. Therefore, we utilized a cleaned version provided in this [link](https://github.com/cuhksz-nlp/R2Gen).


## Training opaque models

## Training SAE

## Name the concepts

## Training CAV

## Constructing CBMs

## Evaluation


## Bibtex
If you find this work helpful, you can cite our paper as follows:
```
@article{gong2025concepts,
  title={Concepts from Neurons: Building Interpretable Medical Image Diagnostic Models by Dissecting Opaque Neural Networks},
  author={Gong, Shizhan and Wang, Huayu and Zhang, Xiaofan and Dou, Qi},
  journal={Information Processing in Medical Imaging},
  year={2025}
}
```


## Contact
For any questions, please contact <a href="mailto:szgong22@cse.cuhk.edu.hk">szgong22@cse.cuhk.edu.hk</a>
