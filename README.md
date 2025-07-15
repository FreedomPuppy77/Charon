# 8th ABAW Valence-Arousal Estimation

## 1. Create environment

Create a python environment using conda, install packages in requirements.txt with `pip install -r requirements.txt`, or manually. 

## 2. Preprocess data and extract feature

- mae pretrain with https://github.com/facebookresearch/mae
- Visual feature extraction is implemented using the feature_VA.py script.
- construct samples dataset using `construct_*.py`.
- split samples dataset to segments using `split_*.py`.

## 3. Train model

Train model with `VA/vaestimation.py`.

## 4. Predict

Predict with `VA/test.py`.

## 5.Citation

```
@inproceedings{Mamba-VA,
  title={Mamba-VA: A Mamba-Based Approach for Continuous Emotion Recognition in Valence-Arousal Space},
  author={Yuheng Liang, Zheyu Wang, Feng Liu, Mingzhou Liu, Yu Yao},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={5651--5656},
  year={2025}
}
```

