# AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer
[**Arxiv**](https://arxiv.org/abs/2412.00837) | [**Project Page**](https://luoxue-star.github.io/AniMer_project_page/) | [**Hugging Face Demo**](https://huggingface.co/spaces/luoxue-star/AniMer)

## Environment Setup
```bash
git clone https://github.com/luoxue-star/AniMer.git
conda create -n AniMer python=3.10
cd AniMer
# install pytorch
pip install torch==2.4.0 torchvision==0.19.0 torchaudio==2.4.0 --index-url https://download.pytorch.org/whl/cu124
pip install -e .[all]
# install pytorch3d
pip install "git+https://github.com/facebookresearch/pytorch3d.git"
```

## Gradio demo
Downloading the checkpoint folder named AniMer from [here](https://drive.google.com/drive/folders/1xYCJyxZeSYPI6RXnHqa_lDbjnIAs8hVi?usp=sharing) to `data/`. Then you can try our model by:
```bash
python app.py
```


## Testing
If you do not want to use gradio app, you can use the following command:
```bash
python demo.py --checkpoint data/AniMer/checkpoints/checkpoint.ckpt --img_folder path/to/imgdir/
```
If you want to reproduce the results in the paper, please switch to the paper branch. 
The reason for this is that we found that the 3D keypoints of the Animal3D dataset may have been exported incorrectly, 
so the version released now is the result of retraining after we fixed it.

## Training
Downloading the pretrained backbone and CtrlAni3D dataset from [here](https://drive.google.com/drive/folders/1_S6ZZ6HYQ4ixAiN8m5uZxh0hFUsRNDxG?usp=sharing). Then, processing the data format to be consistent with Animal3D and replacing the training data path in the configs_hydra/experiment/AniMerStagex.yaml file. 
After that, you can train the model using the following command:
```bash
bash train.sh
```

## Evaluation
Replace the dataset path in amr/configs_hydra/experiment/default_val.yaml and run the following command: 
```bash
python eval.py --config data/AniMer/.hydra/config.yaml --checkpoint data/AniMer/checkpoints/checkpoint.ckpt --dataset DATASETNAME
```

## Acknowledgements
Parts of the code are borrowed from the following repos:
- [ViTPose](https://github.com/ViTAE-Transformer/ViTPose)
- [4DHumans](https://github.com/shubham-goel/4D-Humans)
- [HaMer](https://github.com/geopavlakos/hamer)
- [SupContrast](https://github.com/HobbitLong/SupContrast)

## Citation
If you find this code useful for your research, please consider citing the following paper:
```bibtex
@inproceedings{lyu2025animer,
  title={AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer},
  author={Lyu, Jin and Zhu, Tianyi and Gu, Yi and Lin, Li and Cheng, Pujin and Liu, Yebin and Tang, Xiaoying and An, Liang},
  booktitle={Proceedings of the Computer Vision and Pattern Recognition Conference},
  pages={17486--17496},
  year={2025}
}
```

## Contact
For questions about this implementation, please contact [Jin Lyu](lvjin1766@gmail.com) directly. 