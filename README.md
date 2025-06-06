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
Downloading the checkpoint folder named AniMer from [here](https://drive.google.com/drive/folders/1p6ysA247FAOHI867Bb2z7zxa34i-aXnP?usp=drive_link) to `data/`. Then you can try our model by:
```bash
python app.py
```

## Testing
If you do not want to use gradio app, you can use the following command:
```bash
python demo.py --checkpoint data/AniMer/checkpoints/checkpoint.ckpt --img_folder path/to/imgdir/
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
@misc{lyu2024animeranimalposeshape,
      title={AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer}, 
      author={Jin Lyu and Tianyi Zhu and Yi Gu and Li Lin and Pujin Cheng and Yebin Liu and Xiaoying Tang and Liang An},
      year={2024},
      eprint={2412.00837},
      archivePrefix={arXiv},
      primaryClass={cs.CV},
      url={https://arxiv.org/abs/2412.00837}, 
}
```

## Contact
For questions about this implementation, please contact [Jin Lyu](lvjin1766@gmail.com) directly. 