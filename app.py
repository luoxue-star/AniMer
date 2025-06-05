from pathlib import Path
import torch
import os
import cv2
import numpy as np
import tempfile
from tqdm import tqdm
import torch.utils
import trimesh
import torch.utils.data
import gradio as gr
from typing import Union, List, Tuple, Dict
from amr.models import AMR
from amr.configs import get_config
from amr.utils import recursive_to
from amr.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from amr.utils.renderer import Renderer, cam_crop_to_full
from huggingface_hub import snapshot_download

LIGHT_BLUE = (0.85882353, 0.74117647, 0.65098039)

# Load model config
path_model_cfg = 'data/AniMer/.hydra/config.yaml'
model_cfg = get_config(path_model_cfg)

# Load model
local_dir = "data/AniMer/checkpoints/"
PATH_CHECKPOINT = os.path.join(local_dir, "checkpoint.ckpt")
model = AMR.load_from_checkpoint(checkpoint_path=PATH_CHECKPOINT, map_location="cpu",
                                 cfg=model_cfg, strict=False, weights_only=True)
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = model.to(device)
model.eval()

# Setup the renderer
renderer = Renderer(model_cfg, faces=model.smal.faces)

# Make output directory if it does not exist
OUTPUT_FOLDER = "demo_out"
os.makedirs(OUTPUT_FOLDER, exist_ok=True)


def predict(im):
    return im["composite"]


def inference(img: Dict)-> Tuple[Union[np.ndarray|None], List[str]]:
    img = np.array(img["composite"])[:, :, :-1]
    boxes = np.array([[0, 0, img.shape[1], img.shape[0]]])  # x1, y1, x2, y2

    # Run AniMer on the crop image
    dataset = ViTDetDataset(model_cfg, img, boxes)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=1)
    all_verts = []
    all_cam_t = []
    temp_name = next(tempfile._get_candidate_names())
    for batch in tqdm(dataloader):
        batch = recursive_to(batch, device)
        with torch.no_grad():
            out = model(batch)

        pred_cam = out['pred_cam']
        box_center = batch["box_center"].float()
        box_size = batch["box_size"].float()
        img_size = batch["img_size"].float()
        scaled_focal_length = model_cfg.EXTRA.FOCAL_LENGTH / model_cfg.MODEL.IMAGE_SIZE * img_size.max()
        pred_cam_t_full = cam_crop_to_full(pred_cam, box_center, box_size, img_size,
                                           scaled_focal_length).detach().cpu().numpy()

        # Render the result
        input_patch = (batch['img'][0].cpu() * 255 * (DEFAULT_STD[:, None, None]) + (
                    DEFAULT_MEAN[:, None, None])) / 255.
        input_patch = input_patch.permute(1, 2, 0).numpy()

        verts = out['pred_vertices'][0].detach().cpu().numpy()
        cam_t = pred_cam_t_full[0]
        all_verts.append(verts)
        all_cam_t.append(cam_t)
        regression_img = renderer(out['pred_vertices'][0].detach().cpu().numpy(),
                                  out['pred_cam_t'][0].detach().cpu().numpy(),
                                  batch['img'][0],
                                  mesh_base_color=LIGHT_BLUE,
                                  scene_bg_color=(1, 1, 1),
                                    )
        regression_img = cv2.cvtColor((regression_img * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)

        # Render mesh onto the original image
        if len(all_verts):
            # Return mesh path
            trimeshes = [renderer.vertices_to_trimesh(vvv, ttt.copy(), LIGHT_BLUE) for vvv,ttt in zip(all_verts, all_cam_t)]
            # Join meshes
            mesh = trimesh.util.concatenate(trimeshes)
            # Save mesh to file
            mesh_name = os.path.join(OUTPUT_FOLDER, next(tempfile._get_candidate_names()) + '.obj')
            trimesh.exchange.export.export_mesh(mesh, mesh_name)

            return (regression_img, mesh_name)
        else:
            return (None, [])


demo = gr.Interface(
    fn=inference,
    analytics_enabled=False,
    inputs=gr.ImageEditor(
        sources=("upload", "clipboard"),
        brush=False,
        eraser=False,
        crop_size="1:1",
        layers=False,
        placeholder="Upload an image or select from the examples.",
    ),
    outputs=[
        gr.Image(label="Overlap image"),
        gr.Model3D(display_mode="wireframe", label="3D Mesh"),
    ],
    title="AniMer: Animal Pose and Shape Estimation Using Family Aware Transformer",
    description="""
    Project page: https://luoxue-star.github.io/AniMer_project_page/
    ## Steps for Use
    1. **Input**: Select an example image or upload your own image.
    2. **Crop**: Crop the animal in the image (Otherwise, the result may be poor.)
    3. **Output**:
    - Overlapping Image
    - 3D Mesh 
    """,
    examples=[
        'example_data/000000015956_horse.png', 
        'example_data/n02101388_1188.png',
        'example_data/n02412080_12159.png',
        'example_data/000000101684_zebra.png',
    ],
)

demo.launch()