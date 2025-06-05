from pathlib import Path
import detectron2.config
import detectron2.engine
import torch
import argparse
import os
import cv2
import numpy as np
from tqdm import tqdm
import torch.utils
import torch.utils.data
from amr.models import load_amr
from amr.utils import recursive_to
from amr.datasets.vitdet_dataset import ViTDetDataset, DEFAULT_MEAN, DEFAULT_STD
from amr.utils.renderer import Renderer, cam_crop_to_full
import detectron2
from detectron2 import model_zoo
import warnings
warnings.filterwarnings("ignore")

LIGHT_BLUE = (0.65098039, 0.74117647, 0.85882353)


def main():
    import time
    start = time.time()
    parser = argparse.ArgumentParser(description='AMR demo code')
    parser.add_argument('--checkpoint', type=str,
                        help='Path to pretrained model checkpoint')
    parser.add_argument('--img_folder', type=str, default='example_data/images', help='Folder with input images')
    parser.add_argument('--out_folder', type=str, default='demo_out', help='Output folder to save rendered results')
    parser.add_argument('--side_view', dest='side_view', action='store_true', default=False,
                        help='If set, render side view also')
    parser.add_argument('--save_mesh', dest='save_mesh', action='store_true', default=False,
                        help='If set, save meshes to disk also')
    parser.add_argument('--batch_size', type=int, default=1, help='Batch size for inference/fitting')
    parser.add_argument('--file_type', nargs='+', default=['*.jpg', '*.png', '*JPEG'],
                        help='List of file extensions to consider')

    args = parser.parse_args()

    model, model_cfg = load_amr(args.checkpoint)

    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
    model = model.to(device)
    model.eval()

    # Setup the renderer
    renderer = Renderer(model_cfg, faces=model.smal.faces)

    # Make output directory if it does not exist
    os.makedirs(args.out_folder, exist_ok=True)

    # Load detector
    cfg = detectron2.config.get_cfg()
    cfg.merge_from_file(model_zoo.get_config_file("COCO-Detection/faster_rcnn_X_101_32x8d_FPN_3x.yaml"))
    cfg.MODEL.ROI_HEADS.SCORE_THRESH_TEST = 0.5 
    cfg.MODEL.WEIGHTS = "/data/lvjin/code/detectorn2/model_final_68b088.pkl"
    detector = detectron2.engine.DefaultPredictor(cfg)

    img_paths = sorted([img for end in args.file_type for img in Path(args.img_folder).glob(end)])
    for img_path in img_paths:
        img_cv2 = cv2.imread(str(img_path))
        img_cv2 = cv2.cvtColor(img_cv2, cv2.COLOR_BGR2RGB)

        # Detect humans in image
        det_out = detector(img_cv2)

        det_instances = det_out['instances']
        valid_idx = [i for i, (c, s) in enumerate(zip(det_instances.pred_classes, det_instances.scores)) if ((c in [15, 16, 17, 18, 19, 21, 22]) & (s > 0.7))]
        boxes = det_instances.pred_boxes.tensor[valid_idx].cpu().numpy()

        # Run AniMer on detected animals
        dataset = ViTDetDataset(model_cfg, img_cv2, boxes)
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=8, shuffle=False, num_workers=0)
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
            batch_size = batch['img'].shape[0]
            for n in range(batch_size):
                # Get filename from path img_path
                img_fn, _ = os.path.splitext(os.path.basename(img_path))
                animal_id = int(batch['animalid'][n])
                white_img = (torch.ones_like(batch['img'][n]).cpu() - DEFAULT_MEAN[:, None, None] / 255) / (
                            DEFAULT_STD[:, None, None] / 255)
                input_patch = (batch['img'][n].cpu() * (DEFAULT_STD[:, None, None]) + (
                            DEFAULT_MEAN[:, None, None])) / 255.
                input_patch = input_patch.permute(1, 2, 0).numpy()

                regression_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        batch['img'][n],
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                            )

                final_img = np.concatenate([input_patch, regression_img], axis=1)

                if args.side_view:
                    side_img = renderer(out['pred_vertices'][n].detach().cpu().numpy(),
                                        out['pred_cam_t'][n].detach().cpu().numpy(),
                                        white_img,
                                        mesh_base_color=LIGHT_BLUE,
                                        scene_bg_color=(1, 1, 1),
                                        side_view=True)
                    final_img = np.concatenate([final_img, side_img], axis=1)

                cv2.imwrite(os.path.join(args.out_folder, f'{img_fn}_{animal_id}.png'), 
                            cv2.cvtColor((255 * final_img[:, :, ::-1]).astype(np.uint8), cv2.COLOR_RGB2BGR))

                # Add all verts and cams to list
                verts = out['pred_vertices'][n].detach().cpu().numpy()
                cam_t = pred_cam_t_full[n]

                # Save all meshes to disk
                if args.save_mesh:
                    camera_translation = cam_t.copy()
                    tmesh = renderer.vertices_to_trimesh(verts, camera_translation, LIGHT_BLUE)
                    tmesh.export(os.path.join(args.out_folder, f'{img_fn}_{animal_id}.obj'))


if __name__ == '__main__':
    main()

