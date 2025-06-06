import os

if 'PYOPENGL_PLATFORM' not in os.environ:
    os.environ['PYOPENGL_PLATFORM'] = 'egl'
import torch
from torchvision.utils import make_grid
import numpy as np
import pyrender
import trimesh
import cv2
import math
import torch.nn.functional as F
from typing import List, Tuple


def create_raymond_lights():
    import pyrender
    thetas = np.pi * np.array([1.0 / 6.0, 1.0 / 6.0, 1.0 / 6.0])
    phis = np.pi * np.array([0.0, 2.0 / 3.0, 4.0 / 3.0])

    nodes = []

    for phi, theta in zip(phis, thetas):
        xp = np.sin(theta) * np.cos(phi)
        yp = np.sin(theta) * np.sin(phi)
        zp = np.cos(theta)

        z = np.array([xp, yp, zp])
        z = z / np.linalg.norm(z)
        x = np.array([-z[1], z[0], 0.0])
        if np.linalg.norm(x) == 0:
            x = np.array([1.0, 0.0, 0.0])
        x = x / np.linalg.norm(x)
        y = np.cross(z, x)

        matrix = np.eye(4)
        matrix[:3, :3] = np.c_[x, y, z]
        nodes.append(pyrender.Node(
            light=pyrender.DirectionalLight(color=np.ones(3), intensity=1.0),
            matrix=matrix
        ))

    return nodes


def get_keypoints_rectangle(keypoints: np.array, threshold: float) -> Tuple[float, float, float]:
    """
    Compute rectangle enclosing keypoints above the threshold.
    Args:
        keypoints (np.array): Keypoint array of shape (N, 3).
        threshold (float): Confidence visualization threshold.
    Returns:
        Tuple[float, float, float]: Rectangle width, height and area.
    """
    valid_ind = keypoints[:, -1] > threshold
    if valid_ind.sum() > 0:
        valid_keypoints = keypoints[valid_ind][:, :-1]
        max_x = valid_keypoints[:, 0].max()
        max_y = valid_keypoints[:, 1].max()
        min_x = valid_keypoints[:, 0].min()
        min_y = valid_keypoints[:, 1].min()
        width = max_x - min_x
        height = max_y - min_y
        area = width * height
        return width, height, area
    else:
        return 0, 0, 0


def render_keypoint(img: np.array, keypoint: np.array, threshold=0.1,
                    use_confidence=False, map_fn=lambda x: np.ones_like(x), alpha=1.0) -> np.array:
    if use_confidence and map_fn is not None:
        thicknessCircleRatioRight = 1. / 50 * map_fn(keypoint[:, -1])
    else:
        thicknessCircleRatioRight = 1. / 50 * np.ones(keypoint.shape[0])

    thicknessLineRatioWRTCircle = 0.75
    if keypoint.shape[0] == 26:
        pairs = [0, 24,  1, 24,  2, 24,  3, 14,  4, 15,  5, 16,  6, 17,  7, 18,  8, 12,  9, 13,  10, 7,  11, 7,
                12, 18,  13, 18,  14, 8,  15, 9,  16, 10,  17, 11,  18, 24,  19, 25,  20, 0,  21, 1,  22, 24,
                23, 24,  25, 7]
    elif keypoint.shape[0] == 18:
        pairs = [9, 8,  8, 2,  2, 3,  3, 4,  2, 0,  2, 1,  4, 5, 
                 5, 14,  14, 15,  4, 6,  6, 7,  7, 11,  11, 10,  
                 7, 13,  13, 12,  5, 16,  5, 17]
    else:
        raise ValueError("Keypoint shape not supported")
    pairs = np.array(pairs).reshape(-1, 2) if pairs is not None else None
    colors = [255., 0., 85.,
              255., 0., 0.,
              255., 85., 0.,
              255., 170., 0.,
              255., 255., 0.,
              170., 255., 0.,
              85., 255., 0.,
              0., 255., 0.,
              255., 0., 0.,
              0., 255., 85.,
              0., 255., 170.,
              0., 255., 255.,
              0., 170., 255.,
              0., 85., 255.,
              0., 0., 255.,
              255., 0., 170.,
              170., 0., 255.,
              255., 0., 255.,
              85., 0., 255.,
              0., 0., 255.,
              0., 0., 255.,
              0., 0., 255.,
              0., 255., 255.,
              0., 255., 255.,
              0., 255., 255.,
              255., 225., 255.]
    colors = np.array(colors).reshape(-1, 3)
    poseScales = [1]

    img_orig = img.copy()
    width, height = img.shape[1], img.shape[2]
    area = width * height

    lineType = 8
    shift = 0
    numberColors = len(colors)
    thresholdRectangle = 0.1

    animal_width, animal_height, animal_area = get_keypoints_rectangle(keypoint, thresholdRectangle)
    if animal_area > 0:
        ratioAreas = min(1, max(animal_width / width, animal_height / height))
        thicknessRatio = np.maximum(np.round(math.sqrt(area) * thicknessCircleRatioRight * ratioAreas), 2)
        thicknessCircle = np.maximum(1, thicknessRatio if ratioAreas > 0.05 else -np.ones_like(thicknessRatio))
        thicknessLine = np.maximum(1, np.round(thicknessRatio * thicknessLineRatioWRTCircle))
        radius = thicknessRatio / 2
    else:
        return img

    img = np.ascontiguousarray(img.copy())
    if pairs is not None:
        for i, pair in enumerate(pairs):
            index1, index2 = pair
            if keypoint[index1, -1] > threshold and keypoint[index2, -1] > threshold:
                thicknessLineScaled = int(round(min(thicknessLine[index1], thicknessLine[index2]) * poseScales[0]))
                colorIndex = index2
                color = colors[colorIndex % numberColors]
                keypoint1 = keypoint[index1, :-1].astype(np.int32)
                keypoint2 = keypoint[index2, :-1].astype(np.int32)
                cv2.line(img, tuple(keypoint1.tolist()), tuple(keypoint2.tolist()), tuple(color.tolist()),
                         thicknessLineScaled, lineType, shift)
    for part in range(len(keypoint)):
        faceIndex = part
        if keypoint[faceIndex, -1] > threshold:
            radiusScaled = int(round(radius[faceIndex] * poseScales[0]))
            thicknessCircleScaled = int(round(thicknessCircle[faceIndex] * poseScales[0]))
            colorIndex = part
            color = colors[colorIndex % numberColors]
            center = keypoint[faceIndex, :-1].astype(np.int32)
            cv2.circle(img, tuple(center.tolist()), radiusScaled, tuple(color.tolist()), thicknessCircleScaled,
                       lineType, shift)

    return img


class MeshRenderer:

    def __init__(self, cfg, faces=None):
        self.cfg = cfg
        self.img_res = cfg.MODEL.IMAGE_SIZE
        self.renderer = pyrender.OffscreenRenderer(viewport_width=self.img_res,
                                                   viewport_height=self.img_res,
                                                   point_size=1.0)

        self.camera_center = [self.img_res // 2, self.img_res // 2]
        self.faces = faces

    def visualize(self, vertices, camera_translation, images, focal_length, nrow=3, padding=2):
        images_np = np.transpose(images, (0, 2, 3, 1))
        rend_imgs = []
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(
                self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=focal_length, side_view=False),
                (2, 0, 1))).float()
            rend_img_side = torch.from_numpy(np.transpose(
                self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=focal_length, side_view=True),
                (2, 0, 1))).float()
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
        rend_imgs = make_grid(rend_imgs, nrow=nrow, padding=padding)
        return rend_imgs

    def visualize_tensorboard(self, vertices, camera_translation, images, focal_length, pred_keypoints, gt_keypoints,
                              pred_masks=None, gt_masks=None):
        images_np = np.transpose(images, (0, 2, 3, 1))
        rend_imgs = []
        pred_keypoints = np.concatenate((pred_keypoints, np.ones_like(pred_keypoints)[:, :, [0]]), axis=-1)
        pred_keypoints = self.img_res * (pred_keypoints + 0.5)
        gt_keypoints[:, :, :-1] = self.img_res * (gt_keypoints[:, :, :-1] + 0.5)
        # keypoint_matches = [(1, 12), (2, 8), (3, 7), (4, 6), (5, 9),
        # (6, 10), (7, 11), (8, 14), (9, 2), (10, 1), (11, 0), (12, 3), (13, 4), (14, 5)]
        # rend_img_pytorch3d = self.render_by_pytorch3d(vertices, camera_translation,
        #                                               images_np, focal_length=self.focal_length)
        for i in range(vertices.shape[0]):
            rend_img = torch.from_numpy(np.transpose(
                self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=focal_length, side_view=False),
                (2, 0, 1))).float()
            rend_img_side = torch.from_numpy(np.transpose(
                self.__call__(vertices[i], camera_translation[i], images_np[i], focal_length=focal_length, side_view=True),
                (2, 0, 1))).float()
            keypoints = pred_keypoints[i]
            pred_keypoints_img = render_keypoint(255 * images_np[i].copy(), keypoints) / 255
            keypoints = gt_keypoints[i]
            gt_keypoints_img = render_keypoint(255 * images_np[i].copy(), keypoints) / 255
            rend_imgs.append(torch.from_numpy(images[i]))
            rend_imgs.append(rend_img)
            rend_imgs.append(rend_img_side)
            if pred_masks is not None:
                rend_imgs.append(torch.from_numpy(pred_masks[i]))
            if gt_masks is not None:
                rend_imgs.append(torch.from_numpy(gt_masks[i]))
            rend_imgs.append(torch.from_numpy(pred_keypoints_img).permute(2, 0, 1))
            rend_imgs.append(torch.from_numpy(gt_keypoints_img).permute(2, 0, 1))
        return rend_imgs

    def __call__(self, vertices, camera_translation, image, focal_length, text=None, resize=None, side_view=False,
                 baseColorFactor=(1.0, 1.0, 0.9, 1.0), rot_angle=90):
        renderer = pyrender.OffscreenRenderer(viewport_width=image.shape[1],
                                              viewport_height=image.shape[0],
                                              point_size=1.0)
        material = pyrender.MetallicRoughnessMaterial(
            metallicFactor=0.0,
            alphaMode='OPAQUE',
            baseColorFactor=baseColorFactor)

        camera_translation[0] *= -1.

        mesh = trimesh.Trimesh(vertices.copy(), self.faces.copy())
        if side_view:
            rot = trimesh.transformations.rotation_matrix(
                np.radians(rot_angle), [0, 1, 0])
            mesh.apply_transform(rot)
        rot = trimesh.transformations.rotation_matrix(
            np.radians(180), [1, 0, 0])
        mesh.apply_transform(rot)
        mesh = pyrender.Mesh.from_trimesh(mesh, material=material)

        scene = pyrender.Scene(bg_color=[0.0, 0.0, 0.0, 0.0],
                               ambient_light=(0.3, 0.3, 0.3))
        scene.add(mesh, 'mesh')

        camera_pose = np.eye(4)
        camera_pose[:3, 3] = camera_translation
        camera_center = [image.shape[1] / 2., image.shape[0] / 2.]
        camera = pyrender.IntrinsicsCamera(fx=focal_length, fy=focal_length,
                                           cx=camera_center[0], cy=camera_center[1],
                                           zfar=1000)
        scene.add(camera, pose=camera_pose)

        light_nodes = create_raymond_lights()
        for node in light_nodes:
            scene.add_node(node)

        color, rend_depth = renderer.render(scene, flags=pyrender.RenderFlags.RGBA)
        color = color.astype(np.float32) / 255.0
        valid_mask = (color[:, :, -1] > 0)[:, :, np.newaxis]
        if not side_view:
            output_img = (color[:, :, :3] * valid_mask +
                          (1 - valid_mask) * image)
        else:
            output_img = color[:, :, :3]
        if resize is not None:
            output_img = cv2.resize(output_img, resize)

        output_img = output_img.astype(np.float32)
        renderer.delete()
        return output_img
