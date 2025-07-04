import numpy as np
from tqdm import tqdm
import torch
from amr.utils import recursive_to
from amr.utils.evaluate_metric import Evaluator
from amr.datasets.datasets import EvaluationDataset
import argparse
from torch.utils.data import DataLoader
from amr.models.amr import AMR
from amr.configs import get_config
torch.multiprocessing.set_sharing_strategy('file_system')


def main(args):
    cfg = get_config(args.config)
    default_cfg = get_config(args.default_eval_config)
    model = AMR.load_from_checkpoint(args.checkpoint, cfg=cfg, strict=False)
    model.eval()

    smal_evaluator = Evaluator(smal_model=model.smal, image_size=cfg.MODEL.IMAGE_SIZE)
    cfg_eval_dataset = dict(default_cfg.DATASETS)
    aug_cfg = cfg_eval_dataset.pop("CONFIG", None)  # augmentation config is not used in evaluation

    if args.dataset.upper() == "ALL":
        for key in cfg_eval_dataset.keys():
            print(f"-------- Evaluate {key} dataset ------------")
            eval_one_dataset(cfg_eval_dataset[key], default_cfg, cfg, model, 
                             evaluator=smal_evaluator, 
                             aug_cfg=aug_cfg, 
                             key=key)
            print(f"-------{key} Dataset evaluate finish ------")
    else:
        print(f"-------- Evaluate {args.dataset} dataset ------------")
        eval_one_dataset(cfg_eval_dataset[args.dataset], default_cfg, cfg, model, 
                         evaluator=smal_evaluator, 
                         aug_cfg=aug_cfg, 
                         key=args.dataset)
        print(f"-------{args.dataset} Dataset evaluate finish ------")


def eval_one_dataset(dataset_cfg, default_cfg, cfg, model, evaluator, aug_cfg, key):
    dataset = EvaluationDataset(root_image=dataset_cfg['ROOT_IMAGE'], 
                                json_file=dataset_cfg['JSON_FILE']['TEST'], 
                                augm_config=aug_cfg, focal_length=cfg.SMAL.get("FOCAL_LENGTH", 1000),
                                image_size=cfg.MODEL.IMAGE_SIZE,
                                )
    dataloader = DataLoader(dataset, batch_size=cfg.TRAIN.BATCH_SIZE, num_workers=cfg.GENERAL.NUM_WORKERS)

    bar = tqdm(dataloader)
    pa_mpjpe_list, pck_list, auc_list, pa_mpvpe_list = [], [], [], []
    for i, batch in enumerate(bar):
        batch = recursive_to(batch, args.device)
        with torch.no_grad():
            output = model(batch)

        if key in ["ANIMAL3D", "CONTROL_ANIMAL3D"]:
            pa_mpjpe, pa_mpvpe = evaluator.eval_3d(output, batch)
        else:
            pa_mpjpe, pa_mpvpe = 0., 0.
        pck, auc = evaluator.eval_2d(output, batch, pck_threshold=default_cfg.METRIC.PCK_THRESHOLD)

        pa_mpjpe_list.append(pa_mpjpe)
        pa_mpvpe_list.append(pa_mpvpe)
        auc_list.append(auc)
        pck_list.append(pck)

        bar.set_postfix(PA_MPJPE=pa_mpjpe,
                        PA_MPVPE=pa_mpvpe,
                        AUC=auc,
                        pck=pck,)

    print("---------------- 3D metric -----------------")
    print(f"Avg PA-MPJPE: {np.mean(pa_mpjpe_list)}")
    print(f"Avg PA-MPVPE: {np.mean(pa_mpvpe_list)}")
    
    print("--------------- 2D metric ------------------")
    print(f"AUC: {np.mean(auc_list)}")
    pck_list = np.array(pck_list)
    for _, th in enumerate(default_cfg.METRIC.PCK_THRESHOLD):
        print(f"PCK@{th}: {np.mean(pck_list[:, _])}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="Path to config file", required=True)
    parser.add_argument("--checkpoint", type=str, help="Path to checkpoint file", required=True)
    parser.add_argument("--default_eval_config", type=str, default="amr/configs_hydra/experiment/default_val.yaml")
    parser.add_argument("--dataset", type=str, default="ALL")
    parser.add_argument("--device", type=str, default="cuda", help="Device to use for evaluation")
    args = parser.parse_args()
    main(args)
