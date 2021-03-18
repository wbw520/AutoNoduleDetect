from Classification.tools.tools import AucCal, matrixs
from Classification.data_generation import load_data
import torch
from tqdm.auto import tqdm
from Classification.train import get_args_parser
import numpy as np
import argparse
from timm.models import create_model


@torch.no_grad()
def inference(args, model, data):
    all_pre = []
    all_true = []
    for i_batch, sample_batch in enumerate(tqdm(data["val"])):
        inputs = sample_batch["image"].to(args.gpu, dtype=torch.float32)
        labels = sample_batch["label"].to(args.gpu, dtype=torch.int64)
        outputs = model(inputs)
        outputs = outputs.cpu().detach().numpy()
        labels = labels.cpu().detach().numpy()
        all_pre.append(outputs)
        all_true.append(labels)
    all_pre = np.concatenate(all_pre, axis=0)
    all_true = np.concatenate(all_true, axis=0)
    epoch_auc = AucCal(model_name + "_" + mode + "_" + args.mode).cal_auc(all_pre, all_true)
    print("Nodule_auc:", epoch_auc)
    preds = np.argmax(all_pre, axis=1)
    matrixs(preds, all_true, model_name + "_" + mode + "_" + args.mode)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    mode = "all"
    model_name = args.model
    init_model = create_model(
        model_name,
        pretrained=args.pre_trained,
        num_classes=args.num_classes)
    init_model = init_model.to(args.gpu)
    init_model.load_state_dict(torch.load("saved_model/" + model_name + "_" + mode + ".pt", map_location=args.gpu), strict=True)
    print("load pre-train over")
    init_model.eval()
    init_model.to(args.gpu)
    if mode == "all":
        # args.mode = "left"
        # dataloaders = load_data(args)
        # inference(args, init_model, dataloaders)
        args.mode = "right"
        dataloaders = load_data(args)
        inference(args, init_model, dataloaders)
    else:
        args.mode = mode
        dataloaders = load_data(args)
        inference(args, init_model, dataloaders)
