import numpy as np
import torch
import argparse
from train import get_args_parser
from timm.models import create_model
from data_generation import MakeList, MakeListTest
from tools.transform_func import make_transform
from tools.SIOU import Siou, CalFinalCoordinate
from tools.tools import patch
from tools.make_json import make_json
import copy
from PIL import Image
import cv2
import os
import torch.nn.functional as F
from captum.attr import (
    GradientShap,
    DeepLift,
    DeepLiftShap,
    IntegratedGradients,
    LayerConductance,
    NeuronConductance,
    NoiseTunnel,
    GuidedGradCam,
    LayerGradCam,
    LayerAttribution,
    LayerDeepLiftShap,
    LayerDeepLift
)
import matplotlib.cm as mpl_color_map
from tqdm import tqdm


def apply_colormap_on_image(org_im, activation, colormap_name):
    """
        Apply heatmap on image
    Args:
        org_img (PIL img): Original image
        activation_map (numpy arr): Activation map (grayscale) 0-255
        colormap_name (str): Name of the colormap
    """
    # Get colormap
    activation = cv2.resize(activation, org_im.size, interpolation=cv2.INTER_LINEAR)
    color_map = mpl_color_map.get_cmap(colormap_name)
    no_trans_heatmap = color_map(activation)
    # Change alpha channel in colormap to make sure original image is displayed
    heatmap = copy.copy(no_trans_heatmap)
    heatmap[:, :, 3] = 0.4
    heatmap = Image.fromarray((heatmap*255).astype(np.uint8))
    no_trans_heatmap = Image.fromarray((no_trans_heatmap*255).astype(np.uint8))

    # Apply heatmap on iamge
    heatmap_on_image = Image.new("RGBA", org_im.size)
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, org_im.convert('RGBA'))
    heatmap_on_image = Image.alpha_composite(heatmap_on_image, heatmap)
    return no_trans_heatmap, heatmap_on_image


def show_cam_on_image(name, img, masks, save_name):
    heatmap_only, heatmap_on_image = apply_colormap_on_image(img, masks, 'jet')
    os.makedirs("results/vis/" + name, exist_ok=True)
    heatmap_on_image.save("results/vis/" + name + "/" + f"{save_name}_{args.mode}_{args.model}.png")


def make_grad(name, attribute_f, inputs, img_heat, save_name, size):
    for target_index in range(1, args.num_classes):
        mask = attribute_f.attribute(inputs, target=target_index)
        if mask.size(1) > 1:
            mask = torch.mean(mask, dim=1, keepdim=True)
        mask = F.interpolate(mask, size=(size[1], size[0]), mode="bilinear")
        mask = mask.squeeze(dim=0).squeeze(dim=0)
        mask = mask.cpu().detach().numpy()
        mask = np.maximum(mask, 0)
        mask = mask - np.min(mask)
        mask = mask / np.max(mask)
        mask = np.uint8(255*mask)
        if save:
            show_cam_on_image(name, img_heat, mask, save_name)
        return mask


def cal(name, model, image_orl, size, label_true):
    func = make_transform(args, mode="inference")
    image = func(image_orl)
    image = image.unsqueeze(0)
    image = image.to(args.gpu, dtype=torch.float32)
    output = model(image)
    output = F.softmax(output, dim=1)
    prediction_score, pred_label_idx = torch.topk(output, 1)

    pred_label_idx.squeeze()
    predicted_label = str(pred_label_idx.item())
    if int(predicted_label) != 1 and int(predicted_label) != label_true:
        # print("predict wrong")
        count.append(1)
        return None

    # print('Predicted:', predicted_label, '(', prediction_score.squeeze().item(), ')')

    if args.att_type == "GradCam":
        gradients = LayerGradCam(model, layer=model.layer4)
        att_map = make_grad(name, gradients, image, image_orl, 'GradCam', size)
    if args.att_type == "DeepLIFT":
        gradients = LayerDeepLift(model, layer=model.layer4)
        att_map = make_grad(name, gradients, image, image_orl, 'DeepLIFT', size)
    return att_map


def cal_siou(name, img, att, root, location_inf):
    att_factory = Siou(args)
    cal_final = CalFinalCoordinate(args)
    att[att < args.heat_value] = 0
    att[att > args.heat_value] = 255
    coordinates = att_factory.deal_siou(root, att, location_inf)
    # print("final coordinate:", coordinates)
    seg_coordinates = []
    final_coordinates = []
    for coor in coordinates:
        total, seg = cal_final.coordinate(root, coor)
        seg_coordinates.append(seg)
        final_coordinates.append(total)
    if save:
        patch(np.array(img), name, seg_coordinates, args)
    return final_coordinates


def circle(data, model):
     for i_batch, sample_batch in enumerate(tqdm(data)):
        name = sample_batch[0].split("/")[-1]
        img = Image.open(sample_batch[0] + "/" + args.mode + ".png")
        mask = Image.open(sample_batch[0] + "/" + args.mode + "_mask.png")
        size = mask.size
        att = cal(name, model, img, size, label_true=1)
        if att is None:
            continue
        coordinates = cal_siou(name, img, att, sample_batch[0], sample_batch[2])
        if len(coordinates) == 0:
            continue
        if name not in record:
            record.update({name: coordinates})
        else:
            record[name].extend(coordinates)


def for_vis(args, cal_part):
    model_name = args.model
    model = create_model(
        model_name,
        pretrained=args.pre_trained,
        num_classes=args.num_classes)
    # for name, module in model._modules.items():
    #     print(name)
    model.load_state_dict(torch.load("saved_model/" + model_name + "_" + args.mode + ".pt", map_location=args.gpu), strict=True)

    model.to(args.gpu)
    model.eval()
    args.mode = cal_part
    if test_one:
        root = "/home/wbw/PAN/final_cxr/nodule/CR-01084483_20150513"
        location = np.array([0,0,1,1,0,0,0,1])
        name = root.split("/")[-2]
        img = Image.open(root + "/" + args.mode + ".png")
        mask = Image.open(root + "/" + args.mode + "_mask.png")
        size = mask.size
        att = cal(name, model, img, size, label_true=1)
        cal_siou(name, img, att, root, location)
    else:
        if test:
            files = MakeListTest(args).get_files()
            circle(files, model)
        else:
            PP = MakeList(args, need_part=False)
            PP.file_name = ["nodule"]
            train, val = PP.get_wanted_data()
            circle(train, model)
            circle(val, model)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    count = []
    record = {}
    save = False
    args.mode = "left"
    test_one = False
    test = True
    if not test_one:
        if args.mode == "all":
            for_vis(args, "left")
            args.mode = "all"
            print(len(count))
            count = []
            for_vis(args, "right")
            print(len(count))
        else:
            args.mode = "left"
            for_vis(args, "left")
            print(len(count))
            count = []
            args.mode = "right"
            for_vis(args, "right")
            print(len(count))
    else:
        want_part = "right"
        for_vis(args, want_part)

    args.mode = "all"
    # for v, k in record.items():
    #     img = Image.open(args.data_dir + "/for_test/" + v + "/" + "total.png")
    #     patch(np.array(img), v, k, args)
    make_json(record, {1: "nodule"}, "att_test_25", args.data_dir+"for_test_seg/")



