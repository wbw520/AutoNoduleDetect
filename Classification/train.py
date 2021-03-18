from Classification.data_generation import load_data
from Classification.train_model import *
import torch.optim as optim
import torch
import argparse
from timm.models import create_model


def get_args_parser():
    parser = argparse.ArgumentParser('Set classification model', add_help=False)

    # root set
    parser.add_argument("--data_dir", type=str, default="/home/wangbowen/data/final_cxr/",
                        help="Path to the directory containing the training data.")
    parser.add_argument("--data_deal", type=str, default="/home/wbw/PAN/test",
                        help="Path to the directory need processing.")

    # data setting
    parser.add_argument('--non_nodule_ratio', default=0.7, type=float)
    parser.add_argument('--normal_ratio', default=0.9, type=float)
    parser.add_argument('--aug', default=True, type=bool, help='whether use pre dataset parameter')

    # train settings
    parser.add_argument('--model', default="resnest50d", type=str)
    parser.add_argument('--mode', default="left", type=str)
    parser.add_argument("--batch_size", type=int, default=8,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning_rate", type=float, default=0.0001,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--num_classes", type=int, default=2,
                        help="Number of classes to predict")
    parser.add_argument("--num_epoch", type=int, default=20,
                        help="Number of training steps.")
    parser.add_argument("--img_size", type=int, default=260,
                        help="size of the input image.")
    parser.add_argument('--pre_trained', default=True, type=str, help='whether use pre parameter for backbone')

    # extra settings
    parser.add_argument('--num_workers', default=4, type=int)
    parser.add_argument("--gpu", type=str, default='cuda:0',
                        help="choose gpu device.")
    parser.add_argument("--heat_value", type=int, default=25,
                        help="heat threshold.")
    parser.add_argument("--x_min", type=int, default=0,
                        help="min x size for found boxes.")
    parser.add_argument("--y_min", type=int, default=0,
                        help="min y size for found boxes.")
    parser.add_argument("--SI_standard", type=float, default=0.7,
                        help="threshold for bbox generation")
    parser.add_argument("--att_type", type=str, default="GradCam",
                        help="threshold for bbox generation")
    return parser


def main(aux=False):
    mode = args.mode
    model_name = args.model
    dataloaders = load_data(args)
    init_model = create_model(
        model_name,
        pretrained=args.pre_trained,
        num_classes=args.num_classes)
    init_model = init_model.to(args.gpu)
    optimizer_ft = optim.AdamW(filter(lambda p: p.requires_grad, init_model.parameters()), lr=0.0001)
    criterion = torch.nn.CrossEntropyLoss()
    if args.mode == "all":
        data_list = ["train", "val_right", "val_left"]
    else:
        data_list = ["train", "val"]
    print(model_name)
    train_model(init_model, dataloaders, criterion, optimizer_ft, "../records/saved_model/" + model_name + "_" + mode + ".pt",
                args=args, data_list=data_list, aux=aux)


if __name__ == '__main__':
    parser = argparse.ArgumentParser('model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()
    main()