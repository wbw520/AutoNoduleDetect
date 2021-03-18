import torch.nn as nn
from SemanticSegmentation.tools import load_data, panduan, make_model, read_one
from .date_generator import PrepareList
from .train_model import *
import torch.optim as optim
from .config import args
import numpy as np
import torch
from .sync_batchnorm import convert_model
from .classification_data_factory import DealWith


def train():
    dataloaders = load_data(args)
    init_model = make_model(args, args.model_name)

    if args.multi:
        init_model = convert_model(init_model)
        init_model = init_model.cuda()
        if torch.cuda.device_count() > 1:
            init_model = nn.DataParallel(init_model, device_ids=[0, 1])
    else:
        init_model.to(args.gpu)

    init_model = init_model.to(device)
    optimizer_ft = optim.Adam(filter(lambda p: p.requires_grad, init_model.parameters()), lr=args.learning_rate)
    criterion = torch.nn.CrossEntropyLoss()
    train_model(init_model, dataloaders, criterion, optimizer_ft, "pspnet.pt", aux=True, num_epochs=20)


def classification_generate(show_inference=False):
    init_model = make_model(args, args.model_name, aux=False)
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    init_model.load_state_dict(torch.load(args.model_name + ".pt"), strict=False)
    print("load finished")
    init_model = init_model.to(device)
    init_model.eval()

    if show_inference:
        read_one(init_model, device)
        return

    root_need = args.data_deal
    name = PrepareList(root_need).generate_list()
    DEMO = DealWith(args)
    for i in range(len(name)):
        if panduan(args.root_save+name[i][:-4]):  # skip image already processed
            continue
        if i % 100 == 0:
            print(str(i)+"/"+str(len(name)))
        original_image, model_image = DEMO.get_demo(name[i])
        output_demo = init_model(model_image)
        _, preds = torch.max(output_demo, 1)
        preds = np.squeeze(np.array(preds.cpu()), axis=0)
        DEMO.deal_all(preds, name[i])


if __name__ == '__main__':
    # train()
    classification_generate(show_inference=False)
