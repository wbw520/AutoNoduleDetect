import time
import torch
from tqdm.auto import tqdm
from tools import read_one, IouCal


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def train_model(my_model, dataloaders, criterion, optimizer, save_name, crf_model=None, num_epochs=20, aux=False):
    start_time = time.time()

    val_acc_history = []
    train_acc_history = []
    best_acc = 0

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-" * 10)
        print("start")
        # for every epoch there is train and val phase respectively
        for phase in ["train", "val"]:
            if phase == "train":
                print("start_training round" + str(epoch))
                my_model.train()  # set model to train
            else:
                print("start_val round" + str(epoch))
                my_model.eval()   # set model to evaluation
                iou = IouCal()
                print("will cal iou")

            running_loss = 0.0
            running_corrects = 0
            for i_batch, sample_batch in enumerate(tqdm(dataloaders[phase])):
                L = len(dataloaders[phase])
                i = i_batch
                inputs = sample_batch["image"].to(device, dtype=torch.float32)
                labels = sample_batch["label"].to(device, dtype=torch.int64)
                # zero the gradient parameter
                optimizer.zero_grad()

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == "train"):
                    if aux and phase == "train":
                        outputs, aux_outputs = my_model(inputs)
                        loss1 = criterion(outputs, labels)
                        loss2 = criterion(aux_outputs, labels)
                        loss = loss1 + 0.4*loss2
                    else:
                        outputs = my_model(inputs)
                        if crf_model is not None:
                            outputs = crf_model(outputs)
                        loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)   # (H, W)

                if phase == "train":
                    loss.backward()
                    optimizer.step()
                if phase == "val":
                    iou.evaluate(labels, preds)

                # statistics
                a = loss.item()
                b = torch.sum(preds == labels)  # count num of correct predictions
                b = b.item()/(inputs.size(0)*inputs.size(2)*inputs.size(3))
                # print("{}/{} Loss: {:.4f} Acc: {:.4f} LR: {}".format(i, L-1, a, b, optimizer.param_groups[0]["lr"]))

                running_loss += a
                running_corrects += b

            epoch_loss = running_loss / len(dataloaders[phase])
            epoch_acc = running_corrects / len(dataloaders[phase])
            if phase == "train":
                train_acc_history.append(round(epoch_acc, 3))
            print("{} Loss: {:.4f} Acc: {:.4f}".format(phase, epoch_loss, epoch_acc))

            if phase == "val":
                print("cal iou")
                iou.iou_demo()
                read_one(my_model, device)
                if epoch_acc > best_acc:  # save model with best acc
                    print("get higher acc save model")
                    best_acc = epoch_acc
                    torch.save(my_model.state_dict(), save_name)

                if epoch == num_epochs//2:
                    optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1

                val_acc_history.append(round(epoch_acc, 3))
                print(val_acc_history)
                print(train_acc_history)

    time_elapsed = time.time() - start_time
    print("training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))