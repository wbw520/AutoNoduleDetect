import time
import torch
from tqdm.auto import tqdm


def train_model(my_model, dataloaders, criterion, optimizer, save_name, args, data_list, aux=False):
    start_time = time.time()

    if args.mode == "all":
        record = {"train": [], "val_right": [], "val_left": []}
    else:
        record = {"train": [], "val": []}
    best_acc = 0
    num_epochs = args.num_epoch

    for epoch in range(num_epochs):
        print("Epoch {}/{}".format(epoch, num_epochs-1))
        print("-" * 10)
        print("start")
        print("lr_rate: ", optimizer.param_groups[0]["lr"])
        # for every epoch there is train and val phase respectively
        for phase in data_list:
            if phase == "train":
                print("start_training round " + str(epoch))
                my_model.train()  # set model to train
            else:
                print("start " + phase + " round" + str(epoch))
                my_model.eval()   # set model to evaluation

            running_loss = 0.0
            running_corrects = 0
            for i_batch, sample_batch in enumerate(tqdm(dataloaders[phase])):
                L = len(dataloaders[phase])
                i = i_batch
                inputs = sample_batch["image"].to(args.gpu, dtype=torch.float32)
                labels = sample_batch["label"].to(args.gpu, dtype=torch.int64)
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
                        loss = criterion(outputs, labels)

                _, preds = torch.max(outputs, 1)   # (H, W)

                if phase == "train":
                    loss.backward()
                    optimizer.step()

                # statistics
                a = loss.item()
                b = torch.sum(preds == labels)  # tensor use data to convert to
                b = b.item()/(inputs.size(0))
                running_loss += a
                running_corrects += b

            epoch_loss = round(running_loss / len(dataloaders[phase]), 3)
            epoch_acc = round(running_corrects / len(dataloaders[phase]), 3)
            record[phase].append(epoch_acc)

            if phase != "train":
                if args.mode == "all":
                    if phase == "val_left":
                        if epoch == num_epochs//2:
                            optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
                        current_avg_acc = round((record["val_right"][epoch] + record["val_left"][epoch])/2, 3)
                        if current_avg_acc > best_acc:
                            best_acc = epoch_acc
                            torch.save(my_model.state_dict(), save_name)
                            print("get higher acc save current model")
                else:
                    if epoch == num_epochs//2:
                        optimizer.param_groups[0]["lr"] = optimizer.param_groups[0]["lr"] * 0.1
                    if record["val"][epoch] > best_acc:
                        best_acc = epoch_acc
                        torch.save(my_model.state_dict(), save_name)
                        print("get higher acc save current model")
        for k, v in record.items():
            print(k, ":", v)

    time_elapsed = time.time() - start_time
    print("training complete in {:.0f}m {:.0f}s".format(time_elapsed // 60, time_elapsed % 60))