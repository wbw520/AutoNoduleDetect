from date_generator import *
import cv2
from model import *
from DANET.danet import get_danet
from modeling.deeplab import DeepLab
from FCN import FCN8s, VGGNet


class ColorTransition(object):
    def __init__(self, ignore_label=255):
        self.root = ""
        self.translate = {0: [0, 0, 0],         # class 0  back_ground         black
                          1: [128, 0, 0],       # class 1  left_down           red
                          2: [0, 128, 0],       # class 2  left_middle         green
                          3: [0, 0, 128],       # class 3  left_up             blue
                          4: [128, 128, 0],     # class 4  left_tip            yellow
                          5: [128, 128, 128],   # class 5  right_tip           silver
                          6: [64, 0, 0],        # class 6  right_up            brown
                          7: [0, 128, 128],     # class 7  right_middle        blue_green
                          8: [128, 0, 128]}     # class 8  right_down          purple

    def id2trainId(self, label, reverse=False):
        if reverse:
            w, h = label.shape
            label_copy = np.zeros((w, h, 3), dtype=np.uint8)
            for index, color in self.translate.items():
                label_copy[label == index] = color
        else:
            w, h, c = label.shape
            label_copy = np.zeros((w, h), dtype=np.uint8)
            for index, color in self.translate.items():
                label_copy[np.logical_and(*list([label[:, :, i] == color[i] for i in range(3)]))] = index
        return label_copy

    def traslation(self):
        all_folder = get_name(self.root)
        for i in all_folder:
            current_folder = self.root + i + "/"
            os.makedirs(self.root + i + "label_translated/")
            file_name = get_name(current_folder + "label", mode_folder=False)
            for j in file_name:
                image = cv2.imread(current_folder + j)
                binary = self.id2trainId(image)
                cv2.imwrite(self.root + i + "/label_translated/" + j, binary)


def load_data(args):
    L = MakeList(args.data_dir).make_list()
    data_transform = {"train": transforms.Compose([ImageFactory(), Aug(), ToTensor()]),
                        "val": transforms.Compose([ImageFactory(), ToTensor()])}
    image_dataset = {x: TotalDataset(L[x], data_transform[x]) for x in ["train", "val"]}
    dataloaders = {x: DataLoader(image_dataset[x], batch_size=args.batch_size, shuffle=True, num_workers=4)
                       for x in ["train", "val"]}  # start the generator
    print("load data over")
    return dataloaders


class IouCal(object):
    def __init__(self, num_class=9):
        self.num_class = num_class
        self.hist = np.zeros((self.num_class, self.num_class))
        self.name = ["bg:", "left_down:", "left_middle:", "left_up:", "left_tip:", "right_tip:", "right_up:",
                     "right_middle:", "right_down:"]

    def fast_hist(self, label, pred, num_class):
        return np.bincount(num_class * label.astype(int) + pred, minlength=num_class ** 2).reshape(num_class, num_class)

    def per_class_iou(self, hist):
        return np.diag(hist)/(hist.sum(1) + hist.sum(0) - np.diag(hist))   # IOU = TP / (TP + FP + FN)

    def evaluate(self, labels, preds):
        labels = np.array(labels.cpu())
        preds = np.array(preds.cpu())
        for label, pred in zip(labels, preds):
            self.hist += self.fast_hist(label.flatten(), pred.flatten(), self.num_class)

    def iou_demo(self):
        iou = self.per_class_iou(self.hist)
        STR = ""
        for i in range(len(self.name)):
            STR = STR + self.name[i] + str(round(iou[i], 3)) + " "
        print(STR)
        miou = np.nanmean(iou)
        print("Miou:", round(miou, 3))


def show_single(image_label, origin):
    # show single image
    image_label = ColorTransition().id2trainId(image_label, reverse=True)
    origin = cv2.resize(origin, (800, 1000), interpolation=cv2.INTER_NEAREST)
    image_show = cv2.resize(image_label, (800, 1000), interpolation=cv2.INTER_NEAREST)
    image_show = cv2.addWeighted(origin, 1.0, image_show, 0.6, 0)
    plt.figure(figsize=(10, 10), facecolor="#FFFFFF")
    plt.imshow(image_show)
    plt.axis('on')
    plt.show()


def read_one(model, device):
    # visualize one image for evaluation
    frame = "demo.png"
    image = cv2.imread(frame)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    origin_image = image
    image = cv2.resize(image, (512, 512), interpolation=cv2.INTER_NEAREST)
    image = np.transpose(image, (2, 0, 1))
    image = np.array([image])
    image = torch.from_numpy(image/255)
    demo = image.to(device, dtype=torch.float32)
    output_demo = model(demo)
    _, preds = torch.max(output_demo, 1)
    preds = np.squeeze(np.array(preds.cpu()), axis=0)
    show_single(preds, origin_image)


def panduan(root):
    return os.path.exists(root)


def make_model(model_name, aux=True):
    if "PSP" in model_name:
        model = PspNet(args.num_classes, use_aux=aux)
        print("load PSPNet")
    elif "U" in model_name:
        model = UNet(3, args.num_classes)
        print("load UNet")
    elif "DA" in model_name:
        model = get_danet(args.num_classes)
        print("load DANet")
    elif "Deep" in model_name:
        model = DeepLab(num_classes=args.num_classes, backbone='resnet', output_stride=16)
        print("load DeepLab-V3")
    elif "FCN" in model_name:
        vgg_model = VGGNet(requires_grad=True, show_params=False)
        model = FCN8s(pretrained_net=vgg_model, n_class=args.num_classes)
        print("load FCN")
    else:
        raise Exception("Invalid model name !", model_name)
    return model