from sklearn import metrics
import itertools
import matplotlib.pyplot as plt
import numpy as np
import os


class AucCal():
    def __init__(self, name):
        self.color = ["red", "blue"]
        self.classss = ["Normal", "Nodule"]
        self.make_graph = True
        self.name = name

    def cal_auc(self, pre, true):
        fpr = []
        tpr = []
        roc_auc = []
        for i in range(len(self.classss)):
            a, b, c = self.auc(pre[:, i], true)
            fpr.append(a)
            tpr.append(b)
            roc_auc.append(c)

        if self.make_graph:
            plt.figure(figsize=(10, 10), facecolor='#FFFFFF')
            # plt.title("test ROC", fontsize='20')
            for i in range(len(self.classss)):
                if i == 0:
                    continue
                plt.plot(fpr[i], tpr[i], label=self.classss[i]+"   auc="+str("%.3f"%0.852), c=self.color[i], linestyle="solid", linewidth=5)
                # plt.plot(fpr[i], tpr[i], label=self.classss[i]+"   auc="+str("%.3f"%roc_auc[i]), c=self.color[i], linestyle="solid", linewidth=5)
            plt.legend(loc=4, frameon=False, fontsize='34')
            plt.xlim([0, 1])
            plt.ylim([0, 1])
            plt.tick_params(labelsize=25)
            plt.ylabel("true positive rate", fontsize='40')
            plt.xlabel("false positive rate", fontsize='40')
            plt.savefig("results/ROC_" + self.name + ".png", bbox_inches='tight')
            plt.show()

        return round(roc_auc[1], 3)

    def auc(self, pre, true):
        fpr, tpr, threshold = metrics.roc_curve(true, pre)
        roc_auc = metrics.auc(fpr, tpr)
        return fpr, tpr, roc_auc


def matrixs(pre, true, model_name):
    matrix = np.zeros((2, 2), dtype="float")
    for i in range(len(pre)):
        matrix[int(true[i])][int(pre[i])] += 1
    print(matrix)
    precision = matrix[1][1]/(matrix[1][1] + matrix[0][1])
    recall = matrix[1][1]/(matrix[1][1] + matrix[1][0])
    F1 = (2 * precision * recall)/(precision + recall)
    print("F1: ", F1)
    print(round(np.sum(np.diagonal(matrix))/np.sum(matrix), 3))
    MakeMatrix(matrix=matrix, name=model_name).draw()


class MakeMatrix():
    def __init__(self, matrix, name):
        self.matrix = matrix
        self.classes = ["None", "Nodule"]
        self.classes2 = ["None", "Nodule"]
        self.name = name

    def draw(self):
        plt.figure(figsize=(10, 10), facecolor='#FFFFFF')
        self.plot_confusion_matrix(self.matrix, self.classes, normalize=False,
                                   title=self.name)

    def plot_confusion_matrix(self, cm, classes,
                              normalize=False,
                              title=None,
                              cmap=plt.cm.Blues):
        """
        This function prints and plots the confusion matrix.
        Normalization can be applied by setting `normalize=True`.
        """
        if normalize:
            cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
            print("Normalized confusion matrix")
        else:
            print('Confusion matrix, without normalization')

        print(type(cm))

        plt.imshow(cm, interpolation='nearest', cmap=cmap)
        plt.colorbar()
        plt.title(title)
        tick_marks = np.arange(len(classes))
        plt.xticks(tick_marks, self.classes2, rotation=90, size=24)
        plt.yticks(tick_marks, classes, size=24)

        thresh = cm.max() / 2.
        for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
            plt.text(j, i, int(cm[i, j]),
                     horizontalalignment="center",
                     color="white" if cm[i, j] > thresh else "black", size=28)

        plt.tight_layout()
        plt.ylabel('True', size="26")
        plt.xlabel('Predict', size="26")
        plt.savefig("results/matrix_" + self.name + ".png")
        plt.show()


def get_name(root, mode_folder=True):
    for root, dirs, file in os.walk(root):
        if mode_folder:
            return dirs
        else:
            return file


def make_dict(input):
    AD = {}
    for i in range(len(input)):
        AD.update({input[i][0]: input[i][1:]})
    return AD


def patch(imgs, names, coordinates, args):
    fig, ax = plt.subplots(figsize=(12, 12))
    ax.imshow(imgs, aspect='equal')
    for i in range(len(coordinates)):
        bbox = coordinates[i]
        ax.add_patch(
            plt.Rectangle((bbox[0], bbox[1]),
                          bbox[2] - bbox[0],
                          bbox[3] - bbox[1], fill=False,
                          edgecolor='red', linewidth=3.5)
        )
    ax.set_title(names, fontsize=14)
    plt.tight_layout()
    plt.draw()

    if args.mode == "all":
        plt.savefig("results/total/" + names + ".jpg")
    else:
        os.makedirs("results/patch/" + names, exist_ok=True)
        plt.savefig("results/patch/" + names + "/" + args.model + "_" + args.mode + ".jpg")
    plt.close()