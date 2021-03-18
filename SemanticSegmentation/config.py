import argparse


def get_arguments():
    """Parse all the arguments for model.

    Returns:
      A list of parsed arguments.
    """
    parser = argparse.ArgumentParser(description="PSP-Net Network")

    # root set
    parser.add_argument("--data-dir", type=str, default="/home/wbw/U-Net_multi/",
                        help="Path to the directory containing the training image list.")
    parser.add_argument("--data-deal", type=str, default="/home/wbw/PAN/final_cxr/for_test/",
                        help="Path to the directory need processing.")
    parser.add_argument("--root-save", type=str, default="/home/wbw/PAN/final_cxr/for_test_seg/",
                        help="Path to the directory save processed images.")
    parser.add_argument("--data-single", type=str, default="",
                        help="Path to the directory need single processing.")

    # train settings
    parser.add_argument("--batch-size", type=int, default=8,
                        help="Number of images sent to the network in one step.")
    parser.add_argument("--learning-rate", type=float, default=0.0001,
                        help="Base learning rate for training with polynomial decay.")
    parser.add_argument("--num-classes", type=int, default=9,
                        help="Number of classes to predict (including background).")
    parser.add_argument("--num-epoch", type=int, default=20,
                        help="Number of training steps.")
    parser.add_argument("--aux", type=bool, default=True,
                        help="whether use aux branch for training")
    parser.add_argument("--image-size", type=int, default=512,
                        help="size of the input image.")

    # extra settings
    parser.add_argument("--model-name", type=str, default='DeepLab-V3',
                        help="name of init model.")
    parser.add_argument("--gpu", type=str, default='cuda:0',
                        help="choose gpu device.")
    parser.add_argument("--multi", type=bool, default=False,
                        help="choose gpu device.")
    parser.add_argument("--x-min", type=int, default=30,
                        help="min size for found boxes.")
    parser.add_argument("--y-min", type=int, default=30,
                        help="min size for found boxes.")
    return parser.parse_args()


args = get_arguments()