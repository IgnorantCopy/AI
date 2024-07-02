import os.path
import logging
import time
import sys
import cv2
import numpy as np
from collections import OrderedDict

import torch

from utils import utils_logger
from utils import utils_image as util

from models.network_srresnet import MSRResNet as net


def main():
    utils_logger.logger_info("logger_1", log_path="log_msrresnet.log")
    logger = logging.getLogger("logger_1")

    # --------------------------------
    # basic settings
    # --------------------------------
    testsets = "testsets"
    testset_L = "set1"

    # torch.cuda.current_device()
    # torch.cuda.empty_cache()

    if sys.platform == "darwin":
        logger.info("Running on macOS")
        # device = torch.device('cpu')
        device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")  # for MacOS
    else:
        # torch.backends.cudnn.benchmark = True
        # device = torch.device('cpu')
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Current device is {device}")

    # --------------------------------
    # load model
    # --------------------------------
    model_path = os.path.join("model_zoo", "MSRResNetx4.pth")
    model = net(in_nc=3, out_nc=3, nf=64, nb=16, upscale=4)

    # model_old = torch.load(model_path)
    # state_dict = model.state_dict()
    # for ((key, param),(key2, param2)) in zip(model_old.items(), state_dict.items()):
    #     state_dict[key2] = param
    #     print([key, key2])
    # print([param.size(), param2.size()])
    # torch.save(state_dict, 'model_new.pth')

    model.load_state_dict(torch.load(model_path), strict=True)
    model.eval()
    for k, v in model.named_parameters():
        v.requires_grad = False
    model = model.to(device)

    # number of parameters
    number_parameters = sum(map(lambda x: x.numel(), model.parameters()))
    logger.info("Params number: {}".format(number_parameters))

    # --------------------------------
    # read image
    # --------------------------------
    L_path = os.path.join(testsets, testset_L)
    E_path = os.path.join(testsets, testset_L + "_results")
    util.mkdir(E_path)

    # record runtime
    test_results = OrderedDict()
    test_results["runtime"] = []

    logger.info(L_path)
    logger.info(E_path)
    idx = 0

    # start = torch.cuda.Event(enable_timing=True)
    # end = torch.cuda.Event(enable_timing=True)

    for img in util.get_image_paths(L_path):
        # --------------------------------
        # (1) img_L
        # --------------------------------
        idx += 1
        img_name, ext = os.path.splitext(os.path.basename(img))
        logger.info("{:->4d}--> {:>10s}".format(idx, img_name + ext))

        img_L = util.imread_uint(img, n_channels=3)

        w, h, c = img_L.shape
        img_bicubic = cv2.resize(img_L, (4 * h, 4 * w), interpolation=cv2.INTER_CUBIC)

        img_L = util.uint2tensor4(img_L)  # pytorch b x c x h x w = batchsize x channel x h x w
        img_L = img_L.to(device)

        # start.record()
        # img_E = model(img_L)
        # end.record()
        # torch.cuda.synchronize()
        # test_results['runtime'].append(start.elapsed_time(end))  # milliseconds

        # torch.cuda.synchronize()
        start_time = time.time()
        img_E = model(img_L)
        # torch.cuda.synchronize()
        end_time = time.time()
        test_results["runtime"].append(end_time - start_time)  # seconds
        logger.info("Model forward pass took {:.6f} seconds.".format(end_time - start_time))

        # --------------------------------
        # (2) img_E
        # --------------------------------
        img_E = util.tensor2uint(img_E)

        util.imsave(np.concatenate([img_bicubic, img_E], 1), os.path.join(E_path, img_name + ".png"))
        # util.imsave(img_E, os.path.join(E_path, img_name+ext))
    ave_runtime = sum(test_results['runtime']) / len(test_results['runtime'])  # / 1000.0
    logger.info("------> Average runtime of ({}) is : {:.6f} seconds".format(L_path, ave_runtime))


if __name__ == "__main__":
    main()
