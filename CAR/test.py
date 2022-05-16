from option import args
import argparse
import glob
import os
import torch
import cv2
import numpy as np
from skimage import img_as_ubyte
from torch.autograd import Variable
import torch.nn as nn
from utils import logger, batch_PSNR_SSIM_v1, forward_chop
from model.cncn import Net

torch.manual_seed(args.seed)
parser = argparse.ArgumentParser(description="CNCN")
parser.add_argument("--model", type=str, default='CNCN', help="Mode name")
parser.add_argument('--logdir', type=str, default='./pretrain/cncn_Q10.pt', help="Path to pretrained model")
parser.add_argument("--lq_test_data", type=str, default='./testsets/LQ/LIVE1/Q10', help='Classic, LIVE1')
parser.add_argument("--hq_test_data", type=str, default='./testsets/HQ/LIVE1/Q10', help='Classic, LIVE1')
parser.add_argument("--rgb_range", type=int, default=1.)
parser.add_argument("--save_path", type=str, default='./res/CNCN', help='Save restoration results')
parser.add_argument("--save", type=bool, default=True)
parser.add_argument("--chop", type=bool, default=True)
parser.add_argument("--ensemble", type=bool, default=False)

opt = parser.parse_args()

lg = logger(opt.model, 'res/' + opt.model + '.log')

if not os.path.exists(opt.save_path):
    os.makedirs(opt.save_path)

def normalize(data, rgb_range):
    return data / (255. / rgb_range)


def main():
    lg.info('Loading model ...')
    model = Net()
    model.load_state_dict(torch.load(opt.logdir, map_location='cuda'))
    model = model.cuda()

    model.eval()
    # load data info
    lg.info('Loading data info ...')
    lq_files = glob.glob(os.path.join(opt.lq_test_data, '*.jpeg'))
    lq_files.sort()

    hq_files = glob.glob(os.path.join(opt.hq_test_data, '*.png'))
    hq_files.sort()

    # print(lq_files, hq_files)

    # process data
    psnr_test = 0
    ssim_test = 0
    with torch.no_grad():
        for f_lq, f_hq in zip(lq_files, hq_files):
            Img_lq = cv2.imread(f_lq)
            Img_lq = normalize(np.float32(Img_lq[:, :, 0]), opt.rgb_range)
            Img_lq = np.expand_dims(Img_lq, 0)
            Img_lq = np.expand_dims(Img_lq, 1)
            Img_lq = torch.Tensor(Img_lq)
           
            Img_hq = cv2.imread(f_hq)
            Img_hq = normalize(np.float32(Img_hq[:, :, 0]), opt.rgb_range)
            Img_hq = np.expand_dims(Img_hq, 0)
            Img_hq = np.expand_dims(Img_hq, 1)
            Img_hq = torch.Tensor(Img_hq)

            Img_hq, Img_lq = Variable(Img_hq.cuda()), Variable(Img_lq.cuda())

            with torch.no_grad():
                if opt.ensemble:
                    Out = torch.clamp(forward_chop(Img_lq, model, n_GPUs=args.n_GPUs, ensemble=True), 0., opt.rgb_range)
                else:
                    Out = torch.clamp(forward_chop(Img_lq, model, n_GPUs=args.n_GPUs, ensemble=False), 0., opt.rgb_range)

            psnr_score, ssim_score = batch_PSNR_SSIM_v1(Out, Img_hq) # n c h w
            psnr_test += psnr_score
            ssim_test += ssim_score
            file_name = os.path.split(f_hq)[-1]
            lg.info("%s: PSNR %.4f  SSIM %.4f" % (file_name, psnr_score, ssim_score))

            # save results
            if opt.save:
                image = Out.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                cv2.imwrite(os.path.join(opt.save_path, file_name), img_as_ubyte(image))

    psnr_test /= len(lq_files)
    ssim_test /= len(lq_files)
    lg.info("\nPSNR on test data %f, SSIM on test data %f" % (psnr_test, ssim_test))
    lg.info('Finish!\n')


if __name__ == "__main__":
    main()
