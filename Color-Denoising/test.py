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
parser.add_argument('--logdir', type=str, default='./pretrain/cncn_n10.pt', help="Path to pretrained model") # 49 27
parser.add_argument("--test_data", type=str, default='./testsets/Kodak24', help='test on Kodak24, BSD68 and Urban100')
parser.add_argument("--test_noiseL", type=float, default=10, help='noise level used on test set')
parser.add_argument("--rgb_range", type=int, default=1.)
parser.add_argument("--save_path", type=str, default='./res/CNCN_n10', help='Save restoration results')
parser.add_argument("--save", type=bool, default=False)
parser.add_argument("--chop", type=bool, default=True)
parser.add_argument("--ensemble", type=bool, default=False)
opt = parser.parse_args()
lg = logger(opt.model, 'res/' + 'CNCN_n10.log')


def normalize(data, rgb_range):
    return data / (255. / rgb_range)


def main():
    # Build model
    lg.info('Noise level: %s' % (opt.test_noiseL))
    lg.info('Loading model: %s' % (os.path.split(opt.logdir)[-1]))
    net = Net()
    net = nn.DataParallel(net).module
    net.load_state_dict(torch.load(opt.logdir, map_location='cuda'))
    model = net.cuda()

    model.eval()
    # load data info
    lg.info('Loading data info: %s' % (os.path.split(opt.test_data)[-1]))
    files_source = glob.glob(os.path.join(opt.test_data, '*.png'))
    files_source.sort()

    # process data
    psnr_test = 0
    ssim_test = 0
    with torch.no_grad():
        for f in files_source:
            Img = cv2.imread(f, cv2.COLOR_BGR2RGB)
            Img = normalize(np.float32(Img), opt.rgb_range)
            ISource = torch.from_numpy(np.ascontiguousarray(Img)).permute(2, 0, 1).float().unsqueeze(0)
            # ISource = torch.Tensor(Img)
            torch.manual_seed(args.seed)
            noise = torch.FloatTensor(ISource.size()).normal_(mean=0, std=opt.test_noiseL / (255. / opt.rgb_range))

            # noisy image
            INoisy = ISource + noise
            ISource, INoisy = Variable(ISource.cuda()), Variable(INoisy.cuda())
            with torch.no_grad():
                if opt.ensemble:
                    Out = torch.clamp(forward_chop(INoisy, model, n_GPUs=args.n_GPUs, ensemble=True), 0., opt.rgb_range)
                else:
                    Out = torch.clamp(forward_chop(INoisy, model, n_GPUs=args.n_GPUs, ensemble=False), 0., opt.rgb_range)

            psnr_score, ssim_score = batch_PSNR_SSIM_v1(Out, ISource) # n c h w
            psnr_test += psnr_score
            ssim_test += ssim_score
            file_name = os.path.split(f)[-1]
            lg.info("%s: PSNR %.4f  SSIM %.4f" % (file_name, psnr_score, ssim_score))

            # save results
            if opt.save:
                image = Out.cpu().detach().permute(0, 2, 3, 1).squeeze(0).numpy()
                if not os.path.exists(opt.save_path):
                    os.makedirs(opt.save_path)
                cv2.imwrite(os.path.join(opt.save_path, file_name), img_as_ubyte(image))

    psnr_test /= len(files_source)
    ssim_test /= len(files_source)
    lg.info("\nPSNR on test data %f, SSIM on test data %f" % (psnr_test, ssim_test))
    lg.info('Finish!\n')


if __name__ == "__main__":
    main()
