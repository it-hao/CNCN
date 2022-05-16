import torch
import torch.nn as nn
import torch.nn.functional as F

def make_model(args, parent=False):
    return Net()


def default_conv(in_channels, out_channels, kernel_size, bias=True):
    return nn.Conv2d(
        in_channels, out_channels, kernel_size,
        padding=(kernel_size // 2), bias=bias)


class MGCB(nn.Module):
    def __init__(self, in_channels, reduction):
        super(MGCB, self).__init__()
        self.reduction = reduction
        self.k = in_channels // self.reduction  # 64 // 16 = 4
        self.in_channels = in_channels

        self.phi = nn.Conv2d(in_channels, self.k, kernel_size=1)
        self.softmax = nn.Softmax(dim=1)

        self.theta = nn.Conv2d(in_channels, self.k, kernel_size=1)

    def spatial_pool(self, x):
        b, c, h, w = x.size()
        input_x = x
        # [N, C, H * W]
        input_x = input_x.view(b, c, h * w)
        # [N, k, H, W]
        context_mask = self.phi(x)
        # [N, H * W, k]
        context_mask = context_mask.view(b, self.k, h * w).permute(0, 2, 1)
        # [N, H * W, k]
        context_mask = self.softmax(context_mask)
        # [N, C, k]
        context = torch.matmul(input_x, context_mask)
        # [N, k, H, W]
        theta_x = self.theta(x)
        # [N, k, H*W]
        theta_x = theta_x.view(b, self.k, h * w)
        # [N, C, H*W]
        y = torch.matmul(context, theta_x)
        # [N, C, H, W]
        y = y.view(b, c, h, w)

        return y

    def forward(self, x):
        # [N, C, H, W]
        context = self.spatial_pool(x)

        return context


# channel attention layer
class CALayer(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(CALayer, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.conv_du = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction, 1, padding=0, bias=True),
            nn.ReLU(inplace=True),
            nn.Conv2d(in_channels // reduction, in_channels, 1, padding=0, bias=True),
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.avg_pool(x)
        y = self.conv_du(y)
        return x * y


# Attention 
class Attention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(Attention, self).__init__()
        self.mgcb = MGCB(in_channels, reduction)
        self.ca = CALayer(in_channels, reduction)

    def forward(self, x):
        x1 = self.mgcb(x)
        x2 = self.ca(x)
        return x1 + x2


# residual block
# class ResBlock(nn.Module):
#     def __init__(self, conv, in_channels, kernel_size, bias=True, bn=False, act=nn.ReLU(True), res_MGCBle=1):

#         super(ResBlock, self).__init__()
#         m = []
#         for i in range(2):
#             m.append(conv(in_channels, in_channels, kernel_size, bias=bias))
#             if bn: m.append(nn.BatchNorm2d(in_channels))
#             if i == 0: m.append(act)

#         self.body = nn.Sequential(*m)
#         self.res_MGCBle = res_MGCBle

#     def forward(self, x):
#         res = self.body(x).mul(self.res_MGCBle)
#         res += x

#         return res

# residual attention block
class MGRB(nn.Module):
    def __init__(self, conv, in_channels, kernel_size, reduction, bias=True, bn=False, act=nn.PReLU(), res_MGCBle=1):
        super(MGRB, self).__init__()
        m = []
        for i in range(2):
            m.append(conv(in_channels, in_channels, kernel_size, bias=bias))
            if bn: m.append(nn.BatchNorm2d(in_channels))
            if i == 0: m.append(act)
        m.append(Attention(in_channels, reduction))
        self.body = nn.Sequential(*m)
        self.res_MGCBle = res_MGCBle

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# feature extraction module
class FEM(nn.Module):
    def __init__(self, conv, in_channels, kernel_size, reduction, act, res_MGCBle, n_resblocks):
        super(FEM, self).__init__()
        modules_body = []
        modules_body = [
            MGRB(conv, in_channels, kernel_size, reduction, bias=True, bn=False, act=nn.PReLU(), res_MGCBle=1) \
            for _ in range(n_resblocks)]
        modules_body.append(conv(in_channels, in_channels, kernel_size))
        self.body = nn.Sequential(*modules_body)

    def forward(self, x):
        res = self.body(x)
        res += x
        return res


# define network
class Net(nn.Module):
    def __init__(self, conv=default_conv):
        super(Net, self).__init__()
        # define basic setting
        n_modules = 3
        n_resblocks = 8
        in_channels = 64
        kernel_size = 3
        n_colors = 1
        res_MGCBle = 1
        act = nn.PReLU()
        reduction = 16
        self.n_modules = n_modules

        # define head module
        m_head = [conv(n_colors, in_channels, kernel_size)]

        # define body module
        m_body = [FEM(conv, in_channels, kernel_size, reduction, act, res_MGCBle, n_resblocks), P2NM(),
                  FEM(conv, in_channels, kernel_size, reduction, act, res_MGCBle, (n_resblocks - 2) // 2), P2NM(),
                  FEM(conv, in_channels, kernel_size, reduction, act, res_MGCBle, (n_resblocks - 2) // 2), P2NM(),
                  FEM(conv, in_channels, kernel_size, reduction, act, res_MGCBle, (n_resblocks - 2) // 2), P2NM(),
                  FEM(conv, in_channels, kernel_size, reduction, act, res_MGCBle, n_resblocks)]

        # define tail module
        m_tail = [conv(in_channels, n_colors, kernel_size)]

        self.head = nn.Sequential(*m_head)
        self.body = nn.Sequential(*m_body)
        self.tail = nn.Sequential(*m_tail)

    def forward(self, x):
        res = self.head(x)

        res = self.body(res)

        res = self.tail(res)

        return x + res

    def load_state_dict(self, state_dict, strict=True):
        own_state = self.state_dict()
        for name, param in state_dict.items():
            if name in own_state:
                if isinstance(param, nn.Parameter):
                    param = param.data
                try:
                    own_state[name].copy_(param)
                except Exception:
                    if name.find('tail') == -1:
                        raise RuntimeError('While copying the parameter named {}, '
                                           'whose dimensions in the model are {} and '
                                           'whose dimensions in the checkpoint are {}.'
                                           .format(name, own_state[name].size(), param.size()))
            elif strict:
                if name.find('tail') == -1:
                    raise KeyError('unexpected key "{}" in state_dict'
                                   .format(name))


"""
fundamental functions
"""
def same_padding(images, ksizes, strides, rates):
    assert len(images.size()) == 4
    batch_size, channel, rows, cols = images.size()
    out_rows = (rows + strides[0] - 1) // strides[0]
    out_cols = (cols + strides[1] - 1) // strides[1]
    effective_k_row = (ksizes[0] - 1) * rates[0] + 1
    effective_k_col = (ksizes[1] - 1) * rates[1] + 1
    padding_rows = max(0, (out_rows - 1) * strides[0] + effective_k_row - rows)
    padding_cols = max(0, (out_cols - 1) * strides[1] + effective_k_col - cols)
    # Pad the input
    padding_top = int(padding_rows / 2.)
    padding_left = int(padding_cols / 2.)
    padding_bottom = padding_rows - padding_top
    padding_right = padding_cols - padding_left
    paddings = (padding_left, padding_right, padding_top, padding_bottom)
    images = torch.nn.ZeroPad2d(paddings)(images)
    return images, paddings


def extract_image_patches(images, ksizes, strides, rates, padding='same'):
    """
    Extract patches from images and put them in the C output dimension.
    :param padding:
    :param images: [batch, channels, in_rows, in_cols]. A 4-D Tensor with shape
    :param ksizes: [ksize_rows, ksize_cols]. The size of the sliding window for
     each dimension of images
    :param strides: [stride_rows, stride_cols]
    :param rates: [dilation_rows, dilation_cols]
    :return: A Tensor
    """
    assert len(images.size()) == 4
    assert padding in ['same', 'valid']
    paddings = (0, 0, 0, 0)

    if padding == 'same':
        images, paddings = same_padding(images, ksizes, strides, rates)
    elif padding == 'valid':
        pass
    else:
        raise NotImplementedError('Unsupported padding type: {}.\
                Only "same" or "valid" are supported.'.format(padding))

    unfold = torch.nn.Unfold(kernel_size=ksizes, stride=strides)
    patches = unfold(images)
    return patches, paddings


# patch-wise non-local attention
class P2NM(nn.Module):
    def __init__(self, ksize=7, stride_1=4, stride_2=1, softmax_MGCBle=10, in_channels=64,
                 inter_channels=16):
        super(P2NM, self).__init__()
        self.ksize = ksize
        self.stride_1 = stride_1
        self.stride_2 = stride_2
        self.softmax_MGCBle = softmax_MGCBle
        self.inter_channels = inter_channels
        self.in_channels = in_channels
        self.strides = [1, 2, 4]
        # self.conv33 = nn.Conv2d(in_channels=2*in_channels,out_channels=in_channels,kernel_size=1,stride=1,padding=0)

        self.theta = nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels, kernel_size=1)
        self.phi = nn.ModuleList()
        self.g = nn.ModuleList()

        for i in range(len(self.strides)):
            self.phi.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                                      kernel_size=self.strides[i], stride=self.strides[i]))

            self.g.append(nn.Conv2d(in_channels=self.in_channels, out_channels=self.inter_channels,
                                    kernel_size=self.strides[i], stride=self.strides[i]))

        self.W = nn.Conv2d(in_channels=self.inter_channels * len(self.strides), out_channels=self.in_channels,
                           kernel_size=1)

    def forward(self, b):
        kernel = self.ksize

        theta_x = self.theta(b)

        g_x = []
        phi_x = []
        for i in range(len(self.strides)):
            g_x.append(self.g[i](b))
            phi_x.append(self.phi[i](b))

        raw_int_bs = list(theta_x.size())  # b*c*h*w

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        patch_theta_x, paddings_theta_x = extract_image_patches(theta_x, ksizes=[self.ksize, self.ksize],
                                                                strides=[self.stride_1, self.stride_1], rates=[1, 1])
        patch_theta_x = patch_theta_x.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
        patch_theta_x = patch_theta_x.permute(0, 4, 1, 2, 3)
        patch_theta_x_group = torch.split(patch_theta_x, 1, dim=0)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        patch_phi_x_groups = []
        for i in range(len(self.strides)):
            patch_phi_x, padding_phi_x = extract_image_patches(phi_x[i], ksizes=[self.ksize, self.ksize],
                                                               strides=[self.stride_2, self.stride_2],
                                                               rates=[1, 1],
                                                               padding='same')
            patch_phi_x = patch_phi_x.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
            patch_phi_x = patch_phi_x.permute(0, 4, 1, 2, 3)
            patch_phi_x_group = torch.split(patch_phi_x, 1)
            patch_phi_x_groups.append(patch_phi_x_group)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\
        patch_g_x_groups = []
        for i in range(len(self.strides)):
            patch_g_x, paddings_g_x = extract_image_patches(g_x[i], ksizes=[self.ksize, self.ksize],
                                                            strides=[self.stride_2, self.stride_2], rates=[1, 1])
            patch_g_x = patch_g_x.view(raw_int_bs[0], raw_int_bs[1], kernel, kernel, -1)
            patch_g_x = patch_g_x.permute(0, 4, 1, 2, 3)
            patch_g_x_group = torch.split(patch_g_x, 1)
            patch_g_x_groups.append(patch_g_x_group)
        # \\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\

        f_groups = []
        for i in range(len(self.strides)):
            f_groups.append(torch.split(phi_x[i], 1, dim=0))

        outs = []
        for i in range(len(self.strides)):
            y = []
            for xii, pi, ti, gi in zip(f_groups[i], patch_phi_x_groups[i], patch_theta_x_group, patch_g_x_groups[i]):
                # print(pi.size(), ti.size(), gi.size())
                h, w = xii.shape[2], xii.shape[3]
                _, paddings = same_padding(xii, [self.ksize, self.ksize], [1, 1], [1, 1])
                # ti = ti[0]  # [L1, C, k, k]
                c_s = gi.shape[2]  # channel
                k_s = ti[0].shape[2]  # ksize
                ti = ti.view(ti.shape[0], ti.shape[1], -1)  # [1, L1, C*k*k]
                pi = pi.permute(0, 2, 3, 4, 1)  # [1, C, k, k, L1]
                pi = pi.view(pi.shape[0], -1, pi.shape[4])  # [1, C*k*k, L2]
                score_map = torch.matmul(ti, pi)  # [1, L1, L2]
                score_map = score_map.view(score_map.shape[0], score_map.shape[1], h, w)  # [1, L1, h, w]
                b_s, l_s, h_s, w_s = score_map.shape

                yi = score_map.view(b_s, l_s, -1)  # [1, L1, h*w]
                yi = F.softmax(yi * self.softmax_MGCBle, dim=2).view(l_s, -1)
                gi = gi.view(h_s * w_s, -1)  # [1, h*w, k*k*c]
                yi = torch.mm(yi, gi)  # [1, L1, k*k*c]
                yi = yi.view(b_s, l_s, c_s, k_s, k_s)[0]
                zi = yi.view(1, l_s, -1).permute(0, 2, 1)

                zi = torch.nn.functional.fold(zi, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel), padding=paddings[0],
                                              stride=self.stride_1)
                inp = torch.ones_like(zi)
                inp_unf = torch.nn.functional.unfold(inp, (kernel, kernel), padding=paddings[0], stride=self.stride_1)
                out_mask = torch.nn.functional.fold(inp_unf, (raw_int_bs[2], raw_int_bs[3]), (kernel, kernel),
                                                    padding=paddings[0], stride=self.stride_1)
                zi /= out_mask
                y.append(zi)
            y = torch.cat(y, dim=0)
            outs.append(y)
        out = self.W(torch.cat(outs, dim=1))
        out += b
        return out


# if __name__ == '__main__':
#     net = Net()
#     from torchstat import stat

#     stat(net, (1, 12, 12))
