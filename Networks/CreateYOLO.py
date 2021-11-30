import torch
import torch.nn.functional as F
from torch import nn

from utils.utils import parse_model_cfg


class Mish(nn.Module):
    def forward(self, x):
        return x.mul(torch.tanh(F.softplus(x)))

def create_modules(module_defs, ClassNum, AnchorsNum=3):
    hyperparams = module_defs.pop(0)
    filters = int(hyperparams['channels'])
    output_filters = [filters]
    module_list = nn.ModuleList()
    routs = []
    yolo_index = -1

    for i, mdef in enumerate(module_defs):
        modules = nn.Sequential()

        if mdef['type'] == 'convolutional':
            bn = int(mdef['batch_normalize'])
            filters = int(mdef['filters'])
            kernel_size = int(mdef['size'])
            pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
            group = int(mdef['group'])
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=filters,
                                                   kernel_size=kernel_size,
                                                   stride=int(mdef['stride']),
                                                   padding=pad,
                                                   groups=group,
                                                   bias=not bn))
            if bn:
                modules.add_module('BatchNorm2d', nn.BatchNorm2d(filters, momentum=0.1))
            if mdef['activation'] == 'relu6':
                modules.add_module('activation', nn.ReLU6(inplace=True))
            elif mdef['activation'] == 'leaky':
                modules.add_module('activation', nn.LeakyReLU(0.1, inplace=True))
            elif mdef['activation'] == 'mish':
                modules.add_module('activation', Mish())

        elif mdef['type'] == 'maxpool':
            kernel_size = int(mdef['size'])
            stride = int(mdef['stride'])
            maxpool = nn.MaxPool2d(kernel_size=kernel_size, stride=stride, padding=int((kernel_size - 1) // 2))
            modules = maxpool

        elif mdef['type'] == 'upsample':
            modules = nn.Upsample(scale_factor=int(mdef['stride']), mode='nearest')

        elif mdef['type'] == 'route':
            layers = [int(x) for x in mdef['layers'].split(',')]
            filters = sum([output_filters[i + 1 if i > 0 else i] for i in layers])
            routs.extend([l if l > 0 else l + i for l in layers])

        elif mdef['type'] == 'shortcut':
            filters = output_filters[int(mdef['from'])]
            layer = int(mdef['from'])
            routs.extend([i + layer if layer < 0 else layer])

        elif mdef['type'] == 'yolohead':
            outChannel = AnchorsNum * (5 + ClassNum)
            kernel_size = int(mdef['size'])
            pad = (kernel_size - 1) // 2 if int(mdef['pad']) else 0
            modules.add_module('Conv2d', nn.Conv2d(in_channels=output_filters[-1],
                                                   out_channels=outChannel,
                                                   kernel_size=kernel_size,
                                                   stride=int(mdef['stride']),
                                                   padding=pad,
                                                   bias=True))
        else:
            print('Warning: Unrecognized Layer Type: ' + mdef['type'])

        module_list.append(modules)
        output_filters.append(filters)

    return module_list, routs, hyperparams


class CreateNetwork(nn.Module):
    def __init__(self, ClassNum, cfgfile, AnchorsNum=3):
        super(CreateNetwork, self).__init__()
        if isinstance(cfgfile, str):
            self.module_defs = parse_model_cfg(cfgfile)
        else:
            self.module_defs = cfgfile

        self.module_list, self.routs, self.hyperparams = create_modules(self.module_defs, ClassNum, AnchorsNum)

    def forward(self, x):
        listMean = {}
        listStd = {}
        count = 0
        layer_outputs = []
        output = []
        for i, (mdef, module) in enumerate(zip(self.module_defs, self.module_list)):
            count += 1
            mtype = mdef['type']
            if mtype in ['convolutional', 'upsample', 'maxpool']:
                x = module(x)
                listStd[str(count)] = float(torch.std(x))
                listMean[str(count)] = float(torch.mean(x))
            elif mtype == 'route':
                layers = [int(x) for x in mdef['layers'].split(',')]
                if len(layers) == 1:
                    x = layer_outputs[layers[0]]
                    if 'groups' in mdef:
                        x = x[:, (x.shape[1] // 2):]
                else:
                    x = torch.cat([layer_outputs[i] for i in layers], 1)
            elif mtype == 'shortcut':
                x = x + layer_outputs[int(mdef['from'])]
                listStd[str(count)] = float(torch.std(x))
                listMean[str(count)] = float(torch.mean(x))
            elif mtype == 'yolohead':
                x = module(x)
                output.append(x)
            layer_outputs.append(x if i in self.routs else [])
        return output[0], output[1], output[2], {'std': listStd, 'mean': listMean}
