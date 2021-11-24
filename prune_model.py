import os
from copy import deepcopy
from config import Cfg, CreateNet
from utils.utils import *


def obtain_filters_mask(model, thresh, CBL_idx, prune_idx):
    pruned = 0
    total = 0
    num_filters = []
    filters_mask = []
    for idx in CBL_idx:
        bn_module = model.module_list[idx][1]
        if idx in prune_idx:
            weight_copy = bn_module.weight.data.abs().clone()
            channels = weight_copy.shape[0]  #
            min_channel_num = int(channels * 0.01) if int(channels * 0.01) > 0 else 1
            mask = weight_copy.gt(thresh).float()
            if int(torch.sum(mask)) < min_channel_num:
                _, sorted_index_weights = torch.sort(weight_copy, descending=True)
                mask[sorted_index_weights[:min_channel_num]] = 1.
            remain = int(mask.sum())
            pruned = pruned + mask.shape[0] - remain
            print('layer index: {:>3d} \t total channel: {:>4d} \t '.format(idx, mask.shape[0]))
            print('remaining channel: {:>4d}'.format(remain))
        else:
            mask = torch.ones(bn_module.weight.data.shape)
            remain = mask.shape[0]
        total += mask.shape[0]
        num_filters.append(remain)
        filters_mask.append(mask.clone())
    prune_ratio = pruned / total
    print('Prune channels: {}\tPrune ratio: {:.3f}'.format(pruned, prune_ratio))
    return num_filters, filters_mask


if __name__ == '__main__':
    cfgfile = 'model_data/IRB_yolov4.cfg'
    modelfile = 'Logs/IRBYOLO_SparseTraining/cv4/IRBYOLO_ST_cv4.pth'
    prune_percent = 0.8

    model = CreateNet(cfgfile)

    model_dict = model.state_dict()
    pretrained_dict = torch.load(modelfile, map_location=Cfg.device)
    pretrained_dict = {k: v for k, v in pretrained_dict.items()
                       if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)

    CBL_idx, Conv_idx, prune_idx, _, _ = parse_module_defs2(model.module_defs)
    bn_weights = gather_bn_weights(model.module_list, prune_idx)
    sorted_bn, sorted_index = torch.sort(bn_weights)
    thresh_index = int(len(bn_weights) * prune_percent)
    thresh = sorted_bn[thresh_index]
    print('Global Threshold should be less than {:.4f}.'.format(thresh))

    num_filters, filters_mask = obtain_filters_mask(model, thresh, CBL_idx, prune_idx)
    CBLidx2mask = {idx: mask for idx, mask in zip(CBL_idx, filters_mask)}
    CBLidx2filters = {idx: filters for idx, filters in zip(CBL_idx, num_filters)}

    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i['is_access'] = False

    merge_mask(model, CBLidx2mask, CBLidx2filters)

    for i in CBLidx2mask:
        CBLidx2mask[i] = CBLidx2mask[i].clone().cpu().numpy()

    for i in model.module_defs:
        if i['type'] == 'shortcut':
            i.pop('is_access')

    compact_module_defs = deepcopy(model.module_defs)
    for idx in CBL_idx:
        assert compact_module_defs[idx]['type'] == 'convolutional'
        compact_module_defs[idx]['filters'] = str(CBLidx2filters[idx])
        if compact_module_defs[idx]['group'] != '1':
            if idx - 1 in CBLidx2filters:
                compact_module_defs[idx]['group'] = str(CBLidx2filters[idx - 1])
            else:
                compact_module_defs[idx]['group'] = str(CBLidx2filters[idx - 5])

    cfglist = [model.hyperparams.copy()] + compact_module_defs
    compact_model = CreateNet(cfgfile=cfglist)

    init_weights_from_loose_model(compact_model, model, CBL_idx, Conv_idx, CBLidx2mask)

    pruned_cfg_name = os.path.join(os.path.dirname(modelfile), 'MobileYOLO_{}.cfg'.format(prune_percent))
    if not os.path.exists(os.path.dirname(pruned_cfg_name)):
        os.makedirs(os.path.dirname(pruned_cfg_name))
    pruned_cfg_file = write_cfg(pruned_cfg_name, [model.hyperparams.copy()] + compact_module_defs)
    print('Config file has been saved: {}'.format(pruned_cfg_file))

    compact_model_name = os.path.join(os.path.dirname(modelfile), 'MobileYOLO_{}.pth'.format(prune_percent))
    if not os.path.exists(os.path.dirname(compact_model_name)):
        os.makedirs(os.path.dirname(compact_model_name))
    torch.save(compact_model.state_dict(), compact_model_name)
    print('Compact model has been saved: {}'.format(compact_model_name))
