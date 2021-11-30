import time
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.autograd import Variable
from torch.utils.data import DataLoader
from tqdm import tqdm
from config import Cfg, CreateNet
from metrics import get_AP
from utils.dataloader import YoloDataset, yolo_dataset_collate
from utils.utils import gather_bn_weights, Generate_Train
from utils.yolo_loss import YOLOLoss


def updateBN(module_list, s, prune_idx, epoch, epochs):
    s = s if epoch <= epochs * 0.30 else s * 0.01
    for idx in prune_idx:
        bn_module = module_list[idx][1]
        bn_module.weight.grad.data.add_(s * torch.sign(bn_module.weight.data))


def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']


def fit_one_epoch(net, yolo_losses, epoch, epoch_size, gen, writer):
    start_time = time.time()
    loss_this_epoch, loss_conf, loss_cls, loss_loc = 0, 0, 0, 0
    net.train()
    print('Start training')
    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Cfg.Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            images, targets = batch[0], batch[1]
            with torch.no_grad():
                if Cfg.Cuda:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor)).cuda()
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
                else:
                    images = Variable(torch.from_numpy(images).type(torch.FloatTensor))
                    targets = [Variable(torch.from_numpy(ann).type(torch.FloatTensor)) for ann in targets]
            optimizer.zero_grad()
            outputs = net(images)
            losses = []
            for i in range(3):
                loss_this_epoch_, loss_conf_, loss_cls_, loss_loc_ = yolo_losses(outputs[i], targets)
                loss_this_epoch += loss_this_epoch_
                loss_conf += loss_conf_
                loss_cls += loss_cls_
                loss_loc += loss_loc_
                losses.append(loss_this_epoch_)
            loss = sum(losses)
            loss.backward()

            if Cfg.isPruneTrain:
                updateBN(model.module_list, Cfg.pruneLambda, Cfg.prune_idx, epoch, Cfg.Epoch)

            optimizer.step()
            writer.add_scalar('Train_loss_step', loss, (epoch * epoch_size + iteration))
            time_consuming = time.time() - start_time
            pbar.set_postfix(**{'lr': get_lr(optimizer),
                                'step/s': time_consuming})
            pbar.update(1)
            start_time = time.time()

    loss_Pimg = loss_this_epoch / Cfg.num_train
    writer.add_scalar('Train_total_loss_Epoch', loss_Pimg, epoch)
    writer.add_scalar('Train_loss_conf_Epoch', loss_conf / Cfg.num_train, epoch)
    writer.add_scalar('Train_loss_cls_Epoch', loss_cls / Cfg.num_train, epoch)
    writer.add_scalar('Train_loss_loc_Epoch', loss_loc / Cfg.num_train, epoch)

    if Cfg.isPruneTrain:
        bn_weights = gather_bn_weights(model.module_list, Cfg.prune_idx)
        writer.add_histogram('bn_weights/hist', bn_weights.numpy(), epoch, bins='doane')

    net.eval()
    print('Start evaluation')
    val_ap = get_AP(net)
    writer.add_scalar('Val_AP', val_ap, epoch)


    print('Epoch:' + str(epoch + 1) + '/' + str(Cfg.Epoch))
    t = int(time.time())
    model_savePath = '{}/Epoch{}-AP{:.4f}-Loss{:.4f}-{}.pth'.format(Cfg.WriterDir, (epoch + 1), val_ap, loss_Pimg, t)
    torch.save(model.state_dict(), model_savePath)
    print('Total Loss: {:.4f} || AP: {:.4f} '.format(loss_Pimg, val_ap))


if __name__ == "__main__":

    for cv in range(5):
        Generate_Train(Cfg.train_path, Cfg.val_path, cv, Cfg.classes)

        netflag = 'MobileYOLO_cv{}'.format(cv)
        model_path_train = 'WeightsFile/Model/cv{}.pth'.format(cv)
        cfgfile = 'WeightsFile/Config/cv{}.cfg'.format(cv)

        t = time.time()
        Cfg.WriterDir = 'WeightsFile/{}_{}'.format(netflag, int(t))
        writer = SummaryWriter(log_dir=Cfg.WriterDir)

        model = CreateNet(cfgfile=cfgfile)
        if model_path_train != None:
            model_dict = model.state_dict()
            pretrained_dict = torch.load(model_path_train, map_location=Cfg.device)
            pretrained_dict = {k: v for k, v in pretrained_dict.items()
                               if k in model_dict and np.shape(model_dict[k]) == np.shape(v)}
            model_dict.update(pretrained_dict)
            model.load_state_dict(model_dict)

        net = model.train()

        if Cfg.Cuda:
            cudnn.benchmark = True
            net = net.cuda()

        yolo_loss = YOLOLoss(Cfg.anchors, Cfg.num_classes, (Cfg.h, Cfg.w), Cfg.Cuda)

        if Cfg.Cuda:
            graph_inputs = torch.from_numpy(np.random.rand(1, 3, Cfg.h, Cfg.w)).type(torch.FloatTensor).cuda()
        else:
            graph_inputs = torch.from_numpy(np.random.rand(1, 3, Cfg.h, Cfg.w)).type(torch.FloatTensor)

        optimizer = optim.Adam(net.parameters(), Cfg.lr, weight_decay=5e-4)

        lr_scheduler = optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0=5, T_mult=2)

        gen = DataLoader(YoloDataset(Cfg.train_path, (Cfg.h, Cfg.w)), shuffle=True, batch_size=Cfg.Batch_size,
                         num_workers=4, pin_memory=True, drop_last=True, collate_fn=yolo_dataset_collate)

        epoch_size = max(1, Cfg.num_train // Cfg.Batch_size)

        for epoch in range(Cfg.Epoch):
            fit_one_epoch(net, yolo_loss, epoch, epoch_size, gen, writer)
            lr_scheduler.step()
