import argparse
from asyncore import write
from decimal import ConversionSyntax
import logging
from multiprocessing import reduction
import os
import random
import shutil
import sys
import time
import pdb
import cv2
import matplotlib.pyplot as plt
import imageio
from config import get_config
import numpy as np
import torch
import torch.backends.cudnn as cudnn
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from tensorboardX import SummaryWriter
from torch.utils.data import DataLoader
from torch.nn.modules.loss import CrossEntropyLoss
from torchvision import transforms
from tqdm import tqdm
from skimage.measure import label
from networks.vision_mamba import MambaUnet as ViM_seg
from dataloaders.dataset import (BaseDataSets, RandomGenerator, TwoStreamBatchSampler, ThreeStreamBatchSampler)
from networks.net_factory import BCP_net, net_factory
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d
from networks.wavelet import wavelet_2d_transform, create_2d_wavelet_filter
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='wavenet', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=10000, help='maximum epoch number to train')
parser.add_argument('--max_iterations', type=int, default=30000, help='maximum epoch number to train')
parser.add_argument('--batch_size', type=int, default=24, help='batch_size per gpu')
parser.add_argument('--deterministic', type=int,  default=1, help='whether use deterministic training')
parser.add_argument('--base_lr', type=float,  default=0.01, help='segmentation network learning rate')
parser.add_argument('--patch_size', type=list,  default=[256, 256], help='patch size of network input')
parser.add_argument('--seed', type=int,  default=1337, help='random seed')
parser.add_argument('--num_classes', type=int,  default=4, help='output channel of network')

parser.add_argument(
   '--cfg', type=str, default="../code/configs/vmamba_tiny.yaml", help='path to config file', )

parser.add_argument(
    "--opts",
    help="Modify config options by adding 'KEY VALUE' pairs. ",
    default=None,
    nargs='+',
)
parser.add_argument('--zip', action='store_true',
                    help='use zipped dataset instead of folder dataset')
parser.add_argument('--cache-mode', type=str, default='part', choices=['no', 'full', 'part'],
                    help='no: no cache, '
                    'full: cache all data, '
                    'part: sharding the dataset into nonoverlapping pieces and only cache one piece')
parser.add_argument('--resume', help='resume from checkpoint')
parser.add_argument('--accumulation-steps', type=int,
                    help="gradient accumulation steps")
parser.add_argument('--use-checkpoint', action='store_true',
                    help="whether to use gradient checkpointing to save memory")
parser.add_argument('--amp-opt-level', type=str, default='O1', choices=['O0', 'O1', 'O2'],
                    help='mixed precision opt level, if O0, no amp is used')
parser.add_argument('--tag', help='tag of experiment')
parser.add_argument('--eval', action='store_true',
                    help='Perform evaluation only')
parser.add_argument('--throughput', action='store_true',
                    help='Test throughput only')


# label and unlabel
parser.add_argument('--labeled_bs', type=int, default=12, help='labeled_batch_size per gpu')
parser.add_argument('--labelnum', type=int, default=3, help='labeled data')
parser.add_argument('--u_weight', type=float, default=0.5, help='weight of unlabeled pixels')
# costs
parser.add_argument('--gpu', type=str,  default='0', help='GPU to use')
parser.add_argument('--consistency', type=float, default=0.1, help='consistency')
parser.add_argument('--consistency_rampup', type=float, default=200.0, help='consistency_rampup')
parser.add_argument('--magnitude', type=float,  default='6.0', help='magnitude')
parser.add_argument('--s_param', type=int,  default=6, help='multinum of random masks')

args = parser.parse_args()
config = get_config(args)
dice_loss = losses.DiceLoss(n_classes=4)

class WaveletSplit(nn.Module):
    def __init__(self, wt_type='db1', channels=1):
        super().__init__()
        wt_flt, _ = create_2d_wavelet_filter(wt_type, channels, channels, torch.float)
        self.register_buffer('wt_filter', wt_flt)

    def forward(self, x):
        wt = wavelet_2d_transform(x, self.wt_filter)  # [B, C, 4, H/2, W/2]
        low = wt[:, :, 0, :, :]  # LL
        high = wt[:, :, 1:, :, :].reshape(x.size(0), -1, wt.size(-2), wt.size(-1))  # LH+HL+HH
        return low, high
        
def load_net(net, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])

def load_net_opt(net, optimizer, path):
    state = torch.load(str(path))
    net.load_state_dict(state['net'])
    optimizer.load_state_dict(state['opt'])

def save_net_opt(net, optimizer, path):
    state = {
        'net':net.state_dict(),
        'opt':optimizer.state_dict(),
    }
    torch.save(state, str(path))

def get_ACDC_LargestCC(segmentation):
    class_list = []
    for i in range(1, 4):
        temp_prob = segmentation == i * torch.ones_like(segmentation)
        temp_prob = temp_prob.detach().cpu().numpy()
        labels = label(temp_prob)
        # -- with 'try'
        assert(labels.max() != 0)  # assume at least 1 CC
        largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
        class_list.append(largestCC * i)
    acdc_largestCC = class_list[0] + class_list[1] + class_list[2]
    return torch.from_numpy(acdc_largestCC).cuda()

def get_ACDC_2DLargestCC(segmentation):
    batch_list = []
    N = segmentation.shape[0]
    for i in range(0, N):
        class_list = []
        for c in range(1, 4):
            temp_seg = segmentation[i] #== c *  torch.ones_like(segmentation[i])
            temp_prob = torch.zeros_like(temp_seg)
            temp_prob[temp_seg == c] = 1
            temp_prob = temp_prob.detach().cpu().numpy()
            labels = label(temp_prob)          
            if labels.max() != 0:
                largestCC = labels == np.argmax(np.bincount(labels.flat)[1:])+1
                class_list.append(largestCC * c)
            else:
                class_list.append(temp_prob)
        
        n_batch = class_list[0] + class_list[1] + class_list[2]
        batch_list.append(n_batch)

    return torch.Tensor(batch_list).cuda()
    

def get_ACDC_masks(output, nms=0):
    probs = F.softmax(output, dim=1)
    _, probs = torch.max(probs, dim=1)
    if nms == 1:
        probs = get_ACDC_2DLargestCC(probs)      
    return probs

def get_current_consistency_weight(epoch):
    # Consistency ramp-up from https://arxiv.org/abs/1610.02242
    return 5* args.consistency * ramps.sigmoid_rampup(epoch, args.consistency_rampup)

def update_model_ema(model, ema_model, alpha):
    model_state = model.state_dict()
    model_ema_state = ema_model.state_dict()
    new_dict = {}
    for key in model_state:
        new_dict[key] = alpha * model_ema_state[key] + (1 - alpha) * model_state[key]
    ema_model.load_state_dict(new_dict)

def generate_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_x, patch_y = int(img_x*2/3), int(img_y*2/3)
    w = np.random.randint(0, img_x - patch_x)
    h = np.random.randint(0, img_y - patch_y)
    mask[w:w+patch_x, h:h+patch_y] = 0
    loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def random_mask(img, shrink_param=3):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    x_split, y_split = int(img_x / shrink_param), int(img_y / shrink_param)
    patch_x, patch_y = int(img_x*2/(3*shrink_param)), int(img_y*2/(3*shrink_param))
    mask = torch.ones(img_x, img_y).cuda()
    for x_s in range(shrink_param):
        for y_s in range(shrink_param):
            w = np.random.randint(x_s*x_split, (x_s+1)*x_split-patch_x)
            h = np.random.randint(y_s*y_split, (y_s+1)*y_split-patch_y)
            mask[w:w+patch_x, h:h+patch_y] = 0
            loss_mask[:, w:w+patch_x, h:h+patch_y] = 0
    return mask.long(), loss_mask.long()

def contact_mask(img):
    batch_size, channel, img_x, img_y = img.shape[0], img.shape[1], img.shape[2], img.shape[3]
    loss_mask = torch.ones(batch_size, img_x, img_y).cuda()
    mask = torch.ones(img_x, img_y).cuda()
    patch_y = int(img_y *4/9)
    h = np.random.randint(0, img_y-patch_y)
    mask[h:h+patch_y, :] = 0
    loss_mask[:, h:h+patch_y, :] = 0
    return mask.long(), loss_mask.long()


def mix_loss_origin(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')
    img_l, patch_l = img_l.type(torch.int64), patch_l.type(torch.int64)
    output_soft = F.softmax(output, dim=1)
    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight
    patch_mask = 1 - mask
    loss_dice = dice_loss(output_soft, img_l.unsqueeze(1), mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l.unsqueeze(1), patch_mask.unsqueeze(1)) * patch_weight
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16) 
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)#loss = loss_ce
    return loss_dice, loss_ce

def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
    CE = nn.CrossEntropyLoss(reduction='none')

    # ✅ 保证标签是 [B, H, W]，否则 _one_hot_encoder 会出错
    if img_l.dim() == 4:
        img_l = img_l.squeeze(1)
    if patch_l.dim() == 4:
        patch_l = patch_l.squeeze(1)

    img_l = img_l.long()
    patch_l = patch_l.long()
    _, _, H, W = output.shape  # 获取网络输出的空间大小

    if img_l.shape[-2:] != (H, W):
        img_l = F.interpolate(img_l.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1).long()
    if patch_l.shape[-2:] != (H, W):
        patch_l = F.interpolate(patch_l.unsqueeze(1).float(), size=(H, W), mode='nearest').squeeze(1).long()
    output_soft = F.softmax(output, dim=1)

    image_weight, patch_weight = l_weight, u_weight
    if unlab:
        image_weight, patch_weight = u_weight, l_weight

    patch_mask = 1 - mask  # [B, H, W]

    # ✅ 不手动 one-hot，dice_loss 内部自动 one-hot
    loss_dice = dice_loss(output_soft, img_l, mask.unsqueeze(1)) * image_weight
    loss_dice += dice_loss(output_soft, patch_l, patch_mask.unsqueeze(1)) * patch_weight

    # ✅ CrossEntropyLoss 也支持 target = [B, H, W]
    loss_ce = image_weight * (CE(output, img_l) * mask).sum() / (mask.sum() + 1e-16)
    loss_ce += patch_weight * (CE(output, patch_l) * patch_mask).sum() / (patch_mask.sum() + 1e-16)

    return loss_dice, loss_ce


    
def patients_to_slices(dataset, patiens_num):
    ref_dict = None
    if "ACDC" in dataset:
        ref_dict = {"1": 32, "3": 68, "7": 136,
                    "14": 256, "21": 396, "28": 512, "35": 664, "70": 1312}
    elif "Prostate":
        ref_dict = {"2": 27, "4": 53, "8": 120,
                    "12": 179, "16": 256, "21": 312, "42": 623}
    else:
        print("Error")
    return ref_dict[str(patiens_num)]


def compute_prototypes(feats, preds, num_classes):
    """处理UNet的256维特征"""
    if isinstance(feats, tuple):  # 如果返回(encoder_feats, decoder_feats)
        feats = feats[1]  # 使用decoder最后一层特征
    
    # 确保是单一特征图且维度正确
    if len(feats.shape) == 4:  # [B, C, H, W]
        batch_size, c, h, w = feats.shape
        feats = feats.permute(0, 2, 3, 1).reshape(batch_size, -1, c)  # [B, H*W, C]
        preds = preds.view(batch_size, -1)  # [B, H*W]
    else:
        raise ValueError(f"Unexpected feature shape: {feats.shape}")
    
    prototypes = torch.zeros(batch_size, num_classes, c, device=feats.device)
    
    for b in range(batch_size):
        for cls in range(num_classes):
            mask = (preds[b] == cls)
            if mask.any():
                prototypes[b, cls] = feats[b][mask].mean(dim=0)  # 平均特征
    
    return F.normalize(prototypes, dim=-1)  # [B, num_classes, 256]

def get_reliability_weight(pred_prob, feats, prototypes):
    # """适配256维特征的权重计算"""
    B, C, H, W = pred_prob.shape
    
    # 处理特征输入
    if isinstance(feats, tuple):
        feats = feats[1]  # 使用decoder最后一层特征 [B, 256, H, W]
    
    feats = feats.view(B, -1, H*W)  # [B, 256, H*W]
    prototypes = prototypes.transpose(1, 2)  # [B, 256, num_classes]
    
    # 计算每个位置与各类原型的相似度
    similarity = F.cosine_similarity(
        feats.unsqueeze(3),  # [B, 256, H*W, 1]
        prototypes.unsqueeze(2),  # [B, 256, 1, num_classes]
        dim=1
    )  # [B, H*W, num_classes]
    
    max_similarity, _ = similarity.max(dim=2)  # [B, H*W]
    # 计算不确定性(熵)
    uncertainty=1-(-(pred_prob*torch.log(pred_prob+1e-6)).sum(1).view(B,-1))
    return (max_similarity * uncertainty).view(B, 1, H, W)

    
def pre_train(args, snapshot_path):
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.pre_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
     

    model = BCP_net(in_chns=1, class_num=num_classes)
    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path,
                            split="train",
                            num=None,
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")
    total_slices = len(db_train)
    labeled_slice = patients_to_slices(args.root_path,args.labelnum)
    print("Total slices is: {}, labeled slices is:{}".format(total_slices, labeled_slice))
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, total_slices))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size-args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)

    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    optimizer = optim.SGD(model.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start pre_training")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    best_hd = 100
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()

            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            img_mask, loss_mask = generate_mask(img_a)
            gt_mixl = lab_a * img_mask + lab_b * (1 - img_mask)

            #-- original
            net_input = img_a * img_mask + img_b * (1 - img_mask)
            out_mixl = model(net_input)
            loss_dice, loss_ce = mix_loss_origin(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

            loss = (loss_dice + loss_ce) / 2            

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('info/total_loss', loss, iter_num)
            writer.add_scalar('info/mix_dice', loss_dice, iter_num)
            writer.add_scalar('info/mix_ce', loss_ce, iter_num)     

            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(iter_num, loss, loss_dice, loss_ce))
                
            if iter_num % 20 == 0:
                image = net_input[1, 0:1, :, :]
                writer.add_image('pre_train/Mixed_Image', image, iter_num)
                outputs = torch.argmax(torch.softmax(out_mixl, dim=1), dim=1, keepdim=True)
                writer.add_image('pre_train/Mixed_Prediction', outputs[1, ...] * 50, iter_num)
                labs = gt_mixl[1, ...].unsqueeze(0) * 50
                writer.add_image('pre_train/Mixed_GroundTruth', labs, iter_num)

            if iter_num > 0 and iter_num % 200 == 0:
                model.eval()
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    metric_i = val_2d.test_single_volume(sampled_batch["image"], sampled_batch["label"], model, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    # save_mode_path = os.path.join(snapshot_path, 'iter_{}_dice_{}.pth'.format(iter_num, round(best_performance, 4)))
                    save_best_path = os.path.join(snapshot_path,'{}_best_model.pth'.format(args.model))
                    # save_net_opt(model, optimizer, save_mode_path)
                    save_net_opt(model, optimizer, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                model.train()

            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break
    writer.close()

def self_train(args, pre_snapshot_path, snapshot_path):
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations

    labeled_sub_bs = int(args.labeled_bs / 2)
    unlabeled_sub_bs = int((args.batch_size - args.labeled_bs) / 2)

    # 两个模型：低频处理 & 高频处理
    low_net = BCP_net(in_chns=1, class_num=num_classes)
    high_net = BCP_net(in_chns=3, class_num=num_classes)

    wavelet_split = WaveletSplit(channels=1).cuda()

    optimizer = optim.SGD(list(low_net.parameters()) + list(high_net.parameters()),
                          lr=base_lr, momentum=0.9, weight_decay=0.0001)

    def worker_init_fn(worker_id):
        random.seed(args.seed + worker_id)

    db_train = BaseDataSets(base_dir=args.root_path, split="train",
                            transform=transforms.Compose([RandomGenerator(args.patch_size)]))
    db_val = BaseDataSets(base_dir=args.root_path, split="val")

    labeled_slice = patients_to_slices(args.root_path, args.labelnum)
    labeled_idxs = list(range(0, labeled_slice))
    unlabeled_idxs = list(range(labeled_slice, len(db_train)))
    batch_sampler = TwoStreamBatchSampler(labeled_idxs, unlabeled_idxs, args.batch_size, args.batch_size - args.labeled_bs)

    trainloader = DataLoader(db_train, batch_sampler=batch_sampler, num_workers=4, pin_memory=True, worker_init_fn=worker_init_fn)
    valloader = DataLoader(db_val, batch_size=1, shuffle=False, num_workers=1)

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self-training with dual networks (low/high freq)")
    logging.info(f"{len(trainloader)} iterations per epoch")

    low_net.train()
    high_net.train()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'].cuda(), sampled_batch['label'].cuda()

            # 分解为低频 & 高频
            with torch.no_grad():
                low_img, high_img = wavelet_split(volume_batch)

            low_lab = label_batch[:args.labeled_bs]
            high_lab = label_batch[:args.labeled_bs]

            # 数据划分
            low_img_a, low_img_b = low_img[:labeled_sub_bs], low_img[labeled_sub_bs:args.labeled_bs]
            high_img_a, high_img_b = high_img[:labeled_sub_bs], high_img[labeled_sub_bs:args.labeled_bs]
            u_low_a = low_img[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            u_low_b = low_img[args.labeled_bs + unlabeled_sub_bs:]
            u_high_a = high_img[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs]
            u_high_b = high_img[args.labeled_bs + unlabeled_sub_bs:]

            lab_a, lab_b = low_lab[:labeled_sub_bs], low_lab[labeled_sub_bs:args.labeled_bs]

            img_mask, loss_mask = generate_mask(low_img_a)
            # === 有标签监督 ===
            mixed_low = u_low_a * img_mask + low_img_a * (1 - img_mask)
            mixed_high = u_high_a * img_mask + high_img_a * (1 - img_mask)

            out_low = low_net(mixed_low)
            out_high = high_net(mixed_high)

            plab_a = torch.argmax(out_high.detach(), dim=1)
            plab_b = torch.argmax(out_low.detach(), dim=1)
            unl_dice, unl_ce = mix_loss(out_low, plab_a, lab_a, loss_mask, u_weight=args.u_weight, unlab=True)
            l_dice, l_ce = mix_loss(out_high, lab_b, plab_b, loss_mask, u_weight=args.u_weight)

            consistency_weight = get_current_consistency_weight(iter_num // 150)
            soft_low = F.softmax(out_low, dim=1)
            soft_high = F.softmax(out_high, dim=1)
            consistency_loss = F.mse_loss(soft_low, soft_high.detach()) + F.mse_loss(soft_high, soft_low.detach())
            total_loss = (unl_dice + unl_ce + l_dice + l_ce) / 2 + consistency_loss * consistency_weight

            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step()

            iter_num += 1

            writer.add_scalar('loss/total', total_loss.item(), iter_num)
            writer.add_scalar('loss/dice', (unl_dice + l_dice).item(), iter_num)
            writer.add_scalar('loss/ce', (unl_ce + l_ce).item(), iter_num)
            writer.add_scalar('loss/consistency', consistency_loss.item(), iter_num)
            
            # 修复日志记录：使用实际计算的损失变量
            logging.info('iteration %d: loss: %f, mix_dice: %f, mix_ce: %f'%(
                iter_num, total_loss.item(), (unl_dice + l_dice).item(), (unl_ce + l_ce).item()))
            if iter_num > 0 and iter_num % 200 == 0:
                # 切换到评估模式
                low_net.eval()
                high_net.eval()
                
                metric_list = 0.0
                for _, sampled_batch in enumerate(valloader):
                    # 使用低频网络进行验证
                    metric_i = val_2d.test_single_volume(
                        sampled_batch["image"], sampled_batch["label"], low_net, classes=num_classes)
                    metric_list += np.array(metric_i)
                metric_list = metric_list / len(db_val)
                for class_i in range(num_classes-1):
                    writer.add_scalar('info/val_{}_dice'.format(class_i+1), metric_list[class_i, 0], iter_num)
                    writer.add_scalar('info/val_{}_hd95'.format(class_i+1), metric_list[class_i, 1], iter_num)

                performance = np.mean(metric_list, axis=0)[0]
                writer.add_scalar('info/val_mean_dice', performance, iter_num)

                if performance > best_performance:
                    best_performance = performance
                    save_best_path = os.path.join(snapshot_path, '{}_best_model.pth'.format(args.model))
                    # 保存两个网络的状态字典
                    torch.save({
                        'low_net_state_dict': low_net.state_dict(),
                        'high_net_state_dict': high_net.state_dict()
                    }, save_best_path)

                logging.info('iteration %d : mean_dice : %f' % (iter_num, performance))
                
                # 切换回训练模式
                low_net.train()
                high_net.train()
                
            if iter_num >= max_iterations:
                break
        if iter_num >= max_iterations:
            iterator.close()
            break

    writer.close()


if __name__ == "__main__":
    if args.deterministic:
        cudnn.benchmark = False
        cudnn.deterministic = True
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)

    # -- path to save models
    pre_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/pre_train".format(args.exp, args.labelnum)
    self_snapshot_path = "./model/BCP/ACDC_{}_{}_labeled/self_train".format(args.exp, args.labelnum)
    for snapshot_path in [pre_snapshot_path, self_snapshot_path]:
        if not os.path.exists(snapshot_path):
            os.makedirs(snapshot_path)
    shutil.copy('../code/ACDC_BCP_train.py', self_snapshot_path)

    #Pre_train
    logging.basicConfig(filename=pre_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    pre_train(args, pre_snapshot_path)

    #Self_train
    logging.basicConfig(filename=self_snapshot_path+"/log.txt", level=logging.INFO, format='[%(asctime)s.%(msecs)03d] %(message)s', datefmt='%H:%M:%S')
    logging.getLogger().addHandler(logging.StreamHandler(sys.stdout))
    logging.info(str(args))
    self_train(args, pre_snapshot_path, self_snapshot_path)

    


