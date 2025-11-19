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
from networks.net_factory import BCP_net, net_factory, highnet,lownet
from utils import losses, ramps, feature_memory, contrastive_losses, val_2d
parser = argparse.ArgumentParser()
parser.add_argument('--root_path', type=str, default='./data/ACDC', help='Name of Experiment')
parser.add_argument('--exp', type=str, default='wavenet2', help='experiment_name')
parser.add_argument('--model', type=str, default='unet', help='model_name')
parser.add_argument('--pre_iterations', type=int, default=0, help='maximum epoch number to train')
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
    return torch.from_numpy(np.array(batch_list)).float().cuda()

    

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


def mix_loss(output, img_l, patch_l, mask, l_weight=1.0, u_weight=0.5, unlab=False):
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
            loss_dice, loss_ce = mix_loss(out_mixl, lab_a, lab_b, loss_mask, u_weight=1.0, unlab=True)

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
    base_lr = args.base_lr
    num_classes = args.num_classes
    max_iterations = args.max_iterations
    os.environ['CUDA_VISIBLE_DEVICES'] = args.gpu
    # pre_trained_model = os.path.join(pre_snapshot_path,'{}_best_model.pth'.format(args.model))
    labeled_sub_bs, unlabeled_sub_bs = int(args.labeled_bs/2), int((args.batch_size-args.labeled_bs) / 2)
    
    # 创建两个网络：高频网络和低频网络 ✅ 使用正确的网络
    model_high = highnet(in_chns=1, class_num=num_classes)
    model_low = lownet(in_chns=1, class_num=num_classes)
    
    # 为两个网络分别创建优化器
    optimizer_high = optim.SGD(model_high.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)
    optimizer_low = optim.SGD(model_low.parameters(), lr=base_lr, momentum=0.9, weight_decay=0.0001)

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

    writer = SummaryWriter(snapshot_path + '/log')
    logging.info("Start self_training with dual networks")
    logging.info("{} iterations per epoch".format(len(trainloader)))

    model_high.train()
    model_low.train()

    ce_loss = CrossEntropyLoss()

    iter_num = 0
    max_epoch = max_iterations // len(trainloader) + 1
    best_performance = 0.0
    iterator = tqdm(range(max_epoch), ncols=70)
    
    for _ in iterator:
        for _, sampled_batch in enumerate(trainloader):
            volume_batch, label_batch = sampled_batch['image'], sampled_batch['label']
            volume_batch, label_batch = volume_batch.cuda(), label_batch.cuda()
            
            # 确保标签是Long类型
            # label_batch = label_batch.long()

            # 数据划分
            img_a, img_b = volume_batch[:labeled_sub_bs], volume_batch[labeled_sub_bs:args.labeled_bs]
            uimg_a, uimg_b = volume_batch[args.labeled_bs:args.labeled_bs + unlabeled_sub_bs], volume_batch[args.labeled_bs + unlabeled_sub_bs:]
            lab_a, lab_b = label_batch[:labeled_sub_bs], label_batch[labeled_sub_bs:args.labeled_bs]
            
            # 生成掩码
            umask, uloss_mask = generate_mask(uimg_a)  # 或 random_mask(...)
            u_net_input = uimg_a * umask + uimg_b * (1 - umask)
            
            with torch.no_grad():
                # 由两网分别对 uimg_a/uimg_b 产伪标签
                pre_a_high = model_high(uimg_a); pre_b_high = model_high(uimg_b)
                pre_a_low  = model_low(uimg_a);  pre_b_low  = model_low(uimg_b)
                plab_a_high = get_ACDC_masks(pre_a_high, nms=1)
                plab_b_high = get_ACDC_masks(pre_b_high, nms=1)
                plab_a_low  = get_ACDC_masks(pre_a_low,  nms=1)
                plab_b_low  = get_ACDC_masks(pre_b_low,  nms=1)
            
                # 与 u_net_input 的混合一致，拼成“混合伪标签”
                pseudo_for_high = plab_a_low  * umask + plab_b_low  * (1 - umask)  # 低频网 -> 给高频网用
                pseudo_for_low  = plab_a_high * umask + plab_b_high * (1 - umask)  # 高频网 -> 给低频网用
            
            # 2) 监督损失：直接用有标签样本，别绕混合
            out_l_high = model_high(img_a)
            out_l_low  = model_low(img_a)
            sup_dice_h, sup_ce_h = mix_loss(out_l_high, lab_a, lab_a, torch.ones_like(uloss_mask), u_weight=0, unlab=False)
            sup_dice_l, sup_ce_l = mix_loss(out_l_low,  lab_a, lab_a, torch.ones_like(uloss_mask), u_weight=0, unlab=False)
            sup_loss_high = 0.5 * (sup_dice_h + sup_ce_h)
            sup_loss_low  = 0.5 * (sup_dice_l + sup_ce_l)
            
            # 3) 无标签一致性（伪监督）损失：用对齐的混合输入与混合伪标签
            out_u_high = model_high(u_net_input)
            out_u_low  = model_low(u_net_input)
            unl_dice_h, unl_ce_h = mix_loss(out_u_high, pseudo_for_high, pseudo_for_high, uloss_mask, u_weight=args.u_weight, unlab=True)
            unl_dice_l, unl_ce_l = mix_loss(out_u_low,  pseudo_for_low,  pseudo_for_low,  uloss_mask, u_weight=args.u_weight, unlab=True)
            unsup_loss_high = 0.5 * (unl_dice_h + unl_ce_h)
            unsup_loss_low  = 0.5 * (unl_dice_l + unl_ce_l)
            
            # 4) 两网输出一致性：一定基于同一个输入（u_net_input）
            consistency_loss = F.mse_loss(F.softmax(out_u_high, dim=1), F.softmax(out_u_low, dim=1))
            cw = get_current_consistency_weight(iter_num // 150)
            
            total_loss_high = sup_loss_high + unsup_loss_high + 0.5 * cw * consistency_loss
            total_loss_low  = sup_loss_low  + unsup_loss_low  + 0.5 * cw * consistency_loss

            # 反向传播和优化
            optimizer_high.zero_grad()
            optimizer_low.zero_grad()
            
            total_loss = total_loss_high + total_loss_low
            total_loss.backward()
            
            optimizer_high.step()
            optimizer_low.step()


            iter_num += 1

            # 记录日志
            writer.add_scalar('loss/total_high', total_loss_high.item(), iter_num)
            writer.add_scalar('loss/total_low', total_loss_low.item(), iter_num)
            writer.add_scalar('loss/consistency', consistency_loss.item(), iter_num)

            if iter_num % 20 == 0:
                logging.info('iteration %d: high_loss: %.4f, low_loss: %.4f, consistency: %.4f' % 
                           (iter_num, total_loss_high.item(), total_loss_low.item(), consistency_loss.item()))

            # 验证
            if iter_num > 0 and iter_num % 200 == 0:
                model_high.eval()
                model_low.eval()
                
                metric_list_high = 0.0
                metric_list_low = 0.0
                
                for _, sampled_batch in enumerate(valloader):
                    image_val, label_val = sampled_batch["image"].cuda(), sampled_batch["label"].cuda()
                    label_val = label_val.long()
                    
                    with torch.no_grad():
                        # 高频网络验证
                        metric_i_high = val_2d.test_single_volume(image_val, label_val, model_high, classes=num_classes)
                        metric_list_high += np.array(metric_i_high)
                        
                        # 低频网络验证
                        metric_i_low = val_2d.test_single_volume(image_val, label_val, model_low, classes=num_classes)
                        metric_list_low += np.array(metric_i_low)
                
                metric_list_high = metric_list_high / len(db_val)
                metric_list_low = metric_list_low / len(db_val)
                
                performance_high = np.mean(metric_list_high, axis=0)[0]
                performance_low = np.mean(metric_list_low, axis=0)[0]
                
                # 保存最佳模型
                if performance_high > best_performance:
                    best_performance = performance_high
                    torch.save(model_high.state_dict(), os.path.join(snapshot_path, 'highnet_best_model.pth'))
                    torch.save(model_low.state_dict(), os.path.join(snapshot_path, 'lownet_best_model.pth'))

                logging.info('Iter %d: High_Dice: %.4f, Low_Dice: %.4f, Best: %.4f' % 
                           (iter_num, performance_high, performance_low, best_performance))
                
                model_high.train()
                model_low.train()

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

    


