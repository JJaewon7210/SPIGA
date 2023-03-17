'''
Once the model is trained, we freeze the backbone to train the GAT regressor.
'''

from utils.logger import Logger, savefig
from utils.loss import AdaptiveWingLoss, get_preds_fromhm, fan_NME
from utils.evaluation import accuracy, AverageMeter, calc_metrics, calc_dists
from utils.misc import save_checkpoint
import sys
sys.path.insert(0, '../SPIGA')

from data.loaders.dl_config import AlignConfig
from data.loaders.dataloader import get_dataloader
from inference.framework import SPIGAFramework
from inference.config import ModelConfig
import gc
import torch
import numpy as np
import openpyxl
import cv2
import matplotlib.pyplot as plt
import matplotlib
from progress.bar import Bar
import time
import os

import warnings
warnings.filterwarnings('ignore')

matplotlib.use('TkAgg')

def main():
    global best_acc
    global best_auc
    best_acc = 0.
    best_auc = 0.

    # set checkpoint path
    if not os.path.exists('build/checkpoint/GAT/'):
        os.makedirs('build/checkpoint/GAT/')

    # set logger
    logger = Logger(os.path.join(
        'build/checkpoint/', 'log.txt'), title='sftl54')
    logger.set_names(['Epoch', 'LR', 'Train Loss',
                     'Valid Loss', 'Train Acc', 'Val Acc', 'AUC'])

    # data config
    trainConfig = AlignConfig(database_name='sftl54', mode='train')
    valConfig = AlignConfig(database_name='sftl54', mode='test')

    # dataloader
    trainloader, trainset = get_dataloader(batch_size=8, data_config=trainConfig)
    valloader, valset = get_dataloader(batch_size=1, data_config=valConfig)

    # model config
    modelConfig = ModelConfig(dataset_name='sftl54', load_model_url=False)
    modelConfig.dataset = trainset.database

    # model
    processor = SPIGAFramework(modelConfig)
    processor.train(visual_cnn=False, pose_fc=False, gcn=True)

    # multi processing
    processor.multiprocessing()

    # criterion
    criterion = [torch.nn.SmoothL1Loss().cuda(), AdaptiveWingLoss().cuda()]

    # optimizer
    optimizer = torch.optim.Adam(processor.params_to_update, lr=1e-4)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, T_0 = len(trainloader) * 10, T_mult = 2, eta_min = 1e-6)

    # epoch
    lr, train_loss, valid_loss, train_acc, valid_acc, valid_auc = 0, 0, 0, 0, 0, 0
    for epoch in range(150):
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train(trainloader, processor, criterion, optimizer, scheduler)
        
        logger.append([int(epoch + 1), lr, train_loss, valid_loss, train_acc, valid_acc, valid_auc])
        if epoch != 0 and (epoch+1) % 10 == 0:
            with torch.no_grad():
                valid_loss, valid_acc, valid_auc, all_accs = validate(valloader, processor, criterion)


            is_best = valid_auc >= best_auc
            best_auc = max(valid_auc, best_auc)
            save_checkpoint(
                state={
                    'epoch': epoch+1,
                    'state_dict': processor.model.state_dict(),
                    'best_acc': best_auc,
                    'optimizer': optimizer.state_dict(),
                },
                is_best=is_best,
                checkpoint='build/checkpoint/GAT/',
                snapshot=True
            )


def train(loader, processor: SPIGAFramework, criterion, optimizer, scheduler, debug=False, flip=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()

    processor.train(visual_cnn=False, pose_fc=False, gcn=True)
    gc.collect()
    torch.cuda.empty_cache()
    end = time.time()

    bar = Bar('Training', max=len(loader))
    for i, (sample) in enumerate(loader):
        data_time.update(time.time() - end)

        image = sample['image'].numpy()
        landmarks = sample['landmarks'].numpy()
        bbox = sample['bbox'].numpy()

        # batch_size
        batch_size = np.shape(image)[0]

        # Show Image
        # for img,lnd in zip(image,landmarks):
        #     img = img.numpy()
        #     lnd = lnd.numpy()
        #     for num in range(68):
        #         x = int(lnd[num,0])
        #         y = int(lnd[num,1])
        #         img = cv2.circle(img,(x,y),2,(255,0,0),cv2.FILLED,cv2.LINE_4)
        #     cv2.imshow('img',img)
        #     cv2.waitKey(0)

        #  target to torch.cuda
        target_landmarks = processor._data2device(torch.from_numpy(landmarks))

        # output
        features, outputs = processor.pred(image, bbox)

        loss = 0
            
        # Smooth L1 function for GAT layers.
        for idx, hmap in enumerate(outputs['HeatmapPreds']):
            lnds = get_preds_fromhm(hmap.cpu())
            lnds = tuple(lnd_cpu.to("cuda:0") for lnd_cpu in lnds)
            lnds = torch.stack(lnds)
            loss += 2**(idx)*criterion[0](lnds, target_landmarks) * 4
            
        # Calculate acc (sum of nme for this batch)
        acc, _ = fan_NME(hmap.cpu(), target_landmarks.cpu(), num_landmarks=68)

        # update processor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()

        # update history
        losses.update(loss.item(), batch_size)
        acces.update(acc/batch_size, batch_size)

        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f} | LR: {lr: .7f}'.format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg,
            lr=scheduler.get_lr()[0])
        bar.next()
    bar.finish()

    return losses.avg, acces.avg


def validate(loader, processor: SPIGAFramework, criterion, debug=True, flip=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    end = time.time()

    processor.model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    if debug: 
        random_int = np.random.randint(low=1, high=len(loader)-1)
    
    bar = Bar('Validating', max=len(loader))
    all_dists = torch.zeros((68, loader.dataset.__len__()))
    for i, sample in enumerate(loader):
        data_time.update(time.time() - end)

        image = sample['image'].numpy()
        landmarks = sample['landmarks'].numpy()
        bbox = sample['bbox'].numpy()

        # batch_size
        batch_size = np.shape(image)[0]

        #  target to torch.cuda
        target_landmarks = processor._data2device(torch.from_numpy(landmarks))

        # output
        features, outputs = processor.pred(image, bbox)

        loss = 0
        # Awing losses applied to the point and edges heatmaps. weight = 50
        for idx, hmap in enumerate(outputs['HeatmapPoints']):
            loss += 2**(idx)*criterion[1](hmap, target_heatmap2D) * 50
        for idx, hmap in enumerate(outputs['HeatmapEdges']):
            loss += 2**(idx)*criterion[1](hmap, target_boundaries) * 50

        # Smooth L1 function computed between the annotated and predicted landmarks coordinates. weight = 4
        for idx, hmap in enumerate(outputs['HeatmapPreds']):
            lnds = get_preds_fromhm(hmap.cpu())
            loss += 2**(idx)*criterion[0](lnds, target_landmarks) * 4

        # Calculate acc (sum of nme for this batch)
        acc, batch_dists = fan_NME(hmap.cpu(), target_landmarks.cpu(), num_landmarks=68)
        
        # Debug
        if debug & (random_int == i):
            for k, img, hmap in enumerate(zip(image, outputs['HeatmapPreds'])):
                lnds, _ = get_preds_fromhm(hmap.cpu())
                lnds = lnds.numpy()
                for num in range(68):
                    x = int(lnds[num, 0])
                    y = int(lnds[num, 1])
                    img = cv2.circle(img, (x, y), 2, (255, 0, 0),
                                     cv2.FILLED, cv2.LINE_4)
                cv2.imwrite(f'build/checkpoint/GAT/{k}.png', img)
                
        # update history
        all_dists[:, i * batch_size:(i + 1) * batch_size] = batch_dists
        losses.update(loss.item(), batch_size)
        acces.update(acc/batch_size, batch_size)
        
        batch_time.update(time.time() - end)
        end = time.time()
        bar.suffix = '({batch}/{size}) Data: {data:.6f}s | Batch: {bt:.3f}s | Total: {total:} | ETA: {eta:} | Loss: {loss:.4f} | Acc: {acc: .4f}'.format(
            batch=i + 1,
            size=len(loader),
            data=data_time.val,
            bt=batch_time.val,
            total=bar.elapsed_td,
            eta=bar.eta_td,
            loss=losses.avg,
            acc=acces.avg)
        bar.next()
    bar.finish()
    mean_error = torch.mean(all_dists)

    # This is auc of predicted maps and target.
    auc = calc_metrics(all_dists, path='build/checkpoint/GAT/')
    print(
        "=> Mean Error: {:.2f}, AUC@0.07: {} based on maps".format(mean_error*100., auc))

    return losses.avg, acces.avg, auc, all_dists


if __name__ == '__main__':
    main()
