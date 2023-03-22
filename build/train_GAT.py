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
        'build/checkpoint/', 'log.txt'), title='facedb')
    logger.set_names(['Epoch', 'LR', 'Train Loss',
                     'Valid Loss', 'Train Acc', 'Val Acc', 'AUC'])

    # data config
    trainConfig = AlignConfig(database_name='facedb', mode='train')
    valConfig = AlignConfig(database_name='facedb', mode='test')

    # dataloader
    trainloader, trainset = get_dataloader(batch_size=8, data_config=trainConfig)
    valloader, valset = get_dataloader(batch_size=4, data_config=valConfig)

    # model config
    modelConfig = ModelConfig(dataset_name='facedb', load_model_url=False)
    modelConfig.dataset = trainset.database

    # model
    processor = SPIGAFramework(modelConfig)
    processor.multiprocessing()
    processor.train(visual_cnn=False, pose_fc=False, gcn=True)

    # criterion
    criterion = torch.nn.SmoothL1Loss().cuda()

    # optimizer
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, processor.model.parameters()), lr=1e-4)
    scheduler = torch.optim.lr_scheduler.MultiStepLR(
        optimizer, milestones=[len(trainloader) * 100], gamma=0.1)
    
    # epoch
    lr, train_loss, valid_loss, train_acc, valid_acc, valid_auc = 0, 0, 0, 0, 0, 0
    for epoch in range(150):
        lr = optimizer.param_groups[0]['lr']
        train_loss, train_acc = train(trainloader, processor, criterion, optimizer, scheduler)
        
        logger.append([int(epoch + 1), lr, train_loss, valid_loss, train_acc, valid_acc, valid_auc])
        if epoch != 0 and (epoch+1) % 30 == 0:
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


def train(loader, processor: SPIGAFramework, criterion, optimizer, scheduler, debug=True, flip=False):
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

        # target to torch.cuda
        target_landmarks = processor._data2device(torch.from_numpy(landmarks))

        # output
        features, outputs = processor.pred(torch.from_numpy(image), torch.from_numpy(bbox))

        loss = 0
        # Smooth L1 function for GAT layers.
        for idx, lnds in enumerate(outputs['Landmarks']):
            lnds = lnds * 64
            loss += criterion(lnds, target_landmarks)
            
        # Calculate acc (sum of nme for this batch)
        acc, batch_dists = fan_NME(lnds.cpu().detach(), target_landmarks.cpu(), num_landmarks=68)

        # update processor
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        scheduler.step()
        
        # Debug
        if debug  & (i == 1):
            lnds = outputs['Landmarks'][-1]
            for k, (img, lnd) in enumerate(zip(image, lnds.cpu().detach().numpy())):
                for num in range(68):
                    x = int(lnd[num,0]*4*64) 
                    y = int(lnd[num,1]*4*64) 
                    img = cv2.circle(img,(x,y),2,(255,0,0),cv2.FILLED,cv2.LINE_4)
                cv2.imwrite(f'build/checkpoint/GAT/train_{k}.png', img)
                
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


def validate(loader, processor: SPIGAFramework, criterion, debug=False, flip=False):
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    acces = AverageMeter()
    end = time.time()

    processor.model.eval()
    gc.collect()
    torch.cuda.empty_cache()
    
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
        features, outputs = processor.pred(torch.from_numpy(image), torch.from_numpy(bbox))

        loss = 0
        # Smooth L1 function for GAT layers.
        for idx, lnds in enumerate(outputs['Landmarks']):
            loss += criterion(lnds, target_landmarks)
            
        # Calculate acc (sum of nme for this batch)
        acc, batch_dists = fan_NME(lnds.cpu().detach(), target_landmarks.cpu(), num_landmarks=68)
        
        # Debug
        if debug  & (i == 1):
            lnds = outputs['Landmarks'][-1]
            for k, (img, lnd) in enumerate(zip(image, lnds.cpu().detach().numpy())):
                for num in range(68):
                    x = int(lnd[num,0])*4
                    y = int(lnd[num,1])*4
                    img = cv2.circle(img,(x,y),2,(255,0,0),cv2.FILLED,cv2.LINE_4)
                cv2.imwrite(f'build/checkpoint/GAT/train_{k}.png', img)
                
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
    print("=> Mean Error: {:.2f}, AUC@0.07: {} based on maps".format(mean_error*100., auc))

    return losses.avg, acces.avg, auc, all_dists

if __name__ == '__main__':
    main()
