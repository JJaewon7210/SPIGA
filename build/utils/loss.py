import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import math

def ldmk_loss(input, target, weight=None, size_average=True):
    n, c = input.size()

    loss_ = (input - target) ** 2
    iod = torch.sqrt(torch.sum(
        (target[:, 36*2:37*2] - target[:, 45*2:46*2])**2, 1))
    loss = torch.autograd.Variable(torch.zeros((n, c//2))).float().cuda()
    for i in range(c//2):
        loss[:, i] = torch.sqrt((loss_[:, i*2] + loss_[:, i*2+1])) / (iod+1e-6)

    if size_average:
        loss = torch.mean(loss)
    return loss


def transform(point, center, scale, resolution, rotation=0, invert=False):
    _pt = np.ones(3)
    _pt[0] = point[0]
    _pt[1] = point[1]

    h = 200.0 * scale
    t = np.eye(3)
    t[0, 0] = resolution / h
    t[1, 1] = resolution / h
    t[0, 2] = resolution * (-center[0] / h + 0.5)
    t[1, 2] = resolution * (-center[1] / h + 0.5)

    if rotation != 0:
        rotation = -rotation
        r = np.eye(3)
        ang = rotation * math.pi / 180.0
        s = math.sin(ang)
        c = math.cos(ang)
        r[0][0] = c
        r[0][1] = -s
        r[1][0] = s
        r[1][1] = c

        t_ = np.eye(3)
        t_[0][2] = -resolution / 2.0
        t_[1][2] = -resolution / 2.0
        t_inv = torch.eye(3)
        t_inv[0][2] = resolution / 2.0
        t_inv[1][2] = resolution / 2.0
        t = reduce(np.matmul, [t_inv, r, t_, t])

    if invert:
        t = np.linalg.inv(t)
    new_point = (np.matmul(t, _pt))[0:2]

    return new_point.astype(int)


def get_preds_fromhm(hm, center=None, scale=None, rot=None):
    max, idx = torch.max(
        hm.view(hm.size(0), hm.size(1), hm.size(2) * hm.size(3)), 2)
    idx += 1
    preds = idx.view(idx.size(0), idx.size(1), 1).repeat(1, 1, 2).float()
    preds[..., 0].apply_(lambda x: (x - 1) % hm.size(3) + 1)
    preds[..., 1].add_(-1).div_(hm.size(2)).floor_().add_(1)

    for i in range(preds.size(0)):
        for j in range(preds.size(1)):
            hm_ = hm[i, j, :]
            pX, pY = int(preds[i, j, 0]) - 1, int(preds[i, j, 1]) - 1
            if pX > 0 and pX < 63 and pY > 0 and pY < 63:
                diff = torch.FloatTensor(
                    [hm_[pY, pX + 1] - hm_[pY, pX - 1],
                     hm_[pY + 1, pX] - hm_[pY - 1, pX]])
                preds[i, j].add_(diff.sign_().mul_(.25))

    preds.add_(-0.5)

    preds_orig = torch.zeros(preds.size())
    if center is not None and scale is not None:
        for i in range(hm.size(0)):
            for j in range(hm.size(1)):
                preds_orig[i, j] = transform(
                    preds[i, j], center, scale, hm.size(2), rot, True)

    return preds, preds_orig


def fan_NME(pred_heatmaps, gt_landmarks, num_landmarks=68):
    '''
       Calculate total NME for a batch of data
       Args:
           pred_heatmaps: torch tensor of size [batch, points, height, width]
           gt_landmarks: torch tesnsor of size [batch, points, x, y]
       Returns:
           nme: sum of nme for this batch
    '''
    nme = 0
    pred_landmarks, _ = get_preds_fromhm(pred_heatmaps)
    pred_landmarks = pred_landmarks.numpy()
    gt_landmarks = gt_landmarks.numpy()
    for i in range(pred_landmarks.shape[0]):
        pred_landmark = pred_landmarks[i] * 4.0
        gt_landmark = gt_landmarks[i]

        if num_landmarks == 68:
            left_eye = np.average(gt_landmark[36:42], axis=0)
            right_eye = np.average(gt_landmark[42:48], axis=0)
            norm_factor = np.linalg.norm(left_eye - right_eye)
            # norm_factor = np.linalg.norm(gt_landmark[36]- gt_landmark[45])
        elif num_landmarks == 98:
            norm_factor = np.linalg.norm(gt_landmark[60] - gt_landmark[72])
        elif num_landmarks == 19:
            left, top = gt_landmark[-2, :]
            right, bottom = gt_landmark[-1, :]
            norm_factor = math.sqrt(abs(right - left)*abs(top-bottom))
            gt_landmark = gt_landmark[:-2, :]
        elif num_landmarks == 29:
            # norm_factor = np.linalg.norm(gt_landmark[8]- gt_landmark[9])
            norm_factor = np.linalg.norm(gt_landmark[16] - gt_landmark[17])
        nme += (np.sum(np.linalg.norm(pred_landmark - gt_landmark,
                axis=1)) / pred_landmark.shape[0]) / norm_factor
    return nme

class AdaptiveWingLoss(nn.Module):
    def __init__(self, omega=14, theta=0.5, epsilon=1, alpha=2.1):
        super(AdaptiveWingLoss, self).__init__()
        self.omega = omega
        self.theta = theta
        self.epsilon = epsilon
        self.alpha = alpha

    def forward(self, pred, target):
        '''
        :param pred: BxNxHxH (Batch, LandmarkNum, Image, heatMapSize, heatMapSize)
        :param target: BxNxHxH (Batch, LandmarkNum, Image, heatMapSize, heatMapSize)
        :return:
        '''
        y = target
        y_hat = pred
        delta_y = (y - y_hat).abs()
        delta_y1 = delta_y[delta_y < self.theta]
        delta_y2 = delta_y[delta_y >= self.theta]
        y1 = y[delta_y < self.theta]
        y2 = y[delta_y >= self.theta]
        loss1 = self.omega * \
            torch.log(1 + torch.pow(delta_y1 / self.omega, self.alpha - y1))
        A = self.omega * (1 / (1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))) * (self.alpha - y2) * (
            torch.pow(self.theta / self.epsilon, self.alpha - y2 - 1)) * (1 / self.epsilon)
        C = self.theta * A - self.omega * \
            torch.log(1 + torch.pow(self.theta / self.epsilon, self.alpha - y2))
        loss2 = A * delta_y2 - C
        return (loss1.sum() + loss2.sum()) / (len(loss1) + len(loss2))


if __name__ == "__main__":
    pred = torch.zeros((4,136))
    gt = torch.ones((4,136))
    vpred = torch.autograd.Variable(pred, requires_grad=True).float().cuda()
    vgt = torch.autograd.Variable(gt).float().cuda()
    print(ldmk_loss(vpred, vgt))
