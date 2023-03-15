import os
import copy
import torch
import numpy as np

# Paths
weights_path_dft = 'models/weights'

import inference.pretreatment as pretreat
from models.spiga import SPIGA
from inference.config import ModelConfig


class SPIGAFramework:

    def __init__(self, model_cfg: ModelConfig(), gpus=[0], load3DM=True):

        # Parameters
        self.model_cfg = model_cfg
        self.gpus = gpus

        # Pretreatment initialization
        self.transforms = pretreat.get_transformers(self.model_cfg)

        # SPIGA model
        self.model_inputs = ['image', "model3d", "cam_matrix"]
        self.model = SPIGA(num_landmarks=model_cfg.dataset.num_landmarks,
                           num_edges=model_cfg.dataset.num_edges)

        # Load weights and set model
        weights_path = self.model_cfg.model_weights_path
        if weights_path is None:
            weights_path = weights_path_dft

        if self.model_cfg.load_model_url:
            model_state_dict = torch.hub.load_state_dict_from_url(self.model_cfg.model_weights_url,
                                                                  model_dir=weights_path,
                                                                  file_name=self.model_cfg.model_weights)
        else:
            weights_file = os.path.join(weights_path, self.model_cfg.model_weights)
            model_state_dict = torch.load(weights_file)

        self.model.load_state_dict(model_state_dict)
        self.model = self.model.cuda(gpus[0])
        self.model.eval()
        print('SPIGA model loaded!')

        # Load 3D model and camera intrinsic matrix
        if load3DM:
            loader_3DM = pretreat.AddModel3D(model_cfg.dataset.ldm_ids,
                                             ftmap_size=model_cfg.ftmap_size,
                                             focal_ratio=model_cfg.focal_ratio,
                                             totensor=True)
            params_3DM = self._data2device(loader_3DM())
            self.model3d = params_3DM['model3d']
            self.cam_matrix = params_3DM['cam_matrix']
            
    def multiprocessing(self):
        self.model = torch.nn.DataParallel(self.model).cuda()
        self.model = self.model.module
        
        return
        
    def train(self, visual_cnn = True, pose_fc = True, gcn = True):
        
        # fine tunning the landmark detection model
        for child in self.model.visual_cnn.children():
            for param in child.parameters():
                param.requires_grad = visual_cnn

        # fine tunning the pose estimation model
        for child in self.model.visual_cnn.hgs_core.children():
            for param in child.parameters():
                param.requires_grad = pose_fc
        
        for child in self.model.pose_fc.children():
            for param in child.parameters():
                param.requires_grad = pose_fc
        
        # fine tunning the gcn model
        for child in self.model.gcn.children():
            for param in child.parameters():
                param.requires_grad = gcn
                
        params_to_update = self.model.parameters()
        params_to_update = []
        for name, param in self.model.named_parameters():
            if param.requires_grad == True:
                params_to_update.append(param)
                
        self.params_to_update = params_to_update
        
        return

    def inference(self, image, bboxes):
        """
        @param self:
        @param image: Raw image
        @param bboxes: List of bounding box founded on the image [[x,y,w,h],...]
        @return: features dict {'landmarks': list with shape (num_bbox, num_landmarks, 2) and x,y referred to image size
                                'headpose': list with shape (num_bbox, 6) euler->[:3], trl->[3:]
        """
        batch_crops, crop_bboxes = self.pretreat(image, bboxes)
        outputs = self.net_forward(batch_crops)
        features = self.postreatment(outputs, crop_bboxes, bboxes)
        return features
    
    def pred(self, image, bboxes):
        batch_crops, crop_bboxes = self.pretreat(image, bboxes)
        outputs = self.net_forward(batch_crops)
           
        features = {'landmarks': [], 'headpose': []}
        bboxes = self._data2device(torch.from_numpy(np.array(bboxes)))
        crop_bboxes = self._data2device(torch.from_numpy(np.array(crop_bboxes)))
        img_size = self._data2device(torch.Tensor(self.model_cfg.image_size))
        
        # Landmark outputs
        if 'Landmarks' in outputs.keys():
            for landmarks in outputs['Landmarks']:
                landmarks = landmarks.permute(1, 0, 2)
                landmarks = landmarks* img_size
                landmarks_norm = (landmarks - crop_bboxes[:, 0:2]) / crop_bboxes[:, 2:4]
                landmarks_out = (landmarks_norm * bboxes[:, 2:4]) + bboxes[:, 0:2]
                landmarks_out = landmarks_out.permute(1, 0, 2)
                features['landmarks'].append(landmarks_out)

        # Pose outputs
        if 'Pose' in outputs.keys():
            for pose in outputs['Pose']:
                features['headpose'].append(pose)

        return features, outputs

    def pretreat(self, image, bboxes):
        '''
        model_inputs : [batch_images, batch_model3D, batch_cam_matrix], # [box_num, 3, 256, 256], [box_num, 98, 3], [box_num, 3, 3]
        crop_bboxes  : [box_num, 4]                
        '''
        crop_bboxes = []
        crop_images = []
        for bbox in bboxes:
            sample = {'image': copy.deepcopy(image),
                      'bbox': copy.deepcopy(bbox)}
            sample_crop = self.transforms(sample)
            crop_bboxes.append(sample_crop['bbox'])
            crop_images.append(sample_crop['image'])

        # Images to tensor and device
        batch_images = torch.tensor(np.array(crop_images), dtype=torch.float)
        batch_images = self._data2device(batch_images)
        # Batch 3D model and camera intrinsic matrix
        batch_model3D = self.model3d.unsqueeze(0).repeat(len(bboxes), 1, 1)
        batch_cam_matrix = self.cam_matrix.unsqueeze(0).repeat(len(bboxes), 1, 1)

        # SPIGA inputs
        model_inputs = [batch_images, batch_model3D, batch_cam_matrix]
        return model_inputs, crop_bboxes

    def net_forward(self, inputs):
        outputs = self.model(inputs)
        return outputs

    def postreatment(self, output, crop_bboxes, bboxes):
        features = {}
        crop_bboxes = np.array(crop_bboxes)
        bboxes = np.array(bboxes)

        if 'Landmarks' in output.keys():
            landmarks = output['Landmarks'][-1].cpu().detach().numpy()
            landmarks = landmarks.transpose((1, 0, 2))
            landmarks = landmarks*self.model_cfg.image_size
            landmarks_norm = (landmarks - crop_bboxes[:, 0:2]) / crop_bboxes[:, 2:4]
            landmarks_out = (landmarks_norm * bboxes[:, 2:4]) + bboxes[:, 0:2]
            landmarks_out = landmarks_out.transpose((1, 0, 2))
            features['landmarks'] = landmarks_out.tolist()

        # Pose output
        if 'Pose' in output.keys():
            pose = output['Pose'].cpu().detach().numpy()
            features['headpose'] = pose.tolist()

        return features

    def select_inputs(self, batch):
        inputs = []
        for ft_name in self.model_inputs:
            data = batch[ft_name]
            inputs.append(self._data2device(data.type(torch.float)))
        return inputs

    def _data2device(self, data):
        if isinstance(data, list):
            data_var = data
            for data_id, v_data in enumerate(data):
                data_var[data_id] = self._data2device(v_data)
        if isinstance(data, dict):
            data_var = data
            for k, v in data.items():
                data[k] = self._data2device(v)
        else:
            with torch.no_grad():
                data_var = data.cuda(device=self.gpus[0], non_blocking=True)
        return data_var
