import torch
import torch.nn as nn
import torch.nn.functional as F

from torch import nn
from torch import optim
from torch.autograd import Function
from torch.utils.data import Dataset
from torch.nn import functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.utils.data import DataLoader, random_split

from glob import glob
from tqdm import tqdm
from PIL import Image
from os import rename
from os import listdir
from shutil import move
from PIL import ImageFile
from os.path import splitext

from sklearn.metrics import jaccard_score as iou

import os
import re
import cv2
import json
import numpy as np

import segmentation_models_pytorch as smp


ImageFile.LOAD_TRUNCATED_IMAGES = True


####################################################################################################################################

def get_model(model_name, encoder):

    if model_name == "unet":
        return smp.Unet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=3)
    elif model_name == "linknet":
        return smp.Linknet(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=3)
    elif model_name == "pan":
        return smp.PAN(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=3)
    elif model_name == "deeplab":
        return smp.DeepLabV3(encoder_name=encoder, encoder_weights="imagenet", in_channels=3, classes=3)

def get_int(strx):
    return int(re.search(r'\d+', strx).group())

####################################################################################################################################
###################################################### BEGIN - DATALOADER ##########################################################
####################################################################################################################################

def contour_to_mask_v2(img, contour):

    mask = np.zeros_like(img)
    mask = cv2.fillPoly(mask, pts =[np.array(contour)], color=(1,1,1))

    return mask


class BasicDataset(Dataset):
    def __init__(self, imgs_dir, masks_dir, scale=1, mask_suffix=''):
        self.imgs_dir = imgs_dir
        self.masks_dir = masks_dir
        self.scale = scale
        self.mask_suffix = mask_suffix
        assert 0 < scale <= 1, 'Scale must be between 0 and 1'

        self.ids = [splitext(file)[0] for file in listdir(imgs_dir)
                    if not file.startswith('.')]
        #print(len(self.ids))

    def __len__(self):
        return len(self.ids)

    @classmethod
    def preprocess(cls, pil_img, scale, mask_flag=False):
        
        #w, h = pil_img.size
        #newW, newH = int(scale * w), int(scale * h)
        #assert newW > 0 and newH > 0, 'Scale is too small'
        #pil_img = pil_img.resize((newW, newH))

        img_nd = np.array(pil_img)
        #img_nd = cv2.resize(img_nd, (256,256), interpolation = cv2.INTER_AREA)
        #img_nd = cv2.resize(img_nd, (506,506), interpolation = cv2.INTER_AREA)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)
        
        if mask_flag:
            msk = (img_nd > 100).astype(np.uint8)
            return msk.transpose((2, 0, 1))

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255

        return img_trans

    def __getitem__(self, i):
        
        idx = self.ids[i]
        
        img_file = glob(self.imgs_dir + idx + '.*')

        img = Image.open(img_file[0])

        mask_file_metadata = img_file[0].split(".")[0].split("_")
        
        sbj = str(get_int(mask_file_metadata[1]))
        act = mask_file_metadata[2]
        rtn = mask_file_metadata[3]
        frame = int(mask_file_metadata[4])
        cam = "cam"+str(get_int(mask_file_metadata[5]))

        #print(img_file[0])
        #print("sbj:", sbj)
        #print("act:", act)
        #print("rtn:", rtn)
        #print("frame:", frame)
        #print("cam:", cam)

        
        img = self.preprocess(img, self.scale, mask_flag = False)
        #print("Image shape:",img.shape)
        
        mask_file_json = sbj+"_"+act+"_"+rtn+"_"+cam+"_seg.json"

        with open("predictions/"+mask_file_json) as file: # Use file to refer to the file object
            data_human = json.load(file)

        with open("predictions-robot/"+mask_file_json) as file: # Use file to refer to the file object
            data_robot = json.load(file)

        contour_human = data_human[frame]['mask']
        contour_robot = data_robot[frame]['mask']

        mask_human = contour_to_mask_v2(np.zeros((256,256)), contour_human)
        mask_robot = contour_to_mask_v2(np.zeros((256,256)), contour_robot)
        m0 = np.zeros_like(mask_human)

        final_mask = cv2.merge((m0, mask_robot, mask_human))
        final_mask = np.transpose(final_mask, (2,0,1))

        #print("Human mask shape:", mask_human.shape)
        #print("Robot mask shape:", mask_robot.shape)
        #print("Final mask shape:", final_mask.shape)


        return {
            'image': torch.from_numpy(img).type(torch.FloatTensor),
            'mask': torch.from_numpy(final_mask).type(torch.FloatTensor)
        }


class CarvanaDataset(BasicDataset):
    def __init__(self, imgs_dir, masks_dir, scale=1):
        super().__init__(imgs_dir, masks_dir, scale, mask_suffix='_mask')


####################################################################################################################################
####################################################### END - DATALOADER ###########################################################
####################################################################################################################################

####################################################################################################################################

####################################################################################################################################
###################################################### BEGIN - EVALUATION ##########################################################
####################################################################################################################################

def transform(tensor):
    return tensor.cpu().numpy().flatten()


def evaluation_v2(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.segmentation_head[0].out_channels == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot_iou = 0
    tot_dice = 0
    tot = 0

    #for batch in tqdm(loader):
    print("evaluating...")
    for batch in tqdm(loader):
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        if net.segmentation_head[0].out_channels > 1:
            true_masks = torch.argmax(true_masks, 1)
            tot += F.cross_entropy(mask_pred, true_masks).item()

            #tot_dice += dice_coeff(pred, true_masks).item()
            #tot_iou += iou(transform(pred), transform(true_masks))
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            
            tot_dice += dice_coeff(pred, true_masks).item()
            tot_iou += iou(transform(pred), transform(true_masks))

    net.train()
    return tot / n_val


class DiceCoeff(Function):
    """Dice coeff for individual examples"""

    def forward(self, input, target):
        self.save_for_backward(input, target)
        eps = 0.0001
        self.inter = torch.dot(input.view(-1), target.view(-1))
        self.union = torch.sum(input) + torch.sum(target) + eps

        t = (2 * self.inter.float() + eps) / self.union.float()
        return t

    # This function has only a single output, so it gets only one gradient
    def backward(self, grad_output):

        input, target = self.saved_variables
        grad_input = grad_target = None

        if self.needs_input_grad[0]:
            grad_input = grad_output * 2 * (target * self.union - self.inter) \
                         / (self.union * self.union)
        if self.needs_input_grad[1]:
            grad_target = None

        return grad_input, grad_target


def dice_coeff(input, target):
    """Dice coeff for batches"""
    if input.is_cuda:
        s = torch.FloatTensor(1).cuda().zero_()
    else:
        s = torch.FloatTensor(1).zero_()

    for i, c in enumerate(zip(input, target)):
        s = s + DiceCoeff().forward(c[0], c[1])

    return s / (i + 1)

def iou_pytorch(outputs: torch.Tensor, labels: torch.Tensor):
    # You can comment out this line if you are passing tensors of equal shape
    # But if you are passing output from UNet or something it will most probably
    # be with the BATCH x 1 x H x W shape
    outputs = outputs.squeeze(1)  # BATCH x 1 x H x W => BATCH x H x W
    
    intersection = (outputs & labels).float().sum((1, 2))  # Will be zero if Truth=0 or Prediction=0
    union = (outputs | labels).float().sum((1, 2))         # Will be zzero if both are 0
    
    iou = (intersection + SMOOTH) / (union + SMOOTH)  # We smooth our devision to avoid 0/0
    
    thresholded = torch.clamp(20 * (iou - 0.5), 0, 10).ceil() / 10  # This is equal to comparing with thresolds
    
    return thresholded

def eval_net(net, loader, device):
    """Evaluation without the densecrf with the dice coefficient"""
    net.eval()
    mask_type = torch.float32 if net.segmentation_head[0].out_channels == 1 else torch.long
    n_val = len(loader)  # the number of batch
    tot = 0

    #for batch in tqdm(loader):
    print("evaluating...")
    for batch in loader:
        imgs, true_masks = batch['image'], batch['mask']
        imgs = imgs.to(device=device, dtype=torch.float32)
        true_masks = true_masks.to(device=device, dtype=mask_type)

        with torch.no_grad():
            mask_pred = net(imgs)

        if net.segmentation_head[0].out_channels > 1:
            true_masks = torch.argmax(true_masks, 1)
            tot += F.cross_entropy(mask_pred, true_masks).item()
        else:
            pred = torch.sigmoid(mask_pred)
            pred = (pred > 0.5).float()
            #tot += dice_coeff(pred, true_masks).item()
            iou_actual = iou_pytorch(pred, true_masks)
            tot += iou_actual
            print(iou_actual)

    net.train()
    return tot / n_val


####################################################################################################################################
####################################################### END - EVALUATION ###########################################################
####################################################################################################################################

####################################################################################################################################

####################################################################################################################################
######################################################## BEGIN - TRAIN #############################################################
####################################################################################################################################



def train_net(net,
              device,
              epochs=5,
              batch_size=1,
              lr=0.001,
              val_percent=0.1,
              save_cp=True,
              img_scale=0.5,
              dir_checkpoint="", dir_img=""):

    dataset = BasicDataset(dir_img, "", img_scale)
    n_val = int(len(dataset) * val_percent)
    n_train = len(dataset) - n_val
    train, val = random_split(dataset, [n_train, n_val])
    train_loader = DataLoader(train, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True, drop_last=True)

    #setx = train_loader.dataset
    #print(setx[0])


    writer = SummaryWriter(comment=f'LR_{lr}_BS_{batch_size}_SCALE_{img_scale}')
    global_step = 0
    
    loss_lst = []
    best_loss = 32000000
    
    optimizer = optim.RMSprop(net.parameters(), lr=lr, weight_decay=1e-8, momentum=0.9)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min' if net.segmentation_head[0].out_channels > 1 else 'max', patience=2)
    
    if net.segmentation_head[0].out_channels > 1:
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    for epoch in range(epochs):
        net.train()

        epoch_loss = 0
        with tqdm(total=n_train, desc=f'Epoch {epoch + 1}/{epochs}', unit='img') as pbar:
            for batch in train_loader:
                imgs = batch['image']
                true_masks = batch['mask']

                imgs = imgs.to(device=device, dtype=torch.float32)
                mask_type = torch.float32 if net.segmentation_head[0].out_channels == 1 else torch.long
                true_masks = true_masks.to(device=device, dtype=mask_type)

                masks_pred = net(imgs)
                true_masks = torch.argmax(true_masks, 1)

                loss = criterion(masks_pred, true_masks)
                epoch_loss += loss.item()
                writer.add_scalar('Loss/train', loss.item(), global_step)

                pbar.set_postfix(**{'loss (batch)': loss.item()})

                optimizer.zero_grad()
                loss.backward()
                nn.utils.clip_grad_value_(net.parameters(), 0.1)
                optimizer.step()

                pbar.update(imgs.shape[0])
                global_step += 1
                

        val_lss = evaluation_v2(net, val_loader, device)
        print("Validation result is:")
        print(val_lss)
        loss_lst.append(val_lss)

        if val_lss < best_loss:
            best_loss = val_lss

            cp_file = dir_checkpoint + f'CP_epoch{epoch + 1}.pth'
            torch.save(net.state_dict(), cp_file)

            print("Checkpoint saved!")

    writer.close()
    cp_file = dir_checkpoint + f'CP_epoch{epoch + 1}.pth'
    torch.save(net.state_dict(), cp_file)
    
    return loss_lst



####################################################################################################################################
######################################################### END - TRAIN ##############################################################
####################################################################################################################################

####################################################################################################################################

####################################################################################################################################
######################################################## BEGIN - MAIN ##############################################################
####################################################################################################################################


def create_cpt_folders(root_cpt, models_list, encoders_list):

    for model in models_list:
        for encoder in encoders_list:
            folder = root_cpt+"/"+model+"/"+encoder+"/"
            if not os.path.isdir():
                os.mkdir(folder)

if __name__ == '__main__':

    torch.manual_seed(0)


    epochs=5
    batch_size=8
    learning_rate=0.0001
    load=False
    scale=1
    validation=20.0


    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    root_cpt = "checkpoints2/"

    dir_img = 'single_frames/'
    models_list = ['unet', 'deeplab', 'pan', 'linknet']
    encoders_list = ['efficientnet-b0', 'timm-mobilenetv3_large_100']

    create_cpt_folders(root_cpt, models_list, encoders_list)


    for model in models_list:
        for encoder in encoders_list:
            
            net = get_model(model, encoder)
            net.to(device=device)

            base_name = model+"/"+encoder+"/"
            dir_checkpoint = root_cpt + base_name
            dir_loss = "losses/" + base_name

            losses = train_net(net=net, epochs=epochs, batch_size=batch_size, lr=learning_rate, device=device,
                  img_scale=scale, val_percent=validation / 100, dir_checkpoint = dir_checkpoint, dir_img = dir_img)

            np.savetxt(dir_loss+"losses.txt", np.array(losses), delimiter=',')


    #net.segmentation_head[0].out_channels
    ####################################################################################################################################
    ######################################################### END - MAIN ###############################################################
    ####################################################################################################################################


