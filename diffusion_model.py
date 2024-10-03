import torch
import torchvision.transforms as transforms
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from diffusers import DDPMPipeline, DDPMScheduler, UNet2DModel
import numpy as np
from torch.utils.data import Dataset, DataLoader
from astropy.table import Table
from tqdm import tqdm
import time, os
import pandas as pd
import matplotlib.pyplot as plt
import torch.nn as nn
from torch.optim.lr_scheduler import StepLR
import torch.nn.functional as F

# Data preprocessing and loading
def image_preprocess(data, norm, channel=1):
    '''
    # Linear transformation to the image and parameters (reversible)
    '''
    data = data.astype(np.float32)
    data = data * norm
    data = torch.from_numpy(data) #transfer to tensor
    image_size = data.shape[-1]
    data = data.view(len(data), channel , image_size, image_size) # reshape the data
    return data

class MyDataset(Dataset):
    """
    # customize Dataset class. 
    # Para is the option to return the parameters
    # The calibration is a constant factor to rescaling the image's flux
    """
    def __init__(self, data, norm_factor=1):
        self.fore_image = image_preprocess(data['fore_image'].data, channel=1, norm=norm_factor)
        self.back_image = image_preprocess(data['back_image'].data, channel=1, norm=norm_factor)
        self.noise_image = image_preprocess(data['noise_image'].data, channel=4, norm=norm_factor)
        print('shape of the noise image:', self.noise_image.shape)
        print('shape of the fore/back_image image:', self.fore_image.shape, self.back_image.shape)

    def __len__(self):
        return len(self.fore_image)

    def __getitem__(self, idx):
        # return the images
        fore_image = self.fore_image[idx].clone().detach().float()
        back_image = self.back_image[idx].clone().detach().float()
        noise_image= self.noise_image[idx].clone().detach().float()
        return {'fore_image':fore_image,  'back_image':back_image, 'noise_image':noise_image}


class MyLoss(nn.Module):
    def __init__(self):
        super(MyLoss, self).__init__()
        self.L1loss = nn.L1Loss()

    def forward(self, pred_noise, pred_fore, pred_back, noise, fore_image, back_image):
        loss_fore =  self.L1loss(pred_fore, fore_image)
        loss_back =  self.L1loss(pred_back, back_image)
        loss_noise = self.L1loss(pred_noise, noise)
        loss =  loss_fore + loss_back + loss_noise
        return loss.mean()


class GGL_UNet():
    """
    Initialize, one should set the size and channel of input
    """
    def __init__(self):
        self.cfg = {'model_name': 'GGL_UNet_diffusion_model',
                    'train_set': 'images.fits',
                    'input_channel':4,  #single color band
                    'image_size':64,   # 64x64 image 
                    'batch_size': 128*3,
                    'learning_rate': 1e-3,
                    'epochs': 50,
                    'step_size':10,
                    'gamma': 0.5,
                    }
        self.seed = 41
        self.epsilon = 1e-3
        self.zero_point = 0.0
        self.weight_decay = 0.0
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def Init(self):
        output_folder = 'models/'
        if not os.path.exists(output_folder):
            os.mkdir(output_folder)
        
        model_name = self.cfg['model_name']
        if not os.path.exists(f'models/{model_name}'):
            os.mkdir(f'models/{model_name}')
        self.history = pd.DataFrame(columns=['epoch','train_loss','val_loss','time','learning rate'])
        self.cfg['output_pth'] = f'models/{model_name}'
        self.cfg['output_csv'] = self.cfg['output_pth']+f'/{model_name}.csv'

        self.model = UNet2DModel(
        sample_size=self.cfg['image_size'],
        in_channels=4,
        out_channels=12,
        layers_per_block=2,
        block_out_channels=(32, 64, 128, 256),
        down_block_types=(
            "DownBlock2D",  # a regular ResNet downsampling block
            "DownBlock2D",
            "DownBlock2D",
            "AttnDownBlock2D",),
        up_block_types=(
            "AttnUpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
            "UpBlock2D",
        ))

    def load_training_data(self, frac=0.9):
        self.train_set = Table.read(self.cfg['train_set'])
        num = len(self.train_set)
        if frac*num > 0:
            self.train_dataset = MyDataset(self.train_set[0:int(frac*num)])
            self.train_loader = DataLoader(self.train_dataset, batch_size=self.cfg['batch_size'], shuffle=True)
        # we use the same test and valid set
        self.valid_dataset = MyDataset(self.train_set[int(frac*num):])
        self.valid_loader = DataLoader(self.valid_dataset, batch_size=self.cfg['batch_size'], shuffle=True)

    # save the training infomation
    def logging(self, info):
        print('Epoch [{}/{}], Train Loss: {:.4f}, Valid Loss: {:.4f}'.format(info['epoch']+1, self.cfg['epochs'], info['train_loss'], info['val_loss']))
        self.history = pd.concat([self.history, pd.DataFrame([info])], ignore_index=True)
        self.history.to_csv(self.cfg['output_csv']) 
        # save the best one  checkpoint
        if info['val_loss'] <= min(self.history['val_loss'].values):
            self.model.save_pretrained(self.cfg['output_pth'])
            print('save the best checkpoint of ', self.cfg['output_pth'])

    def input_fun(self, data, mode='def'):
        dim = self.cfg['input_channel']
        fore_image, back_image, noise = data['fore_image'], data['back_image'], data['noise_image']
        # https://numpy.org/doc/stable/reference/random/generated/numpy.random.gamma.html
        nl_min, nl_max =  0.5, 1.5
        alpha = np.random.uniform(0, 1.0, (fore_image.shape[0], dim, 1, 1))
        beta  = np.random.uniform(0, 1.0, (back_image.shape[0], dim, 1, 1))
        gamma = np.random.uniform(nl_min, nl_max, (fore_image.shape[0], dim, 1, 1)) 
        epsilon = np.random.uniform(0, 1.0, (noise.shape[0], dim, 1, 1))
        alpha, beta, gamma, epsilon = alpha.astype('float32'), beta.astype('float32'), gamma.astype('float32'), epsilon.astype('float32')
        alpha, beta, gamma, epsilon = torch.from_numpy(alpha), torch.from_numpy(beta), torch.from_numpy(gamma), torch.from_numpy(epsilon)

        if mode=='def':
            zeta = np.random.choice([0, 1], size=(back_image.shape[0], 1, 1, 1), p=[0.05, 0.95])
            rho  = np.random.choice([0, 1], size=(back_image.shape[0], dim, 1, 1), p=[0.10, 0.90])
            selct = rho * zeta
        if mode=='pos':
            selct = np.ones((back_image.shape[0], dim, 1, 1))
        if mode=='neg':
            selct = np.zeros((back_image.shape[0], dim, 1, 1))
        selct = selct.astype('float32')
        selct = torch.from_numpy(selct)

        # https://lenstronomy.readthedocs.io/en/latest/lenstronomy.LightModel.Profiles.html
        fore_image = alpha*fore_image**gamma
        back_image = alpha*beta*back_image*selct
        noise = alpha*beta*epsilon*noise
        input = fore_image + back_image + noise + self.zero_point
        fore_image[fore_image<self.epsilon] = 0
        back_image[back_image<self.epsilon] = 0

        area_bool = (fore_image>self.epsilon) | (back_image>self.epsilon)
        area_size = torch.sum(area_bool, dim=(-1, -2, -3))
        overlap = fore_image*back_image/(fore_image+back_image+self.epsilon)**2
        overlap = torch.sum(overlap, dim=(-1, -2, -3))
        timesteps = (overlap/area_size/self.epsilon).int()
        #timesteps = torch.mean(100*beta*epsilon, dim=(-1, -2, -3)).int()
        input, norm = self.preprocess(input)
        fore_image, back_image, noise =  fore_image*norm, back_image*norm, noise*norm
        # to GPU
        fore_image, back_image, noise, input, timesteps = fore_image.to(self.device), back_image.to(self.device), noise.to(self.device), input.to(self.device), timesteps.to(self.device)
        return fore_image, back_image, noise, input, timesteps

    #====  original output
    def output_fun(self, input, mask, mask_output=False):
        dim = self.cfg['input_channel']
        mask_lens, mask_sour, mask_noise = mask[:,0:dim,:,:], mask[:,dim:2*dim,:,:], mask[:,2*dim:3*dim,:,:]
        mask_noise = mask_noise + self.epsilon
        mask_lens, mask_sour, mask_noise = mask_lens**2, mask_sour**2, mask_noise**2
        norm =  mask_lens + mask_sour + mask_noise
        mask_lens, mask_sour, mask_noise = mask_lens/norm, mask_sour/norm, mask_noise/norm
        pred_noise = mask_noise*input
        pred_fore  = mask_lens*input
        pred_back  = mask_sour*input
        if mask_output:
            pred_noise, pred_fore, pred_back = mask_noise, mask_lens, mask_sour
        return pred_noise, pred_fore, pred_back

    def output_fun_new(self, input, mask):
        dim = self.cfg['input_channel']
        mask_lens, mask_sour, mask_noise = mask[:,0:dim,:,:], mask[:,dim:2*dim,:,:], mask[:,2*dim:3*dim,:,:]
        pred_noise = mask_noise
        mask_lens = mask_lens + 1e-6
        mask_lens, mask_sour = mask_lens**2, mask_sour**2
        norm =  mask_lens + mask_sour
        mask_lens, mask_sour = mask_lens/norm, mask_sour/norm
        pred_fore  = mask_lens*(input-pred_noise)
        pred_back  = mask_sour*(input-pred_noise)
        return pred_noise, pred_fore, pred_back

    def train(self):
        self.model.train()
        train_loss = 0
        for data in tqdm(self.train_loader, desc='Training'):
            self.optimizer.zero_grad()
            fore_image, back_image, noise, input, timesteps = self.input_fun(data)
            mask = self.model(input, timesteps)["sample"]
            pred_noise, pred_fore, pred_back = self.output_fun(input, mask)
            loss = self.criterion(pred_noise, pred_fore, pred_back, noise, fore_image, back_image)
            loss.backward()
            self.optimizer.step()
            train_loss += loss.item() * fore_image.size(0)
        train_loss /= len(self.train_loader.dataset)
        return train_loss

    # validate function
    def valid(self):
        self.model.eval()
        valid_loss = 0
        with torch.no_grad():
            for data in tqdm(self.valid_loader, desc='Validation'):
                fore_image, back_image, noise, input, timesteps = self.input_fun(data)
                mask = self.model(input, timesteps)["sample"] 
                pred_noise, pred_fore, pred_back = self.output_fun(input, mask)
                loss = self.criterion(pred_noise, pred_fore, pred_back, noise, fore_image, back_image)
                valid_loss += loss.item() * fore_image.size(0)
        valid_loss /= len(self.valid_loader.dataset)
        return valid_loss

    #=== training loop
    def training_loop(self):
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.cfg['learning_rate'], weight_decay=self.weight_decay)
        self.model = self.model.to(self.device)
        num_epochs =  self.cfg['epochs']
        self.criterion = MyLoss()
        scheduler = StepLR(self.optimizer, step_size=self.cfg['step_size'], gamma=self.cfg['gamma'], verbose=True)
        np.random.seed(seed=self.seed) # for the repeatable
        for epoch in range(num_epochs):
            start_time = time.time()
            train_loss = self.train()
            valid_loss = self.valid()
            end_time = time.time()
            # save the history
            info = {'epoch':epoch, 'train_loss':train_loss, 'val_loss':valid_loss, 'time':end_time-start_time, 'learning rate': scheduler.get_lr()[0]}
            self.logging(info)
            scheduler.step()
            print('learning rate', scheduler.get_lr())

    #==== preprocessing of real image
    def preprocess(self, images):
        hs = self.cfg['image_size']//2
        rad = 5
        center = images[:,:,hs-rad:hs+rad,hs-rad:hs+rad]
        center = torch.flatten(center, -2, -1)
        #(norm, indices) = center.median(dim=-1)
        (norm, indices) = center.max(dim=-1)
        norm = torch.abs(norm)
        norm = 1.0/norm
        norm = norm.view(norm.shape[0], norm.shape[1], 1, 1)
        images = images*norm
        return images, norm

    #=== network output
    def pred(self, input, timestep=0, load_model=False, mask_output=False):
        #==== load model
        if load_model:
            self.model = self.model.from_pretrained(self.cfg['output_pth'])
        self.model = self.model.to(self.device)
        self.model.eval()
        with torch.no_grad():
            mask = self.model(input, timestep)["sample"] 
            pred_noise, pred_fore, pred_back = self.output_fun(input, mask, mask_output)
        return pred_noise, pred_fore, pred_back

    # reconstruct the mock images
    def pred_mock(self, index, mode='def', RA=[], DEC=[], timestep=0, load_model=False, save_name='pred_mock'):
        data = self.valid_dataset[index[0]:index[1]]
        fore_image, back_image, noise, input, _ = self.input_fun(data, mode)
        input = input.to(self.device)
        pred_noise, pred_fore, pred_back = self.pred(input, timestep, load_model)
        input,fore_image,pred_fore,back_image,pred_back,noise,pred_noise = input.numpy(),fore_image.numpy(),pred_fore.numpy(),back_image.numpy(),pred_back.numpy(),noise.numpy(),pred_noise.numpy()
        timestep = np.ones(input.shape[0])*timestep
        data = [ input,   fore_image,   pred_fore,  back_image,  pred_back,    noise,   pred_noise,  timestep]
        names= ['input', 'fore_image', 'pred_fore','back_image','pred_back',  'noise', 'pred_noise','timestep']
        
        if len(RA) != 0 and len(DEC) != 0:
            data.append(RA)
            names.append('RA')
            data.append(DEC)
            names.append('DEC')
            
        results = Table(data=data,names=names)
        fname = 'results/' + self.cfg['model_name'] + f'_{save_name}_{mode}.fits'
        results.write(fname, format='fits',overwrite=True)

    # reconstruct the real images
    def pred_real(self, real_images, weight=[], RA=[], DEC=[], timestep=0, load_model=False, save_name='pred_real', mask_output=False):
        if real_images.dtype.byteorder != '=':
            real_images = real_images.byteswap().newbyteorder()
        input = torch.from_numpy(real_images)
        input, norm = self.preprocess(input)
        pred_noise, pred_fore, pred_back = self.pred(input, timestep, load_model, mask_output)
        input,pred_fore,pred_back,pred_noise,norm=input.numpy(),pred_fore.numpy(),pred_back.numpy(),pred_noise.numpy(),norm.numpy()
        timestep = np.ones(input.shape[0])*timestep
        data = [ input,   pred_fore,   pred_back,   pred_noise,   timestep,  norm]
        names= ['input', 'pred_fore', 'pred_back', 'pred_noise', 'timestep','norm']
        
        if len(weight) != 0:
            data.append(weight)
            names.append('weight')
        if len(RA) != 0 and len(DEC) != 0:
            data.append(RA)
            names.append('RA')
            data.append(DEC)
            names.append('DEC')

        results = Table(data=data,names=names)
        fname = 'results/' + self.cfg['model_name'] + f'_{save_name}.fits'
        results.write(fname, format='fits',overwrite=True)


if __name__ == "__main__":
    ggl_unet = GGL_UNet()
    #ggl_unet.device = torch.device('cuda:0') # use GPU
    ggl_unet.device = torch.device('cpu')  # use CPU
    ggl_unet.cfg['train_set'] = 'color_images.fits'
    ggl_unet.Init()
    ggl_unet.load_training_data()
    ggl_unet.training_loop()