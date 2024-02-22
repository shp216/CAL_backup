import math
import os
import random
import shutil

import numpy as np
import torch
import wandb
from diffusers.pipelines import DDPMPipeline
from PIL import Image
from accelerate import Accelerator
from diffusers import get_scheduler
from models.CAL import CAL

from einops import repeat
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm

from data_loaders.data_utils import mask_loc, mask_size, mask_whole_box, mask_random_box_and_cat, mask_all
from diffusion import JointDiffusionScheduler, GeometryDiffusionScheduler

from evaluation.iou import transform, print_results, get_iou, get_mean_iou

from logger_set import LOG
from utils import masked_l2, masked_l2_r,masked_cross_entropy, masked_acc, plot_sample, custom_collate_fn

import safetensors.torch as safetensors
from safetensors.torch import load_model, save_model

import copy
from copy import deepcopy
import warnings
warnings.filterwarnings("ignore")
from models.clip_encoder import CLIPModule

def sample_from_model(batch, model, device, diffusion, geometry_scale):
    #shape = batch['geometry'].shape
    shape = batch['geometry'][:,:,:4].shape
    model.eval()
    
    # generate initial noise
    noisy_batch = {
        'geometry': torch.randn(*shape, dtype=torch.float32, device=device).to(device),
        "image_features": batch['image_features']
    }

    # sample x_0 = q(x_0|x_t)
    for i in range(diffusion.num_cont_steps)[::-1]:
        t = torch.tensor([i] * shape[0], device=device)
        with torch.no_grad():
            # denoise for step t.
            geometry_pred = model(batch, noisy_batch, timesteps=t)

            # cos_r = geometry_pred[:, :, 4]
            # sin_r = geometry_pred[:, :, 5]
            # r = torch.atan2(sin_r, cos_r)

            # 새로운 geometry 텐서를 생성합니다: xywhrz 형태
            #geometry_pred = torch.cat([geometry_pred[:, :, 0:4], r.unsqueeze(-1), geometry_pred[:, :, 6].unsqueeze(-1)], dim=-1)
            
            # sample
            geometry_pred = diffusion.inference_step(geometry_pred,
                                                         timestep=torch.tensor([i], device=device),
                                                         sample=noisy_batch['geometry'])
            
            noisy_batch['geometry'] = geometry_pred.prev_sample * batch['padding_mask'][:, :, :4]
            
            # print('########################################################')
            # print("batch[geometry]: ", batch['geometry'])
            # print("noisy_batch[geometry]: ",noisy_batch['geometry'])
            # print("padding_mask after: ", noisy_batch['geometry'] * batch['padding_mask'])
            # print('########################################################')
    return geometry_pred.pred_original_sample 

class TrainLoopCAL:
    def __init__(self, accelerator: Accelerator, model, diffusion: GeometryDiffusionScheduler, train_data,
                 val_data, opt_conf,
                 log_interval: int,
                 save_interval: int, 
                 device: str = 'cpu',
                 resume_from_checkpoint: str = None, 
                 scaling_size = 5,
                 z_scaling_size=0.01):
        
        self.train_data = train_data
        self.val_data = val_data
        self.accelerator = accelerator
        self.save_interval = save_interval
        self.diffusion = diffusion
        self.opt_conf = opt_conf
        self.log_interval = log_interval
        self.device = device
        self.scaling_size = scaling_size
        self.z_scaling_size = z_scaling_size

        optimizer = torch.optim.AdamW(model.parameters(), lr=opt_conf.lr, betas=opt_conf.betas,
                                      weight_decay=opt_conf.weight_decay, eps=opt_conf.epsilon)
        train_loader = DataLoader(train_data, batch_size=opt_conf.batch_size,
                                  shuffle=True, collate_fn = custom_collate_fn, num_workers=opt_conf.num_workers)
        val_loader = DataLoader(val_data, batch_size=opt_conf.batch_size,
                                shuffle=False, collate_fn = custom_collate_fn, num_workers=opt_conf.num_workers)
        lr_scheduler = get_scheduler(opt_conf.lr_scheduler,
                                     optimizer,
                                     num_warmup_steps=opt_conf.num_warmup_steps * opt_conf.gradient_accumulation_steps,
                                     num_training_steps=(len(train_loader) * opt_conf.num_epochs))
        self.model, self.optimizer, self.train_dataloader, self.val_dataloader, self.lr_scheduler = accelerator.prepare(
            model, optimizer, train_loader, val_loader, lr_scheduler
        )
        LOG.info((model.device, self.device))

        self.total_batch_size = opt_conf.batch_size * accelerator.num_processes * opt_conf.gradient_accumulation_steps
        self.num_update_steps_per_epoch = math.ceil(len(train_loader) / opt_conf.gradient_accumulation_steps)
        self.max_train_steps = opt_conf.num_epochs * self.num_update_steps_per_epoch

        LOG.info("***** Running training *****")
        LOG.info(f"  Num examples = {len(train_data)}")
        LOG.info(f"  Num Epochs = {opt_conf.num_epochs}")
        LOG.info(f"  Instantaneous batch size per device = {opt_conf.batch_size}")
        LOG.info(f"  Total train batch size (w. parallel, distributed & accumulation) = {self.total_batch_size}")
        LOG.info(f"  Gradient Accumulation steps = {opt_conf.gradient_accumulation_steps}")
        LOG.info(f"  Total optimization steps = {self.max_train_steps}")
        
        self.global_step = 0
        self.first_epoch = 0
        self.resume_from_checkpoint = resume_from_checkpoint
        if resume_from_checkpoint:
            LOG.print(f"Resuming from checkpoint {resume_from_checkpoint}")
            accelerator.load_state(resume_from_checkpoint)
            last_epoch = int(resume_from_checkpoint.split("-")[1])
            self.global_step = last_epoch * self.num_update_steps_per_epoch
            self.first_epoch = last_epoch
            self.resume_step = 0
            
        # start a new wandb run to track this script
        wandb.init(
            # set the wandb project where this run will be logged
            project="CAL_train",
            # track hyperparameters and run metadata
            config={
            "epochs": 1000,
            "normalize interval": (-1,1)
            }
        )
        

    def train(self):
        for epoch in range(self.first_epoch, self.opt_conf.num_epochs):
            self.train_epoch_CAL(epoch)
            # orig, pred = self.generate_images()
            # wandb.log({
            #     "pred": [wandb.Image(pil, caption=f'pred_{self.global_step}_{i:02d}.jpg')
            #              for i, pil in pred],
            #     "orig": [wandb.Image(pil, caption=f'orig_{self.global_step}.jpg')
            #              for i, pil in orig]}, step=self.global_step)

    def sample2dev(self, sample): # sample to device
        for k, v in sample.items():
            if isinstance(v, dict):
                for k1, v1 in v.items():
                    sample[k][k1] = v1.to(self.device)
            else:
                sample[k] = v.to(self.device)

    ############################################# Content-Aware Layout Generation part ######################################################

    def train_epoch_CAL(self, epoch):
        self.model.train()
        warnings.filterwarnings("ignore")
        device = self.model.device
        progress_bar = tqdm(total=self.num_update_steps_per_epoch, disable=not self.accelerator.is_local_main_process)
        progress_bar.set_description(f"Epoch {epoch}")
        train_losses = []
        train_mean_ious = []
        for step, (batch, ids) in enumerate(self.train_dataloader):
            self.epoch_step = 0
            # input = torch.randn([16,20,6])
            # Geometry1 = torch.ones_like(input)
            # image_features = torch.randn([16,20,512])

            # batch = {"Geometry": Geometry1,
            #         "image_features": image_features}
            # print("#############################################################")
            # print("batch[geometry].shape: ", batch['geometry'].shape)
            # print("batch[image_features].shape: ", batch['image_features'].shape)
            # print("batch[padding_mask].shape: ", batch['padding_mask'].shape)
            # print("batch[cat].shape: ", batch['cat'].shape)
            # print("#############################################################")


            # Skip steps until we reach the resumed step
            if self.resume_from_checkpoint and epoch == self.first_epoch and step < self.resume_step:
                if step % self.opt_conf.gradient_accumulation_steps == 0:
                    progress_bar.update(1)
                continue
            self.sample2dev(batch)

            # Sample noise that we'll add to the boxes
            geometry_scale = torch.tensor([self.scaling_size, self.scaling_size, self.scaling_size, self.scaling_size, 1, self.z_scaling_size]) # scale에 따라 noise 부여
            noise = torch.randn(batch['geometry'].shape).to(device) * geometry_scale.view(1, 1, 6).to(device)  #[batch, 20, 6]
            bsz = batch['geometry'].shape[0] #batch_size
            # Sample a random timestep for each layout
            t = torch.randint(
                0, self.diffusion.num_cont_steps, (bsz,), device=device
            ).long()

            noisy_geometry = self.diffusion.add_noise_Geometry(batch['geometry'], t, noise)
            # rewrite box with noised version, original box is still in batch['box_cond']
            noisy_batch = {"geometry": noisy_geometry,
                           "image_features": batch['image_features']}


            # Run the model on the noisy layouts
            with self.accelerator.accumulate(self.model):
                geometry_predict = self.model(batch, noisy_batch, t)
                train_loss = masked_l2(batch['geometry'], geometry_predict, batch['padding_mask']) #masked_12를 사용하여 xywh만 loss 계산 가능, masked_l2_r는 r,z, r의 normalize loss를 포함
                l2_norm = sum(p.pow(2.0).sum() for p in self.model.parameters())
                l2_loss = 0.0001 * l2_norm
               
                train_loss = train_loss.mean() + l2_loss
                train_losses.append(train_loss.item())
                
                
                pred_geometry = geometry_predict[:,:,:4]*batch['padding_mask'][:,:,:4]

                self.accelerator.backward(train_loss)
                
                true_geometry = batch["geometry"] 
                true_box, pred_box = transform(true_geometry[:,:,:4], pred_geometry[:,:,:4], self.scaling_size)

                batch_mean_iou = get_mean_iou(true_box, pred_box)
                # print(f"batch_mean_iou", batch_mean_iou)
                train_mean_ious.append(batch_mean_iou)

                
                if self.accelerator.sync_gradients:
                    self.accelerator.clip_grad_norm_(self.model.parameters(), 1.0)
                self.optimizer.step()
                self.lr_scheduler.step()
                self.optimizer.zero_grad()

            # losses.setdefault("mse", []).append(loss_mse.mean().detach().item())
            # losses.setdefault("cls", []).append(loss_cls.mean().detach().item())
            # acc_cat = masked_acc(batch['cat'].detach(),
            #                      cls_predict, batch['mask_cat'].detach())
            # losses.setdefault("acc_cat", []).append(acc_cat.mean().detach().item())
            # losses.setdefault("loss", []).append(loss.detach().item())

            if self.accelerator.sync_gradients:
                progress_bar.update(1)
                self.epoch_step+=1
                self.global_step += 1
                logs = {"loss": train_loss.detach().item(), "lr": self.lr_scheduler.get_last_lr()[0],
                        "step": self.global_step}
                progress_bar.set_postfix(**logs)

        
        # Validation loop
        # self.model.eval()

        val_losses = []
        val_mean_ious = []
        val_mean_ious_1000=[]

        with torch.no_grad():
            for val_step, (val_batch, val_ids) in enumerate(self.val_dataloader):
                self.sample2dev(val_batch)
                val_noise = torch.randn(val_batch['geometry'].shape).to(device).to(device)
                val_t = torch.randint(0, self.diffusion.num_cont_steps, (val_batch['geometry'].shape[0],)).long()

                val_noisy_geometry = self.diffusion.add_noise_Geometry(val_batch['geometry'], val_t, val_noise)
                val_noisy_batch = {"geometry": val_noisy_geometry,
                                "image_features": val_batch['image_features']}

                val_pred_geometry = self.model(val_batch, val_noisy_batch, val_t)
                
                val_loss = masked_l2(val_batch['geometry'], val_pred_geometry, val_batch['padding_mask'])
                val_loss = val_loss.mean()
                val_losses.append(val_loss.item())
                
                val_pred_geometry = val_pred_geometry[:,:,:4]*val_batch['padding_mask'][:,:,:4]
                true_geometry = val_batch["geometry"]
                val_true_box, val_pred_box = transform(true_geometry, val_pred_geometry, self.scaling_size)
                val_mean_iou = get_mean_iou(val_true_box, val_pred_box)      
                val_mean_ious.append(val_mean_iou)
                             
                if epoch % 30 == 0:
                    val_pred_geometry_1000 = sample_from_model(val_batch, self.model, device, self.diffusion, geometry_scale)
                    
                    # Calculate and log mean_iou
                    val_pred_geometry_1000 = val_pred_geometry_1000[:,:,:4]*val_batch['padding_mask'][:,:,:4]
                    true_geometry = val_batch["geometry"]
                    val_true_box, val_pred_box = transform(true_geometry, val_pred_geometry_1000, self.scaling_size)
                    val_mean_iou_1000 = get_mean_iou(val_true_box, val_pred_box)      
                    val_mean_ious_1000.append(val_mean_iou_1000)
        
        
        ## wandb 로그 찍기
        avg_train_loss = sum(train_losses)/len(train_losses)
        avg_train_mean_iou = sum(train_mean_ious) / len(train_mean_ious)
        avg_val_loss = sum(val_losses) / len(val_losses) 
        avg_val_mean_iou = sum(val_mean_ious) / len(val_mean_ious)
 
        wandb.log({"avg_loss_train": avg_train_loss}, step=epoch)
        wandb.log({"avg_mean_iou_train": avg_train_mean_iou}, step=epoch)
        wandb.log({"avg_loss_val": avg_val_loss}, step=epoch)
        wandb.log({"avg_mean_iou_val": avg_val_mean_iou}, step=epoch)
        wandb.log({"lr": self.lr_scheduler.get_last_lr()[0]}, step=epoch)
        
        if epoch % 30 == 0:
            avg_val_mean_iou_1000 = sum(val_mean_ious_1000) / len(val_mean_ious_1000)
            wandb.log({"iou_val_1000": avg_val_mean_iou_1000}, step=epoch)
        
        LOG.info(f"Epoch {epoch}, Avg Validation Loss: {avg_val_loss}, Avg Mean IoU: {val_mean_iou}")        
        
        
        # print("############################################")
        # print(type(self.model.state_dict().items()))
        # for x in self.model.state_dict():
        #     print(x, self.model.state_dict()[x].shape)
        # print("############################################")

        progress_bar.close()
        self.accelerator.wait_for_everyone()
        
        # Save the model at the end of each epoch
        if(epoch % 99 == 0):
            save_path = self.opt_conf.ckpt_dir / f"checkpoint-{epoch}/"

            # delete folder if we have already 5 checkpoints
            if self.opt_conf.ckpt_dir.exists():
                ckpts = list(self.opt_conf.ckpt_dir.glob("checkpoint-*"))
                # sort by epoch
                ckpts = sorted(ckpts, key=lambda x: int(x.name.split("-")[1]))
                if len(ckpts) > 3:
                    LOG.info(f"Deleting checkpoint {ckpts[0]}")
                    shutil.rmtree(ckpts[0])
            
            # print("############################################")
            # print(type(self.model.state_dict().items()))
            # for x in self.model.state_dict():
            #     print(x, self.model.state_dict()[x].shape)
            # print("############################################")
            
            self.accelerator.save_state(save_path)
            
            # self.model.save_pretrained(save_path)
            safetensors.save_model(self.model, save_path / "model.pth")

            LOG.info(f"Saving checkpoint to {save_path}")
        
        