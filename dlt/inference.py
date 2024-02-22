
# import numpy as np
# import pickle
# import torch
# from torch.utils.data import DataLoader
# from tqdm import tqdm
# from PIL import Image
# import os
# import wandb
# from natsort import natsorted

# from data_loaders.magazine import MagazineLayout
# from logger_set import LOG
# from absl import flags, app
# from diffusion import JointDiffusionScheduler, GeometryDiffusionScheduler
# from ml_collections import config_flags
# from models.dlt import DLT
# from utils import set_seed, draw_layout_opacity, custom_collate_fn
# from visualize import create_collage
# from data_loaders.publaynet import PublaynetLayout
# from data_loaders.rico import RicoLayout
# from data_loaders.canva import CanvaLayout

# import safetensors.torch as safetensors
# from safetensors.torch import load_model, save_model

# from models.CAL import CAL
# from accelerate import Accelerator
# from evaluation.iou import transform

# from evaluation.iou import transform, get_iou, get_mean_iou


# FLAGS = flags.FLAGS
# config_flags.DEFINE_config_file("config", "Training configuration.",
#                                 lock_config=False)
# flags.DEFINE_string("workdir", default='test2', help="Work unit directory.")
# flags.DEFINE_string("epoch", default='198', help="Epoch to load from checkpoint.")
# flags.DEFINE_string("cond_type", default='all', help="Condition type to sample from.")
# flags.DEFINE_bool("save", default=False, help="Save samples.")
# flags.mark_flags_as_required(["config"])

# def sample_from_model(batch, model, device, diffusion):
#     #shape = batch['geometry'].shape
#     shape = batch['geometry'][:,:,:4].shape
#     model.eval()
    
#     # generate initial noise
#     noisy_batch = {
#         'geometry': torch.randn(*shape, dtype=torch.float32, device=device),
#         "image_features": batch['image_features']
#     }

#     # sample x_0 = q(x_0|x_t)
#     for i in range(diffusion.num_cont_steps)[::-1]:
#         t = torch.tensor([i] * shape[0], device=device)
#         with torch.no_grad():
#             # denoise for step t.
#             geometry_pred = model(batch, noisy_batch, timesteps=t)
            
#             # sample
#             geometry_pred = diffusion.inference_step(geometry_pred,
#                                                          timestep=torch.tensor([i], device=device),
#                                                          sample=noisy_batch['geometry'])
            

#             # update noise with x_t + update denoised categories.
#             # update noise with x_t + update denoised categories.
#             noisy_batch['geometry'] = geometry_pred.prev_sample * batch['padding_mask'][:, :, :4]
#         print("#################################################################") 
#         print(geometry_pred.pred_original_sample.shape)
#         print("#################################################################") 
#     return geometry_pred.pred_original_sample


# def main(*args, **kwargs):
#     config = init_job()
#     config.optimizer.batch_size = 1
#     LOG.info("Loading data.")
#     if config.dataset == 'publaynet':
#         val_data = PublaynetLayout(config.val_json, 9, config.cond_type)
#     elif config.dataset == 'rico':
#         val_data = RicoLayout(config.dataset_path, 'test', 9, config.cond_type)
#     elif config.dataset == 'magazine':
#         val_data = MagazineLayout(config.val_json, 16, config.cond_type)
#     elif config.dataset == 'canva':
#         val_data = CanvaLayout(config.val_json, config.val_clip_json, max_num_com=config.max_num_comp, scaling_size=config.scaling_size)
#     else:    
#         raise NotImplementedError
#     #assert config.categories_num == val_data.categories_num

#     LOG.info("Creating model and diffusion process...")
    
#     #dlt = DLT(categories_num=config.categories_num).to(config.device)
#     # model = DLT(categories_num=config.categories_num, latent_dim=config.latent_dim,
#     #             num_layers=config.num_layers, num_heads=config.num_heads, dropout_r=config.dropout_r,
#     #             activation='gelu', cond_emb_size=config.cond_emb_size,
#     #             cat_emb_size=config.cls_emb_size)
    
#     model = CAL()
    

    
#     #model = DLT.from_pretrained(config.optimizer.ckpt_dir / f'checkpoint-{config.epoch}', strict=True)
#     load_model(model, config.optimizer.ckpt_dir / f'checkpoint-{config.epoch}' / "model.safetensors", strict=True)
#     #model = torch.load(config.optimizer.ckpt_dir / f'checkpoint-{config.epoch}' / "model.pth")


#     model.to(config.device)
#     # noise_scheduler = JointDiffusionScheduler(alpha=0.0,
#     #                                           seq_max_length=config.max_num_comp,
#     #                                           device=config.device,
#     #                                           discrete_features_names=[('cat', config.categories_num), ],
#     #                                           num_discrete_steps=[config.num_discrete_steps, ],
#     #                                           num_train_timesteps=config.num_cont_timesteps,
#     #                                           beta_schedule=config.beta_schedule,
#     #                                           prediction_type='sample',
#     #                                           clip_sample=False, )
#     # Sort checkpoint files based on the checkpoint numbers
#     checkpoint_dir = config.optimizer.ckpt_dir
#     checkpoint_files = [file for file in os.listdir(checkpoint_dir) if file.startswith("checkpoint-")]
#     checkpoint_files = natsorted(checkpoint_files, key=lambda x: int(x.split("-")[1])) 

#     for checkpoint_file in checkpoint_files:
#         checkpoint_number = checkpoint_file.split("-")[1]
#         LOG.info(f"Loading model from checkpoint-{checkpoint_number}...")
#         load_model(model, config.optimizer.ckpt_dir / f'checkpoint-{config.epoch}' / "model.safetensors", strict=True)
#         model.to(config.device)
        
#         wandb.init(
#             # set the wandb project where this run will be logged
#             project="CAL_val",
#             name=f"epoch_{checkpoint_number}",
#             # track hyperparameters and run metadata
#             config={
#                 "epochs": checkpoint_number,
#             }
#         )
#     noise_scheduler = GeometryDiffusionScheduler(seq_max_length=config.max_num_comp,
#                                               device=config.device,
#                                               num_train_timesteps=config.num_cont_timesteps,
#                                               beta_schedule=config.beta_schedule,
#                                               prediction_type='sample',
#                                               clip_sample=False, )

#     val_loader = DataLoader(val_data, batch_size=config.optimizer.batch_size,
#                             shuffle=False, collate_fn = custom_collate_fn,num_workers=config.optimizer.num_workers)
#     model.eval()

#     all_results = {
#             'ids': [],
#             'dataset_val': [],
#             'predicted_val': [],
#             'iou':[]
#         }
#     #i = 0
#     id_data = []
#     real_data = []
#     pred_data = []
#     iou_data  = []
    
#     for batch, ids in tqdm(val_loader):
#         batch = {k: v.to(config.device) for k, v in batch.items()}
#         with torch.no_grad():
#             pred_geometry = sample_from_model(batch, model, config.device, noise_scheduler)*batch["padding_mask"][:, :, :4]
        
#         real_geometry = batch["geometry"]
#         real_box, pred_box = transform(real_geometry, pred_geometry, scaling_size=5)
        
#         # print_results(real_box, pred_box)
#         iou = get_iou(real_box, pred_box)
#         slide_mean_iou = get_mean_iou(iou)

#         wandb.log({"mean_iou": slide_mean_iou})
#         # visualize(ids, batch)
        
#         # 각 하위 리스트를 하나의 리스트로 합치기
#         valid_ids = [item for sublist in ids for item in sublist]
#         valid_ids = valid_ids[0] if len(valid_ids) == 1 else valid_ids
        
#         # print("ids: ", valid_ids)
#         # print("ids 길이: ", len(valid_ids))
#         # print("###############################################################")
        
#         # 각 하위 리스트를 하나의 리스트로 합치기
#         valid_ids = [item for sublist in ids for item in sublist]
#         valid_ids = valid_ids[0] if len(valid_ids) == 1 else valid_ids
        
#         id_data.append(valid_ids)
#         real_data.append(real_geometry)
#         pred_data.append(pred_geometry)
#         iou_data.append(get_iou(real_box, pred_box))
        
#         # 캔버스 크기 예시
#         canvas_size = (1920, 1080)
#         base_path = "visualize_picture"
#         save_path = 'output_result2'
        
#         # 이미지 합치기 실행
#         # for id, geometry  in zip(ids, batch["geometry"]):
#         for id, geometry  in zip(ids, pred_geometry):
#             collage = create_collage(batch['geometry'].squeeze(), id, geometry, canvas_size, base_path, config.scaling_size)
#             # collage.show()
#             # 역 슬래시를 언더스코어로 변경
#             ppt_name = id[0].split('/')[0]
#             slide_name = id[0].split('/')[1]

#             # '_Shape' 이전까지의 문자열을 얻기 위해 '_Shape'을 기준으로 분리하고 첫 번째 부분을 선택
#             slide_name = slide_name.split('_Shape')[0]

#             # 확장자를 다시 추가 (.png는 예시입니다. 실제 확장자에 따라 변경해야 할 수 있습니다.)
#             save_file_name = slide_name + '.png'
#             save_file_name = os.path.join(save_path, ppt_name, save_file_name)
#             os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
#             collage.save(save_file_name)

#         # 결과 보기 또는 저장
#         # # save samples
#         # box = batch['mask_box'] * bbox_pred + (1 - batch['mask_box']) * batch['box_cond']
#         # cat = batch['mask_cat'] * cat_pred + (1 - batch['mask_cat']) * batch['cat']
#         # box = box.cpu().numpy()
#         # cat = cat.cpu().numpy()

#         # all_results['dataset_val'].append(
#         #     np.concatenate([batch['box_cond'].cpu().numpy(),
#         #                     np.expand_dims(batch['cat'].cpu().numpy(), -1)], axis=-1))
#         # all_results['predicted_val'].append(
#         #     np.concatenate([box, np.expand_dims(cat, -1)], axis=-1))
#         # if config.save:
#         #     for b in range(box.shape[0]):
#         #         tmp_box = box[b]
#         #         tmp_cat = cat[b]
#         #         tmp_box = tmp_box[~(tmp_box == 0.).all(1)]
#         #         tmp_cat = tmp_cat[~(tmp_cat == 0)]
#         #         tmp_box = (tmp_box / 2 + 1) / 2
#         #         canvas = draw_layout_opacity(tmp_box, tmp_cat, None, val_data.idx2color_map, height=512)
#         #         Image.fromarray(canvas).save(config.optimizer.samples_dir / f'{str(i)}_{str(b)}.jpg')
#         #         tmp_box = batch['box_cond'][b].cpu().numpy()
#         #         tmp_cat = batch['cat'][b].cpu().numpy()
#         #         tmp_box = tmp_box[~(tmp_box == 0.).all(1)]
#         #         tmp_cat = tmp_cat[~(tmp_cat == 0)]
#         #         tmp_box = (tmp_box / 2 + 1) / 2
#         #         canvas = draw_layout_opacity(tmp_box, tmp_cat, None, val_data.idx2color_map, height=512)
#         #         Image.fromarray(canvas).save(config.optimizer.samples_dir / f'{str(i)}_{str(b)}_gt.jpg')
#         #i += 1
#     # pickle results
#     # with open(config.optimizer.samples_dir / f'results_{config.cond_type}.pkl', 'wb') as f:
#     #     pickle.dump(all_results, f)
#     all_results["ids"] = id_data
#     all_results["dataset_val"] = real_data
#     all_results["predicted_val"] = pred_data
#     all_results["iou"] = iou_data

#     with open(config.dataset_path / f'inference_canva.pkl', 'wb') as f:
#         pickle.dump(all_results, f)
        
#     wandb.finish()


# def init_job():
#     config = FLAGS.config
#     config.log_dir = config.log_dir / FLAGS.workdir
#     config.optimizer.ckpt_dir = config.log_dir / 'checkpoints'
#     config.optimizer.samples_dir = config.log_dir / 'samples'
#     config.epoch = FLAGS.epoch
#     set_seed(config.seed)
#     assert config.dataset in ['publaynet', 'rico', 'magazine', 'canva']
#     #assert FLAGS.cond_type in ['whole_box', 'loc', 'all']
#     config.cond_type = FLAGS.cond_type
#     config.save = FLAGS.save
#     return config


# if __name__ == '__main__':
#     app.run(main)
    
    

import numpy as np
import pickle
import torch
from torch.utils.data import DataLoader
from tqdm import tqdm
from PIL import Image
import os
import wandb
from natsort import natsorted

from data_loaders.magazine import MagazineLayout
from logger_set import LOG
from absl import flags, app
from diffusion import JointDiffusionScheduler, GeometryDiffusionScheduler
from ml_collections import config_flags
from models.dlt import DLT
from utils import set_seed, draw_layout_opacity, custom_collate_fn
from visualize import create_collage
from data_loaders.publaynet import PublaynetLayout
from data_loaders.rico import RicoLayout
from data_loaders.canva import CanvaLayout

import safetensors.torch as safetensors
from safetensors.torch import load_model, save_model

from models.CAL import CAL
from accelerate import Accelerator
from evaluation.iou import transform

from evaluation.iou import transform, get_iou, get_mean_iou

FLAGS = flags.FLAGS
config_flags.DEFINE_config_file("config", "Training configuration.",
                                lock_config=False)
flags.DEFINE_string("workdir", default='test2', help="Work unit directory.")
flags.DEFINE_string("epoch", default='396', help="Epoch to load from checkpoint.")
flags.DEFINE_string("cond_type", default='all', help="Condition type to sample from.")
flags.DEFINE_bool("save", default=False, help="Save samples.")
flags.mark_flags_as_required(["config"])

def sample_from_model(batch, model, device, diffusion):
    #shape = batch['geometry'].shape
    shape = batch['geometry'][:,:,:4].shape
    model.eval()
    
    # generate initial noise
    noisy_batch = {
        'geometry': torch.randn(*shape, dtype=torch.float32, device=device),
        "image_features": batch['image_features']
    }

    # sample x_0 = q(x_0|x_t)
    for i in range(diffusion.num_cont_steps)[::-1]:
        t = torch.tensor([i] * shape[0], device=device)
        with torch.no_grad():
            # denoise for step t.
            geometry_pred = model(batch, noisy_batch, timesteps=t)
            
            # sample
            geometry_pred = diffusion.inference_step(geometry_pred,
                                                         timestep=torch.tensor([i], device=device),
                                                         sample=noisy_batch['geometry'])
            

            # update noise with x_t + update denoised categories.
            # update noise with x_t + update denoised categories.
            noisy_batch['geometry'] = geometry_pred.prev_sample * batch['padding_mask'][:, :, :4]
            
            # print('########################################################')
            # print("batch[geometry]: ", batch['geometry'])
            # print("noisy_batch[geometry]: ",noisy_batch['geometry'])
            # print("padding_mask after: ", noisy_batch['geometry'] * batch['padding_mask'])
            # print('########################################################')
    return geometry_pred.pred_original_sample


def main(*args, **kwargs):
    config = init_job()
    config.optimizer.batch_size = 1
    LOG.info("Loading data.")
    if config.dataset == 'publaynet':
        val_data = PublaynetLayout(config.val_json, 9, config.cond_type)
    elif config.dataset == 'rico':
        val_data = RicoLayout(config.dataset_path, 'test', 9, config.cond_type)
    elif config.dataset == 'magazine':
        val_data = MagazineLayout(config.val_json, 16, config.cond_type)
    elif config.dataset == 'canva':
        val_data = CanvaLayout(config.val_json, config.val_clip_json, max_num_com=config.max_num_comp)
    else:    
        raise NotImplementedError
    #assert config.categories_num == val_data.categories_num

    LOG.info("Creating model and diffusion process...")
    
    #dlt = DLT(categories_num=config.categories_num).to(config.device)
    # model = DLT(categories_num=config.categories_num, latent_dim=config.latent_dim,
    #             num_layers=config.num_layers, num_heads=config.num_heads, dropout_r=config.dropout_r,
    #             activation='gelu', cond_emb_size=config.cond_emb_size,
    #             cat_emb_size=config.cls_emb_size)
    
    model = CAL()
    

    
    #model = DLT.from_pretrained(config.optimizer.ckpt_dir / f'checkpoint-{config.epoch}', strict=True)
    load_model(model, config.optimizer.ckpt_dir / f'checkpoint-{config.epoch}' / "model.safetensors", strict=True)
    #model = torch.load(config.optimizer.ckpt_dir / f'checkpoint-{config.epoch}' / "model.pth")


    model.to(config.device)
    # noise_scheduler = JointDiffusionScheduler(alpha=0.0,
    #                                           seq_max_length=config.max_num_comp,
    #                                           device=config.device,
    #                                           discrete_features_names=[('cat', config.categories_num), ],
    #                                           num_discrete_steps=[config.num_discrete_steps, ],
    #                                           num_train_timesteps=config.num_cont_timesteps,
    #                                           beta_schedule=config.beta_schedule,
    #                                           prediction_type='sample',
    #                                           clip_sample=False, )
    # Sort checkpoint files based on the checkpoint numbers
    checkpoint_dir = config.optimizer.ckpt_dir
    checkpoint_files = [file for file in os.listdir(checkpoint_dir) if file.startswith("checkpoint-")]
    checkpoint_files = natsorted(checkpoint_files, key=lambda x: int(x.split("-")[1])) 

    for checkpoint_file in checkpoint_files:
        checkpoint_number = checkpoint_file.split("-")[1]
        LOG.info(f"Loading model from checkpoint-{checkpoint_number}...")
        load_model(model, config.optimizer.ckpt_dir / f'checkpoint-{config.epoch}' / "model.safetensors", strict=True)
        model.to(config.device)
        
        wandb.init(
            # set the wandb project where this run will be logged
            project="CAL_val",
            name=f"epoch_{checkpoint_number}",
            # track hyperparameters and run metadata
            config={
                "epochs": checkpoint_number,
            }
        )
    noise_scheduler = GeometryDiffusionScheduler(seq_max_length=config.max_num_comp,
                                              device=config.device,
                                              num_train_timesteps=config.num_cont_timesteps,
                                              beta_schedule=config.beta_schedule,
                                              prediction_type='sample',
                                              clip_sample=False, )

    val_loader = DataLoader(val_data, batch_size=config.optimizer.batch_size,
                            shuffle=False, collate_fn = custom_collate_fn,num_workers=config.optimizer.num_workers)
    model.eval()

    all_results = {
            'ids': [],
            'dataset_val': [],
            'predicted_val': [],
            'iou':[]
        }
    #i = 0
    id_data = []
    real_data = []
    pred_data = []
    iou_data  = []
    
    for batch, ids in tqdm(val_loader):
        batch = {k: v.to(config.device) for k, v in batch.items()}
        with torch.no_grad():
            pred_geometry = sample_from_model(batch, model, config.device, noise_scheduler)*batch["padding_mask"][:, :, :4]
        
        real_geometry = batch["geometry"]
        real_box, pred_box = transform(real_geometry, pred_geometry, scaling_size=config.scaling_size)
        
        # print_results(real_box, pred_box)
        iou = get_iou(real_box, pred_box)
        slide_mean_iou = get_mean_iou(real_box, pred_box)

        wandb.log({"mean_iou": slide_mean_iou})
        # visualize(ids, batch)
        
        # 각 하위 리스트를 하나의 리스트로 합치기
        valid_ids = [item for sublist in ids for item in sublist]
        valid_ids = valid_ids[0] if len(valid_ids) == 1 else valid_ids
        
        # print("ids: ", valid_ids)
        # print("ids 길이: ", len(valid_ids))
        # print("###############################################################")
        
        # 각 하위 리스트를 하나의 리스트로 합치기
        valid_ids = [item for sublist in ids for item in sublist]
        valid_ids = valid_ids[0] if len(valid_ids) == 1 else valid_ids
        
        id_data.append(valid_ids)
        real_data.append(real_geometry)
        pred_data.append(pred_geometry)
        iou_data.append(get_iou(real_box, pred_box))
        
        # 캔버스 크기 예시
        canvas_size = (1920, 1080)
        base_path = "visualize_picture"
        save_path = 'output_result2'
        
        # 이미지 합치기 실행
        # for id, geometry  in zip(ids, batch["geometry"]):
        for id, geometry  in zip(ids, pred_geometry):
            collage = create_collage(batch['geometry'].squeeze(), id, geometry, canvas_size, base_path, config.scaling_size)
            # collage.show()
            # 역 슬래시를 언더스코어로 변경
            ppt_name = id[0].split('/')[0]
            slide_name = id[0].split('/')[1]

            # '_Shape' 이전까지의 문자열을 얻기 위해 '_Shape'을 기준으로 분리하고 첫 번째 부분을 선택
            slide_name = slide_name.split('_Shape')[0]

            # 확장자를 다시 추가 (.png는 예시입니다. 실제 확장자에 따라 변경해야 할 수 있습니다.)
            save_file_name = slide_name + '.png'
            save_file_name = os.path.join(save_path, ppt_name, save_file_name)
            os.makedirs(os.path.dirname(save_file_name), exist_ok=True)
            collage.save(save_file_name)

        # 결과 보기 또는 저장
        # # save samples
        # box = batch['mask_box'] * bbox_pred + (1 - batch['mask_box']) * batch['box_cond']
        # cat = batch['mask_cat'] * cat_pred + (1 - batch['mask_cat']) * batch['cat']
        # box = box.cpu().numpy()
        # cat = cat.cpu().numpy()

        # all_results['dataset_val'].append(
        #     np.concatenate([batch['box_cond'].cpu().numpy(),
        #                     np.expand_dims(batch['cat'].cpu().numpy(), -1)], axis=-1))
        # all_results['predicted_val'].append(
        #     np.concatenate([box, np.expand_dims(cat, -1)], axis=-1))
        # if config.save:
        #     for b in range(box.shape[0]):
        #         tmp_box = box[b]
        #         tmp_cat = cat[b]
        #         tmp_box = tmp_box[~(tmp_box == 0.).all(1)]
        #         tmp_cat = tmp_cat[~(tmp_cat == 0)]
        #         tmp_box = (tmp_box / 2 + 1) / 2
        #         canvas = draw_layout_opacity(tmp_box, tmp_cat, None, val_data.idx2color_map, height=512)
        #         Image.fromarray(canvas).save(config.optimizer.samples_dir / f'{str(i)}_{str(b)}.jpg')
        #         tmp_box = batch['box_cond'][b].cpu().numpy()
        #         tmp_cat = batch['cat'][b].cpu().numpy()
        #         tmp_box = tmp_box[~(tmp_box == 0.).all(1)]
        #         tmp_cat = tmp_cat[~(tmp_cat == 0)]
        #         tmp_box = (tmp_box / 2 + 1) / 2
        #         canvas = draw_layout_opacity(tmp_box, tmp_cat, None, val_data.idx2color_map, height=512)
        #         Image.fromarray(canvas).save(config.optimizer.samples_dir / f'{str(i)}_{str(b)}_gt.jpg')
        #i += 1
    # pickle results
    # with open(config.optimizer.samples_dir / f'results_{config.cond_type}.pkl', 'wb') as f:
    #     pickle.dump(all_results, f)
    all_results["ids"] = id_data
    all_results["dataset_val"] = real_data
    all_results["predicted_val"] = pred_data
    all_results["iou"] = iou_data

    with open(config.dataset_path / f'inference_canva.pkl', 'wb') as f:
        pickle.dump(all_results, f)
        
    wandb.finish()


def init_job():
    config = FLAGS.config
    config.log_dir = config.log_dir / FLAGS.workdir
    config.optimizer.ckpt_dir = config.log_dir / 'checkpoints'
    config.optimizer.samples_dir = config.log_dir / 'samples'
    config.epoch = FLAGS.epoch
    set_seed(config.seed)
    assert config.dataset in ['publaynet', 'rico', 'magazine', 'canva']
    #assert FLAGS.cond_type in ['whole_box', 'loc', 'all']
    config.cond_type = FLAGS.cond_type
    config.save = FLAGS.save
    return config


if __name__ == '__main__':
    app.run(main)
    
    
    
    
    
      
    
    
      