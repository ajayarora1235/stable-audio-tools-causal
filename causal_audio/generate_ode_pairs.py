import torch.distributed as dist
from tqdm import tqdm
import argparse
import torch
import math
import os
import re
from stable_audio_tools import get_pretrained_model
import numpy as np

import pandas as pd
import soundfile as sf

def get_caption(mDf, idx):
  title = mDf.iloc[idx]['title']

  description = mDf.iloc[idx]['description']

  tags = mDf.iloc[idx]['tags:']
  tagsList = tags.split(',')

  tagsList = [tag.strip() for tag in tagsList if tag.strip() != '']
  tagsList = np.random.permutation(tagsList)
  tags_shuffled = ', '.join(tagsList)

  elements = [title, description, tags_shuffled]

  for i in range(len(elements)):
    if np.random.rand() < 0.5:
      elements[i] = ""
  
  if all(x == "" for x in elements):
    nonEmpty = [title, description, tags_shuffled]
    nonEmpty = [x for x in nonEmpty if x != ""]
    if len(nonEmpty) == 0:
      print("No metadata found for index", idx)
      return ""
    return nonEmpty[np.random.randint(0, len(nonEmpty))]
  
  order = np.random.permutation(np.arange(len(elements)))

  caption = ', '.join([elements[i] for i in order if elements[i] != ""])
  
  return caption.strip()

  
def generate(device, count, folder):
  metadata_all = pd.read_parquet('/Users/cameronfranz/Downloads/freesound_parquet.parquet')
  assert len(metadata_all) == 554850
  metadata_all.head()

  # print('Generating for {} epochs'.format(count / len(metadata_all)))

  model, model_config = get_pretrained_model("stabilityai/stable-audio-open-1.0")
  model.pretransform.eval()

  # fake data for now: [('id', 12314), ('audio', sf.read('audio.wav'))]
  fake_audio = sf.read('/Users/cameronfranz/Documents/Projects/AudioTrain/drum.wav')
  fake_data = [[632625, fake_audio]]

  # data_all = pd.from_
  data_all = pd.DataFrame(fake_data)
  data_all.columns = ['id', 'audio']

  data_all = data_all.merge(metadata_all, on='id', how='left')

  batch_size = 10

  for i in range(math.ceil(len(data_all) / batch_size)):
    start = i * batch_size
    end = min(start + batch_size, len(data_all))
    batch_data = data_all.iloc[start:end]

    captions = []
    audio = data_all.iloc[start:end]['audio']
    input_tensor_shape = torch.tensor(())
    for j in range(len(batch_data)):
      caption = get_caption(data_all, i)
      audio = torch.tensor(audio[0]).float()
      audio = audio.transpose(0,1).unsqueeze(0)
      with torch.no_grad():
      latents = model.pretransform.encode(audio)
  
  # 1024 * 64 * 550_000 * 0.5 / (1e9) # 18gb estimated
    

# def init_model(device):
#     model = WanDiffusionWrapper().to(device).to(torch.float32)
#     encoder = WanTextEncoder().to(device).to(torch.float32)
#     model.set_module_grad(
#         {
#             "model": False
#         }
#     )

#     scheduler = FlowMatchScheduler(
#         shift=8.0, sigma_min=0.0, extra_one_step=True)
#     scheduler.set_timesteps(num_inference_steps=50, denoising_strength=1.0)
#     scheduler.sigmas = scheduler.sigmas.to(device)

#     sample_neg_prompt = '色调艳丽，过曝，静态，细节模糊不清，字幕，风格，作品，画作，画面，静止，整体发灰，最差质量，低质量，JPEG压缩残留，丑陋的，残缺的，多余的手指，画得不好的手部，画得不好的脸部，畸形的，毁容的，形态畸形的肢体，手指融合，静止不动的画面，杂乱的背景，三条腿，背景人很多，倒着走'

#     unconditional_dict = encoder(
#         text_prompts=[sample_neg_prompt]
#     )

#     return model, encoder, scheduler, unconditional_dict


# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("--local_rank", type=int, default=-1)
#     parser.add_argument("--output_folder", type=str)
#     parser.add_argument("--caption_path", type=str)
#     parser.add_argument("--guidance_scale", type=float, default=6.0)

#     args = parser.parse_args()

#     launch_distributed_job()
#     global_rank = dist.get_rank()

#     device = torch.cuda.current_device()

#     torch.set_grad_enabled(False)
#     torch.backends.cuda.matmul.allow_tf32 = True
#     torch.backends.cudnn.allow_tf32 = True

#     model, encoder, scheduler, unconditional_dict = init_model(device=device)

#     dataset = TextDataset(args.caption_path)

#     if global_rank == 0:
#         os.makedirs(args.output_folder, exist_ok=True)

#     for index in tqdm(range(int(math.ceil(len(dataset) / dist.get_world_size()))), disable=dist.get_rank() != 0):
#         prompt_index = index * dist.get_world_size() + dist.get_rank()
#         if prompt_index >= len(dataset):
#             continue
#         prompt = dataset[prompt_index]

#         conditional_dict = encoder(
#             text_prompts=prompt
#         )

#         latents = torch.randn(
#             [1, 21, 16, 60, 104], dtype=torch.float32, device=device
#         )

#         noisy_input = []

#         for progress_id, t in enumerate(tqdm(scheduler.timesteps)):
#             timestep = t * \
#                 torch.ones([1, 21], device=device, dtype=torch.float32)

#             noisy_input.append(latents)

#             x0_pred_cond = model(
#                 latents, conditional_dict, timestep
#             )

#             x0_pred_uncond = model(
#                 latents, unconditional_dict, timestep
#             )

#             x0_pred = x0_pred_uncond + args.guidance_scale * (
#                 x0_pred_cond - x0_pred_uncond
#             )

#             flow_pred = model._convert_x0_to_flow_pred(
#                 scheduler=scheduler,
#                 x0_pred=x0_pred.flatten(0, 1),
#                 xt=latents.flatten(0, 1),
#                 timestep=timestep.flatten(0, 1)
#             ).unflatten(0, x0_pred.shape[:2])

#             latents = scheduler.step(
#                 flow_pred.flatten(0, 1),
#                 scheduler.timesteps[progress_id] * torch.ones(
#                     [1, 21], device=device, dtype=torch.long).flatten(0, 1),
#                 latents.flatten(0, 1)
#             ).unflatten(dim=0, sizes=flow_pred.shape[:2])

#         noisy_input.append(latents)

#         noisy_inputs = torch.stack(noisy_input, dim=1)

#         noisy_inputs = noisy_inputs[:, [0, 36, 44, -1]]

#         stored_data = noisy_inputs

#         torch.save(
#             {prompt: stored_data.cpu().detach()},
#             os.path.join(args.output_folder, f"{prompt_index:05d}.pt")
#         )

#     dist.barrier()


# if __name__ == "__main__":
#     main()
