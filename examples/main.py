import torch
import numpy as np
import argparse
import sys
sys.path.append('../')
from diffusionmodels.rectifiedflow import RectifiedFlow
from datasets.godmodeanimation import GodModeAnimation
from diffusers import DiffusionPipeline, DDIMScheduler, DDPMScheduler
from torch.utils.data import DataLoader
import platform
from accelerate import Accelerator
from tqdm import tqdm
import logging
from neuralnets.unet import Unet
import matplotlib.pyplot as plt
import os
import imageio


parser = argparse.ArgumentParser(description='main')
parser.add_argument('--diffusionmodel', type=str, default='RectifiedFlow', choices=['DDPM', 'DDIM', 'RectifiedFlow'])
parser.add_argument('--dataset', type=str, default='godmodeanimation/sword_slash_dataset')
parser.add_argument('--pretrained_model', type=str, default="cerspense/zeroscope_v2_576w")
parser.add_argument('--batch_size', type=int, default=2)
parser.add_argument('--num_workers', type=int, default=0)
parser.add_argument('--epochs', type=int, default=10)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--load_checkpoint', action='store_true')
parser.add_argument('--train_or_test', type=str, default="train", choices=["train", "test"])
parser.add_argument('--precision', type=str, default='fp16', choices=['fp16'])
args = parser.parse_args()

logging.basicConfig(level=logging.INFO)


def main():
    output_folder = f"results/{args.dataset.split('/')[-1]}_{args.diffusionmodel}"
    if not os.path.exists(f"{output_folder}/gif"):
        os.makedirs(f"{output_folder}/gif")
    
    diffusionmodel = eval(args.diffusionmodel)()
    dataset = args.dataset
    data_folder = f"../../../repo/data/{dataset}/npz"

    accelerator = Accelerator(mixed_precision=args.precision)
    device = accelerator.device

    pretrained_model = args.pretrained_model
    pipe = DiffusionPipeline.from_pretrained(pretrained_model, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    tokenizer = pipe.tokenizer
    text_encoder = pipe.text_encoder
    vae = pipe.vae
    vae.requires_grad_(False)
    text_encoder.requires_grad_(False)

    dataset = GodModeAnimation(data_folder, tokenizer, train_or_test='train')
    test_dataset = GodModeAnimation(data_folder, tokenizer, train_or_test='test')
    dataloader = DataLoader(dataset, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers, pin_memory=True, drop_last=True)
    test_dataloader = DataLoader(test_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)

    unet = Unet(resolution=32)
    
    optimizer = torch.optim.AdamW(unet.parameters(), lr=args.learning_rate, weight_decay=0)
    
    if args.load_checkpoint:
        model_path = output_folder+'/model.pth'
        if os.path.exists(model_path):
            checkpoint = torch.load(model_path, map_location=lambda storage, loc: storage)
            unet.load_state_dict(checkpoint['model'])
            optimizer.load_state_dict(checkpoint['optimizer'])
        logging.info('===> Checkpoint loaded.')


    if platform.system() not in ['Darwin']:
        unet, optimizer, dataloader = accelerator.prepare(unet, optimizer, dataloader)
    


    if args.train_or_test == 'test':
        unet.eval()

        images = []
        for iter, data in enumerate(tqdm(test_dataloader)):
            
            video, _, text = data
            video = video.to(device)
            batch_size = video.shape[0]
            save_text = text[0].replace(' ', '_')
            
            with torch.no_grad():
                text_input = tokenizer(text, padding="max_length", max_length=tokenizer.model_max_length, truncation=True, return_tensors="pt")
                text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
                max_length = text_input.input_ids.shape[-1]
                uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
                uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
                text_embeddings = torch.cat([uncond_embeddings, text_embeddings])

            
            x0 = np.random.randn(*video.shape).astype(np.float32)
            x0 = torch.from_numpy(x0).float().to(device)
            out = diffusionmodel.sample(unet, x0, text_embeddings)
            print('out:', text, out.shape, out.min(), out.max())
            out = out[0]
            decoded_latents = vae.decode(out.half() / diffusionmodel.scaled_vae_latent_factor).sample.float().detach().cpu().numpy()
            print('decoded_latents:', decoded_latents.shape, decoded_latents.min(), decoded_latents.max())
            decoded_latents = (decoded_latents + 1) / 2
            out = np.clip(decoded_latents, 0, 1)
            
            out = (out * 255).astype(np.uint8).transpose(0, 2, 3, 1)
            imageio.mimsave(f'{output_folder}/gif/{save_text}.gif', out, loop=0)

            if iter + 1 == 2:
                break
        
        return
    

    logging.info("===> Start training...")
    num_train_timesteps = diffusionmodel.num_train_timesteps
    losses = []
    for epoch in tqdm(range(args.epochs)):
        unet.train()
        for iter, data in enumerate((dataloader)):
            
            video, text_input = data    # text_input instead of text, just for same sequence length
            
            num_frames = video.shape[1]
            batch_size = video.shape[0]
            video = video.to(device)
            
            with torch.no_grad():
                text_embeddings = text_encoder(text_input.input_ids.to(device))[0]
                max_length = text_input.input_ids.shape[-1]
                # if random.random() < args.cfg_probability:
                #     uncond_input = tokenizer([""] * batch_size, padding="max_length", max_length=max_length, return_tensors="pt")
                #     uncond_embeddings = text_encoder(uncond_input.input_ids.to(device))[0]
                #     # print('uncond_embeddings:', uncond_embeddings, uncond_embeddings.shape)
                #     text_embeddings = uncond_embeddings  # Set conditions to None for unconditional


            t = torch.randint(0, num_train_timesteps, (video.shape[0], )).to(device)
            noise = torch.randn_like(video).to(device)

            x_t = diffusionmodel.add_noise(video, t, noise)
            
            pred = unet(x_t, t, text_embeddings)
            loss = diffusionmodel.loss(pred, (video - noise))
            losses.append(loss.item())
            optimizer.zero_grad()
            accelerator.backward(loss)

            params_to_clip = unet.parameters()
            accelerator.clip_grad_norm_(params_to_clip, 1.0)
            
            optimizer.step()

            if iter % 1000 == 0:
                logging.info('loss: {:.4f}'.format(loss.item()))
                # break
            

        plt.figure(1)
        plt.plot(losses)
        plt.savefig(output_folder+'/loss.jpg')
        plt.clf()
        np.savetxt(output_folder+'/losses.txt', losses)
        
        model_save = accelerator.unwrap_model(unet)
        checkpoint = {
            "model": model_save.state_dict(),
            "optimizer": optimizer.state_dict(),
        }
        torch.save(checkpoint, output_folder+'/model.pth')
    



if __name__ == "__main__":
    main()


