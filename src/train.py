from __future__ import print_function
import random
import torch
import torch.nn as nn
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torchvision.utils as vutils
from pathlib import Path
import utils
from models.generator import Generator
from models.discriminator import Discriminator
from datetime import datetime

def train(net_disc, net_gen, loss, n_epochs, loader, device, opt):
    img_list = []
    G_losses = []
    D_losses = []
    iters = 0

    fixed_noise = torch.randn(64, opt.nz, 1, 1, device=device)
    real_label = 1.
    fake_label = 0.

    for epoch in range(n_epochs):
        for i, data in enumerate(loader, 0):
            net_disc.zero_grad()

            # Train with real data
            real_data = data[0].to(device)
            b_size = real_data.size(0)
            label = torch.full((b_size,), real_label, dtype=torch.float, device=device)
            out = net_disc(real_data).view(-1)

            errorD_real = loss(out, label)
            errorD_real.backward()
            D_x = out.mean().item()

            # Train with fake data
            noise = torch.randn(b_size, opt.nz, 1, 1, device=device)

            fake_data = net_gen(noise)
            label.fill_(fake_label)
            out = net_disc(fake_data.detach()).view(-1)
            errorD_fake = loss(out, label)
            errorD_fake.backward()
            D_g_z1 = out.mean().item()

            errorD = errorD_fake + errorD_real

            net_disc.opt.step()

            # Update generator
            net_gen.zero_grad()
            label.fill_(real_label)

            out = net_disc(fake_data).view(-1)
            errorG = loss(out, label)

            errorG.backward()
            D_g_z2 = out.mean().item()

            net_gen.opt.step()

            # Stats
            if i % 50 == 0:
                print('[%d/%d][%d/%d]\tLoss_D: %.4f\tLoss_G: %.4f\tD(x): %.4f\tD(G(z)): %.4f / %.4f'
                    % (epoch+1, n_epochs, i, len(loader),
                        errorD.item(), errorG.item(), D_x, D_g_z1, D_g_z2))
                G_losses.append(errorG.item())
                D_losses.append(errorD.item())

            # Check how the generator is doing by saving G's output on fixed_noise
            if (iters % 500 == 0) or ((epoch == n_epochs-1) and (i == len(loader)-1)):
                with torch.no_grad():
                    fake = net_gen(fixed_noise).detach().cpu()
                img_list.append(vutils.make_grid(fake, padding=2, normalize=True))

            iters += 1

        torch.save(net_gen.state_dict(), f'{opt.output_folder}/netG_epoch_{epoch+1}.pth')
        torch.save(net_disc.state_dict(), f'{opt.output_folder}/netD_epoch_{epoch+1}.pth')
    
    return img_list, G_losses, D_losses
        
def main():
    opt = utils.parse_args()

    Path(opt.output_folder).mkdir(exist_ok=True, parents=True)

    torch.manual_seed(random.randint(1, 10000))

    loader = utils.get_loader(opt)

    device = torch.device('cuda:0' if torch.cuda.is_available() and opt.ngpu > 0 else 'cpu')

    net_gen = Generator(opt).to(device)
    net_gen.apply(utils.weights_init)

    net_disc = Discriminator(opt).to(device)
    net_disc.apply(utils.weights_init)

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f'[{current_time}] Starting training!')

    if opt.dry_run:
        opt.n_epochs = 1

    img_list, G_losses, D_losses = train(net_disc, net_gen, nn.BCELoss(), opt.n_epochs, loader, device, opt)

    current_time = datetime.now().strftime("%H:%M:%S")
    print(f'[{current_time}] Done training!')

    utils.plot_generated_imgs(Path(opt.output_folder), img_list)
    utils.plot_losses(Path(opt.output_folder), D_losses, G_losses)
    utils.plot_real_fake(Path(opt.output_folder), next(iter(loader)), img_list, device)

if __name__ == '__main__':
    main()