import torch.nn as nn
import torch.optim as optim

class Generator(nn.Module):
    def __init__(self, opt):
        super(Generator, self).__init__()
        self.ngpu = opt.ngpu
        self.sequential = nn.Sequential(
            nn.ConvTranspose2d(opt.nz, opt.ngf * 8, 4, 1, 0, bias=False),
            nn.BatchNorm2d(opt.ngf * 8),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(opt.ngf * 8, opt.ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 4),
            nn.ReLU(True),
            
            nn.ConvTranspose2d(opt.ngf * 4, opt.ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf * 2),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( opt.ngf * 2, opt.ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(opt.ngf),
            nn.ReLU(True),
            
            nn.ConvTranspose2d( opt.ngf, opt.nc, 4, 2, 1, bias=False),
            nn.Tanh()
        )
        self.opt = optim.Adam(self.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

    def forward(self, input):
        return self.sequential(input)
