import torch
import torch.nn as nn

################
## Functions ###
################

# custom weight initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.01)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.01)
        m.bias.data.fill_(0)

def define_netD(cond_input_size):
    netD = _netD(cond_input_size)
    netD.apply(weights_init)
    return netD

def define_netG(noise_input_size, cond_input_size):
    netG = _netG(noise_input_size, cond_input_size)
    netG.apply(weights_init)
    return netG

################
## Classes ###
################

class _netG(nn.Module):
    def __init__(self, noise_input_size, cond_input_size):
        super(_netG, self).__init__()
        self.noise_input_size = noise_input_size
        self.cond_input_size = cond_input_size
        
        # first dense block
        # input shape [batch_size x 147]
        self.fc1 = nn.Sequential(
            nn.Linear(self.noise_input_size + self.cond_input_size, 100 * 10),
            nn.BatchNorm1d(100 * 10),
            nn.LeakyReLU(0.2, inplace=True)
        )
        # Convolutional block
        self.conv1 = nn.Sequential(
            # input shape [batch_size x 10 x 100]
            nn.ConvTranspose1d(10, 250, 13, stride=2, padding=6,
                              output_padding=1, bias=True),
            nn.BatchNorm1d(250),
            nn.LeakyReLU(0.2, inplace=True),

            # input shape [batch_size x 250 x 200]
            nn.ConvTranspose1d(250, 100, 13, stride=2, padding=6,
                              output_padding=1, bias=True),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.2, inplace=True),

             # input shape [batch_size x 100 x 400]
            nn.ConvTranspose1d(100, 1, 13, stride=1, padding=6,
                              bias=True),
            nn.BatchNorm1d(1),
            # input shape [batch_size x 1 x 400]
            nn.Tanh()
        )

    def forward(self, noise_input, cond_input):
        x = torch.cat((noise_input, cond_input), 1)
        x = self.fc1(x)
        x = x.view(x.size(0), 10, 100)
        x = self.conv1(x)
        return x


class _netD(nn.Module):
    def __init__(self, cond_input_size):
        super(_netD, self).__init__()
        self.cond_input_size = cond_input_size
        # Convolutional block
        self.conv1 = nn.Sequential(
            # input shape batch_size x 1 (number of channels) x 400 (length of pulse)
            nn.Conv1d(1, 100, 13, stride=5, padding=6, bias=True),
            nn.BatchNorm1d(100),
            nn.LeakyReLU(0.2, inplace=True),
            
            # shape [batch_size x 100 x 80]
            nn.Conv1d(100, 250, 13, stride=5, padding=6, bias=True),
            nn.BatchNorm1d(250),
            nn.LeakyReLU(0.2, inplace=True),
            
            # shape [batch_size x 250 x 16]
            nn.Conv1d(250, 300, 13, stride=4, padding=6, bias=True),
            nn.BatchNorm1d(300),
            nn.LeakyReLU(0.2, inplace=True)
            # shape [batch_size x 300 x 4]
        )
        # after flatten 300 * 4 + 47 (conditional input size)
        # Dense block
        self.fc1 = nn.Sequential(
            nn.Linear(1200 + self.cond_input_size, 200),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Linear(200,1),
            nn.Sigmoid()
        )

    def forward(self, pls_input, cond_input):
        x = self.conv1(pls_input)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, cond_input), 1)
        x = self.fc1(x)
        return x