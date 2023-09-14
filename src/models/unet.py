from collections import OrderedDict
import torch
import torch.nn as nn

#### MY MODEL :
#  1,2,3 - shuffle
#  3,2,1 - 

class UNet(nn.Module):

    def __init__(self, lev,
                       in_channels=3, 
                       out_channels=1, 
                       init_features=32,
                       pooling_type=nn.MaxPool1d):
        super(UNet, self).__init__()

        features = init_features
        self.lev = lev
        self.encoder1 = UNet._block(in_channels, features, name="enc1")
        self.pool1 = pooling_type(kernel_size=2, stride=2)
        self.encoder2 = UNet._block(features, features * 2, name="enc2")
        self.pool2 = pooling_type(kernel_size=2, stride=2)
        self.encoder3 = UNet._block(features * 2, features * 4, name="enc3")
        self.pool3 = pooling_type(kernel_size=2, stride=2)
        self.encoder4 = UNet._block(features * 4, features * 8, name="enc4")
        self.pool4 = pooling_type(kernel_size=2, stride=2) # size 9 

        # VIEW :

        self.bottleneck = UNet._block(features * 8, features * 16, name="bottleneck")

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet._block((features * 8) * 2, features * 8, name="dec4")
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet._block((features * 4) * 2, features * 4, name="dec3")
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet._block((features * 2) * 2, features * 2, name="dec2")
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet._block(features * 2, features, name="dec1")

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        # BOTTLE NECK
        self.flatten = nn.Flatten()
        self.SIZE = lev//2//2//2
        self.NUM_FEATURES = features * 8
        self.fc = nn.Linear(self.SIZE*self.NUM_FEATURES, self.SIZE*self.NUM_FEATURES)
        self.selu = nn.SiLU()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        flat = self.flatten(enc4)
        intermediary = self.selu(self.fc(flat))
        bottleneck = intermediary.view(-1, self.NUM_FEATURES, self.SIZE)
#        dec4 = self.upconv4(bottleneck)
#        print('dec4', dec4.shape)
        dec4 = torch.cat((bottleneck, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _lin_bloc(in_channel, features, name):
        nn.Sequential()

    @staticmethod
    def _block(in_channels, features, name):
        return nn.Sequential(
            OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "selu1", nn.SiLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "selu2", nn.SiLU(inplace=True)),
                ]
            )
        )

##########################################################################################################################################################################################

class UNet_no_last_renorm(nn.Module):
    # No normalisation for the last layers

    def __init__(self, lev,
                       in_channels=3, 
                       out_channels=1, 
                       init_features=32,
                       pooling_type=nn.MaxPool1d):
        super(UNet_no_last_renorm, self).__init__()

        features = init_features
        self.lev = lev
        self.encoder1 = UNet_no_last_renorm._block(in_channels, features, name="enc1", renorm=True)
        self.pool1 = pooling_type(kernel_size=2, stride=2)
        self.encoder2 = UNet_no_last_renorm._block(features, features * 2, name="enc2", renorm=True)
        self.pool2 = pooling_type(kernel_size=2, stride=2)
        self.encoder3 = UNet_no_last_renorm._block(features * 2, features * 4, name="enc3", renorm=True)
        self.pool3 = pooling_type(kernel_size=2, stride=2)
        self.encoder4 = UNet_no_last_renorm._block(features * 4, features * 8, name="enc4", renorm=True)
        self.pool4 = pooling_type(kernel_size=2, stride=2) # size 9 

        # VIEW :

        self.bottleneck = UNet_no_last_renorm._block(features * 8, features * 16, name="bottleneck", renorm=True)

        self.upconv4 = nn.ConvTranspose1d(
            features * 16, features * 8, kernel_size=2, stride=2
        )
        self.decoder4 = UNet_no_last_renorm._block((features * 8) * 2, features * 8, name="dec4", renorm=True)
        self.upconv3 = nn.ConvTranspose1d(
            features * 8, features * 4, kernel_size=2, stride=2
        )
        self.decoder3 = UNet_no_last_renorm._block((features * 4) * 2, features * 4, name="dec3", renorm=True)
        self.upconv2 = nn.ConvTranspose1d(
            features * 4, features * 2, kernel_size=2, stride=2
        )
        self.decoder2 = UNet_no_last_renorm._block((features * 2) * 2, features * 2, name="dec2", renorm=False)
        self.upconv1 = nn.ConvTranspose1d(
            features * 2, features, kernel_size=2, stride=2
        )
        self.decoder1 = UNet_no_last_renorm._block(features * 2, features, name="dec1", renorm=False)

        self.conv = nn.Conv1d(
            in_channels=features, out_channels=out_channels, kernel_size=1
        )
        # BOTTLE NECK
        self.flatten = nn.Flatten()
        self.SIZE = lev//2//2//2
        self.NUM_FEATURES = features * 8
        self.fc = nn.Linear(self.SIZE*self.NUM_FEATURES, self.SIZE*self.NUM_FEATURES)
        self.selu = nn.SiLU()

    def forward(self, x):
        enc1 = self.encoder1(x)
        enc2 = self.encoder2(self.pool1(enc1))
        enc3 = self.encoder3(self.pool2(enc2))
        enc4 = self.encoder4(self.pool3(enc3))
        
        flat = self.flatten(enc4)
        intermediary = self.selu(self.fc(flat))
        bottleneck = intermediary.view(-1, self.NUM_FEATURES, self.SIZE)
#        dec4 = self.upconv4(bottleneck)
#        print('dec4', dec4.shape)
        dec4 = torch.cat((bottleneck, enc4), dim=1)
        dec4 = self.decoder4(dec4)
        dec3 = self.upconv3(dec4)
        dec3 = torch.cat((dec3, enc3), dim=1)
        dec3 = self.decoder3(dec3)
        dec2 = self.upconv2(dec3)
        dec2 = torch.cat((dec2, enc2), dim=1)
        dec2 = self.decoder2(dec2)
        dec1 = self.upconv1(dec2)
        dec1 = torch.cat((dec1, enc1), dim=1)
        dec1 = self.decoder1(dec1)
        return self.conv(dec1)

    @staticmethod
    def _lin_bloc(in_channel, features, name):
        nn.Sequential()

    @staticmethod
    def _block(in_channels, features, name, renorm):
        if renorm:
            OD = OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm1", nn.BatchNorm1d(num_features=features)),
                    (name + "selu1", nn.SiLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "norm2", nn.BatchNorm1d(num_features=features)),
                    (name + "selu2", nn.SiLU(inplace=True)),
                ]
            )
        else:
             OD = OrderedDict(
                [
                    (
                        name + "conv1",
                        nn.Conv1d(
                            in_channels=in_channels,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "selu1", nn.SiLU(inplace=True)),
                    (
                        name + "conv2",
                        nn.Conv1d(
                            in_channels=features,
                            out_channels=features,
                            kernel_size=3,
                            padding=1,
                            bias=False,
                        ),
                    ),
                    (name + "selu2", nn.SiLU(inplace=True)),
                ]
            )           
        return nn.Sequential(OD)
