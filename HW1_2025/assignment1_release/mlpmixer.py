from torch import nn
import torch
import math
import matplotlib.pyplot as plt
import os


class PatchEmbed(nn.Module):
    """ 2D Image to Patch Embedding
    """
    def __init__(self, img_size, patch_size, in_chans=3, 
                 embed_dim=768):
        super(PatchEmbed, self).__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.grid_size = img_size // patch_size
        self.num_patches = self.grid_size * self.grid_size
        
        # Uncomment this line and replace ? with correct values
        self.proj = nn.Conv2d(in_chans, embed_dim, kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        """
        :param x: image tensor of shape [batch, channels, img_size, img_size]
        :return out: [batch. num_patches, embed_dim]
        """
        _, _, H, W = x.shape
        assert H == self.img_size, f"Input image height ({H}) doesn't match model ({self.img_size})."
        assert W == self.img_size, f"Input image width ({W}) doesn't match model ({self.img_size})."
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)  # BCHW -> BNC
        return x


class Mlp(nn.Module):
    """ MLP as used in Vision Transformer, MLP-Mixer and related networks """
    def __init__(
            self,
            in_features,
            hidden_features,
            act_layer=nn.GELU,
            drop=0.,
    ):
        super(Mlp, self).__init__()
        out_features = in_features
        hidden_features = hidden_features

        self.fc1 = nn.Linear(in_features, hidden_features, bias=True)
        self.act = act_layer()
        self.drop1 = nn.Dropout(drop)
        self.fc2 = nn.Linear(hidden_features, out_features, bias=True)
        self.drop2 = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop1(x)
        x = self.fc2(x)
        x = self.drop2(x)
        return x


class MixerBlock(nn.Module):
    """ Residual Block w/ token mixing and channel MLPs
    Based on: 'MLP-Mixer: An all-MLP Architecture for Vision' - https://arxiv.org/abs/2105.01601
    """
    def __init__(
            self, dim, seq_len, mlp_ratio=(0.5, 4.0),
            activation='gelu', drop=0., drop_path=0.):
        super(MixerBlock, self).__init__()
        act_layer = {'gelu': nn.GELU, 'relu': nn.ReLU}[activation]
        tokens_dim, channels_dim = int(mlp_ratio[0] * dim), int(mlp_ratio[1] * dim)
        self.norm1 = nn.LayerNorm(dim, eps=1e-6) # norm1 used with mlp_tokens
        self.mlp_tokens = Mlp(seq_len, tokens_dim, act_layer=act_layer, drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6) # norm2 used with mlp_channels
        self.mlp_channels = Mlp(dim, channels_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        raise NotImplementedError
    

class MLPMixer(nn.Module):
    def __init__(self, num_classes, img_size, patch_size, embed_dim, num_blocks, 
                 drop_rate=0., activation='gelu'):
        super(MLPMixer, self).__init__()
        self.patchemb = PatchEmbed(img_size=img_size, 
                                   patch_size=patch_size, 
                                   in_chans=3,
                                   embed_dim=embed_dim)
        self.blocks = nn.Sequential(*[
            MixerBlock(
                dim=embed_dim, seq_len=self.patchemb.num_patches, 
                activation=activation, drop=drop_rate)
            for _ in range(num_blocks)])
        self.norm = nn.LayerNorm(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes)
        self.num_classes = num_classes
        self.apply(self.init_weights)
    
    def init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Conv2d):
            nn.init.kaiming_normal_(module.weight)
            nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            nn.init.ones_(module.weight)
            nn.init.zeros_(module.bias)

    def forward(self, images):
        """ MLPMixer forward
        :param images: [batch, 3, img_size, img_size]
        """
        # step1: Go through the patch embedding
        x = self.patchemb(images)
        # step 2 Go through the mixer blocks
        x = self.blocks(x)
        # step 3 go through layer norm
        x = self.norm(x)
        # step 4 Global averaging spatially
        x = x.mean(dim =1)
        # Classification
        x = self.head(x)
        return x
    
    def visualize(self, logdir):
        """ 
        Visualize the token mixer layer 
        in the desired directory 
        """
        
        if not.os.path.exists(logdir):
            os.makedirs(logdir)
        
        for i  , block in enumerate(self.blocks):
            weights = block.mlp_tokens.fc1.weight.data.cpu()
            num_kernels = weights.shape[0]
            fig , axes = plt.subplots(1 , num_kernels , figsize = (num_kernels , 1) )
            for j in range(num_kernels):
                ax = axes[j]
                kernel = weights[j , :].view(self.patchemb.num_patches , -1)        # reshape to 2D
                kernel = (kernel - kernel.min()) / ( kernel.max() - kernel.min() )  # Normalize
                ax.imshow(kernel , cmap = 'gray')
                ax.axis('off')
            plt.savefig(os.path.join(logdir , f'block_{i}_token_mixer.png'))
            plt.close(fig)
        
        
        
 
