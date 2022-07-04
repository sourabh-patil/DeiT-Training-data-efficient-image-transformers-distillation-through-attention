from distutils import dist
import torch
import torch.nn as nn
import cv2
import torch.nn.functional as F


class PatchEmbed(nn.Module):
    """Patch Embedding or Tokenizer.

    Parameters
    ----------
    kernel_size : int
        Size of kernel for 2d convolutions.

    stride : int
        Stride used for 2d convolutions.

    padding : int
        Number of pixels padded for 2d convolutions.

    pooling_kernel_size : int
        Size of kernel used for 2d pooling after 2d convolutions.

    pooling_stride : int
        Stride used for 2d pooling.

    pooling_padding : int
        Number of pixels padded for 2d pooling.

    n_conv_layers : int
        Number of convolutions used in Sequential manner to tokenize image.

    n_input_channels : int 
        Number of channels of input image.

    embed_dim : int 
        Embedding dimension for each patch (here we are encoding it in channel dimension of 2d convolution).

    in_planes : int 
        Number of channels of 2d convolutions for middle convolution operations.

    activation : torch.nn class
        Activations to be used after 2d convolutions.

    max_pool : bool
        Flag to use 2d max pool.

    conv_bias : bool 
        Flag to use bias in 2d conv.

    """
    def __init__(self,
                 kernel_size = 7, stride = 2, padding = 3,
                 pooling_kernel_size=3, pooling_stride=2, pooling_padding=1,
                 n_conv_layers=4,
                 n_input_channels=3,
                 embed_dim=384,
                 in_planes=64,
                 activation=nn.ReLU,
                 max_pool=True,
                 conv_bias=False):
        super(PatchEmbed, self).__init__()

        n_filter_list = [n_input_channels] + \
                        [in_planes for _ in range(n_conv_layers - 1)] + \
                        [embed_dim]

        self.conv_layers = nn.Sequential(
            *[nn.Sequential(
                nn.Conv2d(n_filter_list[i], n_filter_list[i + 1],
                          kernel_size=(kernel_size, kernel_size),
                          stride=(stride, stride),
                          padding=(padding, padding), bias=conv_bias),
                nn.Identity() if activation is None else activation(),
                nn.MaxPool2d(kernel_size=pooling_kernel_size,
                             stride=pooling_stride,
                             padding=pooling_padding) if max_pool else nn.Identity()
            )
                for i in range(n_conv_layers)
            ])

        self.flattener = nn.Flatten(2, 3)
        self.apply(self.init_weight)

    def sequence_length(self, n_channels=3, height=1316, width=2652):
        return self.forward(torch.zeros((1, n_channels, height, width))).shape[1]

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_input_channels, img_height, img_width)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, 66, embed_dim)`.  Reason: after 4 layers of [2d conv + max pooling] we get 66 number of patches as we combine spatial dimensions 
        """
        # print(self.conv_layers(x).transpose(-2,-1).shape)
        return self.flattener(self.conv_layers(x)).transpose(-2, -1)

    @staticmethod
    def init_weight(m):
        if isinstance(m, nn.Conv2d):
            nn.init.kaiming_normal_(m.weight)


class Attention(nn.Module):
    """Attention mechanism.

    Parameters
    ----------
    dim : int
        The input and out dimension of per token features.

    n_heads : int
        Number of attention heads.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    attn_p : float
        Dropout probability applied to the query, key and value tensors.

    proj_p : float
        Dropout probability applied to the output tensor.


    Attributes
    ----------
    scale : float
        Normalizing consant for the dot product.

    qkv : nn.Linear
        Linear projection for the query, key and value.

    proj : nn.Linear
        Linear mapping that takes in the concatenated output of all attention
        heads and maps it into a new space.

    attn_drop, proj_drop : nn.Dropout
        Dropout layers.
    """
    def __init__(self, dim, n_heads=12, qkv_bias=True, attn_p=0., proj_p=0.):
        super().__init__()
        self.n_heads = n_heads
        self.dim = dim
        self.head_dim = dim // n_heads
        self.scale = self.head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_p)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        n_samples, n_tokens, dim = x.shape
        # print(x.shape)

        if dim != self.dim:
            raise ValueError

        qkv = self.qkv(x)  # (n_samples, n_patches + 1, 3 * dim)
        # print(qkv.shape) #################################################################################
        qkv = qkv.reshape(
                n_samples, n_tokens, 3, self.n_heads, self.head_dim
        )  # (n_smaples, n_patches + 1, 3, n_heads, head_dim)
        qkv = qkv.permute(
                2, 0, 3, 1, 4
        )  # (3, n_samples, n_heads, n_patches + 1, head_dim)

        q, k, v = qkv[0], qkv[1], qkv[2]
        k_t = k.transpose(-2, -1)  # (n_samples, n_heads, head_dim, n_patches + 1)
        dp = (
           q @ k_t
        ) * self.scale # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = dp.softmax(dim=-1)  # (n_samples, n_heads, n_patches + 1, n_patches + 1)
        attn = self.attn_drop(attn)

        weighted_avg = attn @ v  # (n_samples, n_heads, n_patches +1, head_dim)
        weighted_avg = weighted_avg.transpose(
                1, 2
        )  # (n_samples, n_patches + 1, n_heads, head_dim)
        weighted_avg = weighted_avg.flatten(2)  # (n_samples, n_patches + 1, dim)

        x = self.proj(weighted_avg)  # (n_samples, n_patches + 1, dim)
        x = self.proj_drop(x)  # (n_samples, n_patches + 1, dim)

        return x


class MLP(nn.Module):
    """Multilayer perceptron.

    Parameters
    ----------
    in_features : int
        Number of input features.

    hidden_features : int
        Number of nodes in the hidden layer.

    out_features : int
        Number of output features.

    p : float
        Dropout probability.

    Attributes
    ----------
    fc : nn.Linear
        The First linear layer.

    act : nn.GELU
        GELU activation function.

    fc2 : nn.Linear
        The second linear layer.

    drop : nn.Dropout
        Dropout layer.
    """
    def __init__(self, in_features, hidden_features, out_features, p=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(p)

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, in_features)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches +1, out_features)`
        """
        x = self.fc1(
                x
        ) # (n_samples, n_patches + 1, hidden_features)
        x = self.act(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.fc2(x)  # (n_samples, n_patches + 1, hidden_features)
        x = self.drop(x)  # (n_samples, n_patches + 1, hidden_features)

        return x


class Block(nn.Module):
    """Transformer block.

    Parameters
    ----------
    dim : int
        Embeddinig dimension.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension size of the `MLP` module with respect
        to `dim`.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    norm1, norm2 : LayerNorm
        Layer normalization.

    attn : Attention
        Attention module.

    mlp : MLP
        MLP module.
    """
    def __init__(self, dim, n_heads, mlp_ratio=4.0, qkv_bias=True, p=0., attn_p=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(
                dim,
                n_heads=n_heads,
                qkv_bias=qkv_bias,
                attn_p=attn_p,
                proj_p=p
        )
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        hidden_features = int(dim * mlp_ratio)
        self.mlp = MLP(
                in_features=dim,
                hidden_features=hidden_features,
                out_features=dim,
        )

    def forward(self, x):
        """Run forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.

        Returns
        -------
        torch.Tensor
            Shape `(n_samples, n_patches + 1, dim)`.
        """
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))

        return x


class CCT_tokenizer_ViT(nn.Module):
    """Simplified implementation of the Vision transformer.

    Parameters
    ----------
    img_height : int
        Height of input image.

    img_width : int
        Width of input image.

    in_chans : int
        Number of input channels.

    n_classes : int
        Number of classes.

    embed_dim : int
        Dimensionality of the token/patch embeddings.

    depth : int
        Number of blocks.

    n_heads : int
        Number of attention heads.

    mlp_ratio : float
        Determines the hidden dimension of the `MLP` module.

    qkv_bias : bool
        If True then we include bias to the query, key and value projections.

    p, attn_p : float
        Dropout probability.

    Attributes
    ----------
    patch_embed : PatchEmbed
        Instance of `PatchEmbed` layer (Tokenizer) which applies conv layers to tokenize input image.

    cls_token : nn.Parameter
        Learnable parameter that will represent the first token in the sequence.
        It has `embed_dim` elements.
    
    distill_token : nn.Parameters
        Learnable parameter that will represent the last token in the sequence.
        It also has 'embed_dim' elements.

    pos_emb : nn.Parameter
        Positional embedding of the cls token + all the patches.
        It has `(1 + n_patches + 1) * embed_dim` elements.

    pos_drop : nn.Dropout
        Dropout layer.

    blocks : nn.ModuleList
        List of `Block` modules.

    norm : nn.LayerNorm
        Layer normalization.
    """


    def __init__(
            self,
            img_height = 1316,
            img_width = 2632,
            in_chans=3,
            n_classes=1000,
            embed_dim=768,
            depth=12,
            n_heads=12,
            mlp_ratio=2.,
            qkv_bias=True,
            p=0.,
            attn_p=0.,
            ):
        super().__init__()

        # self.patch_size = patch_size

        self.num_patches = 66

        self.embed_dim = embed_dim

        self.patch_embed = PatchEmbed(embed_dim=self.embed_dim)                   
        

        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.distill_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(
                torch.zeros(1, 1 + self.num_patches + 1, embed_dim)
        )
        self.pos_drop = nn.Dropout(p=p)

        self.blocks = nn.ModuleList(
            [
                Block(
                    dim=embed_dim,
                    n_heads=n_heads,
                    mlp_ratio=mlp_ratio,
                    qkv_bias=qkv_bias,
                    p=p,
                    attn_p=attn_p,
                )
                for _ in range(depth)
            ]
        )

        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)
        self.head = nn.Linear(embed_dim, n_classes)
        self.distill_mlp = nn.Linear(embed_dim, n_classes)


    def forward(self, x):
        """Run the forward pass.

        Parameters
        ----------
        x : torch.Tensor
            Shape `(n_samples, in_chans, img_height, img_width)`.

        Returns
        -------
        logits : torch.Tensor
            Logits over all the classes - `(n_samples, n_classes)`.
        """
        n_samples = x.shape[0]

        # n_patches = x.shape[1]

        # patch_size = x.shape[3]

        # x = torch.reshape(x, (-1,3,self.patch_size,self.patch_size))

        # print(x.shape)

        x = self.patch_embed(x)

        # print(x.shape)

        cls_token = self.cls_token.expand(
                n_samples, -1, -1
        )  # (n_samples, 1, embed_dim)

        # print(cls_token.shape)

        distill_token = self.distill_token.expand(
            n_samples, -1, -1
        ) # (n_samples, 1, embed_dim)

        # print(distill_token.shape)

        # print(cls_token.shape)

        x = torch.cat((cls_token, x), dim=1)  # (n_samples, 1 + n_patches, embed_dim)
        x = torch.cat((x, distill_token), dim=1) # (n_samples, 1 + n_patches + 1, embed_dim)

        # print(x.shape)
        # print(self.pos_embed.shape)

        x = x + self.pos_embed  # (n_samples, 1 + n_patches + 1, embed_dim)

        # print(x.shape)

        x = self.pos_drop(x)

        # print(x.shape)

        for block in self.blocks:
            x = block(x)

        x = self.norm(x)

        # print(x.shape)

        cls_token_final = x[:, 0]  # just the CLS token
        distill_token_final = x[:, -1] # just the distillation token

        student_score = self.head(cls_token_final)
        distill_score = self.distill_mlp(distill_token_final)

        return student_score, distill_score


# inp = torch.rand(2*50,3,188,188)

# p_emb = PatchEmbed(num_patches=50,patch_size=188,embed_dim=384)

# out = p_emb(inp)

# print(out.shape)

# model = VisionTransformer(num_patches=50,patch_size=188,n_classes=4,embed_dim=384,depth=6,n_heads=12,p=0.2,attn_p=0.2)

# print(sum([param.numel() for param in model.parameters()]))

# inp = torch.rand(2,50,3,188,188)

# s, d = model(inp)

# print(s.shape)
# print(d.shape)

# model = VisionTransformer(in_chans=3, n_classes=4, embed_dim=192, depth=3, n_heads=12, mlp_ratio=2, p=0.2, attn_p=0.2)

# inp = torch.rand(2,3,1316,2632)

# s, d = model(inp)

# print(s.shape)
# print(d.shape)

# print(sum([param.numel() for param in model.parameters()]))