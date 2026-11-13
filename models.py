"""
models.py - The two models we compare.

    1. SmallResNet : a CNN-based model (a small ResNet)
    2. ViT         : a Transformer-based model (a small Vision Transformer)

Both are sized to have roughly the same number of parameters (~3M), so the
comparison is fair. Both take a CIFAR-10 image of shape (3, 32, 32) and
output 10 class scores (logits).

Each model also has a `features(x)` method that returns the vector just
before the final classifier. We use it later to visualise what each model
has learned (t-SNE plot in compare.py).
"""

import torch
import torch.nn as nn


# =============================================================================
# 1. CNN-based model: Small ResNet
# =============================================================================

class ResidualBlock(nn.Module):
    """
    A basic ResNet block:

        x --> Conv3x3 -> BN -> ReLU -> Conv3x3 -> BN --> (+) -> ReLU --> out
        |                                                 ^
        +----------------- shortcut ----------------------+

    The shortcut lets the block learn only the "difference" (residual) from
    its input, which makes deeper networks much easier to train.
    """

    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()

        # First 3x3 conv. If stride=2, it halves the height and width.
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3,
                               stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)

        # Second 3x3 conv keeps the size the same.
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3,
                               stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.relu = nn.ReLU(inplace=True)

        # If the shape changes (fewer pixels or more channels), the shortcut
        # needs a 1x1 conv so that it can be added to the main path.
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1,
                          stride=stride, bias=False),
                nn.BatchNorm2d(out_channels),
            )
        else:
            self.shortcut = nn.Identity()

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out = out + self.shortcut(x)          # the residual connection
        return self.relu(out)


class SmallResNet(nn.Module):
    """
    Structure (output shape shown for a 32x32 input):

        Input                                   3 x 32 x 32
        Stem:   Conv3x3(64) + BN + ReLU        64 x 32 x 32
        Stage1: 2 x ResidualBlock(64)          64 x 32 x 32
        Stage2: 2 x ResidualBlock(128), s=2   128 x 16 x 16
        Stage3: 2 x ResidualBlock(256), s=2   256 x  8 x  8
        Global average pooling                256
        Fully connected                        10

    Total: ~2.8M parameters.
    """

    def __init__(self, num_classes=10):
        super().__init__()

        # "Stem": the first conv layer that turns RGB into 64 feature maps.
        self.stem = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )

        # Three stages. Each new stage halves the image size and doubles the
        # channels, so the network sees larger regions as it goes deeper.
        self.stage1 = nn.Sequential(ResidualBlock(64, 64),
                                    ResidualBlock(64, 64))
        self.stage2 = nn.Sequential(ResidualBlock(64, 128, stride=2),
                                    ResidualBlock(128, 128))
        self.stage3 = nn.Sequential(ResidualBlock(128, 256, stride=2),
                                    ResidualBlock(256, 256))

        # Average each 8x8 feature map into one number -> a 256-d vector.
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Final classifier.
        self.head = nn.Linear(256, num_classes)

    def features(self, x):
        """Return the 256-d feature vector (before the classifier)."""
        x = self.stem(x)
        x = self.stage1(x)
        x = self.stage2(x)
        x = self.stage3(x)
        x = self.pool(x)              # (B, 256, 1, 1)
        return torch.flatten(x, 1)    # (B, 256)

    def forward(self, x):
        return self.head(self.features(x))


# =============================================================================
# 2. Transformer-based model: Vision Transformer (ViT)
# =============================================================================

class PatchEmbedding(nn.Module):
    """
    Cut the image into non-overlapping patches and turn each one into a vector
    (a "token").

    With a 32x32 image and patch_size=4 there are (32/4) * (32/4) = 64 patches.
    A Conv2d whose kernel size equals its stride (both = patch_size) does this
    in one step: it is the same as flattening each patch and applying a
    linear layer.
    """

    def __init__(self, patch_size=4, in_channels=3, embed_dim=256):
        super().__init__()
        self.proj = nn.Conv2d(in_channels, embed_dim,
                              kernel_size=patch_size, stride=patch_size)

    def forward(self, x):
        x = self.proj(x)                  # (B, 256, 8, 8)
        x = x.flatten(2)                  # (B, 256, 64)
        return x.transpose(1, 2)          # (B, 64, 256): 64 tokens of size 256


class TransformerBlock(nn.Module):
    """
    One Transformer encoder block (the "pre-norm" version used in ViT):

        x --> LayerNorm -> Multi-Head Self-Attention --> (+) --> y
        |                                                 ^
        +-------------------------------------------------+

        y --> LayerNorm -> MLP (Linear-GELU-Linear) -----> (+) --> out
        |                                                   ^
        +---------------------------------------------------+

    Self-attention lets EVERY patch look at EVERY other patch in a single
    layer, so the model has a global view right from the first block.
    A CNN only sees a small local neighbourhood per layer.
    """

    def __init__(self, dim, num_heads, mlp_dim, dropout=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, num_heads, dropout=dropout,
                                          batch_first=True)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        # Self-attention: query, key and value all come from the same tokens.
        h = self.norm1(x)
        attn_out, _ = self.attn(h, h, h, need_weights=False)
        x = x + attn_out                  # residual connection 1

        x = x + self.mlp(self.norm2(x))   # residual connection 2
        return x


class ViT(nn.Module):
    """
    Structure:

        Input image                              3 x 32 x 32
        Patch embedding (4x4 patches)            64 tokens x 256
        Prepend a learnable [CLS] token          65 tokens x 256
        Add learnable position embeddings        65 tokens x 256
        6 x TransformerBlock (8 heads, MLP 512)  65 tokens x 256
        LayerNorm, take the [CLS] token          256
        Fully connected                          10

    Total: ~3.2M parameters.

    Note: a Transformer has no built-in idea of "which patch is next to which"
    (unlike convolution). It has to learn this from the position embeddings
    and the data. This is the main reason ViTs need more data or training to
    match CNNs on small datasets.
    """

    def __init__(self, image_size=32, patch_size=4, num_classes=10,
                 dim=256, depth=6, num_heads=8, mlp_dim=512, dropout=0.1):
        super().__init__()

        num_patches = (image_size // patch_size) ** 2      # 64

        self.patch_embed = PatchEmbedding(patch_size, 3, dim)

        # The [CLS] token is an extra learnable vector. After the Transformer,
        # its output summarises the whole image and is used for classifying.
        self.cls_token = nn.Parameter(torch.zeros(1, 1, dim))

        # One learnable position vector per token (64 patches + 1 CLS).
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, dim))
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)

        self.dropout = nn.Dropout(dropout)

        # Stack of Transformer blocks.
        self.blocks = nn.Sequential(*[
            TransformerBlock(dim, num_heads, mlp_dim, dropout)
            for _ in range(depth)
        ])

        self.norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, num_classes)

    def features(self, x):
        """Return the 256-d [CLS] feature vector (before the classifier)."""
        batch_size = x.shape[0]

        x = self.patch_embed(x)                              # (B, 64, 256)

        cls = self.cls_token.expand(batch_size, -1, -1)      # (B, 1, 256)
        x = torch.cat([cls, x], dim=1)                       # (B, 65, 256)

        x = x + self.pos_embed                               # add positions
        x = self.dropout(x)

        x = self.blocks(x)                                   # (B, 65, 256)
        x = self.norm(x)
        return x[:, 0]                                       # the CLS token

    def forward(self, x):
        return self.head(self.features(x))