import torch
import torch.nn as nn

class ResidualBlock(nn.Module):
    """Residual block with group normalization and configurable activation."""

    def __init__(self, channels, act_type='gelu'):
        """
        Initialize residual block.
        
        Args:
            channels: Number of input/output channels
            act_type: Activation function type ('relu', 'leakyrelu', 'gelu', 'silu')
        """
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn1 = nn.GroupNorm(8, channels)
        self.conv2 = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.bn2 = nn.GroupNorm(8, channels)

        activation_map = {
            'relu': nn.ReLU(inplace=True),
            'leakyrelu': nn.LeakyReLU(0.2, inplace=True),
            'gelu': nn.GELU(),
            'silu': nn.SiLU(inplace=True)
        }
        self.act = activation_map.get(act_type, nn.GELU())

    def forward(self, x):
        """Forward pass through residual block."""
        residual = x
        out = self.act(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += residual
        out = self.act(out)
        return out


class ChannelAttention(nn.Module):
    """Channel attention module using squeeze-and-excitation."""

    def __init__(self, channels, reduction=8):
        """
        Initialize channel attention.
        
        Args:
            channels: Number of input channels
            reduction: Reduction ratio for bottleneck
        """
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.fc = nn.Sequential(
            nn.Conv2d(channels, channels // reduction, 1, bias=False),
            nn.GELU(),
            nn.Conv2d(channels // reduction, channels, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass through channel attention."""
        avg_out = self.fc(self.avg_pool(x))
        max_out = self.fc(self.max_pool(x))
        out = avg_out + max_out
        return self.sigmoid(out) * x


class SpatialAttention(nn.Module):
    """Spatial attention module."""

    def __init__(self, kernel_size=7):
        """
        Initialize spatial attention.
        
        Args:
            kernel_size: Convolution kernel size
        """
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size,
                              padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        """Forward pass through spatial attention."""
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        y = torch.cat([avg_out, max_out], dim=1)
        y = self.conv(y)
        return self.sigmoid(y) * x


class DegradationAwareModule(nn.Module):
    """Module that incorporates degradation type information."""

    def __init__(self, channels):
        """
        Initialize degradation-aware module.
        
        Args:
            channels: Number of feature channels
        """
        super(DegradationAwareModule, self).__init__()
        self.degradation_embed = nn.Sequential(
            nn.Linear(1, 256),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(256, 512),
            nn.GELU(),
            nn.Dropout(0.1),
            nn.Linear(512, channels)
        )

        # Dual attention mechanism
        self.channel_attn = ChannelAttention(channels * 2, reduction=8)
        self.spatial_attn = SpatialAttention(kernel_size=5)

        # Fusion with residual connection
        self.fusion_conv1 = nn.Conv2d(channels * 2, channels, kernel_size=1)
        self.fusion_conv2 = nn.Conv2d(channels, channels,
                                      kernel_size=3, padding=1)
        self.norm = nn.GroupNorm(8, channels)
        self.act = nn.GELU()

    def forward(self, features, degradation_type):
        """
        Forward pass through degradation-aware module.
        
        Args:
            features: Input feature tensor
            degradation_type: Degradation type tensor
        """
        batch_size = features.size(0)
        degradation_type = degradation_type.view(batch_size, 1).float()
        degradation_embed = self.degradation_embed(degradation_type)

        degradation_embed = degradation_embed.view(
            batch_size, -1, 1, 1).expand_as(features)

        fused = torch.cat([features, degradation_embed], dim=1)
        fused = self.channel_attn(fused)

        out = self.fusion_conv1(fused)

        out = self.spatial_attn(out)

        residual = out
        out = self.act(self.norm(self.fusion_conv2(out)))
        out = out + residual

        return out


class ResidualGroup(nn.Module):
    """Group of residual blocks with mixed activations."""

    def __init__(self, channels, num_blocks=6, act_type='gelu'):
        """
        Initialize residual group.
        
        Args:
            channels: Number of channels
            num_blocks: Number of residual blocks
            act_type: Base activation type
        """
        super(ResidualGroup, self).__init__()

        self.main_blocks = nn.ModuleList()
        for i in range(num_blocks):
            # Cycle through different activation types
            if i % 3 == 0:
                curr_act_type = 'gelu'
            elif i % 3 == 1:
                curr_act_type = 'silu'
            else:
                curr_act_type = 'leakyrelu'
            self.main_blocks.append(
                ResidualBlock(channels, act_type=curr_act_type))

        self.fusion = nn.Conv2d(channels, channels, kernel_size=3, padding=1)
        self.act = nn.GELU() if act_type == 'gelu' else nn.SiLU(inplace=True)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        """Forward pass through residual group."""
        residual = x
        for block in self.main_blocks:
            x = block(x)
        x = self.norm(self.fusion(x))
        return self.act(x + residual)


class FFNBlock(nn.Module):
    """Feed-forward network block for feature enhancement."""

    def __init__(self, channels, expansion_ratio=2.66):
        """
        Initialize FFN block.
        
        Args:
            channels: Number of input/output channels
            expansion_ratio: Expansion ratio for hidden dimension
        """
        super(FFNBlock, self).__init__()
        hidden_dim = int(channels * expansion_ratio)
        self.conv1 = nn.Conv2d(channels, hidden_dim, kernel_size=1)
        self.act = nn.GELU()
        self.conv2 = nn.Conv2d(hidden_dim, channels, kernel_size=1)
        self.norm = nn.GroupNorm(8, channels)

    def forward(self, x):
        """Forward pass through FFN block."""
        residual = x
        x = self.conv1(x)
        x = self.act(x)
        x = self.conv2(x)
        x = self.norm(x)
        return x + residual


class PromptIR(nn.Module):
    """Main PromptIR model for image restoration."""

    def __init__(self, in_channels=3, out_channels=3,
                 base_channels=64, num_blocks=12):
        """
        Initialize PromptIR model.
        
        Args:
            in_channels: Number of input channels
            out_channels: Number of output channels
            base_channels: Base number of channels
            num_blocks: Total number of processing blocks
        """
        super(PromptIR, self).__init__()

        # Initial feature extraction
        self.initial = nn.Sequential(
            nn.Conv2d(in_channels, base_channels, kernel_size=3, padding=1),
            nn.GELU()
        )

        # Encoder with more capacity
        self.down1 = nn.Sequential(
            nn.Conv2d(base_channels, base_channels * 2,
                      kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResidualBlock(base_channels * 2, act_type='gelu')
        )

        self.down2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels * 4,
                      kernel_size=3, stride=2, padding=1),
            nn.GELU(),
            ResidualBlock(base_channels * 4, act_type='gelu')
        )

        self.degradation_module = DegradationAwareModule(base_channels * 4)

        # Main processing blocks organized in groups
        self.num_groups = 3
        self.blocks_per_group = num_blocks // self.num_groups

        self.groups = nn.ModuleList([
            ResidualGroup(base_channels * 4, num_blocks=self.blocks_per_group)
            for _ in range(self.num_groups)
        ])

        # FFN blocks for feature enhancement
        self.ffn_blocks = nn.ModuleList([
            FFNBlock(base_channels * 4) for _ in range(2)
        ])

        # Global channel and spatial attention
        self.channel_attention = ChannelAttention(
            base_channels * 4, reduction=8)
        self.spatial_attention = SpatialAttention(kernel_size=5)

        # Decoder with skip connections and better upsampling
        self.up1 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 4, base_channels * 2,
                               kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResidualBlock(base_channels * 2, act_type='gelu')
        )

        # Skip connection fusion
        self.fusion1 = nn.Sequential(
            nn.Conv2d(base_channels * 4, base_channels * 2, kernel_size=1),
            nn.GELU()
        )

        self.up2 = nn.Sequential(
            nn.ConvTranspose2d(base_channels * 2, base_channels,
                               kernel_size=4, stride=2, padding=1),
            nn.GELU(),
            ResidualBlock(base_channels, act_type='gelu')
        )

        # Skip connection fusion
        self.fusion2 = nn.Sequential(
            nn.Conv2d(base_channels * 2, base_channels, kernel_size=1),
            nn.GELU()
        )

        # Final refinement
        self.refine = nn.Sequential(
            nn.Conv2d(base_channels, base_channels,
                      kernel_size=3, padding=1),
            nn.GELU(),
            nn.Conv2d(base_channels, base_channels,
                      kernel_size=3, padding=1),
            nn.GELU()
        )

        # Output layer
        self.output = nn.Conv2d(base_channels, out_channels,
                                kernel_size=3, padding=1)

        # Learnable residual scaling parameter with better initialization
        self.residual_scale = nn.Parameter(torch.ones(1) * 0.1)

        # Initialize weights properly
        self._initialize_weights()

    def _initialize_weights(self):
        """Initialize model weights."""
        for m in self.modules():
            if isinstance(m, (nn.Conv2d, nn.ConvTranspose2d)):
                nn.init.kaiming_normal_(m.weight, mode='fan_out',
                                        nonlinearity='relu')
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.GroupNorm):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.constant_(m.bias, 0)

    def forward(self, x, degradation_type=None):
        """
        Forward pass through PromptIR model.
        
        Args:
            x: Input tensor
            degradation_type: Degradation type tensor
        """
        input_img = x

        initial_features = self.initial(x)

        # Encoder with skip connections
        down1_features = self.down1(initial_features)
        down2_features = self.down2(down1_features)

        # Apply degradation-aware module
        if degradation_type is not None:
            features = self.degradation_module(
                down2_features, degradation_type)
        else:
            # Default to a middle value when degradation type is unknown
            batch_size = x.size(0)
            default_type = torch.ones(batch_size, device=x.device) * 0.5
            features = self.degradation_module(down2_features, default_type)

        # Apply residual groups
        residual_features = features
        for i, group in enumerate(self.groups):
            features = group(features)
            # Apply FFN blocks after certain groups
            if i == len(self.groups) // 2:
                features = self.ffn_blocks[0](features)

        # Apply second FFN block
        features = self.ffn_blocks[1](features)

        # Add global residual connection
        features = features + residual_features

        # Apply attention mechanisms
        features = self.channel_attention(features)
        features = self.spatial_attention(features)

        # Decoder with skip connections
        up1_features = self.up1(features)
        # Fuse with skip connection
        up1_fused = self.fusion1(
            torch.cat([up1_features, down1_features], dim=1))

        up2_features = self.up2(up1_fused)
        # Fuse with skip connection
        up2_fused = self.fusion2(
            torch.cat([up2_features, initial_features], dim=1))

        # Final refinement
        refined = self.refine(up2_fused)

        # Output layer
        out = self.output(refined)

        # Scaled residual learning with learnable parameter
        return input_img + out * self.residual_scale


class DegradationTypeDetector(nn.Module):
    """Network to detect degradation type (rain vs snow)."""

    def __init__(self):
        """Initialize degradation type detector."""
        super(DegradationTypeDetector, self).__init__()
        self.features = nn.Sequential(
            # Layer 1
            nn.Conv2d(3, 32, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 64x64

            # Layer 2
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(64),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 32x32

            # Layer 3
            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Layer 4 with residual connection
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(128),
            nn.GELU(),

            # Layer 5
            nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(256),
            nn.GELU(),
            nn.MaxPool2d(kernel_size=2, stride=2),  # 16x16

            # Global feature pooling
            nn.AdaptiveAvgPool2d((1, 1))
        )

        # Added dropout for regularization
        self.classifier = nn.Sequential(
            nn.Linear(256, 128),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(128, 64),
            nn.GELU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        """Forward pass through detector."""
        x = self.features(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x
