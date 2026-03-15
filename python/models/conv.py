import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. CONFIGURATION
# ==========================================
# UPDATE THIS PATH to point to your trained .pth file
weights_path = 'best_model_soft.pth' 

# Dimensions extracted from your training config
# n_fft=1024 -> Freq Bins = 513
# Time steps = 128
# Input Channels = 5
INPUT_CHANNELS = 5
FREQ_BINS = 513
FIXED_TIME_STEPS = 128

# Dummy input shape: (Batch_Size, Channels, Freq(Height), Time(Width))
dummy_input_shape = (1, INPUT_CHANNELS, FREQ_BINS, FIXED_TIME_STEPS)

# ==========================================
# 2. MODEL DEFINITION
# (Copied exactly from your provided code)
# ==========================================

class GHPA(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.groups = out_channels // 4
        self.p_h = nn.Parameter(torch.randn(1, self.groups, 10, 1))
        self.p_w = nn.Parameter(torch.randn(1, self.groups, 1, 10))
        self.p_c = nn.Parameter(torch.randn(1, self.groups, 1, 1))  
        self.proj_out = nn.Conv2d(out_channels, out_channels, kernel_size=1)

    def forward(self, x):
        B, C, H, W = x.shape
        x_groups = torch.chunk(x, 4, dim=1)
        p_hw = F.interpolate(self.p_h * self.p_w, size=(H, W), mode='bilinear', align_corners=False)
        x0 = x_groups[0] * p_hw
        p_ch = F.interpolate(self.p_c * self.p_h, size=(H, W), mode='bilinear', align_corners=False)
        x1 = x_groups[1] * p_ch
        p_cw = F.interpolate(self.p_c * self.p_w, size=(H, W), mode='bilinear', align_corners=False)
        x2 = x_groups[2] * p_cw
        x3 = x_groups[3]
        out = torch.cat([x0, x1, x2, x3], dim=1)
        out = self.proj_out(out)
        return out

class GAB(nn.Module):
    def __init__(self, low_channels, high_channels, out_channels):
        super().__init__()
        self.conv_high = nn.Sequential(nn.Conv2d(high_channels, low_channels, 1), nn.BatchNorm2d(low_channels), nn.ReLU())
        self.dilated_convs = nn.ModuleList([
            nn.Conv2d(low_channels, low_channels, 3, padding=1, dilation=1, groups=low_channels),
            nn.Conv2d(low_channels, low_channels, 3, padding=2, dilation=2, groups=low_channels),
            nn.Conv2d(low_channels, low_channels, 3, padding=5, dilation=5, groups=low_channels),
            nn.Conv2d(low_channels, low_channels, 3, padding=7, dilation=7, groups=low_channels)
        ])
        self.proj = nn.Conv2d(low_channels * 4, out_channels, 1)

    def forward(self, low_feat, high_feat):
        high_feat = F.interpolate(high_feat, size=low_feat.shape[2:], mode='bilinear', align_corners=False)
        high_feat = self.conv_high(high_feat)
        fused = low_feat + high_feat
        outs = [conv(fused) for conv in self.dilated_convs]
        out = torch.cat(outs, dim=1)
        out = self.proj(out)
        return out

class ResBlock(nn.Module):
    def __init__(self, in_c, out_c, dropout=0.2):
        super().__init__()
        self.conv1 = nn.Conv2d(in_c, out_c, 3, padding=1); self.bn1 = nn.BatchNorm2d(out_c); self.relu1 = nn.ReLU(); self.drop = nn.Dropout2d(p=dropout)
        self.conv2 = nn.Conv2d(out_c, out_c, 3, padding=1); self.bn2 = nn.BatchNorm2d(out_c); self.relu2 = nn.ReLU()
        self.skip = nn.Identity()
        if in_c != out_c: self.skip = nn.Conv2d(in_c, out_c, 1)

    def forward(self, x):
        identity = self.skip(x)
        out = self.conv1(x); out = self.bn1(out); out = self.relu1(out); out = self.drop(out)
        out = self.conv2(out); out = self.bn2(out)
        out = out + identity
        return self.relu2(out)

class EGE_Audio_UNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.enc1_conv = nn.Conv2d(INPUT_CHANNELS, 32, 3, padding=1); self.enc1_bn = nn.BatchNorm2d(32); self.enc1_relu = nn.ReLU()
        self.enc2_conv = nn.Conv2d(32, 64, 3, padding=1); self.enc2_bn = nn.BatchNorm2d(64); self.enc2_relu = nn.ReLU(); self.enc2_res = ResBlock(64, 64)
        self.enc3_conv = nn.Conv2d(64, 128, 3, padding=1); self.enc3_bn = nn.BatchNorm2d(128); self.enc3_relu = nn.ReLU(); self.enc3_res = ResBlock(128, 128)
        self.enc4_conv = nn.Conv2d(128, 256, 3, padding=1); self.enc4_bn = nn.BatchNorm2d(256); self.enc4_relu = nn.ReLU(); self.enc4_res = ResBlock(256, 256)
        
        self.pool = nn.MaxPool2d(kernel_size=(1, 2))
        
        self.bottleneck = GHPA(256, 256)
        
        self.up4 = nn.ConvTranspose2d(256, 256, (1, 2), stride=(1, 2)); self.dec4 = ResBlock(256, 256); self.gab4 = GAB(256, 256, 256)
        self.up3 = nn.ConvTranspose2d(256, 128, (1, 2), stride=(1, 2)); self.dec3 = ResBlock(128, 128); self.gab3 = GAB(128, 256, 128)
        self.up2 = nn.ConvTranspose2d(128, 64, (1, 2), stride=(1, 2)); self.dec2 = ResBlock(64, 64); self.gab2 = GAB(64, 128, 64)
        self.up1 = nn.ConvTranspose2d(64, 32, (1, 2), stride=(1, 2)); self.dec1 = ResBlock(32, 32); self.gab1 = GAB(32, 64, 32)
        
        self.out_conv = nn.Conv2d(32, 1, 1)

    def _match_size(self, upsampled, target):
        if upsampled.shape[2:] != target.shape[2:]: return F.interpolate(upsampled, size=target.shape[2:], mode='bilinear', align_corners=False)
        return upsampled

    def forward(self, x):
        e1 = self.enc1_relu(self.enc1_bn(self.enc1_conv(x))); p1 = self.pool(e1)
        e2 = self.enc2_res(self.enc2_relu(self.enc2_bn(self.enc2_conv(p1)))); p2 = self.pool(e2)
        e3 = self.enc3_res(self.enc3_relu(self.enc3_bn(self.enc3_conv(p2)))); p3 = self.pool(e3)
        e4 = self.enc4_res(self.enc4_relu(self.enc4_bn(self.enc4_conv(p3)))); p4 = self.pool(e4)
        
        b = self.bottleneck(p4)
        
        u4 = self.up4(b); skip4 = self.gab4(e4, b); u4 = self._match_size(u4, skip4); d4 = self.dec4(u4 + skip4)
        u3 = self.up3(d4); skip3 = self.gab3(e3, d4); u3 = self._match_size(u3, skip3); d3 = self.dec3(u3 + skip3)
        u2 = self.up2(d3); skip2 = self.gab2(e2, d3); u2 = self._match_size(u2, skip2); d2 = self.dec2(u2 + skip2)
        u1 = self.up1(d2); skip1 = self.gab1(e1, d2); u1 = self._match_size(u1, skip1); d1 = self.dec1(u1 + skip1)
        
        out = torch.sigmoid(self.out_conv(d1))
        return out

# ==========================================
# 3. TRACING
# ==========================================
if __name__ == "__main__":
    print(f"1. Initializing model with {INPUT_CHANNELS} input channels...")
    model = EGE_Audio_UNet()

    print(f"2. Loading weights from '{weights_path}'...")
    try:
        # Load weights on CPU to avoid CUDA errors if GPU is missing
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
    except FileNotFoundError:
        print(f"\n[ERROR] Could not find file: {weights_path}")
        print("Please ensure the .pth file is in the same folder or update 'weights_path'.")
        exit()
    except Exception as e:
        print(f"\n[ERROR] Failed to load weights: {e}")
        print("Tip: If the keys don't match, check if you trained with DataParallel (keys starting with 'module.').")
        exit()

    # CRITICAL: Switch to evaluation mode
    # This freezes Dropout and BatchNorm statistics
    model.eval()

    print(f"3. Creating dummy input: {dummy_input_shape}...")
    dummy_input = torch.randn(dummy_input_shape)

    print("4. Tracing model (this might take a moment)...")
    # Tracing executes the model once with the dummy input to record operations
    traced_model = torch.jit.trace(model, dummy_input)

    output_filename = "traced_ege_unet_5ch.pt"
    traced_model.save(output_filename)
    
    print("-" * 40)
    print(f"SUCCESS! Traced model saved to: {output_filename}")
    print(f"Input Specs used: {dummy_input_shape}")
    print("-" * 40)
    print("You can now load this in MATLAB using: mod = torch.load('traced_ege_unet_5ch.pt');")