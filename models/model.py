# models/model.py

import torch
import torch.nn as nn
import torch.nn.functional as F
from timm.models import create_model

# ---------------------------------------------------------------------------
# Charsets
# ---------------------------------------------------------------------------
CHARSET_36 = '0123456789abcdefghijklmnopqrstuvwxyz'
CHARSET_64 = ' 0123456789abcdefghijklmnopqrstuvwxyz!"#$%&\'()*+,-./:;=?@[\\]_`~'

CHARSET   = CHARSET_36
BLANK_IDX = len(CHARSET)

def set_charset(n):
    global CHARSET, BLANK_IDX
    if n == 36:
        CHARSET, BLANK_IDX = CHARSET_36, len(CHARSET_36)
    elif n == 64:
        CHARSET, BLANK_IDX = CHARSET_64, len(CHARSET_64)
    else:
        raise ValueError('charset must be 36 or 64')
    return CHARSET, BLANK_IDX


# ---------------------------------------------------------------------------
# Residual Block
# ---------------------------------------------------------------------------
class _ResBlock(nn.Module):
    def __init__(self, in_ch, out_ch, stride=1):
        super().__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(out_ch),
        )
        self.shortcut = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 1, stride=stride, bias=False),
            nn.BatchNorm2d(out_ch),
        ) if (stride != 1 or in_ch != out_ch) else nn.Identity()
        self.relu = nn.ReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.conv(x) + self.shortcut(x))


# ---------------------------------------------------------------------------
# Enhanced Geometric Feature Extractor
# ---------------------------------------------------------------------------
class EnhancedGeometricExtractor(nn.Module):
    def __init__(self, in_channels: int = 1):
        super().__init__()
        self.conv1 = _ResBlock(in_channels, 64,  stride=2)
        self.conv2 = _ResBlock(64,  128, stride=2)
        self.conv3 = _ResBlock(128, 256, stride=2)
        self.conv4 = _ResBlock(256, 512, stride=1)
        self.reduce = nn.Conv2d(512, 256, kernel_size=1)

        def _head(out_ch):
            return nn.Sequential(
                nn.Conv2d(256, 64, 3, padding=1, bias=False),
                nn.BatchNorm2d(64), nn.ReLU(inplace=True),
                nn.Conv2d(64, out_ch, 1)
            )

        self.boundary_head    = _head(1)
        self.orientation_head = _head(2)
        self.curvature_head   = _head(1)

        self.geo_attention = nn.Sequential(
            nn.Conv2d(260, 64,  kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(64,  260, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x):
        f = self.reduce(self.conv4(self.conv3(self.conv2(self.conv1(x)))))
        boundary    = torch.sigmoid(self.boundary_head(f))
        orientation = torch.tanh(self.orientation_head(f))
        curvature   = torch.relu(self.curvature_head(f))

        geo = torch.cat([f, boundary, orientation, curvature], dim=1)
        geo = geo * self.geo_attention(geo)

        return {
            'features':    geo,
            'boundary':    boundary,
            'orientation': orientation,
            'curvature':   curvature,
        }


# ---------------------------------------------------------------------------
# Adaptive Rectification (Affine + TPS)
# ---------------------------------------------------------------------------
class AdaptiveRectification(nn.Module):
    NUM_CTRL = 8

    def __init__(self, geo_dim: int = 260):
        super().__init__()
        self.pool = nn.AdaptiveAvgPool2d(1)

        # Affine
        self.affine_mlp = nn.Sequential(
            nn.Linear(geo_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(256, 128),     nn.ReLU(inplace=True),
            nn.Linear(128, 6)
        )
        nn.init.normal_(self.affine_mlp[-1].weight, std=0.01)
        self.affine_mlp[-1].bias.data.copy_(
            torch.tensor([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]) +
            torch.tensor([0.0, 0.05, 0.0, 0.05, 0.0, 0.0])
        )

        self.tilt_detector = nn.Sequential(
            nn.Linear(geo_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )
        nn.init.constant_(self.tilt_detector[-2].bias, -2.0)

        # TPS
        self.tps_mlp = nn.Sequential(
            nn.Linear(geo_dim, 256), nn.ReLU(inplace=True), nn.Dropout(0.1),
            nn.Linear(256, 128),     nn.ReLU(inplace=True),
            nn.Linear(128, self.NUM_CTRL * 2)
        )
        nn.init.zeros_(self.tps_mlp[-1].weight)
        nn.init.zeros_(self.tps_mlp[-1].bias)

        self._register_ctrl_points()

    def _register_ctrl_points(self):
        K = self.NUM_CTRL
        xs = torch.linspace(-1, 1, K // 2)
        top    = torch.stack([xs, -torch.ones(K // 2)], dim=1)
        bottom = torch.stack([xs,  torch.ones(K // 2)], dim=1)
        ctrl = torch.cat([top, bottom], dim=0)
        self.register_buffer('ctrl_pts', ctrl)

    @staticmethod
    def _tps_rbf(r2: torch.Tensor) -> torch.Tensor:
        r2 = r2.clamp(min=1e-10)
        return r2 * torch.log(r2)

    def _apply_tps(self, image: torch.Tensor, offsets: torch.Tensor) -> torch.Tensor:
        B, C, H, W = image.shape
        K = self.NUM_CTRL

        offsets = torch.tanh(offsets) * 0.2
        src = self.ctrl_pts.unsqueeze(0).expand(B, -1, -1)
        dst = src + offsets

        gy = torch.linspace(-1, 1, H, device=image.device)
        gx = torch.linspace(-1, 1, W, device=image.device)
        grid_y, grid_x = torch.meshgrid(gy, gx, indexing='ij')
        grid = torch.stack([grid_x, grid_y], dim=-1)
        grid_flat = grid.view(1, H * W, 2).expand(B, -1, -1)

        d2 = ((grid_flat.unsqueeze(2) - src.unsqueeze(1)) ** 2).sum(-1)
        phi = self._tps_rbf(d2)

        delta = dst - src
        d2_ctrl = ((src.unsqueeze(2) - src.unsqueeze(1)) ** 2).sum(-1)
        phi_ctrl = self._tps_rbf(d2_ctrl)

        eye = torch.eye(K, device=image.device).unsqueeze(0) * 1e-6
        w = torch.linalg.solve(phi_ctrl + eye, delta)

        disp = torch.bmm(phi, w)
        warped_grid = grid_flat + disp
        warped_grid = warped_grid.view(B, H, W, 2)

        return F.grid_sample(image, warped_grid,
                             mode='bilinear', padding_mode='border',
                             align_corners=False)

    def forward(self, image: torch.Tensor, geo_features: torch.Tensor, use_tps: bool = False):
        geo_pooled = self.pool(geo_features).flatten(1)

        tilt_score = self.tilt_detector(geo_pooled)

        raw = self.affine_mlp(geo_pooled)
        identity = torch.tensor([1,0,0,0,1,0], device=raw.device, dtype=raw.dtype).unsqueeze(0)
        delta = torch.tanh(raw - identity) * 0.8

        tilt_gate = torch.cat([
            0.3 + 0.7 * tilt_score,
            tilt_score,
            0.5 * torch.ones_like(tilt_score),
            tilt_score,
            0.3 + 0.7 * tilt_score,
            0.5 * torch.ones_like(tilt_score),
        ], dim=1)

        delta = delta * tilt_gate
        theta_vec = identity + delta
        theta = theta_vec.view(-1, 2, 3)

        grid = F.affine_grid(theta, image.size(), align_corners=False)
        rect = F.grid_sample(image, grid, mode='bilinear', padding_mode='border', align_corners=False)

        params = {'affine': theta, 'tps': None}

        if use_tps:
            offsets = self.tps_mlp(geo_pooled).view(-1, self.NUM_CTRL, 2)
            rect = self._apply_tps(rect, offsets)
            params['tps'] = offsets

        return rect, params


# ---------------------------------------------------------------------------
# Geometric-Visual Fusion
# ---------------------------------------------------------------------------
class GeometricVisualFusion(nn.Module):
    def __init__(self, visual_dim=384, geo_dim=260, common_dim=256, num_heads=8):
        super().__init__()
        self.v_proj     = nn.Linear(visual_dim, common_dim)
        self.g_proj     = nn.Linear(geo_dim, common_dim)
        self.cross_attn = nn.MultiheadAttention(embed_dim=common_dim, num_heads=num_heads, batch_first=True)
        self.out_proj   = nn.Linear(common_dim, visual_dim)
        self.layer_norm = nn.LayerNorm(visual_dim)

    def forward(self, visual: torch.Tensor, geo: torch.Tensor) -> torch.Tensor:
        B, D_g, H, W = geo.shape
        geo_flat = geo.flatten(2).transpose(1, 2)           # (B, H*W, D_g)

        v = self.v_proj(visual)
        g = self.g_proj(geo_flat)

        attn_out, _ = self.cross_attn(v, g, g)
        fused = self.out_proj(attn_out)
        fused = self.layer_norm(visual + fused)

        return fused


# ---------------------------------------------------------------------------
# PARSeq-style Decoder (simplified)
# ---------------------------------------------------------------------------
class PARSeqDecoder(nn.Module):
    def __init__(self, visual_dim=384, num_chars=36, max_len=25,
                 num_layers=1, num_heads=6, mlp_ratio=4.0, dropout=0.1,
                 perm_num=6, perm_mirrored=True):
        super().__init__()
        self.max_len = max_len
        self.num_chars = num_chars
        self.BOS_IDX = num_chars
        self.EOS_IDX = num_chars + 1
        self.PAD_IDX = num_chars + 2

        self.pos_queries = nn.Embedding(max_len + 1, visual_dim)
        self.char_embed  = nn.Embedding(num_chars + 3, visual_dim)  # BOS/EOS/PAD

        decoder_layer = nn.TransformerDecoderLayer(
            d_model=visual_dim, nhead=num_heads, dim_feedforward=int(visual_dim*mlp_ratio),
            dropout=dropout, batch_first=True, norm_first=True
        )
        self.decoder = nn.TransformerDecoder(decoder_layer, num_layers=num_layers)

        self.head = nn.Linear(visual_dim, num_chars + 3)  # + BOS/EOS/PAD

    def forward_train(self, memory, tgt_tokens):
        B, L = tgt_tokens.shape
        pos = torch.arange(L, device=memory.device)
        pos_q = self.pos_queries(pos).unsqueeze(0).expand(B, -1, -1)

        tgt_emb = self.char_embed(tgt_tokens) + pos_q

        causal_mask = torch.triu(torch.ones(L, L, device=memory.device), diagonal=1).bool()
        output = self.decoder(tgt_emb, memory, tgt_mask=causal_mask)
        logits = self.head(output)
        return logits

    def forward_inference(self, memory, refine_iters=1):
        B = memory.size(0)
        L = self.max_len + 1
        device = memory.device

        pos = torch.arange(L, device=device)
        pos_q = self.pos_queries(pos).unsqueeze(0).expand(B, -1, -1)

        causal = torch.triu(torch.full((L, L), True, device=device), diagonal=1)

        bos_ids = torch.full((B, L), self.BOS_IDX, dtype=torch.long, device=device)
        char_q  = self.char_embed(bos_ids)
        tgt_emb = pos_q + char_q
        logits  = self.decoder(tgt_emb, memory, tgt_mask=causal)
        logits  = self.head(logits)

        for _ in range(refine_iters):
            pred_ids = logits.argmax(-1)
            is_eos   = (pred_ids >= self.EOS_IDX)
            eos_mask = is_eos.cumsum(dim=1).bool()
            pred_ids = pred_ids.masked_fill(eos_mask, self.PAD_IDX)

            bos      = torch.full((B, 1), self.BOS_IDX, dtype=torch.long, device=device)
            ctx_ids  = torch.cat([bos, pred_ids[:, :-1]], dim=1)
            char_q   = self.char_embed(ctx_ids)
            tgt_emb  = pos_q + char_q
            logits   = self.decoder(tgt_emb, memory, tgt_mask=causal)
            logits   = self.head(logits)

        return logits

    def forward(self, memory, tgt_tokens=None, refine_iters=1):
        if tgt_tokens is not None:
            return self.forward_train(memory, tgt_tokens)
        return self.forward_inference(memory, refine_iters)


# ---------------------------------------------------------------------------
# Full Model
# ---------------------------------------------------------------------------
class PARSeqGeoAware(nn.Module):
    def __init__(self, num_chars=37,
                 use_geometric=True,
                 use_fusion=True,
                 use_rectification=True,
                 use_tps=False,
                 use_attention=False,
                 max_len=25):
        super().__init__()
        self.use_geometric     = use_geometric
        self.use_fusion        = use_fusion
        self.use_rectification = use_rectification
        self.use_tps           = use_tps
        self.use_attention     = use_attention
        self.max_len           = max_len

        self.encoder = create_model(
            'vit_small_patch16_224', pretrained=True,
            num_classes=0, img_size=(64, 256), global_pool=''
        )

        if use_geometric:
            self.gfe = EnhancedGeometricExtractor(in_channels=1)
            self.fusion = GeometricVisualFusion()
            self.geo_concat_proj = nn.Linear(260, 384)

        if use_rectification:
            self.rectification = AdaptiveRectification()

        if use_attention:
            real_chars = num_chars - 1
            self.parseq_decoder = PARSeqDecoder(
                visual_dim=384, num_chars=real_chars, max_len=max_len)
        else:
            self.head = nn.Linear(384, num_chars)

    def forward(self, images, tgt_tokens=None, refine_iters=1, return_features=False):
        geo_output = rect_img = transform_params = None

        if self.use_geometric:
            geo_output = self.gfe(images)
            geo_features = geo_output['features']
        else:
            geo_features = None

        if self.use_rectification and geo_features is not None:
            rect_img, transform_params = self.rectification(
                images, geo_features, use_tps=self.use_tps)
            geo_output = self.gfe(rect_img)
            geo_features = geo_output['features']
            vit_input = rect_img
        else:
            vit_input = images

        visual = self.encoder(vit_input.repeat(1, 3, 1, 1))

        if self.use_geometric and geo_features is not None:
            if self.use_fusion:
                fused = self.fusion(visual, geo_features)
            else:
                geo_global = F.adaptive_avg_pool2d(geo_features, 1).flatten(1)
                geo_proj = self.geo_concat_proj(geo_global)
                fused = visual + geo_proj.unsqueeze(1)
        else:
            fused = visual

        if self.use_attention:
            output = self.parseq_decoder(fused, tgt_tokens, refine_iters)
        else:
            logits = self.head(fused)
            output = logits.permute(1, 0, 2).log_softmax(2)

        if return_features:
            return output, {
                'visual': visual, 'geometric': geo_output,
                'rectified': rect_img, 'transform': transform_params,
                'fused': fused
            }
        return output


# ---------------------------------------------------------------------------
# CTC helpers
# ---------------------------------------------------------------------------
def improved_ctc_decode(log_probs, charset=CHARSET, blank_idx=BLANK_IDX):
    pred = torch.argmax(log_probs, dim=2).permute(1, 0)
    results = []
    for seq in pred.tolist():
        chars, prev = [], None
        for idx in seq:
            if idx == blank_idx:
                prev = blank_idx
            elif idx != prev:
                if idx < len(charset):
                    chars.append(charset[idx])
                prev = idx
        results.append(''.join(chars))
    return results