import torch
import torch.nn as nn
import torch.nn.functional as F

class DropBlock2D(nn.Module):
    def __init__(self, drop_prob=0.1, block_size=5):
        super().__init__()
        self.drop_prob = drop_prob
        self.block_size = block_size

    def forward(self, x):
        if not self.training or self.drop_prob == 0.:
            return x
        N, C, H, W = x.size()
        gamma = self.drop_prob / (self.block_size ** 2)
        mask = (torch.rand(N, C, H - self.block_size + 1, W - self.block_size + 1, device=x.device) < gamma).float()
        mask = F.pad(mask, [self.block_size//2]*4, mode='constant', value=0)
        mask = F.max_pool2d(mask, kernel_size=self.block_size, stride=1, padding=self.block_size//2)
        mask = 1 - mask
        x = x * mask * mask.numel() / mask.sum()
        return x

class Parallel_DFS_SE(nn.Module):
    def __init__(self, C, T=21, hidden=128, se_reduction=16):
        super().__init__()
        self.C = C
        
        self.branch_dfs_se = DFS_SE(C, T, hidden, se_reduction, fusion_weight=0.5)
        
        self.branch_se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C // se_reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // se_reduction, C, 1),
            nn.Sigmoid()
        )
        
        self.fusion_weight = nn.Parameter(torch.tensor([0.5, 0.5]))

    def forward(self, x):
        feat_dfs_se = self.branch_dfs_se(x)
        att_se = self.branch_se(x)
        feat_se = x * att_se
        
        norm_weight = F.softmax(self.fusion_weight, dim=0)
        
        feat_fusion = feat_dfs_se * norm_weight[0] + feat_se * norm_weight[1]
        return feat_fusion

class DFS_SE(nn.Module):
    def __init__(self, C, T=21, hidden=128, se_reduction=16, fusion_weight=0.5):
        super().__init__()
        self.T = T
        self.hidden = hidden
        self.C = C
        self.fusion_weight = fusion_weight
        
        self.se = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C // se_reduction, 1),
            nn.ReLU(inplace=True),
            nn.Conv2d(C // se_reduction, C, 1),
            nn.Sigmoid()
        )
        
        self.node_emb = nn.Conv2d(C, hidden, 1)
        self.policy = nn.Sequential(
            nn.Conv2d(hidden, hidden, 1), nn.ReLU(inplace=True),
            nn.Conv2d(hidden, 9, 1)
        )
        self.dfs_att_gen = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(hidden, C, 1), nn.Sigmoid()
        )

    @staticmethod
    def _make_adj(H, W):
        idx = torch.arange(H * W).view(H, W)
        dirs = [(-1, -1), (-1, 0), (-1, 1),
                (0, -1),          (0, 1),
                (1, -1),  (1, 0), (1, 1)]
        adj = torch.zeros(H * W, 8, dtype=torch.long)
        for k, (dy, dx) in enumerate(dirs):
            y = torch.arange(H).view(-1, 1) + dy
            x = torch.arange(W).view(1, -1) + dx
            neighbor = idx[y.clamp(0, H - 1), x.clamp(0, W - 1)]
            adj[:, k] = neighbor.view(-1)
        return adj

    def forward(self, x):
        N, C, H, W = x.size()
        device = x.device

        att_se = self.se(x)
        x_se = x * att_se

        node = self.node_emb(x_se)
        node_flat = node.permute(0, 2, 3, 1).reshape(N, H * W, self.hidden)

        adj = self._make_adj(H, W).to(device)

        curr = torch.zeros(N, dtype=torch.long, device=device)
        visited = torch.zeros(N, H * W, dtype=torch.bool, device=device)
        path_feat = torch.zeros_like(node_flat)

        for t in range(self.T):
            h_map = node_flat[torch.arange(N), curr].view(N, self.hidden, 1, 1)
            logits = self.policy(h_map).squeeze(-1).squeeze(-1)
            a = F.gumbel_softmax(logits, tau=1, hard=False)

            stop_mask = a[:, 8]
            nei_mask = a[:, :8]
            next_idx = torch.gather(adj[curr], 1, torch.argmax(nei_mask, dim=1, keepdim=True)).squeeze(1)
            next_idx = torch.where(visited[torch.arange(N), next_idx], curr, next_idx)
            next_idx = torch.where(stop_mask.bool(), curr, next_idx)

            visited[torch.arange(N), next_idx] = True
            path_feat[torch.arange(N), next_idx] += node_flat[torch.arange(N), next_idx] * (1 - stop_mask).unsqueeze(1)
            curr = next_idx

        path_feat_avg = path_feat.mean(dim=1).view(N, self.hidden, 1, 1)
        att_dfs = self.dfs_att_gen(path_feat_avg)

        att_fusion = att_se * self.fusion_weight + att_dfs * (1 - self.fusion_weight)

        return x * att_fusion

class PromptLite(nn.Module):
    def __init__(self, C, prompt_len=8):
        super().__init__()
        self.pgm = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C//4, 1, groups=4), nn.ReLU(),
            nn.Conv2d(C//4, prompt_len, 1, groups=4)
        )
        self.pim = nn.Conv2d(C + prompt_len, C, 1, groups=4)

    def forward(self, x):
        b, c, h, w = x.size()
        prompt = self.pgm(x).expand(-1, -1, h, w)
        fusion = torch.cat([x, prompt], 1)
        att = torch.sigmoid(self.pim(fusion)).clamp_min(0.5)
        return x * att

class ReduLayer(nn.Module):
    def __init__(self, C, decay=0.9):
        super().__init__()
        self.decay = decay
        self.proj = nn.Conv2d(C, C, 1, groups=4, bias=False)
        self.gate = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Conv2d(C, C // 4, 1), nn.ReLU(inplace=True),
            nn.Conv2d(C // 4, C, 1), nn.Sigmoid()
        )

    def forward(self, x):
        x_proj = self.proj(x)
        gate = self.gate(x)
        return x + self.decay * gate * x_proj

class PromptReduBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, drop_prob=0.1, block_size=5,
                 use_prompt=True, use_redu=True):
        super().__init__()
        self.block = BasicBlock(in_planes, planes, stride, drop_prob, block_size)
        self.prompt = PromptLite(planes) if use_prompt else nn.Identity()
        self.redu   = ReduLayer(planes)           if use_redu   else nn.Identity()

    def forward(self, x):
        x = self.block(x)
        x = self.prompt(x)
        x = self.redu(x)
        return x

class BasicBlock(nn.Module):
    def __init__(self, in_planes, planes, stride=1, drop_prob=0.1, block_size=5,
                 dfs_T=21, dfs_hidden=128):
        super().__init__()
        self.conv1 = nn.Conv2d(in_planes, planes, 3, stride, 1, bias=False)
        self.bn1   = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, 3, 1, 1, bias=False)
        self.bn2   = nn.BatchNorm2d(planes)
        self.parallel_fusion = Parallel_DFS_SE(planes, T=dfs_T, hidden=dfs_hidden)
        self.dropblock = DropBlock2D(drop_prob, block_size)
        self.shortcut = nn.Sequential()
        if stride != 1 or in_planes != planes:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_planes, planes, 1, stride, bias=False),
                nn.BatchNorm2d(planes)
            )

    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.dropblock(out)
        out = self.bn2(self.conv2(out))
        out = self.parallel_fusion(out)
        out += self.shortcut(x)
        return F.relu(out)

class WideResNet(nn.Module):
    block = BasicBlock

    def __init__(self, depth=34, widen_factor=10, num_classes=100,
                 drop_prob=0.1, block_size=5, dfs_T=21, dfs_hidden=128):
        super().__init__()
        self.dfs_T = dfs_T
        self.dfs_hidden = dfs_hidden
        
        n = (depth - 4) // 6
        k = widen_factor
        stages = [16, 16*k, 32*k, 64*k]
        self.conv1 = nn.Conv2d(3, stages[0], 3, 1, 1, bias=False)
        self.layer1 = self._make_layer(stages[0], stages[1], n, stride=1,
                                       drop_prob=drop_prob, block_size=block_size)
        self.layer2 = self._make_layer(stages[1], stages[2], n, stride=2,
                                       drop_prob=drop_prob, block_size=block_size)
        self.layer3 = self._make_layer(stages[2], stages[3], n, stride=2,
                                       drop_prob=drop_prob, block_size=block_size)
        self.bn  = nn.BatchNorm2d(stages[3])
        self.fc  = nn.Linear(stages[3], num_classes)
        self._init_weights()

    def _init_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight)

    def _make_layer(self, in_planes, planes, blocks, stride=1,
                    drop_prob=0.1, block_size=5):
        layers = [self.block(in_planes, planes, stride,
                             drop_prob=drop_prob, block_size=block_size,
                             dfs_T=self.dfs_T, dfs_hidden=self.dfs_hidden)]
        for _ in range(1, blocks):
            layers.append(self.block(planes, planes,
                                     drop_prob=drop_prob, block_size=block_size,
                                     dfs_T=self.dfs_T, dfs_hidden=self.dfs_hidden))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.conv1(x)
        x = self.layer1(x); x = self.layer2(x); x = self.layer3(x)
        x = F.relu(self.bn(x))
        x = F.adaptive_avg_pool2d(x, 1).flatten(1)
        return self.fc(x)

def create_model(num_classes=100, device='cuda', drop_prob=0.1,
                 use_parb=False, use_prompt=True, use_redu=False,
                 dfs_T=21, dfs_hidden=128):
    if use_parb:
        raise NotImplementedError("ParBBlock is not defined")
    else:
        base_block = BasicBlock
    WideResNet.block = base_block
    model = WideResNet(
        num_classes=num_classes,
        depth=34,
        widen_factor=10,
        drop_prob=drop_prob,
        dfs_T=dfs_T,
        dfs_hidden=dfs_hidden
    ).to(device)

    for stage in [model.layer1, model.layer2, model.layer3]:
        last_blk = stage[-1]
        stage[-1] = PromptReduBlock(
            in_planes=last_blk.conv1.in_channels,
            planes=last_blk.conv1.out_channels,
            stride=1,
            drop_prob=drop_prob,
            use_prompt=use_prompt,
            use_redu=use_redu
        ).to(device)
    return model
