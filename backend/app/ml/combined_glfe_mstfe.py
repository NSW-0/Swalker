# FROZEN COPY for the website backend. Pinned to models/stage2_iter70000.pth.
# Copied verbatim from the research working file on 2026-10-28. Do not edit here:
# the research version keeps changing, and the website must keep loading the
# checkpoint it was trained with. If a new checkpoint needs a new architecture,
# replace this file and the checkpoint together. embed.py loads with strict=True,
# so a mismatch fails at startup instead of giving wrong vectors.

"""
Combined GLFE + MSTFE Gait Recognition Model  (GLTNet)
======================================================
Combines:
  - GLConv layers (global-local spatial richness) from GLFE (Lin et al., ICCV 2021)
  - Multi-scale temporal branches (small + large scale) from MSTFE (Lin et al., ICIP 2021)
  - LTA (Local Temporal Aggregation) from GLFE
  - GeM pooling from GLFE
  - B3D BasicBlock from MT3D (Lin et al., ACM MM 2020)

Input:  silhouette sequences  (B, 1, T, H, W)  e.g. T=30, H=64, W=44
Output: feature vectors       (B, C_out * H_strips)  for gallery/probe matching


╔══════════════════════════════════════════════════════════════════════════╗
║  REVIEW PATCH NOTES  —  2026-09-21                                       ║
╚══════════════════════════════════════════════════════════════════════════╝
Every change below is marked inline with `# [PATCH-n]`. Two tiers:

  TIER A — precision. ON by default, one flag to revert.
    [PATCH-1] TripletLoss ran under autocast in float16, and torch.cdist
              with B=32 > 25 takes its matmul shortcut by default
              (||a||² + ||b||² - 2a·b, a cancellation hazard).
              MEASURED, not assumed — tests/test_p1_loss_gradients.py:
              the resulting distance error is ~0.05 on distances of ~90,
              i.e. about 11% of the triplet margin (0.5). PyTorch
              accumulates the matmul in fp32 and only rounds the final
              distance to fp16, so this is a real but MODEST perturbation
              of the margin decision, not the catastrophic corruption the
              ||a||²-spacing argument alone would predict. Those numbers
              are from CPU; a CUDA path accumulating in fp16 would be
              worse, so treat them as a lower bound.
              Fixed anyway because it costs nothing and removes a variable.
              Revert with TRIPLET_FP32 = False.
    [PATCH-2] CrossEntropy also forced to fp32 (cheap, removes a second
              autocast precision trap on 274-way logits).

  TIER B — recipe judgement calls. OFF by default; current behaviour is
           preserved exactly so a resumed run does not change underneath you.
    [PATCH-3] TripletLoss reduction. The code averages over ALL valid
              triplets including the ones already satisfying the margin
              (loss exactly 0). GaitSet / GLFE / MSTFE all average over the
              NON-ZERO triplets only ("BA+"). Averaging over all of them
              means the reported triplet magnitude decays toward 0 as
              training succeeds — which is most of the 10-20x gap against
              cross-entropy the brief flags as an open question. Switch with
              TRIPLET_REDUCTION = 'ba_plus'.
    [PATCH-4] Loss weighting is now explicit (w_tri, w_ce) instead of an
              implicit 1:1 sum, so Priority 1's ablation is a config change
              rather than a code edit. Defaults are 1.0 / 1.0 = unchanged.
    [PATCH-5] GeM's exponent p is clamped to [1, 16] and marked
              `_no_weight_decay`. p is an nn.Parameter sitting in the same
              Adam group as every conv weight, receiving weight_decay=5e-4.
              Adam's L2 term is normalised by sqrt(v), and p's own gradient
              is tiny, so the decay dominates and drags p toward 0 over
              70k iterations. `x.pow(1/p)` as p→0 is an explosion, and p<0
              inverts the pooling entirely. Clamping is free insurance.
              Revert with GEM_CLAMP_P = False.

  Also added (no behavioural change):
    [PATCH-6] TripletLoss / CombinedLoss now return a stats dict of TENSORS
              (not floats). Nothing syncs with the GPU until the caller asks
              for a float, and train.py's gradient probe needs the live
              tensors to call autograd.grad() on the two loss terms
              separately. NOTE: CombinedLoss.forward() now returns
              `(total, stats)` — a 2-tuple, not the old 3-tuple. train.py in
              this same folder is updated to match.
    [PATCH-7] `no_weight_decay_param_names()` helper on the model, used by
              train.py's optional param-group split.

Architecture, channel dims, n_parts, n_strips and the forward pass are
UNCHANGED. Nothing here alters what shape of model a checkpoint expects, so
every existing .pth still loads.

Authors' note: this is the research sketch — channel dims and n_parts
can be tuned during ablation.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


# ─────────────────────────────────────────────
# 0.  PATCH SWITCHES
# ─────────────────────────────────────────────
# Tier A — leave these alone unless you deliberately want to reproduce the
# old (numerically broken) behaviour bit-for-bit.
TRIPLET_FP32 = True          # [PATCH-1] compute the distance matrix in fp32
CE_FP32      = True          # [PATCH-2] compute cross-entropy in fp32
GEM_CLAMP_P  = True          # [PATCH-5] keep GeM's exponent in a sane range

# Tier B — defaults reproduce the run you have going right now.
TRIPLET_REDUCTION_DEFAULT = 'ba_all'   # 'ba_all' = current, 'ba_plus' = papers


# ─────────────────────────────────────────────
# 1.  UTILITY BLOCKS
# ─────────────────────────────────────────────

class BasicBlock3D(nn.Module):
    """
    B3D-B from MT3D: trunk 3×3×3 conv + two parallel low-rank branches.
    Xi+1 = conv3d(Xi) + conv(1×3×3)(Xi) + conv(3×1×1)(Xi)
    """
    def __init__(self, in_c, out_c):
        super().__init__()
        # trunk
        self.trunk = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(inplace=True),
        )
        # spatial low-rank branch (1×3×3)
        self.spatial_branch = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(1, 3, 3), padding=(0, 1, 1), bias=False),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(inplace=True),
        )
        # temporal low-rank branch (3×1×1)
        self.temporal_branch = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=(3, 1, 1), padding=(1, 0, 0), bias=False),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(inplace=True),
        )
        # channel alignment if needed
        self.shortcut = nn.Conv3d(in_c, out_c, 1, bias=False) if in_c != out_c else nn.Identity()

    def forward(self, x):
        return self.trunk(x) + self.spatial_branch(x) + self.temporal_branch(x)


class LTA(nn.Module):
    """
    Local Temporal Aggregation (GLFE).
    Replaces the first spatial pooling layer.
    Uses a 3D conv with temporal stride=2 and spatial stride=1
    so spatial resolution is preserved while temporal is halved.
    """
    def __init__(self, in_c, out_c, kernel_a=3, stride_b=2):
        super().__init__()
        self.conv = nn.Conv3d(
            in_c, out_c,
            kernel_size=(kernel_a, 1, 1),
            stride=(stride_b, 1, 1),
            padding=(kernel_a // 2, 0, 0),
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        return self.relu(self.bn(self.conv(x)))


class GLConv(nn.Module):
    """
    Global-Local Convolutional Layer (GLFE).

    mode='A'  → element-wise addition  (GLConvA)
               output shape: same as input  (C_out × T × H × W)

    mode='B'  → height-wise concatenation  (GLConvB)
               output shape: C_out × T × 2H × W
               (global branch output cat'd with reassembled local branch)
    """
    def __init__(self, in_c, out_c, n_parts=8, mode='A'):
        super().__init__()
        assert mode in ('A', 'B')
        self.n_parts = n_parts
        self.mode = mode

        # global branch
        self.global_conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(inplace=True),
        )
        # local branch  (shared weights across all n_parts strips)
        self.local_conv = nn.Sequential(
            nn.Conv3d(in_c, out_c, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(out_c),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, C, T, H, W)
        B, C, T, H, W = x.shape
        assert H % self.n_parts == 0, \
            f"H={H} must be divisible by n_parts={self.n_parts}"

        # ── global branch ────────────────────────────
        y_global = self.global_conv(x)          # (B, out_c, T, H, W)

        # ── local branch (strip-wise, shared weights) ─
        strip_h = H // self.n_parts
        strips = x.split(strip_h, dim=3)        # list of n_parts tensors (B,C,T,strip_h,W)
        y_local_strips = [self.local_conv(s) for s in strips]
        y_local = torch.cat(y_local_strips, dim=3)   # (B, out_c, T, H, W)

        # ── combine ──────────────────────────────────
        if self.mode == 'A':
            return y_global + y_local            # (B, out_c, T, H, W)
        else:  # mode == 'B'
            return torch.cat([y_global, y_local], dim=3)  # (B, out_c, T, 2H, W)


class LocalTransform(nn.Module):
    """
    Large-scale temporal branch: aggregate every `stride` consecutive frames
    into a single interval representation (from MT3D).
    Uses 3D conv with kernel/stride = (s, 1, 1).
    """
    def __init__(self, in_c, out_c, stride=3):
        super().__init__()
        self.stride = stride
        self.conv = nn.Conv3d(
            in_c, out_c,
            kernel_size=(stride, 1, 1),
            stride=(stride, 1, 1),
            bias=False,
        )
        self.bn = nn.BatchNorm3d(out_c)
        self.relu = nn.LeakyReLU(inplace=True)

    def forward(self, x):
        # Trim temporal dim so it's divisible by stride
        T = x.shape[2]
        T_trim = (T // self.stride) * self.stride
        # Guard: a sequence shorter than one interval would produce an empty
        # temporal dim and a cryptic conv error further down. Repeat-pad up
        # to one full interval instead. (Cannot trigger during training at
        # SEQ_LEN=30, but eval passes full sequences of arbitrary length.)
        if T_trim < self.stride:
            reps = self.stride - T + 1
            x = torch.cat([x] + [x[:, :, -1:]] * reps, dim=2)
            T_trim = self.stride
        return self.relu(self.bn(self.conv(x[:, :, :T_trim])))


class GeM(nn.Module):
    """
    Generalised-Mean Pooling (GLFE).
    Applied per horizontal strip in the W dimension.
    p is learnable.

    [PATCH-5] p is clamped into [p_min, p_max] at use time and tagged
    `_no_weight_decay`. See the header note — an unconstrained p under
    weight_decay=5e-4 drifts toward 0, and `.pow(1/p)` as p→0 is an
    explosion. The clamp is a no-op while p stays healthy, so it cannot
    change a run where nothing was going wrong.
    """
    def __init__(self, p_init=6.5, eps=1e-6, p_min=1.0, p_max=16.0):
        super().__init__()
        self.p = nn.Parameter(torch.ones(1) * p_init)
        # Tag so train.py's optional param-group split can find it by object
        # identity as well as by name.
        try:
            self.p._no_weight_decay = True
        except AttributeError:      # pragma: no cover - older torch
            pass
        self.eps = eps
        self.p_min = p_min
        self.p_max = p_max

    def forward(self, x):
        # x: (B, C, 1, H, W)  →  (B, C, 1, H, 1)
        p = self.p.clamp(self.p_min, self.p_max) if GEM_CLAMP_P else self.p
        return x.clamp(min=self.eps).pow(p).mean(dim=-1, keepdim=True).pow(1.0 / p)


# ─────────────────────────────────────────────
# 2.  GLFE SPATIAL FRONT-END
# ─────────────────────────────────────────────

class GLFEFrontend(nn.Module):
    """
    Spatial feature extractor from GLFE:
      Conv3d → LTA → GLConvA → MaxPool → GLConvA → GLConvB

    Output: (B, 128, T/2, 64, 22)
      H goes 64 → (maxpool) 32 → (GLConvB height-concat) 64.
    """
    def __init__(self, n_parts=8):
        super().__init__()
        # initial shallow feature extraction
        self.init_conv = nn.Sequential(
            nn.Conv3d(1, 32, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm3d(32),
            nn.LeakyReLU(inplace=True),
        )
        # LTA replaces first spatial pool → preserves spatial, reduces temporal
        self.lta = LTA(in_c=32, out_c=32, kernel_a=3, stride_b=2)

        # GLConv layers
        self.glconv_a1 = GLConv(32,  64,  n_parts=n_parts, mode='A')
        self.maxpool   = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.glconv_a2 = GLConv(64,  128, n_parts=n_parts, mode='A')
        self.glconv_b  = GLConv(128, 128, n_parts=n_parts, mode='B')
        # GLConvB doubles H, bring it back to 128-channel with 1×1×1 conv
        self.proj = nn.Sequential(
            nn.Conv3d(128, 128, kernel_size=1, bias=False),
            nn.BatchNorm3d(128),
            nn.LeakyReLU(inplace=True),
        )

    def forward(self, x):
        # x: (B, 1, T, 64, 44)
        x = self.init_conv(x)          # → (B, 32, T,  64, 44)
        x = self.lta(x)                # → (B, 32, T/2, 64, 44)
        x = self.glconv_a1(x)          # → (B, 64, T/2, 64, 44)
        x = self.maxpool(x)            # → (B, 64, T/2, 32, 22)
        x = self.glconv_a2(x)          # → (B, 128,T/2, 32, 22)
        x = self.glconv_b(x)           # → (B, 128,T/2, 64, 22)  ← H doubled by concat
        x = self.proj(x)               # → (B, 128,T/2, 64, 22)
        return x


# ─────────────────────────────────────────────
# 3.  MSTFE TEMPORAL BRANCHES
# ─────────────────────────────────────────────

class SmallScaleBranch(nn.Module):
    """
    Small temporal scale: adjacent-frame relations.
    Standard 3D conv chain (B3D blocks) on the full-resolution sequence.
    """
    def __init__(self, in_c=128, out_c=128):
        super().__init__()
        self.b3d1 = BasicBlock3D(in_c,  out_c)
        self.pool  = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.b3d2  = BasicBlock3D(out_c, out_c)

    def forward(self, x):
        x = self.b3d1(x)
        x = self.pool(x)
        x = self.b3d2(x)
        return x   # (B, out_c, T', H'', W'')


class LargeScaleBranch(nn.Module):
    """
    Large temporal scale: interval-level relations.
    LocalTransform collapses every `stride` frames into one interval,
    then B3D extracts features over those coarser intervals.
    """
    def __init__(self, in_c=128, out_c=128, stride=3):
        super().__init__()
        self.local_transform = LocalTransform(in_c, in_c, stride=stride)
        self.b3d1 = BasicBlock3D(in_c,  out_c)
        self.pool  = nn.MaxPool3d(kernel_size=(1, 2, 2), stride=(1, 2, 2))
        self.b3d2  = BasicBlock3D(out_c, out_c)

    def forward(self, x):
        x = self.local_transform(x)    # (B, in_c, T//stride, H, W)
        x = self.b3d1(x)
        x = self.pool(x)
        x = self.b3d2(x)
        return x   # (B, out_c, T'', H'', W'')


# ─────────────────────────────────────────────
# 4.  FEATURE MAPPING HEAD
# ─────────────────────────────────────────────

class GaitHead(nn.Module):
    """
    Temporal pooling → GeM spatial pooling per horizontal strip → separate FC.
    Produces one feature vector per strip; all strips concatenated = final descriptor.

    n_strips  : number of horizontal strips
    in_c      : channels from temporal branches (2 × branch_c after concat)
    out_c     : output channels per strip

    Real shapes at the configured settings: the fused tensor arrives with
    H=32 (not 16 as an older comment in GLFEMSTFEModel claimed), so each of
    the 16 strips is 2 rows tall and each LazyLinear sees C*2 = 512 inputs.
    Final descriptor = 256 * 16 = 4096-d. Comment corrected; no code change.
    """
    def __init__(self, n_strips, in_c, out_c=256):
        super().__init__()
        self.n_strips = n_strips
        self.gem = GeM(p_init=6.5)
        # LazyLinear auto-infers input size on first forward pass
        # (handles variable strip_h without manual shape tracking)
        self.fcs = nn.ModuleList([
            nn.LazyLinear(out_c) for _ in range(n_strips)
        ])
        self.bn = nn.LazyBatchNorm1d()

    def forward(self, x):
        # x: (B, C, T, H, W)
        # ── temporal pooling (max over T) ──────────
        x = x.max(dim=2).values          # (B, C, H, W)
        x = x.unsqueeze(2)               # (B, C, 1, H, W)

        # ── per-strip GeM + FC ─────────────────────
        strip_h = x.shape[3] // self.n_strips
        assert strip_h >= 1 and x.shape[3] % self.n_strips == 0, (
            f"GaitHead: H={x.shape[3]} is not divisible by n_strips="
            f"{self.n_strips}; strips would silently drop rows."
        )
        strips = x.split(strip_h, dim=3)  # list of (B, C, 1, strip_h, W)

        out = []
        for i, s in enumerate(strips):
            s = self.gem(s)              # (B, C, 1, strip_h, 1)
            s = s.flatten(1)             # (B, C * strip_h)
            s = self.fcs[i](s)           # (B, out_c)
            out.append(s)

        out = torch.cat(out, dim=1)      # (B, out_c * n_strips)
        out = self.bn(out)
        return out


# ─────────────────────────────────────────────
# 5.  FULL COMBINED MODEL
# ─────────────────────────────────────────────

class GLFEMSTFEModel(nn.Module):
    """
    Full combined model.

    Architecture summary
    --------------------
    Input silhouettes (B, 1, T, 64, 44)
         │
    ┌────┴────────────────────────────────┐
    │  GLFE Frontend (GLConv + LTA)       │  spatial richness from GLFE
    └────┬────────────────────────────────┘
         │  (B, 128, T/2, 64, 22)
         ├──────────────────────────────────────────┐
         │                                          │
    ┌────┴──────────────┐              ┌────────────┴──────────────┐
    │ Small-Scale Branch│              │  Large-Scale Branch        │
    │ (adjacent frames) │              │  (interval-level, stride=3)│
    └────┬──────────────┘              └────────────┬──────────────┘
         │  (B,128,T',32,11)                        │  (B,128,T'',32,11)
         │                                          │
         └──────────── temporal-pool ───────────────┘
                     concat on channels
                    (B, 256, T_min, 32, 11)
                            │
                    ┌───────┴───────┐
                    │  GaitHead     │  GeM + strip FC (16 strips × 2 rows)
                    └───────┬───────┘
                            │
                    feature vector  (B, 256 * 16 = 4096)

    Training loss: triplet (BA strategy) + cross-entropy
    Evaluation:    Euclidean distance on feature vectors
    """

    def __init__(
        self,
        n_parts=8,          # GLConv strips
        branch_c=128,       # channels in temporal branches
        head_out_c=256,     # FC output channels per strip
        local_stride=3,     # large-scale branch interval stride
        n_classes=74,       # number of training IDs (CASIA-B LT setting)
    ):
        super().__init__()
        self.n_parts = n_parts

        # ── spatial frontend ─────────────────────────
        self.frontend = GLFEFrontend(n_parts=n_parts)

        # ── temporal branches ─────────────────────────
        self.small_branch = SmallScaleBranch(in_c=128, out_c=branch_c)
        self.large_branch  = LargeScaleBranch(in_c=128, out_c=branch_c,
                                               stride=local_stride)

        # ── head ──────────────────────────────────────
        # Fused tensor arrives at H=32 (frontend 64, halved by the MaxPool
        # inside each temporal branch), so 16 strips = 2 rows each.
        self.n_strips = 16
        self.head = GaitHead(
            n_strips=self.n_strips,
            in_c=branch_c * 2,   # small + large concatenated on C
            out_c=head_out_c,
        )

        # ── classification head (only used during training) ──
        feat_dim = head_out_c * self.n_strips
        self.classifier = nn.Linear(feat_dim, n_classes)

    # [PATCH-7]
    def no_weight_decay_param_names(self):
        """
        Names of parameters that should NOT receive weight decay:
          - every BatchNorm affine weight/bias (1-D params)
          - every bias
          - GeM's learnable exponent p
        Used by train.py's optional param-group split (WEIGHT_DECAY_SPLIT).
        Returns a set of names matching self.named_parameters().
        """
        names = set()
        for name, param in self.named_parameters():
            if param.ndim <= 1:
                # covers BatchNorm{1d,3d} weight/bias, Linear bias, and
                # GeM.p (shape (1,)) in one rule
                names.add(name)
            elif getattr(param, '_no_weight_decay', False):
                names.add(name)
        return names

    def forward(self, x):
        """
        x: (B, 1, T, H, W)
        Returns:
            feat    : (B, feat_dim)  ← used for triplet loss + evaluation
            logits  : (B, n_classes) ← used for cross-entropy loss (training only)
        """
        # ── 1. spatial frontend ──────────────────────
        s = self.frontend(x)         # (B, 128, T/2, 64, 22)

        # ── 2. temporal branches ─────────────────────
        s_small = self.small_branch(s)  # (B, 128, T', 32, 11)
        s_large = self.large_branch(s)  # (B, 128, T'',32, 11)

        # align temporal dim (take min length via adaptive pool)
        T_min = min(s_small.shape[2], s_large.shape[2])
        if s_small.shape[2] != T_min:
            s_small = F.adaptive_max_pool3d(s_small, (T_min, s_small.shape[3], s_small.shape[4]))
        if s_large.shape[2] != T_min:
            s_large = F.adaptive_max_pool3d(s_large, (T_min, s_large.shape[3], s_large.shape[4]))

        # ── 3. fuse branches ─────────────────────────
        fused = torch.cat([s_small, s_large], dim=1)   # (B, 256, T_min, 32, 11)

        # ── 4. head → feature vector ─────────────────
        feat = self.head(fused)      # (B, head_out_c * n_strips)

        # ── 5. classifier (training only) ────────────
        logits = self.classifier(feat)

        return feat, logits


# ─────────────────────────────────────────────
# 6.  LOSS FUNCTIONS
# ─────────────────────────────────────────────

class TripletLoss(nn.Module):
    """
    Batch-All (BA) triplet loss with margin.
    Vectorized implementation — no Python loops.

    reduction
    ---------
    'ba_all'  : mean over EVERY valid (anchor, positive, negative) triplet,
                including the ones already satisfying the margin whose loss
                is exactly 0. This is what the original file did and is the
                default here so a resumed run is unchanged.

                Consequence worth understanding: once most triplets are
                satisfied, this average is dominated by zeros and shrinks
                toward 0 even though the remaining hard triplets are just as
                wrong as before. The *reported* triplet magnitude therefore
                measures "fraction of easy triplets", not "how wrong the
                embedding is", and the gradient it produces is scaled down
                by that same fraction.

    'ba_plus' : mean over the NON-ZERO triplets only. This is what GaitSet's
                reference implementation does, and what GLFE / MSTFE
                inherit. Magnitude stays on the scale of the margin instead
                of decaying, so it stays commensurate with cross-entropy
                without any hand-tuned weight.

    The returned stats dict always reports BOTH so you can watch the gap
    without changing what you optimise.
    """

    VALID_REDUCTIONS = ('ba_all', 'ba_plus')

    def __init__(self, margin=0.2, reduction=TRIPLET_REDUCTION_DEFAULT,
                 eps=1e-12, fp32=None):
        super().__init__()
        assert reduction in self.VALID_REDUCTIONS, (
            f"reduction must be one of {self.VALID_REDUCTIONS}, got {reduction!r}"
        )
        self.margin = margin
        self.reduction = reduction
        self.eps = eps
        self.fp32 = TRIPLET_FP32 if fp32 is None else fp32

    def _pairwise_dist(self, feat):
        # [PATCH-1] Two contributing problems, both removed here:
        #
        #   (a) dtype. Under `with autocast('cuda')`, feat arrives as fp16.
        #       cdist is not on autocast's fp32 promotion list, so the whole
        #       distance matrix was computed and STORED in fp16. A 4096-d
        #       BatchNorm'd embedding gives pairwise distances around 90,
        #       where fp16's representable spacing is 0.0625.
        #   (b) algorithm. torch.cdist's default compute_mode
        #       ('use_mm_for_euclid_dist_if_necessary') switches to the
        #       ||a||² + ||b||² - 2a·b matmul identity once B > 25.
        #       BATCH_SIZE is 32, so it always took that path — a
        #       cancellation hazard that subtracts two large nearly-equal
        #       quantities.
        #
        # Measured effect (tests/test_p1_loss_gradients.py): max distance
        # error ~0.05, i.e. ~11% of the margin (0.5). Enough to move the
        # margin decision on triplets sitting near the boundary; NOT enough
        # to explain a Rank-1 collapse on its own. Fixed because it costs
        # nothing and turns a variable into a constant.
        if self.fp32:
            f = feat.float()
            return torch.cdist(f, f, p=2,
                               compute_mode='donot_use_mm_for_euclid_dist')
        return torch.cdist(feat, feat, p=2)

    def forward(self, feat, labels):
        """
        feat  : (B, D)
        labels: (B,)  integer class labels

        Returns (loss, stats) where stats holds detached TENSORS — nothing
        forces a GPU sync until the caller converts one to a Python float.
        """
        B = feat.shape[0]
        dist = self._pairwise_dist(feat)          # (B, B)

        # positive/negative masks
        mask_pos = (labels.unsqueeze(1) == labels.unsqueeze(0))  # (B, B)
        mask_neg = ~mask_pos                                     # (B, B)
        # NOTE: the diagonal is inside mask_pos, so mask_neg already
        # excludes self-pairs. Correct as written; verified by test.

        # exclude diagonal from positive mask (anchor vs itself)
        eye = torch.eye(B, dtype=torch.bool, device=feat.device)
        mask_pos = mask_pos & ~eye

        # broadcast: d_ap (B, B, 1), d_an (B, 1, B)
        d_ap = dist.unsqueeze(2)   # (B, B, 1)
        d_an = dist.unsqueeze(1)   # (B, 1, B)

        # all triplet losses: (B, B, B)
        raw = F.relu(d_ap - d_an + self.margin)

        # valid mask: anchor-positive pairs × anchor-negative pairs
        valid = mask_pos.unsqueeze(2) & mask_neg.unsqueeze(1)  # (B, B, B)

        n_valid = valid.sum()
        if n_valid.item() == 0:
            # No same-identity pair in this batch at all. Return a true zero
            # that still carries a graph edge so backward() does not fail.
            zero = feat.sum() * 0.0
            stats = {
                'tri_ba_all':  zero.detach(),
                'tri_ba_plus': zero.detach(),
                'n_valid':     n_valid.detach(),
                'n_active':    torch.zeros((), device=feat.device),
                'frac_active': torch.zeros((), device=feat.device),
                'mean_d_ap':   torch.zeros((), device=feat.device),
                'mean_d_an':   torch.zeros((), device=feat.device),
                'batch_has_positives': torch.zeros((), device=feat.device),
            }
            return zero, stats

        losses = raw[valid]
        n_active = (losses > self.eps).sum()

        ba_all  = losses.mean()
        ba_plus = losses.sum() / n_active.clamp(min=1).to(losses.dtype)

        loss = ba_plus if self.reduction == 'ba_plus' else ba_all

        stats = {
            'tri_ba_all':  ba_all.detach(),
            'tri_ba_plus': ba_plus.detach(),
            'n_valid':     n_valid.detach(),
            'n_active':    n_active.detach(),
            'frac_active': (n_active.float() / n_valid.float()).detach(),
            'mean_d_ap':   dist[mask_pos].mean().detach(),
            'mean_d_an':   dist[mask_neg].mean().detach(),
            'batch_has_positives': torch.ones((), device=feat.device),
        }
        return loss, stats


class CombinedLoss(nn.Module):
    """
    L = w_tri * L_triplet + w_ce * L_crossentropy

    [PATCH-4] The weights are explicit now. Defaults (1.0, 1.0) reproduce
    the original unweighted sum exactly.

    [PATCH-6] forward() returns `(total, stats)` — a 2-TUPLE. The old
    signature returned `(total, l_tri_float, l_ce_float)`. train.py in this
    folder is updated; if you have any other caller, update it too.

    `stats` values are detached tensors except 'tri_t' / 'ce_t', which are
    the live graph-attached loss terms. train.py's gradient probe calls
    torch.autograd.grad() on those two separately to measure how much of the
    embedding's gradient each term is actually responsible for.
    """

    def __init__(self, margin=0.2, w_tri=1.0, w_ce=1.0,
                 label_smoothing=0.1,
                 triplet_reduction=TRIPLET_REDUCTION_DEFAULT):
        super().__init__()
        self.triplet = TripletLoss(margin=margin, reduction=triplet_reduction)
        self.ce      = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.w_tri   = float(w_tri)
        self.w_ce    = float(w_ce)

    def extra_repr(self):
        return (f"w_tri={self.w_tri}, w_ce={self.w_ce}, "
                f"margin={self.triplet.margin}, "
                f"triplet_reduction={self.triplet.reduction!r}, "
                f"triplet_fp32={self.triplet.fp32}")

    def forward(self, feat, logits, labels):
        l_tri, tstats = self.triplet(feat, labels)
        # [PATCH-2] 274-way logits in fp16 through a log-softmax is an
        # avoidable precision loss for essentially no compute.
        l_ce = self.ce(logits.float() if CE_FP32 else logits, labels)

        total = self.w_tri * l_tri + self.w_ce * l_ce

        stats = dict(tstats)
        stats['tri_t'] = l_tri          # graph-attached, for the grad probe
        stats['ce_t']  = l_ce           # graph-attached, for the grad probe
        stats['tri']   = l_tri.detach()
        stats['ce']    = l_ce.detach()
        stats['total'] = total.detach()
        stats['w_tri'] = self.w_tri
        stats['w_ce']  = self.w_ce
        return total, stats


def stats_to_floats(stats):
    """
    Convert a stats dict to plain Python floats for logging. Call this ONLY
    where you actually print — every conversion is a GPU sync, and doing it
    every iteration (as the original `.item()` calls did) serialises the
    training loop against the GPU for no benefit.
    """
    out = {}
    for k, v in stats.items():
        if k in ('tri_t', 'ce_t'):
            continue
        out[k] = float(v) if torch.is_tensor(v) else v
    return out


# ─────────────────────────────────────────────
# 7.  QUICK SANITY CHECK
# ─────────────────────────────────────────────

if __name__ == "__main__":
    print("=" * 66)
    print("Combined GLFE + MSTFE — Sanity Check (patched)")
    print("=" * 66)

    device = torch.device("cpu")   # CPU-only check

    model = GLFEMSTFEModel(
        n_parts=8,
        branch_c=64,        # reduced for memory check
        head_out_c=128,
        local_stride=3,
        n_classes=74,
    ).to(device)

    # small input to fit in container memory
    dummy  = torch.randn(4, 1, 16, 64, 44)
    labels = torch.tensor([0, 0, 1, 1])

    with torch.no_grad():
        feat, logits = model(dummy)

    print(f"Input shape  : {tuple(dummy.shape)}")
    print(f"Feature shape: {tuple(feat.shape)}")
    print(f"Logits shape : {tuple(logits.shape)}")

    crit = CombinedLoss(margin=0.5)
    total, stats = crit(feat, logits, labels)
    f = stats_to_floats(stats)
    print(f"\nCombinedLoss : total={f['total']:.4f}  "
          f"tri={f['tri']:.4f}  ce={f['ce']:.4f}")
    print(f"  triplet BA(all)  = {f['tri_ba_all']:.6f}")
    print(f"  triplet BA+(nz)  = {f['tri_ba_plus']:.6f}")
    print(f"  valid triplets   = {int(f['n_valid'])}, "
          f"active = {int(f['n_active'])} "
          f"({100 * f['frac_active']:.1f}%)")
    print(f"  mean d(a,p)      = {f['mean_d_ap']:.4f}")
    print(f"  mean d(a,n)      = {f['mean_d_an']:.4f}")

    total_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"\nTrainable parameters : {total_params:,}")
    no_wd = model.no_weight_decay_param_names()
    print(f"Params excluded from weight decay if WEIGHT_DECAY_SPLIT=True: "
          f"{len(no_wd)}")
    print("=" * 66)
    print("All checks passed.")
