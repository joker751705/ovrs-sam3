# OVRS-SAM3 设计说明

适用分支：`master`
项目仓库：`jk-jin/ovrs-sam3`
当前任务：开放词汇遥感语义分割

> 本文描述项目当前采用的模型与训练设计。代码实现发生结构性变化时，应同步更新本文。

## 1. 项目目标

OVRS-SAM3 接收一批遥感图像和当前数据集的类别名称，输出每个类别的像素级分割 logits。项目组合三类能力：

* SAM3 提供稳定的多尺度图像特征、文本提示编码、类条件 transformer encoder 和分割解码器。
* RemoteCLIP 提供面向遥感场景的局部图文对齐。
* Encoder refiner 在低分辨率融合两者，通过三阶段语义—细节双路融合上采样生成掩码。

当前只实现 semantic 模式，不支持实例分割、hybrid 模式或非空几何提示训练。

整体流程如下：

```text
图像与类别名称
  ├─ SAM3 图像 backbone → 288/144/72 多尺度 FPN
  ├─ SAM3 文本编码器与 transformer encoder layer 1..6
  │    → 每个图像-类别对的完整 6 层 encoder feature
  │    → SAM 文本 token 的 masked mean
  └─ RemoteCLIP
       ├─ 504×504 图像 → 36×36 dense image feature
       └─ 每类 64 个文本模板 → template text feature

完整 6 层 encoder feature
  → prompt cross-attention
  → 72×72 cross-attended encoder feature

RemoteCLIP 局部相似度图
  → 64 通道模板分数图经 1×1 Conv 投影
  → 与 CLIP dense feature map 分别 L2 归一化后融合
  → 两次拼接前均做逐像素通道 L2 归一化
  → clip_score_embed_36 [B, C, 256, 36, 36]

72×72 cross-attended encoder feature
  → 双线性下采样
  → base_feature_36 [B, C, 256, 36, 36]
  → 直接作为初始 feature_36

clip_score_embed_36 直接作为初始 score_embed_36，不接收 SAM3 FPN 注入

feature_36 + score_embed_36
  → Refiner layer 1..4（全类别同时运行）
  → refiner_features_36 [B, C, 256, 36, 36]

随后按 prompt_chunk_size 逐提示块执行：

original_encoder_feature_72_chunk
  → 冻结的 SAM3 Pixel Decoder（torch.no_grad() 中，forward_multiscale）
  → original_pixel_feature_72 [B×C_chunk, 256, 72, 72]
  → original_pixel_feature_144 [B×C_chunk, 256, 144, 144]
  → original_pixel_feature_288 [B×C_chunk, 256, 288, 288]

original_pixel_feature_288
  → 冻结的 SAM3 semantic_seg_head（仅当固定蒸馏权重 > 0 时执行）
  → sam3_teacher_logits

refiner_feature_36_chunk
  → stage_72: Refiner + O72 语义支路，Refiner + FPN72 细节支路
  → stage_144: Refiner + O144 语义支路，Refiner + FPN144 细节支路
  → stage_288: Refiner + O288 语义支路，Refiner + FPN288 细节支路
  → stage_288 输出直接进入冻结 SAM3 semantic_seg_head
  → final_logits_chunk

每个 chunk 立即计算 loss 和 backward，不再拼接全类别 logits。
```

## 2. 张量约定

| 记号       | 含义                             |
| -------- | ------------------------------ |
| `B`      | batch 中的图像数                    |
| `M`      | 原始前向类别数（背景排除后、逗号拆分前）            |
| `P`      | 逗号拆分后的展开提示数                     |
| `P_chunk` | 当前提示块中的提示数                      |
| `K`      | 每类 RemoteCLIP 文本模板数，固定为 64     |
| `D`      | SAM3 hidden dimension，固定为 256  |
| `D_clip` | RemoteCLIP 投影维度，ViT-L/14 为 768 |
| `L`      | Refiner 层数，当前固定为 4              |

当前固定输入下的主要张量为：

| 张量                            | 形状                                 | 说明                             |
| ----------------------------- | ---------------------------------- | ------------------------------ |
| `backbone_fpn`                | `[B, 256, 288/144/72, 288/144/72]` | SAM3 多尺度图像特征，顺序固定为 288、144、72 |
| `cross_attended_encoder_features_72` | `[B, P, 256, 72, 72]`       | 完整 6 层 encoder 与 prompt cross-attention 后的提示条件特征 |
| `sam_fpn_288`                  | `[B, 256, 288, 288]`               | SAM3 backbone 图像级 FPN288 |
| `sam_fpn_144`                  | `[B, 256, 144, 144]`               | SAM3 backbone 图像级 FPN144 |
| `sam_fpn_72`                   | `[B, 256, 72, 72]`                 | SAM3 backbone 图像级 FPN72 |
| `sam_text_mean`               | `[B, P, 256]`                      | SAM 文本 token 的 masked mean     |
| `remoteclip_feat_map`         | `[B, 768, 36, 36]`                 | RemoteCLIP dense image feature |
| `template_clip_text`          | `[P, 64, 768]`                     | 每提示 64 个模板的文本特征                |
| `clip_score_maps_36`          | `[B, P, 64, 36, 36]`               | 局部图文相似度图                       |
| `clip_score_embed_36`         | `[B, P, 256, 36, 36]`              | 纯 RemoteCLIP score embedding，直接作为 Refiner 初始 score stream |
| `score_embed_36`              | `[B, P, 256, 36, 36]`              | 经 Refiner 更新后的 score stream |
| `refiner_features_36`         | `[B, P, 256, 36, 36]`              | Refiner 的图像特征流                 |
| `original_pixel_feature_72`   | `[B×P_chunk, 256, 72, 72]`         | 冻结 Pixel Decoder 最低分辨率输出 |
| `original_pixel_feature_144`  | `[B×P_chunk, 256, 144, 144]`       | 冻结 Pixel Decoder 中间分辨率输出 |
| `original_pixel_feature_288`  | `[B×P_chunk, 256, 288, 288]`       | 冻结 Pixel Decoder 最高分辨率输出（同时用于 stage_288 语义支路和 detached teacher） |
| `final_pixel_feature_288`     | `[B×P_chunk, 256, 288, 288]`       | RefinerPyramidDecoder stage_288 直接输出，随后进入 frozen semantic_seg_head |
| `final_logits`                | `[B, P_chunk, 288, 288]`（逐 chunk） | 可训练路径输出的最终语义分割 logits |
| `sam3_teacher_logits`         | `[B, P_chunk, 288, 288]`（逐 chunk） | 冻结 SAM3 semantic head 输出的 detached teacher logits |

训练损失和评测会在必要时用最近邻插值把标签映射到 logits 尺度。

## 3. SAM3 分支

### 3.1 图像特征

SAM3 接收 1008×1008 的标准化图像。ViT patch size 为 14，主干 token grid 为 72×72。SimpleFPN 产生 288×288、144×144、72×72 和 36×36 四级特征；当前 `scalp=1` 丢弃最低分辨率的 36×36 级，因此主路径保留前三个尺度。

SAM3 图像 backbone 在训练中冻结并运行于 `eval()`。图像特征使用 `torch.no_grad()` 计算并 detach。

### 3.2 类条件 encoder 与提示展开

原始类别名称支持逗号分隔的多个提示词。例如 `"ship, vessel"` 展开为两个独立提示，`"bridge"` 保持一个。展开后的提示按 `prompt_chunk_size` 分块（默认每块 4 个提示），以控制显存。

模型在展开后的提示空间运行。训练标签通过 `prompt_to_class_id` 把同一原始类别的所有提示映射到相同标签；推理时通过像素级最大值合并回原始类别。

每个图像与每个提示组成一个 prompt pair。冻结的 SAM3 文本编码器和 6 层 transformer encoder 为每个 pair 生成类条件图像特征。所有 6 层在 `torch.no_grad()` 中一次运行完毕。

完整 encoder 输出后，执行一次 prompt cross-attention（同样在 `no_grad()` 中），得到 cross-attended full-encoder feature。SAM 文本向量通过有效 token 的 masked mean 得到，padding token 不参与平均。

所有提示块按原始顺序重新拼接。

### 3.3 共享冻结 Pixel Decoder

冻结的 Pixel Decoder 通过 `forward_multiscale()` 返回 72、144、288 三个尺度的特征。三个特征全部冻结、无梯度。

```text
类条件 72×72 特征（替换 FPN 最后一层）
  → pixel_feature_72
  → 上采样到 144×144 + FPN144 → 3×3 Conv + GroupNorm + ReLU
  → pixel_feature_144
  → 上采样到 288×288 + FPN288 → 3×3 Conv + GroupNorm + ReLU
  → pixel_feature_288
```

Pixel Decoder 内部继续使用 `interpolation_mode="nearest"`（SAM3 原始设置）。每次 chunk 只调用一次 Pixel Decoder，且必须在 `torch.no_grad()` 中。Pixel Decoder 参数冻结且保持 `eval()`。

O72/O144/O288 始终用于 Refiner Pyramid Decoder。原始 semantic head
始终冻结，但仅当固定蒸馏权重大于零且存在可蒸馏类别时按需运行以产生
teacher logits。

## 4. RemoteCLIP 分支

### 4.1 Dense 图像编码

RemoteCLIP 使用 ViT-L/14。原始图像单独缩放到 504×504，并使用 CLIP mean/std 归一化，得到 36×36 patch grid。

前面的 transformer blocks 正常执行；最后一个 block 使用 dense value-branch：

1. 计算 QKV 投影；
2. 只取 V 分支；
3. 经过 attention output projection；
4. 向空间 token 注入 class token 信息；
5. 执行 MLP 残差；
6. 经过 `ln_post` 和原始 visual projection。

最终输出 `[B, 768, 36, 36]`。配置指定的中间层特征只作为 debug 数据保留，不进入当前主路径。

### 4.2 模板文本编码

每个类别使用 64 个固定遥感文本模板，生成 `[C, 64, 768]` 的模板特征。文本编码支持 micro-batch 和 non-reentrant activation checkpoint。

缓存规则必须服从参数是否可训练：

* RemoteCLIP 文本分支完全冻结时，可以缓存 detach 后的模板特征。
* 文本分支可训练且全局梯度开启时，每个训练 step 重新编码并保留计算图。
* 验证位于 `torch.no_grad()` 中，可以在一次验证过程中复用当前权重对应的缓存。

不能用模块的 `training` 属性判断是否需要梯度，因为 RemoteCLIP 在部分微调时仍保持 `eval()`。

### 4.3 Score embedding

64 个模板文本特征和 36×36 dense RemoteCLIP 图像特征分别做
L2 归一化，逐像素计算余弦相似度并乘固定系数 20，得到
[B, C, 64, 36, 36] 模板分数图。

模板分数图展平 batch 与类别维后，经过 64→256 的 1×1 Conv、
GroupNorm 和 GELU，得到中间特征 1。

中间特征 1 与 RemoteCLIP dense feature map 在每个空间位置分别沿
通道维执行 L2 归一化。归一化后的 256 通道中间特征与归一化后的
768 通道 CLIP 特征拼接，经 1024→256 的 1×1 Conv、GroupNorm 和
GELU 得到中间特征 2。

中间特征 1 与中间特征 2 再次分别执行逐像素通道 L2 归一化，
拼接为 512 通道。随后依次经过普通 3×3 Conv 512→256 和普通
3×3 Conv 256→256；每层卷积后均使用 GroupNorm 和 GELU。

最终输出 [B, C, 256, 36, 36] 的 clip_score_embed_36。该特征不再
接收 SAM3 FPN 注入，直接作为 Refiner 的初始 score stream。

## 5. Class-conditioned encoder refiner

Refiner 在 36×36 上同时维护图像 feature 流和 score embedding 流。默认使用 4 层、8 个 attention heads、12×12 窗口和 6 像素 shift。

### 5.1 Feature stream 与 score stream 初始化

Cross-attended full-encoder feature（72×72）双线性下采样到 36×36，得到
`base_feature_36`，直接作为 feature stream。

score_embeddings.py 生成的 clip_score_embed_36 直接作为 score
stream。进入所有 Refiner Attention 层之前不再融合任何 SAM3 FPN
特征，也不再设置额外残差系数。

SAM3 FPN 只在后续 RefinerPyramidDecoder 的 72/144/288 三个
高分辨率细节支路中使用。

### 5.2 单层 refiner

每层采用 pre-norm，并依次执行：

1. **ClassScoreAttention**：在每个空间位置跨类别做注意力。Q/K 由图像 feature、SAM 文本均值和 score embedding 拼接后投影；feature 与 score 使用独立 value/output 分支。
2. **Regular WindowScoreAttention**：每个类别内部执行非移位窗口注意力。
3. **Shifted WindowScoreAttention**：使用 shift mask 和相对位置偏置连接相邻窗口。
4. **Feature FFN**：逐 token 更新图像流。
5. **Score FFN**：逐 token 更新分数流。

每个注意力和 FFN 子层均采用 pre-norm，并在末端线性投影后直接执行残差相加。Refiner 内部不设置固定或可学习残差系数。类间注意力和窗口注意力的更新尺度由各自的 feature/score output projection 学习，两路 FFN 的更新尺度由各自第二个线性投影层学习。

全部 Refiner 层结束后，对最终 feature stream 执行一次逐空间位置、沿通道维度的 LayerNorm，再将归一化后的 refiner_features_36 送入 RefinerPyramidDecoder。最终 score stream 不执行额外输出 LayerNorm。

子层执行顺序为：

```text
pre-norm → attention/FFN → output projection → dropout → direct residual
```

### 5.3 多尺度金字塔解码器

Refiner 在所有类别上统一执行后，最终 36×36 feature stream 先经过一次通道 LayerNorm，再进入 `RefinerPyramidDecoder`，
三个尺度分别实例化独立的 `SemanticDetailFusionStage`（stage_72 / stage_144 / stage_288）。
stage_288 输出直接进入冻结 `semantic_seg_head`，不再有最终融合模块。

**输入**：

- 类条件张量（`[N, 256, H, W]`，`N = B×C_chunk`）：refiner_feature_36, original_pixel_feature_72/144/288
- 图像级 FPN（`[B, 256, H, W]`）：sam_fpn_72, sam_fpn_144, sam_fpn_288

**SemanticDetailFusionStage（每个尺度）**：

四路独立的 256→128 投影（`1×1 Conv + GroupNorm`，无激活）：

```text
semantic_refiner_proj:  upsampled_refiner [N, 256, H, W] → [N, 128, H, W]
detail_refiner_proj:    upsampled_refiner [N, 256, H, W] → [N, 128, H, W]
pixel_proj:             original_pixel     [N, 256, H, W] → [N, 128, H, W]
fpn_proj:               sam_fpn            [B, 256, H, W] → [B, 128, H, W]
```

语义和细节分支各自使用独立的 Refiner 投影（`semantic_refiner_proj` 和 `detail_refiner_proj`），参数不共享。

FPN 先按图像投影到 128 通道，再通过广播与 Refiner 逐类别相加。不使用 `repeat` 或 `repeat_interleave` 复制 256 通道 FPN。

两条独立支路（不能共享参数），结构相同：

```text
block: 普通 3×3 Conv (128→128) → GN(8,128) → GELU
     → 1×1 Conv (128→128) → GN(8,128)
output = input + block(input)   # block 内部残差
```

语义支路融合独立 Refiner 语义投影与 Pixel Decoder；细节支路融合独立 Refiner 细节投影与原始 FPN。

两条支路分别使用独立的 128→256 `1×1 Conv`（无 norm、无激活）恢复到 256 通道。

两路直接相加，再经过一个 256→256 的 `1×1 Conv`（`fusion_out_proj`）输出：

```python
fused_out = semantic_out + detail_out
output = fusion_out_proj(fused_out)
```

每个 stage 不再包含可学习的末尾融合系数。256 通道输出后无 GroupNorm、GELU、ReLU 或原始特征残差。无 final fusion 模块。

整个 pyramid decoder 使用一次 non-reentrant checkpoint（`self.training and self.use_checkpoint` 时开启），不嵌套给每个 stage。

## 6. 冻结 SAM3 分割头与梯度边界

Prompt cross-attention 在完整 6 层 encoder 之后、Refiner 之前执行一次（通过 `apply_prompt_cross_attention()`），位于 `torch.no_grad()` 中。

Pixel Decoder 参数始终冻结（`requires_grad=False`）并保持 `eval()`。每个 class chunk 只执行一次冻结 Pixel Decoder，该调用位于 `torch.no_grad()` 中，一次返回 O72、O144、O288。

Refiner 特征不再经过 Pixel Decoder。`RefinerPyramidDecoder` 使用 O72/O144/O288 为语义支路提供 Pixel Decoder 特征，并使用原始 backbone FPN 为细节支路提供高频细节。

- **原始 O288**：在 `no_grad()` 中经过冻结 semantic head，产生 detached teacher。
- **stage_288 输出**：在梯度开启状态下经过同一个冻结 semantic head，产生 student。semantic head 参数无梯度更新，但 student 梯度可以穿过冻结卷积回传至 pyramid decoder 和 Refiner。

原始 semantic head 始终冻结。其参数本身无梯度更新，但作为 student 调用时梯度可穿过其权重回传。

轻量上采样器消费 O72、O144、O288 和原始 FPN；teacher 只消费 O288；student semantic head 消费 stage_288 输出。

## 7. 训练设计

### 7.1 冻结与微调

以下 SAM3 模块冻结并保持 `eval()`：

* backbone；
* transformer encoder（完整 6 层，在 `no_grad()` 中执行）；
* geometry encoder；
* segmentation head（Pixel Decoder 参数冻结并保持 `eval()`。原始分支在 `no_grad()` 中执行，Refiner 分支在梯度开启状态下执行。semantic head 在原始分支中产生 detach 的 teacher logits，在 student 分支中产生可回传梯度的 student logits）。

完整 SAM3 encoder 和前置 prompt cross-attention 不保留计算图，均在 `torch.no_grad()` 中执行。

`core.encoder_refiner` 完整训练。其内部的 Refiner 层、`RefinerPyramidDecoder`（stage_72/144/288）同属一个参数组，由现有 `trainable_modules=["core.encoder_refiner"]` 自动覆盖，使用基础学习率 `1e-4`。最终掩码 logits 由冻结的 SAM3 `semantic_seg_head` 产生。

RemoteCLIP 图像和文本分支默认使用 `attention` 微调模式，仅训练注意力 Q/V 与位置嵌入，同时保持 `eval()` 以关闭 dropout 和 patch dropout。

OpenCLIP 常把 Q/K/V 存在同一个融合参数中。项目对该参数注册梯度 mask，使 K 区域梯度为 0；同时把整个融合参数组的 weight decay 强制设为 0。恢复 optimizer 状态后会重新应用这一不变量。

默认 AdamW 基础学习率为 `1e-4`：

* encoder refiner 使用 1.0 倍学习率；
* RemoteCLIP text/image 使用 0.01 倍学习率，即 `1e-6`；
* normalization 参数不使用 weight decay；
* 梯度裁剪上限为 0.1；
* warmup 保持前 1000 步，线性从 0.1 倍到全额学习率，后续余弦衰减。

### 7.2 数据增强

训练图像确定性短边缩放到 1008，再随机裁剪 1008×1008，`cat_max_ratio=1.0`（不限制单一类别占比）。SSD 风格颜色增强（亮度、对比度、饱和度、色相各以 0.5 概率独立启用，色相使用 `/180` 的 HSV 归一化单位），`image` 和 `raw_image` 共享同一次采样参数。只使用 0.5 概率水平翻转。不使用随机尺度、垂直翻转和 90° 旋转。

验证和测试不应用颜色增强或随机裁剪。TTA 默认关闭。

### 7.3 损失

每个类别通道独立使用 binary mask 监督，不使用跨类别 softmax。

**主损失：朴素 BCE**（监督 `final_logits`）：

所有有效像素（label ≠ 255）等权参与全局均值。不做正负像素分离、不按类别是否出现分组加权。每个 `[B, P, H, W]` 位置只要 label ≠ 255 就对损失有相同贡献。多提示类别的所有提示使用同一原始标签作为监督，每个提示通道等权。

```python
# 全局分母 = Σ valid_pixels × P
bce_per_pixel = BCEWithLogits(final_logits, target)
loss_final_bce = (bce_per_pixel * valid_mask).sum() / total_valid_pixels
```

标签 255 被排除（不参与 BCE），对所有提示统一处理。

**Dice 损失**（`final_dice_weight=0.0`）：只对图像中存在的提示计算，默认关闭。开启时按全局 `N_present` 做逐 chunk 贡献归一化。

**SAM3 teacher 掩码蒸馏**（固定权重 `sam3_mask_distill_weight=0.1`）：

蒸馏权重在整个训练过程中保持不变。只要当前 batch 存在可蒸馏提示，就计算 teacher logits。

蒸馏监督范围：

1. 冻结的 SAM3 semantic head 产生的 teacher logits 做 sigmoid，得到 soft probability 目标。
2. student 使用 raw `final_logits`。
3. 用 `binary_cross_entropy_with_logits` 逐像素计算蒸馏损失。
4. 只对 GT 中存在的图像—提示对计算。
5. 所有 `label != 255` 的有效像素参与蒸馏。
6. 对每个存在原始类别，额外包含 GT 外侧指定宽度的边界环。
7. 外侧边界环只保留 `label == 255` 的位置。
8. 外环宽度由 `sam3_mask_distill_boundary_width` 控制，默认 2。
9. 宽度 0 表示不增加外环，仅蒸馏全部有效像素。
10. 远离物体的 255 区域不参与蒸馏。
11. 不存在类别不参与蒸馏。
12. teacher 和 student 都在 288×288 分辨率，不做尺度变换。
13. teacher 必须 detach。
14. 分母是全部存在提示对应的有效像素与类别外环像素总数（全局分母，不按 chunk 单独计算）。
15. 多个提示映射到同一类别时复用相同外环，并在全局分母中按提示独立计数。

总损失：

```python
total_loss = (
    1.0 * loss_final_bce
    + 0.0 * loss_final_dice
    + 0.1 * loss_sam3_mask_distill_bce
)
```

`loss_sam3_mask_distill_bce` 为未经权重的原始蒸馏 BCE；`loss_sam3_mask_distill_weighted` 为真正加入总损失的加权贡献。`0.1` 是整个训练期间固定不变的蒸馏权重。

**训练显存设计**：

* Refiner36 对所有类别一次计算。
* 高分辨率按 chunk 计算。
* 每个 chunk 立即计算 loss 和 backward。
* 使用 detached leaf proxy 累积高分辨率梯度。
* 所有 chunk 完成后把 proxy.grad 一次传回真实 Refiner 图。
* optimizer/scaler/scheduler 每个 batch 只更新一次。
* 不保留多个 chunk 的 72/144/288 特征或 logits。
* 不使用 `retain_graph=True`。

## 8. 推理与评测

推理时模型在提示空间输出 `prompt_logits` [B, P, H, W]。Adapter 先做 sigmoid 得到 `raw_prompt_score_map`，再通过像素级最大值把同一原始类别的所有提示合并为 `raw_final_score_map` [B, M, H, W]。可选的逐类别相对阈值在原始类别空间执行，归一化后只把未通过位置置 0；保留位置继续使用原始 sigmoid 分数。最终对类别维取 argmax。

标签空间中两个机制职责不同：

* `reduce_zero_label` 用于从数据集标签空间彻底删除原始 0 类，并把其余类别只重映射一次；类别名称、前向通道和评测元数据必须使用同一重映射结果。
* `background_cfg` 用于有实际背景语义的数据集。背景可以不进入模型前向，但仍属于评测类别空间，并由统一后处理映射回来。

两条路径不能对同一标签连续执行两次 0 类删除或索引平移。

Evaluator 输出整体 mIoU、mAcc、pixel accuracy 和逐类别指标。`metric_groups` 可以按 `class_ids` 或 `class_names` 定义命名类别组，并分别计算组内 mIoU/mAcc。完整 iSAID→LoveDA 配置使用前景类别组作为 checkpoint monitor。

TTA 当前只支持 `scale=1.0` 和空间翻转。多个视图必须先平均 `raw_prompt_score_map`（提示空间），再合并提示到原始类别，最后统一执行一次非线性相对阈值过滤。

## 9. Checkpoint、恢复与实验追踪

训练只在显式提供 `--resume-from` 时恢复完整状态，不自动扫描 `work_dir`。未提供该参数时不会加载任何已有训练产物；若目标目录已经包含训练 checkpoint，则直接报错，避免混合实验。

完整训练 checkpoint 自包含：

* `global_iter`；
* model、optimizer、AMP scaler 和 scheduler；
* Python、NumPy、Torch CPU 与各 CUDA device 的 RNG 状态；
* 可恢复随机 batch sampler 的排列、增强种子、游标和 generator 状态；
* checkpoint manager 的 best score；
* train/validation 统计及 validation 状态。

实验追踪状态（W&B）不保存在 checkpoint 中，每次程序启动创建新的 W&B run。`trainer/global_iter` 仍然作为 W&B 图表横轴。

NumPy RNG 数组以 Tensor 保存，因此统一加载入口可以安全使用 `torch.load(..., weights_only=True)`。写入 iteration checkpoint、`latest.pth` 和 `best.pth` 时使用临时文件与原子替换。

`latest.pth` 只在保存或完成一次 checkpoint finalization 时更新，不随普通日志输出更新。`best.pth` 只在 monitor 指标严格改善时更新。

恢复顺序为：严格加载模型与训练状态、恢复 sampler/hook、构建 DataLoader iterator、初始化 W&B（新 run）、准备缓存，最后恢复 RNG。若 checkpoint 标记验证尚未完成，恢复后先重放该次验证，再继续训练。

W&B 每次创建新 run，不复用旧 run ID 或 `last_history_step`。JSONL 在恢复时追加，在全新训练时重建。

两种加载模式必须区分：

```bash
# 完整、严格地继续训练
python tools/train.py configs/train/isaid_loveda_full.py \
  --resume-from work_dirs/full/isaid_loveda/latest.pth

# 只加载模型参数，重新开始 optimizer、scheduler、RNG、sampler 和 W&B
python tools/train.py configs/train/isaid_loveda_full.py \
  --load-model-from /path/to/checkpoint.pth \
  --work-dir /path/to/new_work_dir

# 只评测模型参数
python tools/train.py configs/test/loveda.py \
  --eval-only \
  --load-model-from /path/to/checkpoint.pth
```

Ctrl+C 使用 Python 默认 `KeyboardInterrupt`。训练和验证均立即退出，不保存新的 checkpoint。已有周期 checkpoint 保留不变。非人工异常仍可保存 exception checkpoint。

旧格式或缺少完整运行状态的权重不能用于 `--resume-from`，但可以通过 `--load-model-from` 只加载模型参数。

训练日志中记录以下蒸馏相关项：

| 键 | 含义 |
| --- | --- |
| `loss_final_bce` | 朴素 BCE，所有有效像素等权全局均值（跨 chunk 累加后求均值） |
| `loss_sam3_mask_distill_bce` | 未经权重的原始蒸馏 BCE（跨 chunk 累加后求均值） |
| `loss_sam3_mask_distill_weighted` | 真正加入总损失的加权蒸馏贡献（跨 chunk 累加后求均值） |

## 10. 配置与主要文件

完整训练入口为：

```bash
python tools/train.py configs/train/isaid_loveda_full.py
```

关键配置：

| 文件                                            | 职责                               |
| --------------------------------------------- | -------------------------------- |
| `configs/_base_/model/ovrs_sam3.py`           | 模型、RemoteCLIP、refiner、冻结策略和 loss |
| `configs/_base_/optimizer/ovrs_sam3_adamw.py` | AdamW 参数组和学习率倍率                  |
| `configs/_base_/schedule/full_20k.py`         | 20K iteration 计划                 |
| `configs/_base_/dataloader/`                  | 公共训练/评测 DataLoader 与 transforms  |
| `configs/datasets/`                           | 数据集路径、类别与标签空间                    |
| `configs/train/isaid_loveda_full.py`          | 完整训练组合、LoveDA 指标组、W&B 与可视化       |

主要实现：

| 文件                                    | 职责                                       |
| ------------------------------------- | ---------------------------------------- |
| `models/sam3_image.py`                | 类别 chunk、缓存、SAM3 encoder、低分辨率 refiner、逐 chunk 高分辨率解码 |
| `models/encoder_refiner.py`           | 全类别 Refiner、最终 feature LayerNorm 与多尺度金字塔解码接口 |
| `models/refiner_pyramid_decoder.py`   | 三阶段语义—细节双路融合上采样，stage_288 直接输出最终高分辨率特征 |
| `models/encoder_refiner_attention.py` | 跨类别/窗口注意力、双流 FFN、pre-norm 与直接残差更新            |
| `models/maskformer_segmentation.py`   | prompt attention、Pixel Decoder 多尺度输出和原始 semantic head |
| `models/score_embeddings.py`          | 64 模板相似度图、归一化 CLIP 融合和空间卷积增强 |
| `models/openclip_image_encoder.py`    | 36×36 dense RemoteCLIP 图像特征              |
| `models/openclip_text_encoder.py`     | 模板文本编码、micro-batch 与梯度控制                 |
| `losses/semantic_criterion.py`        | Streaming 正负平衡 BCE、Dice 和 SAM3 teacher 蒸馏 |
| `engine/trainer.py`                   | 高分辨率逐 chunk loss/backward 与 proxy 梯度回传 |
| `engine/checkpoint.py`                | 安全、原子、严格的 checkpoint 保存与加载               |
| `engine/runtime_state.py`             | RNG 捕获与恢复                                |
| `data/resumable_sampler.py`           | 可精确恢复的数据顺序与增强种子                          |
| `engine/experiment_hooks.py`          | JSONL 与 W&B 生命周期                         |
| `engine/evaluator.py`                 | 语义指标、命名指标组、背景映射与 TTA                     |

## 11. 实现不变量与限制

修改代码时必须保持：

1. 类别 chunk 完整、无重复且按原顺序拼接。
2. SAM3 encoder、refiner 和 RemoteCLIP grid 分别固定为 72×72、36×36 和 36×36。
3. SAM3 hidden dimension 固定为 256。
4. 模板数固定为 64，RemoteCLIP 图文投影维度一致。
5. clip_score_embed_36 完全由 RemoteCLIP 模板分数图和 dense RemoteCLIP feature map 生成。
6. 模板分数中间特征与 dense CLIP 特征在拼接前必须分别沿通道维执行逐像素 L2 归一化。
7. 两路 256 通道中间特征在第二次拼接前也必须分别执行逐像素 L2 归一化。
8. 进入 Refiner Attention 前不得注入 SAM3 FPN。
9. SAM3 FPN 只允许在 RefinerPyramidDecoder 的 72/144/288 高分辨率细节支路中使用。
10. 可训练 RemoteCLIP 文本特征不能跨 optimizer step 缓存。
11. 验证不得重新开启 RemoteCLIP 图像分支的 autograd。
12. Refiner 必须先在全部类别上执行，再按 chunk 做高分辨率解码。Refiner 不能放进 chunk 循环。
13. Pixel Decoder 每 chunk 只调用一次且必须在 `torch.no_grad()` 中。
14. 三个 Pixel Decoder 尺度（72/144/288）全部来自同一次 `forward_multiscale` 调用。
15. O288 同时用于 stage_288 语义支路和 detached teacher。
16. clip_score_embed_36 保持纯 RemoteCLIP 输出，用于 debug。
17. teacher 只来自原始 O288 并且必须 detach。teacher 和 student 都为 288×288。
18. 蒸馏只监督存在类别。每个存在提示均监督全部 GT 有效像素，并额外监督其原始类别 GT 外侧 `sam3_mask_distill_boundary_width` 像素范围内且标签为 255 的边界环。远处 255 区域不参与蒸馏。多个提示映射到同一类别时复用相同外环，并在全局分母中按提示独立计数。
19. 最终掩码 logits 由冻结的 SAM3 `semantic_seg_head` 产生。
20. Refiner 的类间注意力、常规窗口注意力、移位窗口注意力和双流 FFN 均采用 pre-norm，并在末端线性投影后直接执行残差相加，不允许重新引入固定或可学习残差系数。全部 Refiner 层结束后，必须对最终 feature_36 执行一次通道 LayerNorm；最终 score_embed_36 不执行额外输出 LayerNorm。
21. TTA 必须先平均提示空间分数（`raw_prompt_score_map`），再合并提示到原始类别，最后进行相对阈值过滤。
22. `reduce_zero_label` 与 `background_cfg` 各自只执行其定义的一次标签空间变换。
23. 完整恢复必须严格校验 checkpoint schema；模型权重迁移必须走独立入口。
24. 训练不使用完整 `[B,C,288,288]` 计算图。逐 chunk backward 通过 proxy leaf 隔离。
25. optimizer.zero_grad / scaler.step / scaler.update / scheduler.step 每 batch 只执行一次。
26. 不使用 `retain_graph=True`。
27. backbone_fpn 顺序固定为 `[288, 144, 72]`，即 `backbone_fpn[0]` 为 288、`[1]` 为 144、`[2]` 为 72。
28. 原始 FPN 在 256 通道时不按类别复制；FPN 先按图像投影到 128 通道，再按类别广播。
29. FPN 投影模块可训练，每个 chunk 重新计算，不跨 chunk 缓存计算图。
30. 三个 stage 固定 `branch_dim=128`，3×3 卷积使用普通卷积（无分组）。
31. stage_288 直接返回，无最终融合模块。
32. 最终 logits 由冻结 `semantic_seg_head` 生成。

当前限制：

* 只支持 semantic 模式；
* 不支持非空几何 prompt；
* 不支持动态 SAM3/refiner 空间尺寸或多尺度 TTA；
* `clip_mid_features` 只用于 debug；
* 当前默认训练集为 iSAID，验证集为 LoveDA。

## 12. Checkpoint 非兼容变更

Checkpoint schema 版本为 4。实验追踪状态不再保存在 checkpoint 中；`WandbHook` 不再产生持久化状态。

本次模型参数结构发生了非兼容变化：

* 删除 `pyramid_decoder.final_fusion_288.*`（`FinalPixelFeatureFusion288`）。
* 删除旧 `pyramid_decoder.stage_*.detail_dim=64` 单细节支路。
* 新增每层 `fpn_proj`、`semantic_block`、`detail_block`、双输出投影和独立的 `detail_scale`。
* 三个 stage 从旧 64 通道单支路改成新的 128 通道语义—细节双路结构。
* 旧 stage 中同名参数（如 `refiner_proj`、`pixel_proj`）的 shape 也已变化（64→128 通道）。
* 旧 checkpoint 不能通过 `--resume-from` 严格恢复。
* 可以用 `--load-model-from` 非严格迁移未变化参数。
* 必须使用新的 work directory。
* 不创建旧参数映射或兼容层。
* `_CHECKPOINT_VERSION` 继续保持 4，因为 checkpoint 容器格式没有改变。
* 原始 SAM3 Pixel Decoder 和 semantic head 参数名称、形状保持不变。

后续一次重构中：

* 删除 `pyramid_decoder.stage_*.refiner_proj.*`（共享 Refiner 投影）。
* 删除 `pyramid_decoder.stage_*.detail_scale`（可学习残差系数）。
* 新增 `pyramid_decoder.stage_*.semantic_refiner_proj.*`（独立语义 Refiner 投影）。
* 新增 `pyramid_decoder.stage_*.detail_refiner_proj.*`（独立细节 Refiner 投影）。
* 新增 `pyramid_decoder.stage_*.fusion_out_proj.*`（256→256 最终融合 1×1 Conv）。
* 不复制旧 `refiner_proj` 到两个新投影。
* 蒸馏损失不再使用余弦衰减，权重在训练全程保持固定。
* 配置字段 `sam3_mask_distill_decay_start_iter` 和 `sam3_mask_distill_decay_end_iter` 已删除。
* `_CHECKPOINT_VERSION` 继续保持 4。

本次重构中：

* 删除 `core.encoder_refiner.fpn_score_fusion_36.*`（FPN score 注入卷积模块）。
* 删除 `core.encoder_refiner.fpn_score_injection_scale`（FPN 注入残差系数）。
* 删除 `core.encoder_refiner.clip_score_embed.score_encoder.*`（旧多尺度 score encoder）。
* 新增 `core.encoder_refiner.clip_score_embed.score_stem.*`（64→256 1×1 Conv）。
* 新增 `core.encoder_refiner.clip_score_embed.score_clip_fusion.*`（score+CLIP 融合 1×1 Conv）。
* 新增 `core.encoder_refiner.clip_score_embed.spatial_fusion.*`（空间双 3×3 Conv）。
* 模板数从 32 扩展至 64。
* RemoteCLIP score embedding 生成流程完全重构，使用两次 L2 归一化拼接融合。
* 旧训练 checkpoint 不能通过 `--resume-from` 恢复。
* 可以用 `--load-model-from` 非严格迁移未变化参数。
* 必须使用新的 work directory。
* 不创建旧参数映射或兼容层。
* `_CHECKPOINT_VERSION` 继续保持 4，因为 checkpoint 容器格式没有改变。

本次 Refiner 残差与输出归一化重构：

* 删除 `core.encoder_refiner.layers.*.class_feature_scale`。
* 删除 `core.encoder_refiner.layers.*.class_score_scale`。
* 删除 `core.encoder_refiner.layers.*.regular_feature_scale`。
* 删除 `core.encoder_refiner.layers.*.regular_score_scale`。
* 删除 `core.encoder_refiner.layers.*.shifted_feature_scale`。
* 删除 `core.encoder_refiner.layers.*.shifted_score_scale`。
* 删除 `core.encoder_refiner.layers.*.ffn_feature_scale`。
* 删除 `core.encoder_refiner.layers.*.ffn_score_scale`。
* 删除配置字段 `encoder_refiner_cfg.residual_scale_init` 及其完整构建链路。
* 删除 `residual/refiner_internal/*` 训练、JSONL 和 W&B 日志。
* 新增 `core.encoder_refiner.feature_output_norm.weight`。
* 新增 `core.encoder_refiner.feature_output_norm.bias`。
* Refiner 内部继续保持 pre-norm；所有注意力和 FFN 子层改为直接残差相加。
* 全部 Refiner 层结束后，对最终 feature_36 执行一次通道 LayerNorm。
* 最终 score_embed_36 不增加输出 LayerNorm。
* 不增加旧字段兼容、参数占位、参数映射或固定缩放常量。
* 旧 checkpoint 不能通过 `--resume-from` 严格恢复。
* 可以通过 `--load-model-from` 非严格加载未变化参数，但旧输出投影是在 LayerScale 存在时训练的，不保证删除系数后具有等价数值行为。
* 非严格加载旧 checkpoint 时，新的 feature_output_norm 使用默认初始化：weight 为 1、bias 为 0。
* 新实验必须使用新的 work directory。
* `_CHECKPOINT_VERSION` 保持为 4，因为 checkpoint 容器格式没有变化。
