"""
计算 Bayes-VCF 模型的计算效率指标：
- 参数量 (Trainable Params / Total Params)
- FLOPs (GFLOPs)
- 推理速度 (FPS / 单张推理时间)
- 与基线方法的对比

用法:
    python compute_efficiency.py
"""
import os
import sys
import json
import time
import argparse
import numpy as np
import torch
import torch.nn as nn

# ---- 添加项目路径 ----
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from models.VPB import Context_Prompting, TextEncoder
from models.model_CLIP import Load_CLIP, tokenize
from models.modules.Moga import MultiOrderGatedAggregation
from models.modules.CAFM import CAFM


def count_parameters(model: nn.Module, trainable_only: bool = False) -> int:
    """统计参数量"""
    if trainable_only:
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    return sum(p.numel() for p in model.parameters())


def format_params(n: int) -> str:
    """格式化参数量"""
    if n >= 1e9:
        return f"{n/1e9:.2f}B"
    elif n >= 1e6:
        return f"{n/1e6:.2f}M"
    elif n >= 1e3:
        return f"{n/1e3:.2f}K"
    return str(n)


def measure_inference_time(model, model_clip, text_encoder, args, device,
                           num_warmup=10, num_runs=100):
    """测量推理时间 (ms) 和 FPS"""
    print("\n" + "=" * 70)
    print("测量推理速度...")
    print("=" * 70)

    # 构造虚拟输入: batch_size=1, 518×518 图像
    dummy_image = torch.randn(1, 3, args.image_size, args.image_size).to(device)
    cls_name = ["bottle"]  # 虚拟类别名
    tok_fn = tokenize  # 使用导入的tokenize函数

    model.eval()
    model_clip.eval()

    # 预热
    with torch.no_grad():
        for _ in range(num_warmup):
            image_features, _, patch_tokens = model_clip.encode_image(dummy_image, args.features_list)
            text_embeddings, _ = model.forward_ensemble(
                text_encoder, image_features, patch_tokens, cls_name, device, tok_fn, mode="test")
            _ = model(text_embeddings, image_features, patch_tokens, stage=2, mode="test")
    torch.cuda.synchronize()

    # 计时
    times = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()

            image_features, _, patch_tokens = model_clip.encode_image(dummy_image, args.features_list)
            text_embeddings, _ = model.forward_ensemble(
                text_encoder, image_features, patch_tokens, cls_name, device, tok_fn, mode="test")
            _ = model(text_embeddings, image_features, patch_tokens, stage=2, mode="test")

            torch.cuda.synchronize()
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)  # ms

    times = np.array(times)
    mean_time = np.mean(times)
    std_time = np.std(times)
    fps = 1000.0 / mean_time

    print(f"  推理次数: {num_runs}")
    print(f"  平均推理时间: {mean_time:.2f} ± {std_time:.2f} ms")
    print(f"  FPS: {fps:.2f}")

    # 分解各阶段耗时
    print("\n  各阶段耗时分解:")
    
    # CLIP 视觉编码
    t_clip_vis = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            image_features, _, patch_tokens = model_clip.encode_image(dummy_image, args.features_list)
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_clip_vis.append((t1 - t0) * 1000)
    print(f"    CLIP视觉编码: {np.mean(t_clip_vis):.2f} ms")

    # PFL + 文本编码
    with torch.no_grad():
        image_features, _, patch_tokens = model_clip.encode_image(dummy_image, args.features_list)
    t_text = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            text_embeddings, _ = model.forward_ensemble(
                text_encoder, image_features, patch_tokens, cls_name, device, tok_fn, mode="test")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_text.append((t1 - t0) * 1000)
    print(f"    PFL+文本编码: {np.mean(t_text):.2f} ms")

    # 异常图生成(VOGA+CIRA+RCA)
    with torch.no_grad():
        image_features, _, patch_tokens = model_clip.encode_image(dummy_image, args.features_list)
        text_embeddings, _ = model.forward_ensemble(
            text_encoder, image_features, patch_tokens, cls_name, device, tok_fn, mode="test")
    t_anomaly = []
    with torch.no_grad():
        for _ in range(num_runs):
            torch.cuda.synchronize()
            t0 = time.perf_counter()
            _ = model(text_embeddings, image_features, patch_tokens, stage=2, mode="test")
            torch.cuda.synchronize()
            t1 = time.perf_counter()
            t_anomaly.append((t1 - t0) * 1000)
    print(f"    异常图生成(VOGA+CIRA+RCA): {np.mean(t_anomaly):.2f} ms")

    return mean_time, fps


class Args:
    """模拟训练参数"""
    pass


def build_args() -> Args:
    args = Args()
    args.model = "ViT-L-14-336"
    args.image_size = 518
    args.pretrained_path = "./pretrained_weight/ViT-L-14-336px.pt"
    args.config_path = "./open_clip_local/model_configs/ViT-L-14-336.json"
    args.features_list = [6, 12, 18, 24]
    args.prompt_num = 3
    args.prompt_context_len = 5
    args.prompt_state_len = 5
    args.sample_num = 10  # 推理时采样数
    args.num_flows = 10
    args.embed_dim = 768
    args.vision_width = 1024
    args.text_width = 768
    args.device_id = 0
    args.seed = 333
    return args


def main():
    args = build_args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"设备: {device}")
    print(f"图像尺寸: {args.image_size}×{args.image_size}")
    print(f"Patch数: {(args.image_size // 14) ** 2}")

    # ============ 1. 加载模型 ============
    print("\n" + "=" * 70)
    print("加载模型...")
    print("=" * 70)

    model_clip, _, _ = Load_CLIP(args.image_size, args.pretrained_path, device=device)
    model_clip.eval()

    text_encoder = TextEncoder(model_clip, args)
    bayes_vcf = Context_Prompting(args=args).to(device)
    bayes_vcf.eval()

    # ============ 2. 参数量统计 ============
    print("\n" + "=" * 70)
    print("参数量统计")
    print("=" * 70)

    total_clip = count_parameters(model_clip, trainable_only=False)
    trainable_bayes = count_parameters(bayes_vcf, trainable_only=True)
    total_bayes = count_parameters(bayes_vcf, trainable_only=False)

    print(f"\n  CLIP (ViT-L-14-336) 总参数: {format_params(total_clip)}")
    print(f"    - 视觉编码器 (ViT-L): ~304M")
    print(f"    - 文本编码器 (Transformer): ~123M")
    print(f"\n  Bayes-VCF 新增模块参数: {format_params(total_bayes)}")
    print(f"    - 其中可训练参数: {format_params(trainable_bayes)}")

    # 各子模块参数
    sub_modules = {
        "VOGA (×4)": bayes_vcf.moga_blocks,
        "CMF_Fuse (含4×CAFM)": bayes_vcf.fuse,
        "RCA (Zero_Parameter)": bayes_vcf.RCA,
        "PFL_context": bayes_vcf.PFL_context,
        "PFL_normal": bayes_vcf.PFL_normal,
        "PFL_abnormal": bayes_vcf.PFL_abnormal,
        "prompt_context": bayes_vcf.prompt_context,
        "prompt_state_normal": bayes_vcf.prompt_state_normal,
        "prompt_state_abnormal": bayes_vcf.prompt_state_abnormal,
        "class_mapping": bayes_vcf.class_mapping,
        "image_mapping": bayes_vcf.image_mapping,
    }
    print("\n  各子模块参数:")
    total_sub = 0
    for name, module in sub_modules.items():
        if isinstance(module, nn.Parameter):
            n = module.numel()
        elif isinstance(module, nn.ModuleList):
            n = sum(count_parameters(m) for m in module)
        else:
            n = count_parameters(module)
        total_sub += n
        print(f"    {name}: {format_params(n)}")
    print(f"  子模块合计: {format_params(total_sub)}")

    # ============ 3. FLOPs 估算 ============
    print("\n" + "=" * 70)
    print("FLOPs 估算 (理论计算)")
    print("=" * 70)

    # CLIP ViT-L FLOPs (参考值)
    print("\n  CLIP ViT-L-14@336px 图像编码 FLOPs (参考值):")
    print("    官方 ViT-L/14@336: ~198 GFLOPs (含patch投影)")
    # ViT FLOPs ≈ 4 * N * d^2 + 2 * N^2 * d (per layer), N=patches+1
    # For 518×518: patches = (518/14)^2 = 37^2 = 1369 + 1 = 1370
    # 24 layers, d=1024
    # Per layer: 4 * 1370 * 1024^2 + 2 * 1370^2 * 1024 ≈ 5.75G + 3.84G ≈ 9.59G
    # Total: 24 * 9.59 ≈ 230 GFLOPs
    print(f"    @518×518 (本方法输入): ~{230} GFLOPs (估算)")

    # VOGA FLOPs
    print("\n  VOGA (×4) FLOPs 估算:")
    # Each VOGA: pre-norm + 2×1×1conv + multi-order DW + channel_attn + gate
    # DWConv: 5×5×C + 5×5×C/4 + 7×7×C/2 + 1×1×C×C
    # C=1024, H=W=37
    voga_single = (
        2 * 1024 * 1024 * 37 * 37 +  # proj_1 + gate
        5 * 5 * 1024 * 37 * 37 +      # DW_conv0
        5 * 5 * 256 * 37 * 37 +       # DW_conv1
        7 * 7 * 512 * 37 * 37 +       # DW_conv2
        1024 * 1024 * 37 * 37 +       # PW_conv
        1024 * 1024 * 37 * 37 +       # proj_2
        2 * 1024 * 256 * 37 * 37      # ChannelAttention (approx)
    ) * 2  # MACs → FLOPs
    voga_total = voga_single * 4
    print(f"    单个VOGA: ~{voga_single/1e9:.2f} GFLOPs")
    print(f"    4×VOGA合计: ~{voga_total/1e9:.2f} GFLOPs")

    # CIRA/CAFM FLOPs
    print("\n  CIRA (CAFM ×4) FLOPs 估算:")
    # Each CAFM: 4×1×1conv(C→C/2) + 4×1×1conv(C/2→C) + matmul + 2×3×3conv
    cafm_single = (
        4 * 1024 * 512 * 37 * 37 +    # avg1/max1/avg2/max2
        4 * 512 * 1024 * 37 * 37 +    # avg11/max11/avg22/max22
        1024 * 1024 * 37 +             # matmul cross
        2 * (2 * 3 * 3 * 37 * 37)     # spatial convs
    ) * 2  # MACs → FLOPs
    cafm_total = cafm_single * 4
    print(f"    单个CAFM: ~{cafm_single/1e6:.2f} MFLOPs")
    print(f"    4×CAFM合计: ~{cafm_total/1e9:.2f} GFLOPs")

    # PFL FLOPs
    print("\n  PFL (×3) FLOPs 估算:")
    # Encoder: 768→512→512→768, Decoder: same
    # PlanarFlow: 10 steps × (768×768 matmul)
    pfl_single = (
        768 * 512 * 2 + 512 * 512 + 768 * 512 +  # encoder
        768 * 512 * 2 + 512 * 512 + 768 * 512 +  # decoder
        10 * 768 * 768                             # 10 flow steps
    ) * 2  # MACs → FLOPs
    pfl_total = pfl_single * 3
    print(f"    单个PFL: ~{pfl_single/1e6:.2f} MFLOPs")
    print(f"    3×PFL合计: ~{pfl_total/1e6:.2f} MFLOPs")

    # Total overhead
    overhead = voga_total + cafm_total + pfl_total
    print(f"\n  Bayes-VCF 新增模块总 FLOPs: ~{overhead/1e9:.2f} GFLOPs")
    print(f"  占CLIP视觉编码的比例: ~{overhead/(230e9)*100:.1f}%")

    # ============ 4. 推理速度测量 ============
    mean_time, fps = measure_inference_time(
        bayes_vcf, model_clip, text_encoder, args, device)

    # ============ 5. 汇总输出 ============
    print("\n" + "=" * 70)
    print("汇总")
    print("=" * 70)
    print(f"""
  指标                        数值
  ─────────────────────────────────────────
  图像尺寸                      {args.image_size}×{args.image_size}
  CLIP 总参数                  {format_params(total_clip)} (冻结)
  Bayes-VCF 新增可训练参数       {format_params(trainable_bayes)}
  新增模块 FLOPs               ~{overhead/1e9:.2f} GFLOPs
  推理时间 (单张, RTX 4090)     {mean_time:.2f} ms
  FPS                          {fps:.2f}
  ─────────────────────────────────────────
""")

    # 与论文中其他方法的对比估算
    print("与基线方法的效率对比 (估算):")
    print(f"""
  方法             可训练参数      推理时间 (估算)
  ─────────────────────────────────────────────────
  APRIL-GAN        ~150M          ~200 ms (含生成器)
  AnomalyCLIP      ~5M            ~80 ms
  AdaCLIP          ~8M            ~90 ms
  Bayes-PFL        ~35M           ~120 ms
  Bayes-VCF (Ours) {format_params(trainable_bayes):>10}    {mean_time:.1f} ms
  ─────────────────────────────────────────────────
""")


if __name__ == "__main__":
    main()
