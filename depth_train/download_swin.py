import os

# 1. 强制关闭离线模式
os.environ["HF_HUB_OFFLINE"] = "0"

# 2. 强制将所有 Hugging Face 请求重定向到国内顶级镜像站
os.environ["HF_ENDPOINT"] = "https://hf-mirror.com"

print("网络环境已配置，准备从国内镜像拉取 Swin Transformer 权重...")

try:
    import timm
    # 这一步会触发下载并自动将其存入 ~/.cache/huggingface/hub/
    model = timm.create_model('swin_tiny_patch4_window7_224', pretrained=True)
    print("🎉 下载成功！预训练权重已安全存入本地缓存。")
except Exception as e:
    print(f"❌ 下载失败，报错信息：\n{e}")