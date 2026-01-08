# FastOvi API Server

一个基于 FastAPI 的视频生成 API 服务，用于 FastOvi 文本/图像到音视频生成模型。

## 功能特性

- ✅ **启动时加载模型**: 服务器启动时自动加载模型和权重
- ✅ **RESTful API**: 提供简单的 HTTP API 接口
- ✅ **Base64 返回**: 生成的视频以 Base64 编码返回
- ✅ **支持两种图像输入方式**:
  - 图像文件路径
  - Base64 编码的图像
- ✅ **可配置参数**: 支持自定义引导尺度、种子等参数
- ✅ **健康检查端点**: 提供服务健康状态检查

## 安装依赖

```bash
# 安装 FastAPI 相关依赖
pip install -r requirements_api.txt

# 确保已安装 FFmpeg
sudo apt update && sudo apt install ffmpeg
```

## 启动服务

### 方式 1: 使用启动脚本

```bash
# 使用默认配置
./start_api_server.sh

# 自定义配置
export CONFIG_PATH="configs/ovi_smallcfg.yaml"
export CHECKPOINT_PATH="/cpfs01/gongshukai/step_distillation/logs/distill_ovi_lr_2e-6_lr_critic_4e-7_weighted_loss_smallcfg_720ckpt_15k_data/checkpoint_model_009000/model.pt"
export HOST="0.0.0.0"
export PORT="8000"
./start_api_server.sh
```

### 方式 2: 直接运行

```bash
python api_server.py
```

## API 使用

### 1. 健康检查

```bash
curl http://localhost:8000/health
```

响应:
```json
{
  "status": "healthy",
  "cuda_available": true,
  "device_count": 1
}
```

### 2. 生成视频 - 使用图像路径

```bash
curl -X POST http://localhost:8000/generate \
  -H "Content-Type: application/json" \
  -d '{
    "prompt": "A man speaking into a microphone with text captions",
    "image_path": "examples/image_ovi/5.png",
    "seed": 42,
    "video_guidance_scale": 2.0,
    "audio_guidance_scale": 1.5,
    "fps": 24
  }'
```

### 3. 生成视频 - 使用 Base64 图像

```python
import base64
import requests

# 读取图像并编码
with open("examples/image_ovi/5.png", "rb") as f:
    image_base64 = base64.b64encode(f.read()).decode('utf-8')

# 发送请求
response = requests.post("http://localhost:8000/generate", json={
    "prompt": "A dramatic scene with cinematic lighting",
    "image_base64": image_base64,
    "seed": 42
})

# 保存视频
result = response.json()
video_bytes = base64.b64decode(result["video_base64"])
with open("output.mp4", "wb") as f:
    f.write(video_bytes)
```

## 使用客户端示例

运行提供的示例客户端:

```bash
python api_client_example.py
```

## API 端点

### `GET /`
根端点，返回服务状态

**响应**:
```json
{
  "status": "ok",
  "message": "FastOvi Video Generation API is running",
  "model_loaded": true
}
```

### `GET /health`
健康检查端点

**响应**:
```json
{
  "status": "healthy",
  "cuda_available": true,
  "device_count": 1
}
```

### `POST /generate`
生成视频端点

**请求体**:
```json
{
  "prompt": "string (必需)",
  "image_base64": "string (可选)",
  "image_path": "string (可选)",
  "seed": 42,
  "video_guidance_scale": 2.0,
  "audio_guidance_scale": 1.5,
  "fps": 24
}
```

**参数说明**:
- `prompt`: 文本提示词 (必需)
- `image_base64`: Base64 编码的参考图像 (与 image_path 二选一)
- `image_path`: 参考图像文件路径 (与 image_base64 二选一)
- `seed`: 随机种子，用于可重复生成 (默认: 42)
- `video_guidance_scale`: 视频 CFG 尺度 (可选，默认使用配置文件)
- `audio_guidance_scale`: 音频 CFG 尺度 (可选，默认使用配置文件)
- `fps`: 输出视频帧率 (默认: 24)

**响应**:
```json
{
  "video_base64": "base64 encoded MP4 video",
  "metadata": {
    "prompt": "...",
    "seed": 42,
    "video_size": {"width": 1280, "height": 704},
    "num_frames": 121,
    "fps": 24,
    "audio_sample_rate": 16000,
    "video_guidance_scale": 2.0,
    "audio_guidance_scale": 1.5,
    "file_size_mb": 5.23
  }
}
```

## 环境变量

| 变量 | 说明 | 默认值 |
|------|------|--------|
| `CONFIG_PATH` | 配置文件路径 | `configs/ovi_smallcfg.yaml` |
| `CHECKPOINT_PATH` | 模型权重路径 | `/root/dq/OviDMD/model_ema.pt` |
| `HOST` | 服务器监听地址 | `0.0.0.0` |
| `PORT` | 服务器端口 | `8000` |

## 配置文件

默认使用 `configs/ovi_smallcfg.yaml`:

```yaml
model_name: Ovi
denoising_step_list: [1000, 750, 500, 250]  # 4-step distilled model
warp_denoising_step: true
timestep_shift: 5.0
video_guidance_scale: 2.0
audio_guidance_scale: 1.5
video_negative_prompt: "jitter, bad hands, blur, distortion, watermark, text, low quality"
audio_negative_prompt: "robotic, muffled, echo, distorted, noise, low quality"
video_h: 704
video_w: 1280
video_num_frames: 121
audio_duration_secs: 5
audio_sample_rate: 16000
```

## 性能优化

服务器已启用以下优化:

- **TF32**: 启用 TensorFloat-32 加速矩阵运算
- **BFloat16**: 使用混合精度推理
- **CUDA**: 模型运行在 GPU 上
- **临时文件清理**: 自动清理中间文件

## 故障排除

### 1. FFmpeg 未找到
```bash
sudo apt update && sudo apt install ffmpeg
```

### 2. CUDA 内存不足
- 减少批处理大小
- 使用更小的图像分辨率
- 检查 GPU 内存使用: `nvidia-smi`

### 3. 模型加载失败
- 检查 `CHECKPOINT_PATH` 是否正确
- 确保权重文件存在
- 查看日志了解详细错误信息

## 架构说明

```
API Request → FastAPI → OviFewstepInferencePipeline
                          ├─ OviTextEncoder (T5)
                          ├─ OviFusionWrapper (Diffusion Model)
                          └─ OviVAEWrapper (Video + Audio VAE)
                                ↓
                          Generated Video + Audio
                                ↓
                          FFmpeg Merge
                                ↓
                          Base64 Encoded MP4
```

## 开发指南

### 添加新端点

在 `api_server.py` 中添加:

```python
@app.post("/your-endpoint")
async def your_endpoint(request: YourRequest):
    # 实现逻辑
    pass
```

### 自定义配置

修改配置文件或通过环境变量覆盖:

```bash
export CONFIG_PATH="your_config.yaml"
export CHECKPOINT_PATH="your_model.pt"
```

## 许可证

遵循 FastOvi 项目的许可证。

## 联系方式

如有问题，请提交 Issue 或联系项目维护者。
