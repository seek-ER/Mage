# 视频上传测试功能 - 测试指南

## 快速开始

### 1. 安装依赖

```bash
cd /Users/lingzhen/project/Mage
pip install gradio>=6.0
```

### 2. 启动 Web UI

```bash
# 基础启动
python mage_vl/app.py

# 或指定端口
python mage_vl/app.py --port 7861

# 或创建公开链接
python mage_vl/app.py --share
```

### 3. 访问界面

打开浏览器访问：`http://localhost:7860`

## 功能测试

### 测试 1：基础视频上传（frames 后端）

1. 上传示例视频 `mage_vl/assets/examples/soccer-broadcast.mp4`
2. 输入问题："Describe this video."
3. 选择后端：**frames**
4. 点击"分析视频"
5. **预期结果**：
   - 5-10秒内返回结果
   - Visual tokens: ~1000-2000
   - 处理时间: 2-5秒
   - 采样帧画廊显示 32 帧

### 测试 2：codec 后端

1. 上传相同视频
2. 选择后端：**codec**
3. 选择引擎：**traditional**
4. 点击"分析视频"
5. **预期结果**：
   - 结果正常显示
   - Visual tokens 应低于 frames 后端（约减少 75%）
   - 处理时间可能略长

### 测试 3：自定义问题

1. 上传视频
2. 输入具体问题：
   - "How many people are in the video?"
   - "What sport is being shown?"
   - "Describe the scoreboard."
3. 点击"分析视频"
4. **预期结果**：回答应针对性回答具体问题

### 测试 4：参数调整

1. 上传视频
2. 调整参数：
   - 帧数：16 → 64
   - 最大 token 数：128 → 512
3. 观察结果变化
4. **预期结果**：
   - 更多帧数 = 更多 visual tokens
   - 更多 max_new_tokens = 可能更长的回答

### 测试 5：错误处理

1. 上传无效文件（如 .txt 重命名为 .mp4）
2. 点击"分析视频"
3. **预期结果**：显示清晰的错误信息

4. 不上传视频，直接点击"分析视频"
5. **预期结果**：提示"Please upload a video."

6. 清空问题文本框
7. 点击"分析视频"
8. **预期结果**：提示"Please enter a question."

## 性能基准

| 视频长度 | 帧数 | 预期时间 | 预期 Tokens |
|---------|------|---------|------------|
| 10s     | 32   | 2-5s    | 800-1500   |
| 30s     | 32   | 3-7s    | 800-1500   |
| 60s     | 64   | 5-12s   | 1500-3000  |
| 120s    | 64   | 8-20s   | 1500-3000  |

## CLI 快速测试（可选）

如果需要先用 CLI 验证视频功能正常：

```bash
# frames 后端
python mage_vl/inference_base.py \
  --mode offline \
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
  --video-backend frames \
  --num-frames 32 \
  --question "Describe this video."

# codec 后端
python mage_vl/inference_base.py \
  --mode offline \
  --video mage_vl/assets/examples/soccer-broadcast.mp4 \
  --video-backend codec \
  --codec-engine traditional \
  --num-frames 32 \
  --question "Describe this video."
```

## 常见问题

### Q: 模型加载很慢（30-60秒）
**A**: 首次加载需要下载模型权重。使用 `--preload microsoft/Mage-VL` 参数在启动时预加载。

### Q: codec 后端报错
**A**: 确保已安装 `ffmpeg` 和 `ffprobe`：
```bash
# Ubuntu/Debian
sudo apt-get install ffmpeg

# macOS
brew install ffmpeg
```

### Q: GPU 内存不足
**A**: 减少参数：
- 降低 `num_frames`（如 16 或 24）
- 降低 `max_pixels`（如 100000）
- 使用较短的视频

### Q: 视频无法播放
**A**: 确保视频格式为 MP4 (H.264)。某些浏览器可能不支持所有视频格式。

## 界面截图说明

界面分为三个主要区域：

1. **左侧 - 模型配置**
   - 模型选择（默认 microsoft/Mage-VL）
   - 自定义模型路径（可选）
   - 视频后端选择（frames / codec）
   - 编解码引擎（仅 codec 后端）
   - 参数调整（帧数、最大像素、最大 token 数）

2. **右侧 - 视频输入**
   - 视频上传区域（拖拽或浏览）
   - 问题输入框
   - 分析视频 / 清除按钮

3. **底部 - 结果展示**
   - 模型回答文本
   - Visual tokens 数量
   - 处理时间
   - 采样帧画廊（仅 frames 后端）

## 下一步

如果测试成功，您可以：
1. 上传自己的视频进行测试
2. 尝试不同的问题和参数组合
3. 比较 frames 和 codec 后端的效果差异
4. 使用 `--share` 参数生成公开链接分享给他人
