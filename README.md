# Local Proxy

本地 LLM 代理服务器，替代 CCX，支持多国产模型供应商。

Codex 使用 OpenAI Responses API，但国产模型仅支持 Chat Completions API。
本代理在中间完成协议翻译，并将请求路由到对应供应商。

---

## 快速开始

### 1. 安装依赖

```bash
pip install -r requirements.txt
```

### 2. 配置 API Key

编辑 `config.json`，或启动后在 Web UI 中配置。

预置了 6 个供应商，只需填入对应 API Key：

| 供应商 | 申请地址 |
|--------|---------|
| DeepSeek | https://platform.deepseek.com |
| 智谱 (GLM) | https://open.bigmodel.cn |
| MiniMax | https://platform.minimaxi.com |
| 月之暗面 (Kimi) | https://platform.moonshot.cn |
| 通义千问 (Qwen) | https://dashscope.aliyun.com |
| 豆包 (Doubao) | https://console.volcengine.com/ark |

### 3. 启动

```bash
python proxy.py
```

或双击 `start.bat`。

启动后访问 http://localhost:2000 进入管理界面。

---

## 配合 Codex 使用

### 配置 cc switch

```bash
# API Base URL 指向代理
cc switch --base-url http://localhost:2000

# API Key 随便填（代理不验证，用自己的 Key）
cc switch --api-key sk-local

# 选择模型（会自动路由到对应供应商）
cc switch deepseek-chat
cc switch glm-4
cc switch qwen-max
```

### 工作流程

```
cc switch glm-4              # 告诉 Codex 用 glm-4
    ↓
Codex → 代理 (localhost:2000)  # model="glm-4"
    ↓
代理查 model_map             # glm-4 → zhipu / glm-4
    ↓
代理 → 智谱 API              # 用智谱的 Key，翻译协议
    ↓
智谱返回 → 代理翻译 → Codex
```

### 切换模型

直接 `cc switch <模型名>`，代理自动路由到对应供应商。

可用的模型名见 Web UI 的 Model Map 列表，或 `http://localhost:2000/v1/models`。

---

## Web UI 说明

访问 http://localhost:2000

### System Status

显示当前端口、激活模型、供应商、已配置 Key 数量。

### Model Map

- **Default Model**：请求未匹配到映射时的兜底模型
- **Model Mappings**：显示名 → 供应商:真实模型名 的映射关系
- **Add New Mapping**：添加新的模型映射（显示名 / 供应商 / 真实模型名）

### Providers

- 每个供应商一张卡片，绿色圆点 = 已配置 Key，灰色 = 未配置
- 填写 Base URL 和 API Key 后点 Save
- Remove 删除供应商（模型映射保留但指向空）
- 底部可添加新供应商

### Settings

- Port：监听端口，修改需重启
- Max Retries：上游请求失败重试次数
- Log Level：日志级别

### 🌐 中英文切换

Header 右侧按钮，偏好保存在浏览器中。

---

## 配置文件结构

```json
{
  "port": 2000,
  "default_model": "deepseek-chat",
  "max_retries": 3,
  "log_level": "INFO",
  "providers": {
    "deepseek": {
      "base_url": "https://api.deepseek.com",
      "api_key": "sk-xxx"
    },
    "zhipu": {
      "base_url": "https://open.bigmodel.cn/api/paas/v4",
      "api_key": ""
    }
  },
  "model_map": {
    "deepseek-chat": { "provider": "deepseek", "model": "deepseek-chat" },
    "glm-4":          { "provider": "zhipu",    "model": "glm-4" },
    "gpt-4o":         { "provider": "deepseek", "model": "deepseek-chat" }
  }
}
```

- `providers`：供应商列表，`base_url` 为 API 地址，`api_key` 为密钥
- `model_map`：模型名到供应商的映射，支持多对一（多个显示名指向同一真实模型）

---

## API 端点

| 端点 | 方法 | 说明 |
|------|------|------|
| `/health` | GET | 健康检查 |
| `/api/config` | GET | 获取配置（Key 已脱敏） |
| `/api/config` | POST | 更新配置 |
| `/v1/models` | GET | 模型列表 |
| `/switch` | GET/POST | 查看/切换默认模型 |
| `/v1/responses` | POST | Responses API → Chat Completions |
| `/v1/*` | ALL | 其他端点直通上游 |

---

## 添加新供应商

1. 在 Web UI 的 Providers tab 底部输入供应商名（小写字母+数字+下划线）
2. 点击 + Add Provider
3. 填入 Base URL 和 API Key
4. 点击 Save
5. 切到 Model Map tab，添加使用该供应商的模型映射

---

## 常见问题

### 代理启动了但 cc switch 连不上

确认 `cc switch` 的 base-url 是 `http://localhost:2000`（不是 https，不带 /v1）。

### API Key 填什么

代理不验证请求中的 Key，会用自己的供应商 Key。随便填，如 `sk-local`。

### 模型返回 4xx 错误

- 检查对应供应商的 API Key 是否正确配置
- 检查 Base URL 格式（末尾不要带 `/v1`）
- 查看控制台日志确认路由到了哪个供应商

### 端口冲突

修改 `config.json` 中的 `port` 或 Settings tab 中的 Port，重启后生效。