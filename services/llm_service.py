"""
============================================================
统一 LLM 调用层(services/llm_service.py)
============================================================
【业务定位】把"5 家不同 LLM"统一成同一个 chat() 函数,前端无感切换
  - DeepSeek / 智谱 GLM-4 / 通义千问 / OpenAI / Ollama
  - 全部走 OpenAI Chat Completions 协议,纯 urllib 实现,无需 openai 库

【配置方式】在项目根目录建 .env 文件(本模块自带 .env 加载)
    LLM_PROVIDER=deepseek
    LLM_API_KEY=sk-xxx
    LLM_MODEL=deepseek-chat
    LLM_BASE_URL=https://api.deepseek.com/v1   # 可选

【降级策略】无 key 或调用失败 → fallback_reply() 走"本地规则回复"
  保证前端不会因为没配 key 就报错,演示/汇报时也能跑通。
============================================================
"""

# ── 标准库 ──
import os
import json
import urllib.request
import urllib.error
import ssl
import socket


# ============================================================
# Provider 配置:5 家 LLM 的"基础信息"
# ============================================================
PROVIDERS = {
    "deepseek": {
        "name": "DeepSeek",                       # 显示名
        "default_base_url": "https://api.deepseek.com/v1",
        "default_model": "deepseek-chat",        # 默认模型
        "env_key": "DEEPSEEK_API_KEY",            # 从哪个环境变量取 key
    },
    "zhipu": {
        "name": "智谱 GLM",
        "default_base_url": "https://open.bigmodel.cn/api/paas/v4",
        "default_model": "glm-4-flash",           # 免费档,响应快
        "env_key": "ZHIPU_API_KEY",
    },
    "qwen": {
        "name": "通义千问",
        "default_base_url": "https://dashscope.aliyuncs.com/compatible-mode/v1",
        "default_model": "qwen-plus",
        "env_key": "QWEN_API_KEY",
    },
    "openai": {
        "name": "OpenAI",
        "default_base_url": "https://api.openai.com/v1",
        "default_model": "gpt-4o-mini",
        "env_key": "OPENAI_API_KEY",
    },
    "ollama": {
        "name": "Ollama(本地)",
        "default_base_url": "http://127.0.0.1:11434/v1",
        "default_model": "qwen2.5:7b",
        "env_key": None,                          # 本地无 key
    },
}


# ============================================================
# .env 加载(轻量版,不依赖 python-dotenv)
# ============================================================
def _load_env_file():
    """
    从项目根目录的 .env 文件加载环境变量
    - 注释行(#)跳过
    - 已存在的环境变量不会被覆盖(setdefault)
    """
    try:
        env_path = os.path.join(os.path.dirname(os.path.dirname(__file__)), ".env")
        if not os.path.exists(env_path):
            return
        with open(env_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                k = k.strip()
                v = v.strip().strip('"').strip("'")    # 去掉首尾引号
                os.environ.setdefault(k, v)             # 不覆盖已有
    except Exception:
        pass


# 模块加载时自动执行
_load_env_file()


# ============================================================
# 状态查询:给前端展示"哪些 provider 已配置/可用"
# ============================================================
def get_status():
    """
    返回当前所有 provider 的状态
    {
      "current": "deepseek",                  # 当前激活的 provider
      "providers": [
        {"id":"deepseek","name":"DeepSeek","configured":True,...},
        ...
      ]
    }
    """
    out = []
    for pid, info in PROVIDERS.items():
        key = None
        if info["env_key"]:
            key = os.environ.get(info["env_key"]) or os.environ.get("LLM_API_KEY")
        base_url = os.environ.get("LLM_BASE_URL") or info["default_base_url"]
        # 当前激活的 provider 允许用通用 LLM_API_KEY
        if pid == os.environ.get("LLM_PROVIDER", "zhipu"):
            key = key or os.environ.get("LLM_API_KEY")
        out.append({
            "id":             pid,
            "name":           info["name"],
            "configured":     bool(key) if info["env_key"] else True,
            "default_model":  info["default_model"],
            "base_url":       base_url,
        })
    current = os.environ.get("LLM_PROVIDER", "zhipu")
    return {"current": current, "providers": out}


# ============================================================
# 取某 provider 的最终配置(base_url / model / api_key)
# ============================================================
def _get_config(provider: str):
    info = PROVIDERS.get(provider)
    if not info:
        raise ValueError(f"Unknown provider: {provider}")
    base_url = os.environ.get("LLM_BASE_URL") or info["default_base_url"]
    model    = os.environ.get("LLM_MODEL")    or info["default_model"]
    key      = None
    if info["env_key"]:
        key = os.environ.get(info["env_key"])
        if not key:
            key = os.environ.get("LLM_API_KEY")    # 通用兜底
    return {"base_url": base_url, "model": model, "api_key": key, "name": info["name"]}


# ============================================================
# 核心调用:用 urllib 调 chat completions
# ============================================================
def chat(messages, provider=None, model=None, temperature=0.7, timeout=60):
    """
    调用 LLM,返回字符串回复

    :param messages:    [{"role": "system/user/assistant", "content": "..."}]
    :param provider:    不传 → 用 LLM_PROVIDER 环境变量(默认 zhipu)
    :param model:       不传 → 用 provider 默认模型
    :param temperature: 0~1,越大越有创造性
    :param timeout:     超时秒数
    :return: LLM 回复的纯文本
    :raise: RuntimeError(网络错/HTTP错/格式错)
    """
    provider = provider or os.environ.get("LLM_PROVIDER", "zhipu")
    cfg = _get_config(provider)
    if model:
        cfg["model"] = model

    # Ollama 本地无需 key,其他必须配 key
    if provider != "ollama" and not cfg["api_key"]:
        raise RuntimeError(
            f"{cfg['name']} 未配置 API Key。"
            f"请在 .env 中设置 {PROVIDERS[provider]['env_key']} 或通用 LLM_API_KEY"
        )

    # 拼接 URL
    url = cfg["base_url"].rstrip("/") + "/chat/completions"

    # OpenAI 标准请求体
    payload = {
        "model":       cfg["model"],
        "messages":    messages,
        "temperature": temperature,
        "stream":      False,
    }
    headers = {"Content-Type": "application/json"}
    if cfg["api_key"]:
        headers["Authorization"] = f"Bearer {cfg['api_key']}"

    data = json.dumps(payload).encode("utf-8")
    req  = urllib.request.Request(url, data=data, headers=headers, method="POST")

    # 国内代理环境证书链经常不全 → 宽松 SSL
    ctx = ssl.create_default_context()
    try:
        with urllib.request.urlopen(req, timeout=timeout, context=ctx) as resp:
            body = resp.read().decode("utf-8")
    except urllib.error.HTTPError as e:
        err = e.read().decode("utf-8", errors="ignore")
        raise RuntimeError(f"{cfg['name']} HTTP {e.code}: {err[:300]}")
    except (urllib.error.URLError, socket.timeout) as e:
        raise RuntimeError(f"{cfg['name']} 网络错误: {e}")

    # 解析响应
    obj = json.loads(body)
    try:
        return obj["choices"][0]["message"]["content"]
    except (KeyError, IndexError) as e:
        raise RuntimeError(f"{cfg['name']} 响应格式异常: {body[:300]}")


# ============================================================
# 降级:无 key 时用本地规则回复,保证前端不报错
# ============================================================
def fallback_reply(message: str) -> str:
    """
    没配 LLM key 时用 8 大关键词规则回复
    - 命中关键词 → 返回该主题的"专业穿搭建议"
    - 都不命中 → 返回引导式默认回复
    """
    msg = (message or "").strip()
    if not msg:
        return "请告诉我你想了解什么穿搭问题吧～"

    rules = [
        # 关键词列表     主题回复(多行)
        (["天气", "温度", "冷", "热", "下雨", "下雪"], [
            "🌤️ 天气穿搭小贴士:\n",
            "• 25°C+:轻薄 T 恤 / 短裙 / 透气鞋\n",
            "• 15-25°C:薄外套 + 长裤,早晚加件针织开衫\n",
            "• 5-15°C:风衣 / 夹克 + 围巾,层次感更出片\n",
            "• 0°C 以下:羽绒服 + 雪地靴,内搭注意叠穿",
        ]),
        (["约会", "见对象", "相亲"], [
            "💕 约会穿搭建议:\n",
            "• 休闲约会:简约白T + 高腰牛仔裤 + 小白鞋,清爽自然\n",
            "• 正式约会:碎花连衣裙 / 衬衫 + 西装裤,温柔大方\n",
            "• 颜色:浅粉、米白、雾霾蓝都是不出错的选择\n",
            "• 配饰:细项链 / 耳钉提气质,避免太夸张",
        ]),
        (["职场", "上班", "通勤", "面试"], [
            "💼 职场穿搭建议:\n",
            "• 男士:衬衫 + 西装裤 + 皮鞋,深蓝/灰/白最安全\n",
            "• 女士:西装外套 + 烟管裤 / 半身裙 + 低跟鞋\n",
            "• 避免:太过鲜艳的颜色、破洞元素、运动鞋\n",
            "• 配饰:简约腕表 / 细项链,专业感拉满",
        ]),
        (["小个子", "显高", "矮", "身高"], [
            "📏 小个子显高公式:\n",
            "• 上短下长:短款上衣 + 高腰下装,腿长立刻 +10cm\n",
            "• 同色系穿搭:视觉延伸,显高显瘦\n",
            "• 鞋子:尖头鞋 / 厚底乐福,避免复杂鞋带款\n",
            "• 裤子:九分微喇 / 直筒,露出脚踝最关键",
        ]),
        (["显瘦", "减肥", "微胖", "胖"], [
            "✨ 显瘦穿搭技巧:\n",
            "• 颜色:深色为主,V 领/方领拉长脖颈\n",
            "• 版型:H 型 / A 字型最遮肉,避开 oversize\n",
            "• 面料:硬挺有型 > 软塌贴身\n",
            "• 配饰:细腰带强调腰线,提升精致度",
        ]),
        (["颜色", "配色", "搭配", "撞色"], [
            "🎨 万能配色法则:\n",
            "• 基础款:黑+白+灰+米色,任意组合不出错\n",
            "• 同色系深浅搭配:高级感拉满\n",
            "• 撞色:小面积点缀即可(包包/鞋子/配饰)\n",
            "• 亚洲肤色友好:莫兰迪色系 / 雾霾蓝 / 浅卡其",
        ]),
        (["男生", "男士", "男朋友"], [
            "👔 男士穿搭速成:\n",
            "• 基础款:白T + 牛仔裤 + 帆布鞋,永不过时\n",
            "• 进阶:衬衫叠穿、针织开衫、工装裤\n",
            "• 关键:合身 > 品牌,合身的优衣库胜过不合身的奢侈品\n",
            "• 鞋子:小白鞋 / 乐福鞋 / 沙漠靴,三双轮换",
        ]),
        (["女生", "女士", "女朋友", "裙子"], [
            "👗 女生穿搭灵感:\n",
            "• 通勤:衬衫 + 半裙 + 玛丽珍,温柔又专业\n",
            "• 周末:卫衣 + 直筒裤 + 运动鞋,舒适随性\n",
            "• 约会:碎花裙 + 草编包,浪漫满分\n",
            "• 显高:短上衣 + 高腰裤,黄金比例 3:7",
        ]),
    ]

    # 顺序匹配:命中第一个含关键词的规则就返回
    for keywords, answer in rules:
        for kw in keywords:
            if kw in msg:
                return "".join(answer)

    # 没命中 → 引导式默认回复
    return (
        "👗 我是 AI 穿搭助手,可以根据你的需求给出搭配建议。\n\n"
        "试试问我:\n"
        "• 今天天气适合穿什么?\n"
        "• 约会怎么穿才好看?\n"
        "• 职场穿搭有什么建议?\n"
        "• 小个子怎么搭配显高?\n\n"
        "💡 当前为本地规则回复。如需 AI 大模型回复(完全免费):\n"
        "1. 打开 https://bigmodel.cn/ 注册智谱 AI 账号\n"
        "2. 进入「API Keys」创建一把 key(实名后完全免费,无额度限制)\n"
        "3. 把 key 填到项目根目录的 .env 文件的 ZHIPU_API_KEY= 后面\n"
        "4. 重启 Flask 即可启用真正的 AI 助手(智谱 GLM-4-Flash 模型,响应快、国内直连)"
    )
