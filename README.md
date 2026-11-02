# 网球训练 Agent

上传一段击球视频，得到动作指标、短板诊断、训练计划，以及和自己上次的对比。

一个**用 CV 做动作量化、用 Agent 做编排**的项目：数字来自代码，模型只负责决定调什么工具、怎么表达。

---

## 先说不做什么

诚实的边界比功能列表更能说明项目成熟度：

- **不是诊断工具。** 指标没有教练标注做 ground truth，只能保证"可复现"，不能保证"能区分好坏动作"。
- **不是实时系统。** 单段视频离线分析，约 40 秒。
- **不是通用网球助手。** 知识库 43 条自建语料，覆盖不到的问题会明说"未收录"，不编。

---

## 能力

| 能力 | 说明 |
|---|---|
| 动作分析 | MediaPipe 33 关键点 → 腕速峰值切分击球 → 5 项体高归一化指标，产出标注视频与关键帧 |
| 纵向对比 | SQLite 持久化训练记录，按 `user_id` 隔离，对比首次与最近的指标稳定性（CV） |
| 训练计划 | 按 CV 排名定位最不稳定的指标，生成分周计划；短板由代码判定，模型只负责组织语言 |
| 知识问答 | 43 条自建知识库（技术动作 + 比赛规则），纯 Python BM25，回答强制标注出处 |
| 装备推荐 | 8 款球拍，硬约束过滤 + 多目标打分 |

编排：主调度器 + 2 个子智能体（技术分析师 4 工具 / 装备顾问 1 工具），星型拓扑，子图独立编译、上下文隔离。

```
                 ┌──────────────┐
   用户 ────────►│  主调度器     │
                 └──────┬───────┘
              task() ┌──┴───────────────┐ task()
                     ▼                  ▼
            ┌─────────────────┐  ┌─────────────┐
            │  技术分析师      │  │  装备顾问    │
            │ analyze_stroke  │  │ filter_     │
            │ get_progress    │  │  rackets    │
            │ make_training_  │  └─────────────┘
            │   plan          │
            │ search_technique│
            └─────────────────┘
```

工具统一返回结构化结果，上层用代码校验而不是解析自然语言：

```json
{"ok": true, "data": {...}, "evidence": {...}, "confidence": 0.96, "latency_ms": 4120}
```

---

## 快速开始

```bash
# 1. 环境（依赖见 requirements.txt）
conda create -n tennis python=3.11 && conda activate tennis
pip install -r requirements.txt

# 2. 配置模型（OpenAI 兼容协议，DeepSeek / 通义 / OpenAI 均可）
cp .env.example .env   # 填 OPENAI_API_KEY、OPENAI_BASE_URL、LLM_QWEN_MAX

# 3. 启动
python -m tennis_coach.api.server     # http://127.0.0.1:8010
```

> `LLM_QWEN_MAX` 是沿用自上游项目的变量名，实际可填任何模型名。

命令行跑单段视频：

```bash
python -m tennis_coach.run_minimal path/to/forehand.mp4
# 产出 data/runs/<视频名>/：pose.json / metrics.json / annotated.mp4 / keyframes/ / contact_sheet.jpg
```

评测：

```bash
python -m tennis_coach.evals.run_eval              # 16 条用例
python -m tennis_coach.evals.run_eval --repeat 3   # 区分抖动与确定性失败
python -m tennis_coach.tools.knowledge             # 检索冒烟（hit@1）
```

MCP 服务端（可选，供 Claude Desktop / Cursor 调用）：

```json
{
  "mcpServers": {
    "tennis-coach": {
      "command": "python",
      "args": ["-m", "tennis_coach.mcp_server"],
      "cwd": "/path/to/paper_agent"
    }
  }
}
```

---

## 评测

16 条用例，断言**两层轨迹**而不是回答文本：

```
主智能体层：该调度哪个子智能体        → expect_dispatch / forbid
子智能体层：里面实际调了哪个工具      → expect_tool / forbid_tool
输出层：  结论是否带依据              → must_contain ["依据"]
```

| 类型 | 数量 | 内容 |
|---|---|---|
| route | 7 | 该调度谁 |
| boundary | 2 | 越界与工具隔离 |
| gate | 1 | 该拒答 |
| scope | 2 | 闲聊/越界不该调度 |
| clarify | 1 | 缺参数先追问 |
| multi | 3 | 多轮，共用 thread_id |

```
单轮 15/16 · 检索 hit@1 13/13
```

`tool_isolation_gear` 存在路由抖动（全量跑 1 次失败，单独复跑 3/3 通过），
属于模型层面的不确定性，已记录而非隐藏。

评测抓到过的真实缺陷：

1. **门禁误伤**：`evaluate()` 对所有工具生效，装备推荐被当成"零击球"拒掉
2. **单项被拒后整轮终止**："分析视频 + 选球拍"里视频被拒后，选球拍那半句被丢了
3. **关键词假通过**：检索返回空，模型凭记忆答对了关键词，用例照样通过 —— 直到断言"必须带依据"才暴露

---

## 设计取舍

**① 推理能进代码就不要留给 Prompt**
短板判定由代码按 CV 排名，模型只能执行；质量门禁的拒答文案是确定性字符串，不经过模型。CoT 依赖模型愿意好好想，代码不依赖。

**② 结构限制优先于 Prompt 约束**
子智能体拿到历史视频路径后会"顺手"重跑分析、导致重复记账。修法不是加 prompt 规则，而是让返回值里只给文件名——**先让错误路径走不通，再补话术**。

**③ 判断"不该做"也是技术判断**
- 装备顾问换便宜模型：15 条用例全过、快 12%，但 1/6 会把"无匹配"说成"工具异常"并编造市场价 → 用数据否掉，回退
- MCP 全量迁移：工具是自有业务逻辑，序列化 + 长任务超时的代价换不来功能 → 只保留对外服务端

---

## 已知局限

| 局限 | 说明 |
|---|---|
| 指标区分度未验证 | 击球计数与人工核对一致（8 次），但"好动作 vs 坏动作"缺标注集 |
| 跨人对比无意义 | 指标未做人体尺度校准，不同人的绝对值不可横向比较 |
| 2D 单目 | `contact_offset` 丢失朝向纵深分量，侧视角下信息量偏低 |
| `pose_landmarker_full` 反而更差 | 同段视频 6 次 vs lite 的 8 次，抖动更大，固定用 lite |
| MCP 只覆盖 CLI | Web 服务仍用进程内工具，每次调用新起进程（约 2.5s） |
| `user_id` 不是认证 | 由客户端提供、服务端信任，多租户场景下可伪造 |
| 语料与装备为示例级 | 43 条知识、8 款球拍，非权威来源与真实 catalogue |

每一条都可以在面试里展开成"我知道边界在哪、下一步做什么"。

---

## 目录结构

```
tennis_coach/
├── core/          ToolResult 契约 / 上下文压缩 / 质量门禁 / SQLite 记忆
├── pose/          关键点提取 / 指标计算 / 可视化
├── tools/         5 个 @tool
├── knowledge/     43 条知识库
├── agent/         最小 ReAct 图 / 多智能体编排 / 事件流 / MCP 客户端
├── evals/         16 条用例 + 轨迹断言
├── api/           FastAPI + WebSocket + 对话式前端
└── mcp_server.py  MCP 服务端（5 工具，stdio）
```

设计笔记见 [`docs/Tennis-Agent-七阶段笔记.md`](docs/Tennis-Agent-七阶段笔记.md)。

---

## 来源与许可

架构范式参考开源项目 [MultiAgent-Search](https://github.com/jayergood/MultiAgent-Search)（星型多智能体、子智能体作为工具）。

**业务层为原创实现**：视频分析管线、动作指标、质量门禁、跨会话记忆、知识检索、回归评测、Web 服务与 MCP 封装。

姿态模型为 [MediaPipe Pose Landmarker](https://developers.google.com/mediapipe/solutions/vision/pose_landmarker)（Apache 2.0）。
