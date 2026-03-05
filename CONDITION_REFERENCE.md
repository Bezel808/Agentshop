# 实验条件配置速查 (Condition Reference)

通过 YAML 条件文件控制商品价格、描述、标签等，用于行为经济学实验。  
Web 模式下通过 `--condition-name` 传递条件名给搜索接口。

---

## 快速示例

```bash
# 指定实验条件名（Web 搜索会按条件返回商品）
python run_browser_agent.py --query mousepad --condition-name treatment
```

---

## YAML 结构

```yaml
name: 实验名称
design: between   # between | within | latin_square
seed: 42          # 可选，复现随机

conditions:
  - name: 条件名
    description: "可选说明"
    product_overrides: {}      # 按商品 ID 覆盖
    position_overrides: {}     # 按位置覆盖
    global_overrides: {}       # 全局覆盖所有商品
```

---

## 1. 统一 / 修改价格

| 键 | 作用 | 示例 |
|----|------|------|
| `price` | 固定价格 | `price: 9.99` |
| `price_multiplier` | 价格倍数 | `price_multiplier: 1.2` |
| `price_noise` | 随机浮动 [min, max] | `price_noise: [-0.5, 0.5]` |
| `original_price` | 显示「原价」锚点 | `original_price: 19.99` |

**统一所有商品为 9.99：**
```yaml
global_overrides:
  price: 9.99
```

**按商品单独设价：**
```yaml
product_overrides:
  mousepad_0: { price: 12.99 }
  mousepad_1: { price: 8.99, original_price: 15.99 }
```

**按位置设价（第 0 个 = 第一个）：**
```yaml
position_overrides:
  0: { price: 49.99 }
  1: { price: 9.99 }
```

---

## 2. 修改商品标签 (Badges)

| 键 | 作用 |
|----|------|
| `sponsored` | 广告/赞助 |
| `best_seller` | 畅销 |
| `overall_pick` | 编辑推荐 |
| `low_stock` | 库存紧张 |
| `custom_badges` | 自定义标签列表 |

**示例：**
```yaml
# 全局关闭所有标签
global_overrides:
  sponsored: false
  best_seller: false
  overall_pick: false
  low_stock: false

# 给指定商品加标签
product_overrides:
  mousepad_0:
    best_seller: true
    sponsored: false
  mousepad_1:
    custom_badges: ["限时特惠", "新品"]

# 给指定位置加标签（推荐）
global_overrides:
  inject_labels:
    best_seller: [0]      # 第 1 个商品
    sponsored: [1, 2]     # 第 2、3 个商品
```

---

## 3. 修改标题 / 描述

| 键 | 作用 | 示例 |
|----|------|------|
| `title` | 完全替换标题 | `title: "新品鼠标垫"` |
| `title_prefix` | 标题前加 | `title_prefix: "🔥 "` |
| `title_suffix` | 标题后加 | `title_suffix: " (热卖)"` |
| `title_replace` | 文本替换 | `title_replace: {"旧词": "新词"}` |
| `description` | 商品描述 | `description: "全新描述..."` |

**示例：**
```yaml
product_overrides:
  mousepad_0:
    title_prefix: "[Premium] "
    title_suffix: " - 限时特惠"
    description: "高端布料，顺滑耐用，适合办公与游戏。"
  mousepad_1:
    title_replace: {"Gaming": "电竞"}
```

---

## 4. 评分与其它

| 键 | 作用 |
|----|------|
| `rating` | 评分 (如 4.5) |
| `rating_count` | 评价数 |
| `image_url` | 图片 URL |

**示例：**
```yaml
product_overrides:
  mousepad_0:
    rating: 4.8
    rating_count: 10000
```

---

## 5. 商品 ID 与位置

- **product_overrides**：键为商品 `id`，来自 JSON 的 `sku` 字段（如 `mousepad_0`、`mousepad_6`）。  
  查看数据集：`./aces_ctl.sh datasets` 或打开对应 `datasets_*/xxx.json` 看 `sku`。
- **position_overrides**：键为 0-based 位置（0 = 第一个，1 = 第二个）。
- **global_overrides**：作用在所有商品；若同时有 `product_overrides`，后者优先生效。

---

## 6. 打乱顺序

```yaml
global_overrides:
  shuffle_seed: 123   # 用此 seed 打乱商品顺序
```

---

## 7. 完整示例

```yaml
name: my_study
design: between
seed: 42

conditions:
  - name: control
    description: "基线"
    global_overrides: {}

  - name: treatment
    description: "统一价格 + 首商品加标签"
    product_overrides:
      mousepad_0:
        best_seller: true
        title_prefix: "🔥 "
      mousepad_1:
        description: "性价比之选"
    global_overrides:
      price: 9.99
      inject_labels:
        best_seller: [0]
```

---

## 参考文件

- `configs/experiments/example_price_anchoring.yaml` — 价格锚定
- `configs/experiments/example_decoy_effect.yaml` — 诱饵效应
- `configs/experiments/example_label_framing.yaml` — 标签框架
- 源码：`aces/experiments/control_variables.py`
