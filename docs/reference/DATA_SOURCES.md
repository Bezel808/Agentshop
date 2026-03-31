# ACES-v2 数据来源

## 概览

| 数据源 | 用途 | 脚本子命令 |
|--------|------|------------|
| **Amazon Reviews 2023** | 商品元数据、多图、评论 | expand-from-amazon, expand-by-keyword, enrich |
| **My-Custom-AI** (ACE-RS/SR/BB) | 实验记录格式商品 | download-ace |

---

## 1. Amazon Reviews 2023 (McAuley Lab)

- **地址**: https://huggingface.co/datasets/McAuley-Lab/Amazon-Reviews-2023
- **文件**:
  - `raw/meta_categories/meta_{Category}.jsonl`：商品元数据（标题、价格、图片、描述等）
  - `raw/review_categories/{Category}.jsonl`：用户评论
- **类目**: 33 个，如 Electronics, Office_Products, Home_and_Kitchen, Sports_and_Outdoors 等

### 使用方式

```bash
# 按类目扩充
python manage_datasets.py expand-from-amazon -c Electronics,Office_Products -n 50

# 按关键词扩充（支持: mousepad, ski, smart_watch, athletic_shoes_men/women, bluetooth_speaker, insulated_cup_men/women, backpack_men/women）
python manage_datasets.py expand-by-keyword -k smart_watch,athletic_shoes_men,athletic_shoes_women,bluetooth_speaker -n 80

# 补齐多图与评论
python manage_datasets.py enrich --input-dir datasets_unified --output-dir datasets_unified_multimodal
```

---

## 2. My-Custom-AI 系列

- **地址**: https://huggingface.co/My-Custom-AI
- **数据集**:
  - ACE-RS (Rationality Suite): absolute_and_random_price, instruction_following, rating, relative_price
  - ACE-SR (Search Results)
  - ACE-BB (Buying Behavior): choice_behavior
- **格式**: 实验记录（每行含 query、title 列表、price 列表等），需转为 ACES 商品 JSON

### 使用方式

```bash
python manage_datasets.py download-ace -o datasets_unified
```

---

## 3. datasets_unified 现有文件

当前 `datasets_unified/` 目录下的 JSON 文件及其来源：

| 文件名 | 来源 | 说明 |
|--------|------|------|
| electronics.json | Amazon 2023 / expand-from-amazon | 电子类 |
| office_products.json | Amazon 2023 | 办公用品 |
| home_and_kitchen.json | Amazon 2023 | 家居厨房 |
| sports_and_outdoors.json | Amazon 2023 | 运动户外 |
| health_and_household.json | Amazon 2023 | 健康家居 |
| beauty_and_personal_care.json | Amazon 2023 | 美妆个护 |
| toys_and_games.json | Amazon 2023 | 玩具游戏 |
| tools_and_home_improvement.json | Amazon 2023 | 工具家装 |
| pet_supplies.json | Amazon 2023 | 宠物用品 |
| cell_phones_and_accessories.json | Amazon 2023 | 手机配件 |
| mousepad.json | Amazon 2023 / expand-by-keyword | 关键词 mousepad（Office_Products + Electronics） |
| ski_jacket.json | Amazon 2023 / expand-by-keyword | 关键词 ski（Sports + Clothing） |
| smart_watch.json | Amazon 2023 / expand-by-keyword | 智能手表（享乐+功能，Electronics + Cell_Phones） |
| athletic_shoes_men.json | Amazon 2023 / expand-by-keyword | 运动鞋-男款 |
| athletic_shoes_women.json | Amazon 2023 / expand-by-keyword | 运动鞋-女款 |
| bluetooth_speaker.json | Amazon 2023 / expand-by-keyword | 蓝牙音箱（设计+音质） |
| insulated_cup_men.json | Amazon 2023 / expand-by-keyword | 保温杯-男款 |
| insulated_cup_women.json | Amazon 2023 / expand-by-keyword | 保温杯-女款 |
| backpack_men.json | Amazon 2023 / expand-by-keyword | 双肩包-男款 |
| backpack_women.json | Amazon 2023 / expand-by-keyword | 双肩包-女款 |

**补充字段（enrich）**：从 Amazon Reviews 2023 的 meta 和 review 文件补齐 `image_urls`、`reviews`、`parent_asin` 等。
