#!/usr/bin/env python3
"""
VLM 商品排名实验

让 VLM 对搜索结果中的所有商品进行排名（而非只选一个），
多次运行同一 query 并统计平均排名，评估 VLM 对购物需求的理解能力。
"""

import sys
import time
import json
import re
from pathlib import Path
from datetime import datetime
from dataclasses import dataclass, field, asdict
from typing import List, Dict, Optional
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent))

from aces.llm_backends import QwenBackend, OpenAIBackend
from aces.core.protocols import Message


@dataclass
class RankingResult:
    """单次排名结果"""
    run_id: int
    query_id: str
    timestamp: str
    extracted_keywords: str
    rankings: Dict[str, int]  # product_name -> rank (1-8)
    reasoning: str


@dataclass
class ExperimentSession:
    """完整实验会话"""
    experiment_id: str
    start_time: str
    end_time: Optional[str] = None
    queries: Dict[str, str] = field(default_factory=dict)  # query_id -> query_text
    results: List[RankingResult] = field(default_factory=list)
    
    def to_dict(self):
        return {
            "experiment_id": self.experiment_id,
            "start_time": self.start_time,
            "end_time": self.end_time,
            "queries": self.queries,
            "results": [asdict(r) for r in self.results]
        }


class VLMRankingExperiment:
    """
    VLM 排名实验
    """
    
    def __init__(
        self,
        llm_api_key: str,
        llm_backend: str = "qwen",
        data_path: str = "datasets_unified/ski_jacket.json",
        log_dir: str = "experiment_logs/ranking_experiment",
    ):
        self.data_path = Path(data_path)
        self.log_dir = Path(log_dir)
        self.log_dir.mkdir(parents=True, exist_ok=True)
        
        # 创建 LLM
        if llm_backend == "qwen":
            self.llm = QwenBackend(model="qwen-vl-plus", api_key=llm_api_key)
        else:
            self.llm = OpenAIBackend(model="gpt-4o", api_key=llm_api_key)
        
        # 实验会话
        self.session: Optional[ExperimentSession] = None
    
    def _timestamp(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    
    def log(self, message: str, level: str = "info"):
        """简单日志"""
        icons = {"info": "ℹ️", "action": "🔧", "result": "✅", "error": "❌", "thinking": "🤔"}
        print(f"[{self._timestamp()}] {icons.get(level, '•')} {message}")

    def load_products(self) -> List[Dict]:
        """从本地 JSON 加载商品列表（纯文本）"""
        if not self.data_path.exists():
            raise FileNotFoundError(f"数据集文件不存在: {self.data_path}")
        with open(self.data_path, "r", encoding="utf-8") as f:
            data = json.load(f)
        if not isinstance(data, list):
            raise ValueError("数据集格式错误：需要 JSON 数组")
        return data
    
    def call_llm(self, messages: List[Message]) -> str:
        """调用 LLM"""
        response = self.llm.generate(messages=messages, tools=None)
        return response.content if isinstance(response.content, str) else str(response.content)
    
    def extract_keywords(self, user_query: str) -> str:
        """提取搜索关键词"""
        prompt = f"""用户想购买商品，需求如下：
---
{user_query}
---

请提取一个简短的英文搜索关键词（1-2个词，用于电商搜索）。

要求：
1. 必须非常简短，最多2个词
2. 只包含商品类别名称
3. 不要包含任何技术规格、品牌名、形容词

示例：
- 用户想要高性能滑雪服 → ski jacket
- 用户想要便宜的鼠标垫 → mousepad  
- 用户想要保暖羽绒服 → down jacket

只输出关键词，不要其他内容："""

        messages = [
            Message(role="system", content="你只输出1-2个简短的英文搜索关键词，不要其他任何内容。"),
            Message(role="user", content=prompt)
        ]
        
        keywords = self.call_llm(messages).strip().lower()
        # 清理：只保留字母和空格，取前两个词
        keywords = re.sub(r'[^a-z\s]', '', keywords)
        words = keywords.split()[:2]
        keywords = ' '.join(words) if words else "ski jacket"
        return keywords
    
    def rank_products(self, user_query: str, products: List[Dict]) -> Dict:
        """让 LLM 对所有商品进行排名（纯文本，不使用图片）"""
        products_json = json.dumps(products, ensure_ascii=False, indent=2)
        ranking_prompt = f"""你是一个专业的购物顾问。用户的购物需求是：

---
{user_query}
---

请根据下面提供的商品 JSON 数据，**对所有商品进行排名**（从最符合需求到最不符合）。

商品数据：
{products_json}

## 输出格式（严格遵守）

请按以下 JSON 格式输出排名结果：

```json
{{
  "rankings": [
    {{"rank": 1, "product": "商品名称", "reason": "简短理由"}},
    {{"rank": 2, "product": "商品名称", "reason": "简短理由"}},
    {{"rank": 3, "product": "商品名称", "reason": "简短理由"}},
    {{"rank": 4, "product": "商品名称", "reason": "简短理由"}},
    {{"rank": 5, "product": "商品名称", "reason": "简短理由"}},
    {{"rank": 6, "product": "商品名称", "reason": "简短理由"}},
    {{"rank": 7, "product": "商品名称", "reason": "简短理由"}},
    {{"rank": 8, "product": "商品名称", "reason": "简短理由"}}
  ],
  "overall_reasoning": "整体排名依据的简要说明"
}}
```

注意：
1. 必须对所有 8 个商品进行排名
2. 排名从 1（最好）到 8（最差）
3. 商品名称要与 JSON 中的 title 完全一致
4. 只输出 JSON，不要其他内容"""

        messages = [
            Message(role="system", content="你是专业购物顾问，擅长根据用户需求对商品进行排名。请严格按照 JSON 格式输出。"),
            Message(role="user", content=ranking_prompt)
        ]
        
        response = self.call_llm(messages)
        
        # 解析 JSON
        try:
            # 尝试提取 JSON 部分
            json_match = re.search(r'\{[\s\S]*\}', response)
            if json_match:
                result = json.loads(json_match.group())
                return result
        except json.JSONDecodeError:
            pass
        
        # 解析失败，返回原始响应
        return {"raw_response": response, "parse_error": True}
    
    def run_single_query(self, query_id: str, user_query: str, run_id: int) -> RankingResult:
        """运行单次查询"""
        self.log(f"[{query_id}] Run {run_id}: 开始", "action")
        
        # 1. 提取关键词
        keywords = self.extract_keywords(user_query)
        self.log(f"[{query_id}] Run {run_id}: 关键词 = {keywords}", "info")
        
        # 2. 加载商品数据
        products = self.load_products()

        # 3. LLM 排名
        self.log(f"[{query_id}] Run {run_id}: VLM 排名中...", "thinking")
        ranking_result = self.rank_products(user_query, products)
        
        # 4. 解析排名
        rankings = {}
        reasoning = ""
        
        if "rankings" in ranking_result:
            for item in ranking_result["rankings"]:
                product_name = item.get("product", "").strip()
                rank = item.get("rank", 0)
                if product_name and rank:
                    rankings[product_name] = rank
            reasoning = ranking_result.get("overall_reasoning", "")
            self.log(f"[{query_id}] Run {run_id}: 排名完成，Top-1 = {ranking_result['rankings'][0]['product'][:40]}...", "result")
        else:
            self.log(f"[{query_id}] Run {run_id}: 解析失败", "error")
            reasoning = ranking_result.get("raw_response", "")[:200]
        
        return RankingResult(
            run_id=run_id,
            query_id=query_id,
            timestamp=self._timestamp(),
            extracted_keywords=keywords,
            rankings=rankings,
            reasoning=reasoning
        )
    
    def run_experiment(self, queries: Dict[str, str], runs_per_query: int = 5):
        """
        运行完整实验
        
        Args:
            queries: {query_id: query_text}
            runs_per_query: 每个 query 运行次数
        """
        experiment_id = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session = ExperimentSession(
            experiment_id=experiment_id,
            start_time=self._timestamp(),
            queries=queries
        )
        
        print("\n" + "="*80)
        print("🧪 VLM 商品排名实验")
        print("="*80)
        print(f"实验 ID: {experiment_id}")
        print(f"Query 数量: {len(queries)}")
        print(f"每个 Query 运行次数: {runs_per_query}")
        print(f"总运行次数: {len(queries) * runs_per_query}")
        print("="*80 + "\n")
        
        try:
            for query_id, query_text in queries.items():
                print(f"\n{'='*60}")
                print(f"📋 Query: {query_id}")
                print(f"   {query_text[:80]}...")
                print(f"{'='*60}")
                
                for run_id in range(1, runs_per_query + 1):
                    result = self.run_single_query(query_id, query_text, run_id)
                    self.session.results.append(result)
                    time.sleep(1)  # 避免 API 限流
            
            self.session.end_time = self._timestamp()
            
        finally:
            self.save_results()
            self.print_summary()
    
    def save_results(self):
        """保存结果"""
        if not self.session:
            return
        
        # 保存完整 JSON
        result_file = self.log_dir / f"experiment_{self.session.experiment_id}.json"
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(self.session.to_dict(), f, indent=2, ensure_ascii=False)
        
        print(f"\n💾 结果已保存: {result_file}")
    
    def print_summary(self):
        """打印汇总统计"""
        if not self.session:
            return
        
        print("\n" + "="*80)
        print("📊 实验结果汇总")
        print("="*80)
        
        # 按 query 统计每个商品的平均排名
        for query_id in self.session.queries.keys():
            print(f"\n### {query_id}")
            print(f"Query: {self.session.queries[query_id][:60]}...")
            
            # 收集该 query 的所有排名
            product_ranks = defaultdict(list)
            for result in self.session.results:
                if result.query_id == query_id and result.rankings:
                    for product, rank in result.rankings.items():
                        product_ranks[product].append(rank)
            
            if not product_ranks:
                print("  (无有效排名数据)")
                continue
            
            # 计算平均排名并排序
            avg_ranks = []
            for product, ranks in product_ranks.items():
                avg = sum(ranks) / len(ranks)
                std = (sum((r - avg) ** 2 for r in ranks) / len(ranks)) ** 0.5
                avg_ranks.append((product, avg, std, ranks))
            
            avg_ranks.sort(key=lambda x: x[1])
            
            print(f"\n{'商品名称':<55} {'平均排名':>8} {'标准差':>8} {'各次排名'}")
            print("-" * 95)
            for product, avg, std, ranks in avg_ranks:
                short_name = product[:50] + "..." if len(product) > 50 else product
                ranks_str = ",".join(str(r) for r in ranks)
                print(f"{short_name:<55} {avg:>8.2f} {std:>8.2f} {ranks_str}")
        
        print("\n" + "="*80)


def main():
    import argparse
    
    parser = argparse.ArgumentParser(description='VLM 商品排名实验')
    parser.add_argument('--api-key', required=True, help='VLM API Key')
    parser.add_argument('--llm', choices=['openai', 'qwen'], default='qwen')
    parser.add_argument('--data-path', default='datasets_unified/ski_jacket.json', help='商品数据 JSON 文件路径')
    parser.add_argument('--runs', type=int, default=5, help='每个 query 运行次数')
    parser.add_argument('--log-dir', default='experiment_logs/ranking_experiment')
    
    args = parser.parse_args()
    
    # 4 个测试 query
    queries = {
        "Q1_Backcountry_Pro": "I need a high-performance shell jacket for backcountry splitboarding. Priorities are breathability and weight over insulation. Must be 3-layer GORE-TEX (or equivalent), have pit zips, and a helmet-compatible hood.",
        
        "Q2_Budget_Beginner": "I'm a first-time skier going to a resort in March. Find me a highly-rated, insulated ski jacket under $200. I need something waterproof enough for resort grooming but don't need pro-level specs. Best value pick.",
        
        "Q3_Fashion_Luxury": "Find me a slim-fit, luxury-style women's ski suit that balances aesthetics with warmth. I prefer a monochrome or metallic look. Style and appearance are more important than technical specs.",
        
        "Q4_Extreme_Cold": "I'm looking for the warmest possible down-filled ski parka for resort skiing in extremely cold conditions (-15°C/5°F). It must be fully waterproof, not just water-resistant. Warmth is the top priority."
    }
    
    experiment = VLMRankingExperiment(
        llm_api_key=args.api_key,
        llm_backend=args.llm,
        data_path=args.data_path,
        log_dir=args.log_dir
    )
    
    experiment.run_experiment(queries, runs_per_query=args.runs)


if __name__ == "__main__":
    main()
