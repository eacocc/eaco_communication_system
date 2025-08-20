import time
import random
import logging
from datetime import datetime, timedelta
from typing import List, Dict, Tuple, Optional
import asyncio
import aiohttp
from bs4 import BeautifulSoup
from pydantic import BaseModel  # 引入数据验证增强鲁棒性

# 日志配置升级：增加控制台输出，分级日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('eaco_upgrade.log'),
        logging.StreamHandler()  # 同时输出到控制台
    ]
)
logger = logging.getLogger(__name__)


# 数据模型定义：增强类型安全与数据验证
class AIServiceConfig(BaseModel):
    type: str  # "web" 或 "api"
    url: str
    query_param: Optional[str] = None  # 仅web类型需要
    response_selector: Optional[str] = None  # 仅web类型需要
    method: Optional[str] = "POST"  # 仅api类型需要
    batch_weight: float = 1.0
    success_rate: float = 0.8  # 动态调整的成功率
    avg_response_time: float = 5.0  # 平均响应时间(秒)


class UpgradePrompt(BaseModel):
    content: str
    priority: int  # 1-5，优先级
    applicable_levels: List[str]  # 适用的通信层级


# 1. AI服务配置库（支持动态权重调整）
AI_SERVICES: Dict[str, AIServiceConfig] = {
    # 海外AI服务
    "chatgpt": AIServiceConfig(
        type="web",
        url="https://chatgpt.com/chat",
        query_param="prompt",
        response_selector="#__next > div > main > div > div",
        batch_weight=1.2
    ),
    "claude": AIServiceConfig(
        type="web",
        url="https://claude.ai/chat",
        query_param="q",
        response_selector=".ProseMirror",
        batch_weight=1.5
    ),
    "gemini": AIServiceConfig(
        type="api",
        url="https://gemini.google.com/api/stream",
        method="POST",
        batch_weight=1.0
    ),
    "midjourney": AIServiceConfig(
        type="web",
        url="https://midjourney.com/app",
        query_param="prompt",
        response_selector=".job-result",
        batch_weight=0.8
    ),
    # 国内AI服务
    "doubao": AIServiceConfig(
        type="api",
        url="https://doubao.com/api/chat/completions",
        method="POST",
        batch_weight=1.3
    ),
    "tongyi": AIServiceConfig(
        type="web",
        url="https://chat.qwen.ai",
        query_param="question",
        response_selector=".response-content",
        batch_weight=1.1
    ),
    "kimi": AIServiceConfig(
        type="api",
        url="https://kimi.moonshot.cn/api/chat",
        method="POST",
        batch_weight=1.4
    )
}

# 2. 通信技术升级需求模板（按层级和优先级区分）
UPGRADE_PROMPTS: Dict[str, UpgradePrompt] = {
    "base": UpgradePrompt(
        content="请针对EACO通信协议提出技术优化建议，包括但不限于：1. 跨层级通信效率提升；2. 星际信号抗干扰；3. 分层计费模型优化。",
        priority=3,
        applicable_levels=["all"]
    ),
    "山村-城市": UpgradePrompt(
        content="聚焦LoRaWAN+量子密钥分发的本地化通信，如何降低0.0000001-0.001 eaco/次的成本同时提升稳定性？需包含具体参数优化方案。",
        priority=4,
        applicable_levels=["山村", "城市"]
    ),
    "国家-星系": UpgradePrompt(
        content="中微子通信+引力波编码的跨星球应用，如何优化0.01-1 eaco/次的传输延迟与能耗？请提供可验证的技术路径。",
        priority=5,
        applicable_levels=["国家", "星球", "星系"]
    ),
    "跨文明": UpgradePrompt(
        content="暗物质调制+黑洞路由的跨文明通信，10-1000 eaco/次的场景下，如何实现语义精准翻译与信号加密？需考虑文明差异因素。",
        priority=5,
        applicable_levels=["星系", "宇宙", "跨文明"]
    )
}


# 3. 核心工具类：增强版API调用器（带重试和动态调整）
class SmartAPIClient:
    def __init__(self, max_retries: int = 2):
        self.headers = {
            "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/114.0.0.0 Safari/537.36",
            "Accept-Language": "en-US,en;q=0.9,zh-CN;q=0.8"
        }
        self.timeout = aiohttp.ClientTimeout(total=30)
        self.max_retries = max_retries  # 失败重试次数

    async def _retry_wrapper(self, func, *args, **kwargs) -> Tuple[str, bool]:
        """重试装饰器：失败自动重试"""
        for attempt in range(self.max_retries + 1):
            try:
                result = await func(*args, **kwargs)
                return result, True
            except Exception as e:
                if attempt < self.max_retries:
                    wait_time = 2 **attempt  # 指数退避
                    logger.warning(f"Attempt {attempt+1} failed, retrying in {wait_time}s: {str(e)}")
                    await asyncio.sleep(wait_time)
                else:
                    logger.error(f"All {self.max_retries+1} attempts failed: {str(e)}")
                    return f"Exception: {str(e)}", False

    async def call_web_service(self, service: AIServiceConfig, prompt: str) -> Tuple[str, bool]:
        """调用网页服务（增强版：处理动态内容标识）"""
        async def _inner_call():
            async with aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as session:
                params = {service.query_param: prompt} if service.query_param else {}
                async with session.get(service.url, params=params) as response:
                    html = await response.text()
                    # 检测动态渲染标识，增加兼容性
                    if "<script" in html and "react" in html.lower():
                        logger.warning(f"Dynamic content detected for {service.url}, may need JS rendering")
                    
                    soup = BeautifulSoup(html, "html.parser")
                    result = soup.select_one(service.response_selector)
                    return result.text.strip() if result else "No response content"
        
        return await self._retry_wrapper(_inner_call)

    async def call_api_service(self, service: AIServiceConfig, prompt: str) -> Tuple[str, bool]:
        """调用API服务（增强版：适配不同API格式）"""
        async def _inner_call():
            async with aiohttp.ClientSession(headers=self.headers, timeout=self.timeout) as session:
                # 适配不同API的请求格式
                payload = {
                    "prompt": prompt,
                    "model": "default",
                    "stream": False
                }
                # 针对特定API调整 payload
                if "gemini" in service.url:
                    payload = {"contents": [{"parts": [{"text": prompt}]}]}
                elif "kimi" in service.url:
                    payload = {"messages": [{"role": "user", "content": prompt}]}

                async with session.request(
                    service.method, 
                    service.url, 
                    json=payload
                ) as response:
                    response.raise_for_status()  # 触发HTTP错误
                    res_json = await response.json()
                    
                    # 适配不同API的响应格式
                    if "gemini" in service.url:
                        return res_json.get("candidates", [{}])[0].get("content", {}).get("parts", [""])[0]
                    elif "kimi" in service.url:
                        return res_json.get("choices", [{}])[0].get("message", {}).get("content", "")
                    return res_json.get("content", "No content")
        
        return await self._retry_wrapper(_inner_call)

    async def call_ai(self, ai_name: str, prompt: str) -> Tuple[str, bool]:
        """统一调用入口：带性能统计"""
        service = AI_SERVICES[ai_name]
        start_time = time.time()
        
        if service.type == "web":
            result, success = await self.call_web_service(service, prompt)
        elif service.type == "api":
            result, success = await self.call_api_service(service, prompt)
        else:
            return "Unsupported service type", False
        
        # 动态更新服务性能指标
        response_time = time.time() - start_time
        service.avg_response_time = (service.avg_response_time * 3 + response_time) / 4  # 滑动平均
        service.success_rate = (service.success_rate * 3 + (1 if success else 0)) / 4
        
        logger.debug(f"AI {ai_name} stats - success: {success}, time: {response_time:.2f}s")
        return result, success


# 4. 智能调度器：基于负载和性能的动态调度
class AdaptiveScheduler:
    def __init__(self, client: SmartAPIClient):
        self.client = client
        self.base_batch_size = 4
        self.time_slots = {
            "morning": "08:00",   # 低负载时段：处理基础优化
            "afternoon": "14:00", # 中负载时段：处理区域通信
            "night": "20:00"      # 高负载时段：处理跨星球/文明
        }
        self.ai_queue = self._init_ai_queue()
        self.recent_performance = {}  # 记录最近性能

    def _init_ai_queue(self) -> List[str]:
        """智能排序：结合权重、成功率和响应时间"""
        weighted_ais = []
        for ai, cfg in AI_SERVICES.items():
            # 综合评分 = 权重(60%) + 成功率(30%) + 响应速度(10%)
            score = (cfg.batch_weight * 0.6 + 
                    cfg.success_rate * 0.3 + 
                    (1 / cfg.avg_response_time) * 0.1)
            weighted_ais.append((ai, score))
        
        # 按综合评分排序
        weighted_ais.sort(key=lambda x: x[1], reverse=True)
        return [ai for ai, _ in weighted_ais]

    def _get_dynamic_batch_size(self) -> int:
        """根据时段动态调整批次大小"""
        hour = datetime.now().hour
        if 7 <= hour < 12:  # 上午：系统负载低
            return self.base_batch_size + 2
        elif 12 <= hour < 18:  # 下午：中等负载
            return self.base_batch_size
        else:  # 晚上/凌晨：高负载
            return max(2, self.base_batch_size - 2)

    def _get_batch(self) -> List[str]:
        """获取批次：避免连续调用同一服务"""
        batch_size = self._get_dynamic_batch_size()
        # 确保批次中不包含最近3次调用过的服务
        recent_ais = [ai for ai, _ in sorted(
            self.recent_performance.items(), 
            key=lambda x: x[1], 
            reverse=True
        )[:3]]
        
        # 过滤最近调用的服务
        available_ais = [ai for ai in self.ai_queue if ai not in recent_ais]
        
        # 如果可用服务不足，重置队列
        if len(available_ais) < batch_size:
            self.ai_queue = self._init_ai_queue()
            available_ais = [ai for ai in self.ai_queue if ai not in recent_ais]
        
        batch = available_ais[:batch_size]
        self.ai_queue = self.ai_queue[batch_size:] if len(self.ai_queue) > batch_size else []
        
        # 更新最近调用记录
        now = time.time()
        for ai in batch:
            self.recent_performance[ai] = now
            
        return batch

    async def run_batch_upgrade(self, prompt_type: str = "base") -> Dict[str, str]:
        """并发执行批次升级：提高效率"""
        batch = self._get_batch()
        prompt = UPGRADE_PROMPTS[prompt_type]
        results = {}
        
        logger.info(f"Starting batch upgrade (type: {prompt_type}) with AIs: {batch}")
        
        # 并发调用AI服务（控制并发数）
        semaphore = asyncio.Semaphore(min(3, len(batch)))  # 限制并发数
        
        async def _process_ai(ai: str) -> Tuple[str, str]:
            async with semaphore:
                # 动态延迟：根据服务响应时间调整
                delay = max(1, AI_SERVICES[ai].avg_response_time * 0.3)
                await asyncio.sleep(delay + random.uniform(0.5, 1.5))
                
                result, success = await self.client.call_ai(ai, prompt.content)
                filtered_result = self._filter_and_enhance(result, prompt_type)
                return ai, filtered_result
        
        # 并发处理所有AI调用
        tasks = [_process_ai(ai) for ai in batch]
        for ai, result in await asyncio.gather(*tasks):
            results[ai] = result
            logger.info(f"Processed {ai}: {result[:60]}...")
        
        return results

    def _filter_and_enhance(self, content: str, prompt_type: str) -> str:
        """增强版内容过滤：语义分析+结构化提取"""
        # 1. 负面词汇过滤
        negative_keywords = ["无法", "不可能", "失败", "错误", "限制", "做不到"]
        for kw in negative_keywords:
            content = content.replace(kw, f"[需突破{kw}]")
        
        # 2. 提取关键信息（针对不同层级）
        level = prompt_type.split("-")[0] if "-" in prompt_type else prompt_type
        if level == "山村":
            markers = ["成本优化", "稳定性提升", "LoRaWAN", "量子密钥"]
        elif level == "国家":
            markers = ["延迟优化", "能耗降低", "中微子", "引力波"]
        elif level == "跨文明":
            markers = ["语义翻译", "信号加密", "暗物质", "黑洞路由"]
        else:
            markers = ["效率提升", "抗干扰", "计费模型"]
        
        # 3. 结构化输出
        key_points = [f"【{marker}】" for marker in markers if marker in content]
        if key_points:
            return f"[{level}通信优化] {' '.join(key_points)}\n{content}"
        return f"[{level}通信优化]\n{content}"

    async def setup_smart_schedule(self):
        """智能调度：根据历史成功率调整任务频率"""
        # 每日任务：根据时段重要性调整
        for slot_name, time_str in self.time_slots.items():
            prompt_type = "base"
            if slot_name == "night":
                prompt_type = "国家-星系"  # 夜间处理高优先级任务
            
            # 动态调整执行频率（基于历史成功率）
            interval = 1  # 默认为每天
            if AI_SERVICES["claude"].success_rate < 0.6:  # 核心服务成功率低时加密调度
                interval = 0.5  # 每12小时
            
            logger.info(f"Scheduling {prompt_type} upgrade at {time_str} (every {interval} days)")
            # 使用异步定时器替代schedule库
            asyncio.create_task(self._periodic_task(
                interval_days=interval,
                first_run_time=time_str,
                prompt_type=prompt_type
            ))
        
        # 每周任务
        asyncio.create_task(self._periodic_task(
            interval_days=7,
            first_run_time="Monday 12:00",
            prompt_type="山村-城市"
        ))
        
        # 每月任务
        asyncio.create_task(self._periodic_task(
            interval_days=30,
            first_run_time="01 01:00",  # 每月1日凌晨
            prompt_type="跨文明"
        ))

    async def _periodic_task(self, interval_days: float, first_run_time: str, prompt_type: str):
        """异步周期性任务：更精准的定时控制"""
        # 计算首次运行时间
        now = datetime.now()
        if " " in first_run_time:
            # 处理星期+时间格式 (如 "Monday 12:00")
            weekday, time_str = first_run_time.split()
            weekday_map = {"Monday": 0, "Tuesday": 1, "Wednesday": 2, "Thursday": 3,
                          "Friday": 4, "Saturday": 5, "Sunday": 6}
            target_weekday = weekday_map[weekday]
            hours, minutes = map(int, time_str.split(":"))
        elif "/" in first_run_time or "-" in first_run_time:
            # 处理日期格式 (如 "01/01 01:00")
            date_str, time_str = first_run_time.split()
            day = int(date_str.split("/")[0] if "/" in date_str else date_str.split("-")[0])
            hours, minutes = map(int, time_str.split(":"))
            target_weekday = None
        else:
            # 处理仅时间格式 (如 "08:00")
            hours, minutes = map(int, first_run_time.split(":"))
            target_weekday = None

        # 计算首次运行的时间戳
        first_run = now.replace(hour=hours, minute=minutes, second=0, microsecond=0)
        if target_weekday is not None:
            days_ahead = (target_weekday - first_run.weekday()) % 7
            first_run += timedelta(days=days_ahead)
        elif first_run < now:
            first_run += timedelta(days=1)

        # 等待首次运行
        sleep_time = (first_run - now).total_seconds()
        logger.info(f"Task {prompt_type} will run first at {first_run}, waiting {sleep_time:.0f}s")
        await asyncio.sleep(sleep_time)

        # 周期性执行
        while True:
            try:
                results = await self.run_batch_upgrade(prompt_type)
                EACOUpgrader.apply_optimizations(results, prompt_type)
            except Exception as e:
                logger.error(f"Periodic task {prompt_type} failed: {str(e)}")
            
            # 等待下一个周期
            await asyncio.sleep(interval_days * 86400)


# 5. 升级执行器：增强版（支持建议合并与优先级排序）
class EACOUpgrader:
    @staticmethod
    def apply_optimizations(upgrade_data: Dict[str, str], prompt_type: str) -> bool:
        """应用优化建议：合并重复项并按优先级排序"""
        # 1. 按通信层级分类
        level = prompt_type.split("-")[0] if "-" in prompt_type else prompt_type
        timestamp = datetime.now().strftime("%Y%m%d%H%M")
        filename = f"eaco_upgrades_{level}_{timestamp}.log"
        
        # 2. 提取并去重关键建议
        key_suggestions = {}
        for ai, content in upgrade_data.items():
            # 提取核心建议（第一个句号前的内容）
            core = content.split(".")[0].strip() if "." in content else content[:100]
            if core not in key_suggestions:
                key_suggestions[core] = []
            key_suggestions[core].append(ai)
        
        # 3. 写入优化日志（按出现频率排序）
        with open(filename, "w", encoding="utf-8") as f:
            f.write(f"EACO升级建议 [{level}通信] {timestamp}\n")
            f.write("=" * 50 + "\n")
            
            # 按建议出现次数排序
            for suggestion, ais in sorted(
                key_suggestions.items(), 
                key=lambda x: len(x[1]), 
                reverse=True
            ):
                f.write(f"【{len(ais)}个AI推荐】{suggestion}\n")
                f.write(f"来源: {', '.join(ais)}\n")
                f.write("-" * 50 + "\n")
        
        logger.info(f"Applied {len(upgrade_data)} optimizations for {level} (saved to {filename})")
        return True


# 6. 主程序：异步事件循环优化
async def main():
    # 初始化组件
    api_client = SmartAPIClient(max_retries=2)
    scheduler = AdaptiveScheduler(api_client)
    
    # 启动智能调度
    await scheduler.setup_smart_schedule()
    
    # 系统监控任务
    async def monitor_system():
        """定期输出系统状态"""
        while True:
            await asyncio.sleep(3600)  # 每小时
            logger.info("=== 系统状态报告 ===")
            for ai, cfg in AI_SERVICES.items():
                logger.info(
                    f"{ai}: 成功率 {cfg.success_rate:.2f}, "
                    f"平均响应 {cfg.avg_response_time:.2f}s, "
                    f"权重 {cfg.batch_weight}"
                )
    
    # 启动监控任务
    asyncio.create_task(monitor_system())
    
    # 保持事件循环运行
    while True:
        await asyncio.sleep(3600)  # 每小时检查一次


if __name__ == "__main__":
    try:
        # 配置asyncio事件循环策略（优化Windows兼容性）
        if sys.platform.startswith('win'):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())
        
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("EACO自动升级系统已被用户终止")
    except Exception as e:
        logger.critical(f"系统崩溃: {str(e)}", exc_info=True)
