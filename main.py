import ray
import time
import random

# 初始化 Ray 分布式计算引擎
ray.init(ignore_reinit_error=True)

# ==========================================
# 1. 定义 Worker Agent (模拟异构旧设备节点)
# ==========================================
@ray.remote
class WorkerAgent:
    def __init__(self, node_id):
        self.node_id = node_id
        # 初始资源状态
        self.electricity_tokens = 0     # 初始电费 Token
        
        # 拆分三大核心硬件的占用率 (%)
        self.cpu_usage = random.uniform(1.0, 5.0)
        self.gpu_usage = 0.0  
        self.npu_usage = 0.0  
        
        self.temperature = random.uniform(35.0, 45.0) # 模拟硬件温度 (°C)
        self.current_task = "Idle"
        self.reward_points = 0 # 累积奖励积分
        self.end_points = 0 #最终分数

    def get_state(self):
        """获取当前节点状态"""
        return {
            "id": self.node_id,
            "tokens": self.electricity_tokens,
            "cpu": round(self.cpu_usage, 2),
            "gpu": round(self.gpu_usage, 2),
            "npu": round(self.npu_usage, 2),
            "temp": round(self.temperature, 2),
            "task": self.current_task,
            "reward": self.end_points
        }

    def run_low_power_task(self):
        """
        饥饿机制：利用 NPU 进行低功耗边缘推理，赚取“电费 Token”
        """
        self.current_task = "Edge Inference (NPU赚Token)"
        self.cpu_usage = random.uniform(5.0, 15.0) 
        self.gpu_usage = 0.0 # 关闭 GPU 省电
        self.npu_usage = random.uniform(40.0, 80.0) # NPU 高效工作
        
        # NPU 功耗低，发热小，甚至能让整体系统散热降温
        self.temperature = max(35.0, self.temperature - random.uniform(2.0, 5.0)) 
        self.electricity_tokens += random.randint(4, 7) # 赚取 Token 奖励
        time.sleep(0.5) 

    def run_heavy_task(self, task_name):
        """
        高性能核心抢占：消耗 Token 执行重负载任务 (区分 GPU 与 CPU 职责)
        """
        if self.electricity_tokens >= 10:
            self.current_task = f"Heavy ({task_name})"
            
            if task_name == "LLM Prefill (GPU)":
                # GPU 负责极耗资源的 Prompt 预处理和大型矩阵乘法
                self.cpu_usage = random.uniform(20.0, 40.0) # CPU 负责发指令和数据调度
                self.gpu_usage = random.uniform(85.0, 99.0) # GPU 满载
                self.npu_usage = 0.0
                
                self.temperature += random.uniform(20.0, 35.0) # GPU 导致温度狂飙
                self.electricity_tokens -= 12 # GPU 极度费电，消耗更多 Token
                self.reward_points += 10 # 高难度任务高回报
                
            elif task_name == "Data Tokenization (CPU)":
                # CPU 负责大语言模型的数据分词预处理、I/O 等逻辑密集型任务
                self.cpu_usage = random.uniform(85.0, 98.0) # CPU 满载
                self.gpu_usage = 0.0
                self.npu_usage = 0.0
                
                self.temperature += random.uniform(10.0, 20.0) # CPU 发热中等
                self.electricity_tokens -= 8 # CPU 费电一般
                self.reward_points += 6 # 中等回报

            
            time.sleep(1.0) 
            return True
        else:
            self.current_task = "Hungry (等待 NPU 赚取)"
            return False

    def apply_penalty(self, penalty_amount):
        """Master 下发的惩罚机制"""
        self.electricity_tokens = max(0, self.electricity_tokens - penalty_amount)
    
    def apply_reward(self):
        """Master 下发的奖励机制"""
        self.end_points += self.reward_points
        self.reward_points = 0 

# ==========================================
# 2. 定义 Master Agent (资源调度中枢)
# ==========================================
class MasterAgent:
    def __init__(self, num_workers=3):
        print(f"👑 Master Agent 启动，已接管 {num_workers} 台异构节点设备...\n")
        self.workers = [WorkerAgent.remote(f"Node-{i+1}") for i in range(num_workers)]

    def get_all_states(self):
        """同步拉取所有 Worker 的状态"""
        return ray.get([worker.get_state.remote() for worker in self.workers])

    def orchestrate(self, rounds=5):
        """核心调度逻辑（博弈与分配）"""
        for r in range(rounds):
            print(f"========== 🔄 资源调度回合 {r + 1} ==========")
            states = self.get_all_states()

            # --- 步骤 A: 监控与惩罚奖励机制 ---
            for worker, state in zip(self.workers, states):
                worker.apply_reward.remote() 
                
                # 触发惩罚的阈值判定：总温度过高，或者 GPU/CPU 长时间撞温度墙
                if state['temp'] > 85.0:
                    print(f"  [过热警告] {state['id']} 整体温度过高 ({state['temp']}°C)！扣除 15 Token 强制降频。")
                    worker.apply_penalty.remote(15)
                elif state['gpu'] > 95.0:
                    print(f"  [负载警告] {state['id']} GPU 撞击温度墙限制 (GPU: {state['gpu']}%)！扣除 10 Token。")
                    worker.apply_penalty.remote(10)
                elif state['cpu'] > 95.0:
                    print(f"  [负载警告] {state['id']} CPU 满载时间过长 (CPU: {state['cpu']}%)！扣除 5 Token。")
                    worker.apply_penalty.remote(5)

            # 刷新状态
            states = self.get_all_states()

            # --- 步骤 B: 任务分配与“饥饿感”机制 ---
            for worker, state in zip(self.workers, states):
                if state['tokens'] < 10:
                    # 触发“饥饿感”，设备必须去打工赚电费 (NPU 边缘计算)
                    print(f"  [打工] {state['id']} 处于饥饿状态 (Tokens: {state['tokens']})，切换 NPU 低功耗边缘推理赚取 Token。")
                    worker.run_low_power_task.remote()
                else:
                    # Token 充足，根据大模型流水线分配对应硬件任务
                    task_name = random.choice(["LLM Prefill (GPU)", "Data Tokenization (CPU)"])
                    print(f"  [抢占] {state['id']} 算力充足 (Tokens: {state['tokens']})，分配重负载任务: {task_name}")
                    worker.run_heavy_task.remote(task_name)

            # 等待异步任务执行一小会
            time.sleep(1.5)

            # --- 步骤 C: 打印本回合结算面板 (加入异构数据) ---
            print("\n  📊 回合结算状态摘要:")
            final_states = self.get_all_states()
            # 格式化表头对齐
            print(f"    {'节点':<8} | {'Tokens':<6} | {'CPU %':<6} | {'GPU %':<6} | {'NPU %':<6} | {'温度 °C':<7} | {'奖励分':<6} | 当前状态")
            print("    " + "-"*85)
            for s in final_states:
                print(f"    {s['id']:<8} | {s['tokens']:<6} | {s['cpu']:<6.2f} | {s['gpu']:<6.2f} | {s['npu']:<6.2f} | {s['temp']:<7.2f} | {s['reward']:<6} | {s['task']}")
            print("\n")

# ==========================================
# 3. 运行主程序
# ==========================================
if __name__ == "__main__":
    # 初始化 3 台旧设备的模拟集群
    master = MasterAgent(num_workers=5)
    
    # 运行 10 个周期的博弈调度
    master.orchestrate(rounds=10)
    
    # 关闭 Ray 集群
    ray.shutdown()