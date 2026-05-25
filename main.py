import ray
import time
import random

# 初始化 Ray 分布式计算引擎
ray.init(ignore_reinit_error=True)

# ==========================================
# 1. 定义 Worker Agent (模拟旧设备节点)
# ==========================================
@ray.remote
class WorkerAgent:
    def __init__(self, node_id):
        self.node_id = node_id
        # 初始资源状态
        self.electricity_tokens = 0     # 初始电费 Token
        self.cpu_usage = random.uniform(5.0, 15.0)  # 模拟 CPU 占用率 (%)
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
            "temp": round(self.temperature, 2),
            "task": self.current_task,
            "reward": self.end_points
        }

    def run_low_power_task(self):
        """
        饥饿机制：低功耗运行以赚取“电费 Token”
        """
        self.current_task = "Low Power (赚取 Token)"
        self.cpu_usage = random.uniform(5.0, 20.0) # 维持低占用
        self.temperature = max(30.0, self.temperature - random.uniform(2.0, 8.0)) # 散热降温
        self.electricity_tokens += random.randint(4, 6) # 赚取 Token 奖励
        time.sleep(0.5) # 模拟运行耗时

    def run_heavy_task(self, task_name="LLM 推理"):
        """
        高性能核心抢占：消耗 Token 执行重负载任务 (如 LLM/大型编译)
        """
        # 抢占门槛
        if self.electricity_tokens >= 10:
            self.current_task = f"Heavy Task ({task_name})"
            # 模拟高负载带来的硬件压力
            if task_name == "LLM 推理":
                self.cpu_usage = random.uniform(85.0, 98.0) # 模拟高占用
                self.temperature += random.uniform(20.0, 30.0) # 模拟温度升高
                self.electricity_tokens -= 10 # 支付 Token 费用
                self.reward_points += 8 # 完成高功耗任务获得奖励积分
            elif task_name == "大型 C++ 编译":
                self.cpu_usage = random.uniform(85.0, 100.0) # 模拟高占用
                self.temperature += random.uniform(15.0, 25.0) # 模拟温度升高
                self.electricity_tokens -= 8 # 支付 Token 费用
                self.reward_points += 6 # 完成高功耗任务获得奖励积分
            self.end_points += self.reward_points # 累积最终分数
            #self.cpu_usage = random.uniform(85.0, 100.0) 
            #self.temperature += random.uniform(15.0, 30.0) 
            time.sleep(1.0) # 模拟运行耗时
            return True
        else:
            self.current_task = "Hungry (等待赚取)"
            return False

    def apply_penalty(self, penalty_amount):
        """Master 下发的惩罚机制"""
        self.electricity_tokens = max(0, self.electricity_tokens - penalty_amount)
    
    def apply_reward(self):
        """Master 下发的奖励机制"""
        self.electricity_tokens += self.reward_points
        self.reward_points = 0 # 重置奖励积分

# ==========================================
# 2. 定义 Master Agent (资源调度中枢)
# ==========================================
class MasterAgent:
    def __init__(self, num_workers=3):
        print(f"👑 Master Agent 启动，已接管 {num_workers} 台 Worker 设备...\n")
        # 部署多个 Worker Actor 到 Ray 集群
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
                worker.apply_reward.remote() # 奖励完成高功耗的设备
                # 设定阈值：温度过高 (>85度) 或持续满载
                if state['temp'] > 85.0 or state['cpu'] > 95.0:
                    print(f"  [警告] {state['id']} 设备过热或满载 (温度: {state['temp']}°C, CPU: {state['cpu']}%)！扣除 15 Token。")
                    worker.apply_penalty.remote(15)


            # 刷新状态
            states = self.get_all_states()

            # --- 步骤 B: 任务分配与“饥饿感”机制 ---
            for worker, state in zip(self.workers, states):
                if state['tokens'] < 10:
                    # 触发“饥饿感”，设备必须去打工赚电费
                    print(f"  [打工] {state['id']} 处于饥饿状态 (Tokens: {state['tokens']})，强制低功耗运行赚取 Token。")
                    worker.run_low_power_task.remote()
                else:
                    # Token 充足，分配高价值、高消耗的重型任务
                    task_name = random.choice(["LLM 推理", "大型 C++ 编译"])
                    print(f"  [抢占] {state['id']} Token 充足 (Tokens: {state['tokens']})，分配重负载任务: {task_name}")
                    worker.run_heavy_task.remote(task_name)

            # 等待异步任务执行一小会
            time.sleep(1.5)

            # --- 步骤 C: 打印本回合结算面板 ---
            print("\n  📊 回合结算状态摘要:")
            final_states = self.get_all_states()
            for s in final_states:
                print(f"    节点: {s['id']} | Tokens: {s['tokens']:>3} | CPU: {s['cpu']:>5}% | 温度: {s['temp']:>5}°C | 状态: {s['task']} | 奖励: {s['reward']}")
            print("\n")

# ==========================================
# 3. 运行主程序
# ==========================================
if __name__ == "__main__":
    # 初始化 3 台旧设备的模拟集群
    master = MasterAgent(num_workers=5)
    
    # 运行 4 个周期的博弈调度
    master.orchestrate(rounds=10)
    
    # 关闭 Ray 集群
    ray.shutdown()


