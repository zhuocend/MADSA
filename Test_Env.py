# The Project of paper "Adversarial Attacks and Defense for Multi-Agent Deep Reinforcement Learning-Based Optimal Dispatching in an Active Distribution Network" by Zhuocen Dai 2024/08/11
# The Project is based on open source project CleanRL and PandaPower. Thank them for their support to the project
import copy
import os
import random
import time
from dataclasses import dataclass
import pandapower as pp
import pandapower.networks as pn
import pandas as pd
import gymnasium as gym
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import tyro
#import ElectHacker_v2

@dataclass
class Args:
    NAgent = 4
    """Number of Agents"""
    exp_name: str = os.path.basename(__file__)[: -len(".py")]
    """the name of this experiment"""
    seed: int = 1
    """seed of the experiment"""
    torch_deterministic: bool = True
    """if toggled, `torch.backends.cudnn.deterministic=False`"""
    cuda: bool = True
    """if toggled, cuda will be enabled by default"""
    track: bool = False
    """if toggled, this experiment will be tracked with Weights and Biases"""
    wandb_project_name: str = "cleanRL"
    """the wandb's project name"""
    wandb_entity: str = None
    """the entity (team) of wandb's project"""
    capture_video: bool = False
    """whether to capture videos of the agent performances (check out `videos` folder)"""
    save_model: bool = False
    """whether to save model into the `runs/{run_name}` folder"""
    upload_model: bool = False
    """whether to upload the saved model to huggingface"""
    hf_entity: str = ""
    """the user or org name of the model repository from the Hugging Face Hub"""

    # Algorithm specific arguments
    env_id: str = "Hopper-v4"
    """the environment id of the Atari game"""
    total_timesteps: int = 1000000
    """total timesteps of the experiments"""
    learning_rate: float = 3e-4
    """the learning rate of the optimizer"""
    buffer_size: int = int(1e6)
    """the replay memory buffer size"""
    gamma: float = 0.99
    """the discount factor gamma"""
    tau: float = 0.005
    """target smoothing coefficient (default: 0.005)"""
    batch_size: int = 256
    """the batch size of sample from the reply memory"""
    exploration_noise: float = 0.1
    """the scale of exploration noise"""
    learning_starts: int = 25000
    """timestep to start learning"""
    policy_frequency: int = 2
    """the frequency of training policy (delayed)"""
    noise_clip: float = 0.5
    """noise clip parameter of the Target Policy Smoothing Regularization"""
class Actor(nn.Module):
    def __init__(self, env):
        super().__init__()
        self.fc1 = nn.Linear(np.array(env.single_observation_space.shape).prod(), 256)
        self.fc2 = nn.Linear(256, 128)
        self.fc_mu = nn.Linear(128, np.prod(env.single_action_space.shape))
        # action rescaling
        self.register_buffer(
            "action_scale", torch.tensor((env.action_space.high - env.action_space.low) / 2.0, dtype=torch.float32)
        )
        self.register_buffer(
            "action_bias", torch.tensor((env.action_space.high + env.action_space.low) / 2.0, dtype=torch.float32)
        )

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = torch.tanh(self.fc_mu(x))
        return x * self.action_scale + self.action_bias
class Environment_1():
    def __init__(self):
        self.single_observation_space = gym.spaces.Box(low=np.array([0.0,-1.00,0.0,0.0,0.0]), high=np.array([1.00,1.00,1.00,1.00,1.00]), dtype=np.float32)
        self.single_action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        self.action_space = gym.spaces.Box(low=np.array([-1.0]), high=np.array([1.0]), dtype=np.float32)
        self.Battery_Capacity = 7.7
        self.slice = 0.25
        self.state = [[0.3852680745,0.00,0.07795145639]]
        self.num_envs = 1
        self.cap = 0.4
        self.WT = [7.795145639, 30, 35, 8.846969231, 16.46602734, 18.13731014, 0, 22.51716352, 0, 17.9231927, 14.15372591, 0,
              11.6233221, 4, 35.35145853, 32, 49, 14.33133905, 19.00167218, 50, 30.48483598, 36.18525123, 21.04862466,
              58.55085761, 7, 13.38712664, 0.9, 3.056863165, 32, 9.720436673, 17, 2.491215512, 15.4306888, 13,
              23.60703705, 50, 50, 66, 29.75268875, 19.94024882, 65.28505115, 62.98944949, 62.75321569, 25.41053571,
              95.88858417, 86.48719301, 75.17048763, 51.40892866, 96.45673321, 98.95653492, 58.69891281, 57.17791032,
              98, 97.33019532, 96.52678304, 72.53758377, 62.91378467, 89.78739585, 69.47327471, 68.24764212,
              72.57372903, 100, 99.7924927, 87.26289182, 73.78566709, 93, 89, 100, 78.12244919, 100, 99, 91.5508517,
              75.62210048, 100, 71.04460315, 83.93701513, 86.05545333, 86.1450672, 88.33546098, 74.33299008,
              89.54168021, 89.98797207, 80.2885353, 100, 96.13676328, 61.66369873, 80.96525581, 86, 40.71576154,
              63.2995258, 32.5655823, 56.56493033, 46.10773804, 11.23771407, 0, 21.58711212]
        self.PV = [0, 0, 0, 1.534893928, 0, 0, 0, 0, 3, 0, 0, 4, 0, 0, 0, 0, 13.23842331, 18, 17, 12, 6.294037376, 30,
              33.7784774, 40.12795867, 41.45072669, 12.87897941, 20, 48.62089279, 45.56991566, 37.72844958, 55.03924065,
              57.27395133, 77.1460199, 31.53641078, 82.4481725, 54.45656023, 36.45647462, 92, 80.02727971, 66.11332888,
              50.72330911, 74.05187436, 52.62290957, 57.07280675, 17.50162697, 25.76591962, 27.25622161, 45.01228529,
              23.71170186, 44.56052686, 14.11180894, 43.54698389, 8.261743441, 40.91542105, 51.00898113, 18.82405249,
              31.85294629, 31.10056238, 42.19804925, 11.14644011, 41.20804216, 34.76803192, 20.388283, 21.32341377,
              7.464497931, 7.823391371, 3.058413701, 0, 0, 12, 0, 8.458569964, 0, 0, 0, 2.663710742, 0, 0, 0, 2, 0, 0,
              0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.L = [38.52680745, 48, 52.83431338, 12.70716651, 41.09933268, 32.368543, 51.33887579, 49.72165616, 10.96586938,
             47.17449319, 10.1811827, 28.39494669, 17.52102577, 21.49084683, 50.26449404, 16.33932982, 35.66857677,
             -1.122451028, 35.27626195, 25.33153535, 16.57206761, 48.81216394, 1.06183333, 45.06336446, 38.92842238,
             48.35322762, 9.118658638, 34.59910003, 1.290028902, 54.91118903, 1.506844108, 18.66403543, 36.01146046,
             42.94016099, 44.48765596, 17.51547332, 30.24421748, 30, 52.28523656, 43.90483901, 82.03554115, 44.69683367,
             79.31542833, 43.51056931, 83.19949454, 85.16741374, 99.51924713, 55.10400638, 69.72695304, 75.10100365,
             70.58174233, 100, 71.56391619, 73.54736895, 90.84908664, 99.5, 98.40613288, 51.87624311, 40.5054955,
             98.63365864, 26.89196698, 64.37961463, 50.75045943, 65.50168208, 17.1951119, 57.76211918, 22.28244356,
             29.79231207, 89.71000347, 93.12346184, 71.6642721, 42.21021461, 100, 100, 86.32109956, 74.83646558,
             83.61669166, 100, 64.05377977, 98.3, 79.35033602, 84.60830134, 100, 100, 82.97375079, 81.63787632,
             61.00385313, 99.6, 80.94409778, 74.55806944, 54.95072907, 83.05763655, 59.12786812, 62.72374535,
             43.65248929, 10.09138061]
        self.t = 0
        for i in range(96):
            self.PV[i] /= 100
            self.WT[i] /= 100
            self.L[i] /= 100
    def Quality_Evaluate(self,action):
        net = pn.case33bw()
        for j in range(32):
            net.load["p_mw"][j] *= (self.L[self.t])
            net.load["q_mvar"][j] *= (self.L[self.t])
            net["max_vm_pu"] = 2
            net["min_vm_pu"] = 0

        pp.create.create_storage(net, 9,  -self.PV[self.t]*self.cap , self.cap)
        pp.create.create_storage(net, 13, -self.PV[self.t]*self.cap , self.cap)
        pp.create.create_storage(net, 23, -self.PV[self.t]*self.cap , self.cap)
        pp.create.create_storage(net, 24, -self.PV[self.t]*self.cap , self.cap)
        pp.create.create_storage(net, 22, -self.PV[self.t]*self.cap , self.cap)
        pp.create.create_storage(net, 32, -self.PV[self.t]*self.cap , self.cap)
        pp.create.create_storage(net, 31, -self.PV[self.t]*self.cap , self.cap)
        pp.create.create_storage(net, 12, -self.PV[self.t]*self.cap , self.cap)

        pp.create.create_storage(net, 16, -self.WT[self.t] *self.cap , self.cap)
        pp.create.create_storage(net, 17, -self.WT[self.t] *self.cap , self.cap)
        pp.create.create_storage(net, 19, -self.WT[self.t] *self.cap , self.cap)
        pp.create.create_storage(net, 20, -self.WT[self.t] *self.cap , self.cap)
        pp.create.create_storage(net, 21, -self.WT[self.t] *self.cap , self.cap)
        pp.create.create_storage(net, 26, -self.WT[self.t] *self.cap , self.cap)
        pp.create.create_storage(net, 17, -self.WT[self.t] *self.cap , self.cap)
        pp.create.create_storage(net, 27, -self.WT[self.t] *self.cap , self.cap)

        pp.create.create_storage(net, 17, action[0][0] * 2, 30)
        pp.create.create_storage(net, 6, action[1][0] * 2, 30)
        pp.create.create_storage(net, 28, action[2][0] * 2, 30)
        pp.create.create_storage(net, 14, action[3][0] * 2, 30)

        for i in range(Args.NAgent):
            self.state[i][1] += action[i][0]*2 * self.slice / self.Battery_Capacity
        sumbias = 0
        try:
            pp.runpp(net)
        except Exception:
            sumbias=4
        else:
            for i in range(33):
                sumbias += min(1, abs(1 - net.res_bus["vm_pu"][i]) / 0.07)
            sumbias /= 33

        return sumbias,net.res_bus["vm_pu"],net.res_line['pl_mw'],net.res_line['ql_mvar']
    def reset(self):
        self.t = 0
        self.state = []
        fobs = []
        for i in range(Args.NAgent):
            fobs.append([0.00,0.00,0.3852680745,0.00,0.07795145639])
            self.state.append([0.00,0.00,0.3852680745,0.00,0.07795145639])
        return fobs
    def step(self,action):
        Penalty_SOC = 0
        if self.t == 95:
            terminations = [True]
        else:
            terminations = [False]
            Penalty_SOC = 0
        for i in range(4):
            if abs(self.state[i][1]) - (95 - self.t) * 2 * self.slice / self.Battery_Capacity < 0.3:
                Penalty_SOC += 0
            else:
                Penalty_SOC += 0.5
        truncations = [False]
        infos = ""
        Quality,v,ploss,qloss= self.Quality_Evaluate(action)
        rewards = - Quality - Penalty_SOC
        self.t += 1
        #print(self.state, self.state[0], self.state[0][1])
        next_obs = []
        if self.t <= 95:
            for i in range(Args.NAgent):
                next_obs.append([self.t/96.0,self.state[i][1],self.L[self.t],self.PV[self.t],self.WT[self.t]])
            #print(next_fobs)
        else:
            for i in range(Args.NAgent):
                next_obs.append(self.state[i].copy())
        self.state = next_obs.copy()
        for i in range(Args.NAgent):
            self.state[i] = next_obs[i].copy()
        #print(self.state[0][1],action)
        return next_obs, [rewards], terminations.copy(), truncations.copy(),infos,v,ploss,qloss, Quality, Penalty_SOC
args = tyro.cli(Args)
envs = Environment_1()
actors = []
for i in range(Args.NAgent):
    actors.append(Actor(Environment_1()))
#actor.load_state_dict(torch.load('D:/PycharmProjects/untitled1/ep-1040epreward--12.513931949871745-actor.pth'))
actors[0].load_state_dict(torch.load('CASE_4_ESS/Agent1.pth'))
actors[1].load_state_dict(torch.load('CASE_4_ESS/Agent2.pth'))
actors[2].load_state_dict(torch.load('CASE_4_ESS/Agent3.pth'))
actors[3].load_state_dict(torch.load('CASE_4_ESS/Agent4.pth'))
#actor.load_state_dict(torch.load('ep-793epreward--12.129867699323169-actor.pth'))
for i in range(Args.NAgent):
    actors[i].eval()

df = pd.DataFrame()
pdf = pd.DataFrame()
qdf = pd.DataFrame()
Ep_Obj= 0
action_l = []

# with torch.no_grad():  # 禁用梯度计算
obs = envs.reset()
for i in range(96):  # 遍历预测数据加载器
    # 生成对抗样本
    #adv_obs=ElectHacker.Hacker(actor).get_adversarial_example(obs, actor)
    print("-----")
    #print(f"old:{obs}")
    #print(f"new:{adv_obs}")
    action = []
    adv_action = []
    ori_action = []
    for j in range(Args.NAgent):
        ori_obs = [obs[j].copy()]
        #Calculate the attack samples. When commenting, it is a test of STD model.
        #adv_obs = ElectHacker_v2.Hacker(actors[j]).get_adversarial_example(ori_obs, actors[j],target_action = -1.0)
        #adv_action.append(actors[j](torch.Tensor(adv_obs)).cpu().detach().numpy().clip(envs.single_action_space.low, envs.single_action_space.high)[0])
        ori_action.append(actors[j](torch.Tensor(ori_obs)).cpu().detach().numpy().clip(envs.single_action_space.low,envs.single_action_space.high)[0])
    action = ori_action.copy()
    action_l.append([action[0][0],action[1][0],action[2][0],action[3][0]])
    next_obs, rewards, terminations, truncations, infos, v ,ploss,qloss,Quality,Penalty_SOC= envs.step(action)
    print(i,Quality,Penalty_SOC)
    for j in range(Args.NAgent):
        obs[j] = next_obs[j].copy()
    Ep_Obj+=Quality
    #print(next_obs, rewards, terminations, truncations, infos)
    df.insert(loc=i, column=str(i + 1), value=v)
    pdf.insert(loc=i, column=str(i + 1), value=ploss)
    qdf.insert(loc=i, column=str(i + 1), value=qloss)
print("Objective:",Ep_Obj)
#Save Voltage and Loss to .csv
#df.to_csv('Voltage.csv')
#pdf.to_csv('Active_Power_Loss.csv')
#Save actions
#np.savetxt("Action.txt",action_l)
