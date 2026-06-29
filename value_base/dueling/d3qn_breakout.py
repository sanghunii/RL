import random
import numpy as np

import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym
from gymnasium.wrappers import AtariPreprocessing, FrameStack

from experience_replay.replay_buffer import ReplayBuffer
from value_base.dueling.network.dueling_network import DuelingQNetwork



# 1. Environment Setup Helper
def make_env(env_id):
    env = gym.make(env_id, frameskip=1) # frameskip=1로 설정하고 Preprocessing에서 4프레임 건너뛰기 처리
    # 표준 Atari 전처리: 84x84 리사이즈, 그레이스케일, 라이프 상실 시 에피소드 종료 처리 등
    env = AtariPreprocessing(env, frame_skip=4, grayscale_obs=True, scale_obs=False, terminal_on_life_loss=True)
    # 최근 4프레임을 스택으로 쌓아 (4, 84, 84) 차원 생성
    env = FrameStack(env, num_stack=4)
    return env


# 2. Evaluation Function
def evaluate(policy_net, env_id, n_eval_episodes=3, device='cuda'):
    """평가 시에는 탐험(Epsilon)을 끄고 네트워크를 eval 모드로 전환합니다."""
    eval_env = make_env(env_id)
    policy_net.eval() # 평가 모드 전환 (Dropout, BatchNorm 등이 있다면 필수)
    
    total_rewards = []
    for _ in range(n_eval_episodes):
        state, _ = eval_env.reset()
        episode_reward = 0
        done = False
        
        while not done:
            # LazyFrames를 numpy array로 변환 후 텐서화
            state_tensor = torch.tensor(np.array(state), dtype=torch.uint8).unsqueeze(0).to(device)
            with torch.no_grad(): # 평가 시에는 그래디언트 계산 방지
                q_values = policy_net.get_q_values(state_tensor)
                action = q_values.argmax(dim=1).item()
                
            next_state, reward, terminated, truncated, _ = eval_env.step(action)
            done = terminated or truncated
            episode_reward += reward
            state = next_state
            
        total_rewards.append(episode_reward)
        
    eval_env.close()
    policy_net.train() # 평가 종료 후 다시 훈련 모드로 복귀
    return np.mean(total_rewards)


# 3. Main Training Loop (Double DQN Logic)
def train():
    # --- Hyperparameters ---
    ENV_ID = "ALE/Breakout-v5"
    BATCH_SIZE = 32
    GAMMA = 0.99
    LR = 1e-4
    BUFFER_SIZE = 100000  # RAM 용량에 따라 조절 (100k ~ 1M)
    TARGET_UPDATE_FREQ = 1000
    MAX_STEPS = 5_000_000
    EVAL_FREQ = 50_000
    LEARNING_STARTS = 10000 # 버퍼에 데이터가 쌓일 때까지 무작위 행동
    
    EPS_START = 1.0
    EPS_END = 0.05
    EPS_DECAY_STEPS = 1_000_000

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    

    env = make_env(ENV_ID)
    n_actions = env.action_space.n

    policy_net = DuelingQNetwork(n_actions).to(device)
    target_net = DuelingQNetwork(n_actions).to(device)
    target_net.load_state_dict(policy_net.state_dict())
    target_net.eval() # 타겟 네트워크는 학습되지 않으므로 항상 eval 모드

    optimizer = optim.Adam(policy_net.parameters(), lr=LR)
    buffer = ReplayBuffer(BUFFER_SIZE)

    state, _ = env.reset()
    episode_reward = 0
    episode_count = 0

    for step in range(1, MAX_STEPS + 1):
        # 1. Epsilon-Greedy Action Selection
        epsilon = max(EPS_END, EPS_START - (EPS_START - EPS_END) * (step / EPS_DECAY_STEPS))
        
        if step < LEARNING_STARTS or random.random() < epsilon:
            action = env.action_space.sample()
        else:
            state_tensor = torch.tensor(np.array(state), dtype=torch.uint8).unsqueeze(0).to(device)
            with torch.no_grad():
                q_values = policy_net.get_q_values(state_tensor)
                action = q_values.argmax(dim=1).item()

        # 2. Step Environment
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        
        # 보상 클리핑 (Atari 학습 안정화를 위해 -1, 0, 1로 클리핑)
        clipped_reward = np.sign(reward)
        
        buffer.push(np.array(state), action, clipped_reward, np.array(next_state), done)
        state = next_state
        episode_reward += reward

        if done:
            state, _ = env.reset()
            episode_reward = 0
            episode_count += 1

        # 3. Training Step
        if step > LEARNING_STARTS and step % 4 == 0: # Atari는 보통 4스텝마다 1번씩 학습
            states, actions, rewards, next_states, dones = buffer.sample(BATCH_SIZE)
            
            states = torch.tensor(states, dtype=torch.uint8).to(device)
            actions = torch.tensor(actions, dtype=torch.long).unsqueeze(1).to(device)
            rewards = torch.tensor(rewards).unsqueeze(1).to(device)
            next_states = torch.tensor(next_states, dtype=torch.uint8).to(device)
            dones = torch.tensor(dones).unsqueeze(1).to(device)

            # --- Double DQN Logic ---
            # 현재 상태의 Q값 계산
            q_values = policy_net.get_q_values(states)
            current_q = q_values.gather(1, actions)

            with torch.no_grad():
                # 다음 상태에서의 행동은 Policy Net(Online Net)이 결정 (argmax Q)
                next_q_values_policy = policy_net.get_q_values(next_states)
                next_actions = next_q_values_policy.argmax(dim=1, keepdim=True)
                
                # 결정된 행동의 Q값 평가는 Target Net이 수행
                next_q_values_target = target_net.get_q_values(next_states)
                target_q_next = next_q_values_target.gather(1, next_actions)
                
                # 타겟 Q 계산 (종료 상태이면 reward만)
                target_q = rewards + (GAMMA * target_q_next * (1 - dones))

            # Loss 계산 (Huber Loss가 이상치에 더 강건함)
            loss = nn.SmoothL1Loss()(current_q, target_q)

            optimizer.zero_grad()
            loss.backward()
            
            # Gradient Clipping: 가파른 그래디언트 폭발 방지 (기울기를 잘라내기 전 원래의 그래디언트 방향과 크기를 유지하면서 한도를 씌움)
            nn.utils.clip_grad_norm_(policy_net.parameters(), max_norm=10.0)
            
            optimizer.step()


        # 4. Target Network Update
        if step % TARGET_UPDATE_FREQ == 0:
            target_net.load_state_dict(policy_net.state_dict())

        # 5. Evaluation
        if step % EVAL_FREQ == 0:
            avg_eval_reward = evaluate(policy_net, ENV_ID, device=device)
            print(f"Step: {step} | Eval Reward: {avg_eval_reward:.2f}")

    env.close()

if __name__ == "__main__":
    train()