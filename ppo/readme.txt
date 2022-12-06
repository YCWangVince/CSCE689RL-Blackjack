The PPO is implemented in ppo_agent.py under the original code structure of rlcard.

Settings:
pip3 install rlcard
pip3 install rlcard[torch]
cd ppo
pip3 install -e .

How to run ppo:
python3 examples/run_rl.py --env blackjack --algorithm ppo --log_dir experiments/blackjack_ppo_result/

Mingyang Wan 128003788 for CSCE689 Reinforcement Learning @ TAMU
