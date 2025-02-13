import sys
print(sys.path)
from argparse import ArgumentParser

from luxai_s2.env import LuxAI_S2


from agent import Agent

from lux.config import EnvConfig
from run_matches.interactions import interact

parser = ArgumentParser()
parser.add_argument("--nr_steps", default=1000, type=int)
parser.add_argument("--seed", default=212457496, type=int)
parser.add_argument("--debug_mode", default=False, type=bool)
args = parser.parse_args()

env = LuxAI_S2()
env.reset()

# recreate our agents and run
env_cfg = EnvConfig()
agents = {player: Agent(player, env_cfg) for player in env.agents}
interact(env, agents, args.nr_steps, seed=args.seed)
