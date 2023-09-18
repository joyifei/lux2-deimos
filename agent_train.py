from functools import wraps
import sys
import os.path as osp

work_directory = osp.dirname(osp.abspath(__file__))
sys.path.append(osp.join(work_directory, "implementations"))

import numpy as np
import torch as th
import tree
from lux.config import EnvConfig
from lux.kit import GameState, obs_to_game_state

from implementations.env.parsers.action_parser_full_act import ActionParser
from implementations.env.parsers.feature_parser import FeatureParser, LuxFeature
from implementations.policy.impl.multi_task_softmax_policy_impl import (_gen_pi, _sample_til_valid)
from implementations.policy.net import Net
from implementations.env.player import Player
from implementations.impl_config import EnvParam, ModelParam
import traceback
from stable_baselines3.common.policies import ActorCriticPolicy

import json


def print_exc(f):

    @wraps(f)
    def wrapper(*args, **kwargs):
        try:
            return f(*args, **kwargs)
        except Exception as e:
            print(f"torch.__version__={th.__version__}", file=sys.stderr)
            print(e, file=sys.stderr)
            traceback.print_exc()
            raise

    return wrapper


from typing import Tuple
class AgentTrain(ActorCriticPolicy):

    def __init__(self, observation_space, action_space, lr_schedule, use_sde, player='player_0', **kwargs ) -> None:
        super().__init__(observation_space, action_space, lr_schedule)
        self.player = player
        self.opp_player = "player_1" if self.player == "player_0" else "player_0"
        np.random.seed(0)
        self.env_cfg: EnvConfig = EnvConfig()
        self.feature_parser = FeatureParser()
        self.action_parser = ActionParser()

        with open(osp.join(work_directory, "conf.json")) as f:
            conf = json.load(f)
        model_related = conf['model_related']
        model_param = ModelParam()
        if 'model_param' in model_related:
            model_param = ModelParam(**model_related['model_param'])
        self.net = Net(model_param)
        self.net.load_state_dict(th.load(
            osp.join(work_directory, "model.pt"),
            map_location=th.device('cpu'),
        ))

        self.early_setup_player = Player(self.player, self.env_cfg)
    def getLuxFeature(self, obs):
        player_obs = {key.replace(self.player+'_',''): value for key, value in obs.items() if key.startswith(self.player)}
        action_features = { key.replace('action_', ''): value for key, value in player_obs.items() if key.startswith('action_')}
        global_feature = next( value for key, value in player_obs.items() if key.startswith('global_'))
        map_feature = next( value for key, value in player_obs.items() if key.startswith('map_') )

        feature = LuxFeature(global_feature, map_feature, action_features)
        return feature

    def getVa(self, obs):
        #get a new dictionary with items in obs start with 'va_' + self.player, and the key is replaced with the original key without 'va_' + self.player+"_"
        va = {key.replace('va_'+self.player+'_', ''): value for key, value in obs.items() if key.startswith('va_'+self.player)}
        return va

    def forward(self, obs, deterministic: bool = False):

        feature = self.getLuxFeature(obs)
        va = self.getVa(obs)

        np2torch = lambda x, dtype: x if isinstance(x, th.Tensor) else th.tensor(x)[None].type(dtype)

        logp, value, action, entropy, _ = self.net.forward(
            np2torch(feature.global_feature, th.float32),
            np2torch(feature.map_feature, th.float32),
            tree.map_structure(lambda x: np2torch(x, th.int16), feature.action_feature),
            tree.map_structure(lambda x: np2torch(x, th.bool), va),
        )
        #action = tree.map_structure(lambda x: x.detach().cpu().numpy()[0] if x.is_cuda else x.detach().numpy()[0], action)

        action = self.flatten_action(action)
        #temp_action = self.unflatten_action(action)
        action = action.reshape(1,-1)
        return action, value, logp
    def flatten_action(self, action):
        factory_flattened = action['factory_act'].view(-1)

        # Flatten unit_act
        unit_flattened = action['unit_act'].view(-1)

        # Get bid and spawn values
        bid_value = action['bid'].view(-1)


        # Concatenate all flattened components
        flattened_action = th.cat([factory_flattened,
                                   unit_flattened,
                                   bid_value,
                                   action['factory_spawn']['location'].view(-1),
                                   action['factory_spawn']['water'].view(-1),
                                   action['factory_spawn']['metal'].view(-1)])
        return flattened_action
    #provide a unflatten function to convert the flattened action back to the original action
    def unflatten_action(self, action):
        #get first self.map
        factory_act = action[:self.action_space['factory_act'].n]
        unit_act = action[self.action_space['factory_act'].n:self.action_space['factory_act'].n + self.action_space['unit_act'].n]
        bid = action[self.action_space['factory_act'].n + self.action_space['unit_act'].n:self.action_space['factory_act'].n + self.action_space['unit_act'].n + 1]
        factory_spawn_location = action[self.action_space['factory_act'].n + self.action_space['unit_act'].n + 1:self.action_space['factory_act'].n + self.action_space['unit_act'].n + 3]
        factory_spawn_water = action[self.action_space['factory_act'].n + self.action_space['unit_act'].n + 3:self.action_space['factory_act'].n + self.action_space['unit_act'].n + 4]
        factory_spawn_metal = action[self.action_space['factory_act'].n + self.action_space['unit_act'].n + 4:self.action_space['factory_act'].n + self.action_space['unit_act'].n + 5]
        factory_spawn = {'location': factory_spawn_location, 'water': factory_spawn_water, 'metal': factory_spawn_metal}
        return {'factory_act': factory_act, 'unit_act': unit_act, 'bid': bid, 'factory_spawn': factory_spawn}
    def early_setup(self, step: int, obs, remainingOverageTime: int = 60):
        if EnvParam.rule_based_early_step:
            return self.early_setup_player.early_setup(step, obs, remainingOverageTime)
        else:
            return self.act(step, obs, remainingOverageTime)

    # @print_exc
    def act(self, step: int, obs, remainingOverageTime: int = 60):
        """
        optionally do forward simulation to simulate positions of units, lichen, etc. in the future
        from lux.forward_sim import forward_sim
        forward_obs = forward_sim(obs, self.env_cfg, n=2)
        forward_game_states = [obs_to_game_state(step + i, self.env_cfg, f_obs) for i, f_obs in enumerate(forward_obs)]
        """


        action,value, logp = self.forward(obs)

        action = self.action_parser._parse(game_state, self.player, action)

        return action
