import numpy as np
import torch
from typing import Optional
from tensordict import TensorDict
from torchrl.envs import EnvBase
from torchrl.data import (
    MultiDiscreteTensorSpec,
    DiscreteTensorSpec,
    UnboundedContinuousTensorSpec,
    CompositeSpec,
)


class SimpleEnv(EnvBase):
    batch_size = torch.Size([1])

    def __init__(self, size=(10, 10), ts=100) -> None:
        super().__init__()

        self.width, self.height = size
        self.timestep_limit = ts
        self.action_spec = DiscreteTensorSpec(4, shape=(1, 1),dtype=torch.int64)
        self.observation_spec = CompositeSpec(
            ag_pos=DiscreteTensorSpec(self.width * self.height, shape=(1, 1), dtype=torch.int64),
            visited_tile=DiscreteTensorSpec(2, shape=(1, self.width * self.height),dtype=torch.int64),
            shape=torch.Size([1])
        )

        # Observation: ag_pos discrete
        # visited_tile: 1,0 size of the num tiles
        
        self.reward_spec = UnboundedContinuousTensorSpec(shape=(1, 1),dtype=torch.float32)
        self.done_spec = DiscreteTensorSpec(2, shape=(1, 1),dtype=bool)

        self._action_to_direction = {
            0: torch.tensor([1, 0]),
            1: torch.tensor([0, 1]),
            2: torch.tensor([-1, 0]),
            3: torch.tensor([0, -1]),
        }
        self.reset()

    def _step(self, td: TensorDict):
        self.timesteps += 1
        is_done = self.timesteps >= self.timestep_limit
        self._move(self.agent_pos, td["action"])
        events = []
        if not self.check_visisted_tile(self.agent_pos, self.agent_visited_fields):
            self.agent_visited_fields[0, self.agent_pos] = 1
            events.append("new_field")
        obs = self._get_obs()
        r1 = 0.2 if "new_field" in events else 0.0
        self.agent_R += r1
        done = torch.tensor([[is_done]])
        out = self.construct_out_td(td, self.agent_R, obs, done)
        
        print(self.agent_R)
        return out

    def _move(self, coords: torch.Tensor, action: torch.Tensor):
        orig_coord = coords[0, 0].item()
        print(f"pos {orig_coord}")
        # convert to x, y
        x_y_coords = torch.tensor([orig_coord // self.width, orig_coord % self.width])
        print(f"x y coord{x_y_coords}")
        action_key = action[0, 0].item()
        movement = self._action_to_direction[action_key]
        print(f"action {movement}")
        new_x_y_coords = x_y_coords + movement
        
        new_x_y_coords[0] = 0 if new_x_y_coords[0] < 0 else (self.height-1 if new_x_y_coords[0] >= self.height else new_x_y_coords[0])
        new_x_y_coords[1] = 0 if new_x_y_coords[1] < 0 else (self.width-1 if new_x_y_coords[1] >= self.width else new_x_y_coords[1])
        print(f"new xy {new_x_y_coords}")
        new_coords = torch.tensor(
            [[new_x_y_coords[0] * self.width + new_x_y_coords[1]]]
        )
        # Bound it in the arena
        new_coords = self.observation_spec["ag_pos"].project(new_coords)
        print(f"new coords {new_coords}")
        self.agent_pos = new_coords
        return new_coords

    def check_visisted_tile(self, agent_pos, record):
        # agent_pos: 1,1
        # record: 1,100
        return record[0, agent_pos[0, 0]]

    def construct_out_td(self, td, rewards, obs, done):
        next_td = TensorDict(
            {
                "done": done,
                "reward": rewards,
                # "observation": obs,
                "ag_pos": self.agent_pos,
                "visited_tile": self.agent_visited_fields,
            },
            batch_size=torch.Size([1]),
        )
        return next_td

    def _reset(self, td: TensorDict = None):
        self.agent_pos = torch.tensor([[0]])
        self.agent_R = torch.tensor([[0.0]])
        self.agent_visited_fields = torch.zeros(self.width * self.height, dtype=torch.int64).unsqueeze(0)
        self.agent_visited_fields[0,self.agent_pos[0,0]] = 1
        self.timesteps = 0

        out = TensorDict(
            {
                "done": torch.tensor([[False]]),
                "reward": self.agent_R,
                # "observation": self._get_obs(),
                "ag_pos": self.agent_pos,
                "visited_tile": self.agent_visited_fields
            },
            batch_size=torch.Size([1]),
        )
        return out

    def _set_seed(self, seed: Optional[int]):
        rng = torch.manual_seed(seed)
        self.rng = rng

    def _get_obs(self):
        # return a tensordict containing agent position and visited tiles
        out = TensorDict(
            {"ag_pos": self.agent_pos, "visited_tile": self.agent_visited_fields},
            batch_size=torch.Size([1]),
        )
        return out
