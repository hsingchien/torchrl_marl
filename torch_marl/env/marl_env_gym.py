
from gymnasium.spaces import Discrete, MultiDiscrete, Dict, Box, MultiBinary
from gymnasium import Env
import numpy as np
from tabulate import tabulate
from pprint import pprint
from PIL import Image, ImageDraw, ImageFont

class MultiAgentArena_v3(Env):
  ## Returning default configuration dictionary

  DEFAULT_CONFIG = {
    "width": 10,
    "height": 10,
    # "num_agents": 2,
    "num_chasers": 1,
    "num_escapers": 1,
    # 'roles': {'agent_0': 'chaser', 'agent_1': 'escaper'},
    'roles': None,
    "timestep_limit": 100,
    # Defined in the sequence of left, forward, backward, right
    "range_of_view": {'chaser': [3,2,0,2,], 'escaper': [2,2,1,2,]},
    "random_initialization_position": True,
    "assigned_initialization_position": [[1, 1], [3, 2]],
    "use_1d_obs": False,
    "reward_dictionary": {"chaser": {'explore': 0.1, 'collision': 0.3, 'other': -0.1},
                          "escaper": {'explore': 0.1, 'collision': -0.3, 'other': -0.1}},
    'num_events': 3,
    "using_old_action_map": False,
    "ACTION_MAP": {
      0: [-1, 0],  # Move UP
      1: [0, 1],  # Move RIGHT
      2: [1, 0],  # Move DOWN
      3: [0, -1],  # Move LEFT
      # 4: [0, 0],  # NOOP
    },
    # If enabling this, each agent will have a different grid world of observations
    "separate_grid_representation_for_other_agent": False,
    # If this is enabled, the exploration reward will decay over time
    "decaying_rewards_for_chaser_before_colliding": False,
    # This is to use together with the above one, if this is enabled, the collision reward will be muted for the chaser
    # It can be declared as False or an integer number for time steps
    "mute_chaser_collision": False,
    # This is to use together with the above one, if this is enabled, the collision reward will be muted for the escaper
    "mute_escaper_collision": False,
    # Decaying rate for the exploration reward
    # The first one is for padding, the last one is for the last step and every last step
    "decaying_rate_for_chaser_exploration": [1, 0.5, 0.25, 0.125, 0.0625, 0.03125, 0.015625, 0],
    # If this is enabled, the collision reward will surpass the exploration reward
    "collision_suprpass_explore": True,
    # If this is enabled, the collision reward will be calculated together for the prey and predator in collision
    "collision_share_between_pairs": True,
    # If this is enabled, the collision reward will be calculated together for the team of predators,
    # This is not implemented yet
    "collision_shared_within_chasers": False,
    # Make visit count visible to the agents
    "visit_count_visible": False,
    ## Here, we are going to define a few partial observability styles:
    # Full means that the agent can always observe the entire field
    # "full+coordinates" means that the agent can observe the entire field with numerical coordinates
    # 'partial' means that the agnets can only see a limited distances, like 7x7 squares instead of 10x10 full field
    # 'partial+coordinates' means that the agnets can only see a limited distance, with coordinate information
    "partial_observability_style": "full",
  }

  def __init__(self, config=None):
    super().__init__()
    config = config or {}
    self.config = {**self.DEFAULT_CONFIG, **config}
    pprint("Initializing MultiAgentArena_v38 with config:")
    pprint(self.config)
    # Update the action map dict if they are not integer values
    if '0' in self.config["ACTION_MAP"].keys():
      self.ACTION_MAP = {int(key): item for key, item in self.config["ACTION_MAP"].items()}
    else:
      self.ACTION_MAP = self.config["ACTION_MAP"]
    self.num_events = self.config["num_events"]

    # Dimensions of the grid
    # Directly access keys from the merged config
    self.width = self.config["width"]
    self.height = self.config["height"]
    # self.num_agents = self.config["num_agents"]
    self.num_chasers = self.config["num_chasers"]
    self.num_escapers = self.config["num_escapers"]
    self.num_agents = self.num_chasers + self.num_escapers
    self.range_of_view = self.config["range_of_view"]
    # self.vision_field = {role: [self.range_of_view[role][0] + self.range_of_view[role][2],
    #                            self.range_of_view[role][1] + self.range_of_view[role][3]]
    #                      for role in self.range_of_view.keys()}
    self.collision_surpass_explore = self.config["collision_suprpass_explore"]
    self.collision_share_between_pairs = self.config["collision_share_between_pairs"]
    self.collision_shared_within_chasers = self.config["collision_shared_within_chasers"]
    self.mute_chaser_collision = self.config["mute_chaser_collision"]
    self.mute_escaper_collision = self.config["mute_escaper_collision"]
    self.visit_count_visible = self.config["visit_count_visible"]
    self.separate_grid_representation_for_other_agent = self.config["separate_grid_representation_for_other_agent"]
    self.decaying_rewards_for_chaser_before_colliding = self.config["decaying_rewards_for_chaser_before_colliding"]
    self.decaying_rate_for_chaser_exploration = self.config["decaying_rate_for_chaser_exploration"]

    # End an episode after this many timesteps.
    self.timestep_limit = self.config["timestep_limit"]

    # configuring variables
    # for cId in range(self.num_chasers):
    self.players = [f"agent_{i}" for i in range(self.num_agents)]
    self._agent_ids = self.players
    # We are going to assign the agents an identity. Also, we need to notice that the chaser is initialized first
    # such that they will have default higher priority in movement
    if isinstance(self.config['roles'], dict):
      self.roles = self.config['roles']
    else:
      self.roles = {f"agent_{i}": 'chaser' for i in range(self.num_chasers)}
      self.roles.update({f"agent_{i}": 'escaper' for i in range(self.num_chasers, self.num_agents)})
      # raise NotImplementedError
    self.reward_value_dict = {
      f"agent_{i}": self.config['reward_dictionary'][self.roles[f"agent_{i}"]] for i in range(self.num_agents)}
    self.observation_space = {f"agent_{i}": None for i in range(self.num_agents)}
    # self.global_observation_space = None
    # 0=NOOP, 1=up, 2=right, 3=down, 4=left
    self.action_space = {f"agent_{i}": Discrete(len(self.ACTION_MAP)) for i in range(self.num_agents)}
    #
    self.partial_observability_style = self.config["partial_observability_style"]
    self.coordinates = {f"agent_{i}": None for i in range(self.num_agents)}
    # reset environment

    self.terminateds = {}
    self.truncateds = {}
    self.current_step = None
    # The following are per agent
    self.rewards = {}
    self.step_rewards = {}
    self.agent_actions = {}
    self.agent_positions = {}
    self.visit_counts = {}
    self.agent_events = {}
    self.agent_action_history = {}
    self.agent_event_history = {}
    self.agent_position_history = {}
    self.agent_continual_step_exploration = {}
    self._infos = {}
    self.reset()

  def reset(self, *, seed=None, options=None):
    # Reset termination flag
    self.terminateds["__all__"] = False
    self.truncateds["__all__"] = False
    # Moving current_step initialization outside the loop
    self.current_step = 0
    if self.config['random_initialization_position']:
      positions = np.random.choice(np.arange(np.multiply(self.width, self.height)), self.num_agents, replace=False)
      positions = [list(np.unravel_index(position, (self.width, self.height))) for position in positions]
    else:
      positions = self.config['assigned_initialization_position']
    for pId, player in enumerate(self.players):
      self.agent_positions[player] = positions[pId]
      self.rewards[player] = 0
      self.step_rewards[player] = 0
      self.agent_actions[player] = 0 # Default action is UP
      self.agent_action_history[player] = []  # Initialized with no states as the first
      self.agent_events[player] = None
      self.agent_event_history[player] = []
      self.agent_position_history[player] = []
      self.agent_continual_step_exploration[player] = []
      self._infos[player] = {
        'event': None,
        'action': None,
        'position': None,
        'old_position': None,
        'collision_pairs': None,
      }
      self.visit_counts[player] = np.zeros((self.width, self.height), dtype=np.uint16)
      self.coordinates[player] = np.arange(self.width * self.height).reshape((self.width, self.height)).astype(np.uint16) + 1

      # Return the initial observation in the new episode.
      # Agent positions in the grid world
      if 'full' in self.partial_observability_style:
        # Define the field of view as the world size
        observation_space = {
          "grid_world": Box(low=0, high=255, shape=(self.width, self.height), dtype=np.uint8,)
        }
      elif 'partial' in self.partial_observability_style:
        role = self.roles[player]
        # Define the field of view as a smaller square
        observation_space = {
          "grid_world": Box(low=0, high=255,
                            shape=(self.range_of_view[role][0] + self.range_of_view[role][2] + 1,
                                   self.range_of_view[role][1] + self.range_of_view[role][3] + 1,),
                            dtype=np.uint8, )
        }
      else:
        raise NotImplementedError

      # In this setting, the agent will have another box to save the observed location of the other agent
      if self.separate_grid_representation_for_other_agent:
        for player_i in self.players:
          if player_i != player:
            observation_space[f'grid_{player_i}'] = observation_space['grid_world']

      # In this setting, the agent will have access to the coordinates surrounding itself
      if 'coordinates' in self.partial_observability_style:
        observation_space['coordinates'] = observation_space['grid_world']

      # The visit counts is the number of visits on each tile
      if self.visit_count_visible:
        observation_space["visit_count"] = observation_space['grid_world']
      self.observation_space[player] = Dict(observation_space)

   
    return self.get_observation_space(), self._infos

  def step(self, action_dict: dict):
    # To ensure the order follows the order of the players
    if action_dict and (self.players != list(action_dict.keys())):
      action_dict = dict((player, action_dict[player]) for player in self.players)
    old_positions = list(self.agent_positions[player] for player in self.players)
    new_positions = [None] * self.num_agents
    # Here, we are introducing a matrix to represent the "interaction" between agents
    collision_matrix = np.zeros((self.num_agents, self.num_agents), dtype=bool)
    # The interaction between different agents will represent the collision
    exploration_array = np.zeros((self.num_agents), dtype=bool)

    events = [None] * self.num_agents

    # We first move all agents and record events of agents
    for pId, player in enumerate(self.players):
      if (len(action_dict) == 0) and (self.current_step == 0):
        break
      act = action_dict[player]
      new_position, collision_pairs = self._move(player, old_positions[pId], act)
      events[pId] = "other"
      self.agent_positions[player] = new_position
      new_positions[pId] = new_position

      # Voluntary collision detected, then update the collision matrix, and the events for the voluntary party
      if collision_pairs:
        collision_matrix[self.players.index(collision_pairs[0]), self.players.index(collision_pairs[1])] = 1
        events[pId] = ' collide '.join(collision_pairs)

    # We then check if there is a collision and exploration
    for pId, (player, act) in enumerate(action_dict.items()):
      # If involuntary collision happened, we need to update the event based on the collision matrix
      if np.any(collision_matrix[pId, :]) or np.any(collision_matrix[:, pId]):
        events[pId] = 'collided' if 'collide' not in events[pId] else events[pId]

        ## If collision surpass exploration, our rule will be that the agent got no exploration check
        if self.collision_surpass_explore:
          pass
        ## If NOT surpassing, check if the position changes, then if there is a new tile
        elif (old_positions[pId] != new_positions[pId]) and self.check_new_tile(player, new_positions[pId]):
          exploration_array[pId] = 1
          events[pId] = 'explore' if events[pId] == 'other' else events[pId] + '|explore'

      # If no collision happened to this agent, update it normally
      elif (old_positions[pId] != new_positions[pId]) and self.check_new_tile(player, new_positions[pId]):
        exploration_array[pId] = 1
        events[pId] = 'explore'

      self.agent_positions[player] = new_positions[pId]
      self.agent_actions[player] = act
      self.agent_action_history[player].append(act)
      self.agent_events[player] = events[pId]
      self.agent_event_history[player].append(events[pId])
      self._infos[player] = {
        'event': events[pId],
        'action': self.agent_actions[player],
        'position': self.agent_positions[player],
        'old_position': old_positions[pId],
        'collision_pairs': collision_pairs,
      }

    # Update the collision events based on the defined rules
    ## If collide is shared between pairs, we will calculate the collision reward together
    if self.collision_share_between_pairs:
      collision_matrix = (collision_matrix + collision_matrix.T).astype(bool)

    ## If the collision is shared within a team, we will share the collision reward within the team of predators
    if self.collision_shared_within_chasers:
      raise NotImplementedError

    # Update the step reward
    for pId, (player, act) in enumerate(action_dict.items()):
      has_collision = collision_matrix[pId, :].sum() > 0
      collision_reward = collision_matrix[pId, :].sum() * self.reward_value_dict[player]['collision']

      ## If we check if we would like to mute the collision reward
      if self.roles[player] == 'chaser':
        mute_check_on = self.mute_chaser_collision
        past_collision_counts = np.sum(
          ['collide' in item for item in self.agent_event_history[player][-self.mute_chaser_collision - 1:-1]])
      elif self.roles[player] == 'escaper':
        mute_check_on = self.mute_escaper_collision
        past_collision_counts = np.sum(
          ['collide' in item for item in self.agent_event_history[player][-self.mute_escaper_collision - 1:-1]])
      if mute_check_on and has_collision and past_collision_counts:
        collision_reward = 0
        has_collision = False
        self.agent_events[player] += ' | muted'
        self.agent_event_history[player][-1] = self.agent_events[player]

      has_exploration = exploration_array[pId] > 0
      exploration_reward = exploration_array[pId] * self.reward_value_dict[player]['explore']
      if self.decaying_rewards_for_chaser_before_colliding and (self.roles[player] == 'chaser') and (exploration_reward > 0):
        exploration_reward *= self.calculate_decayed_reward_fraction(self.agent_event_history[player],
                                                                     self.decaying_rate_for_chaser_exploration)
      # other reward is zero if there is a collision or exploration
      other_reward = self.reward_value_dict[player]['other'] * (not (has_collision or has_exploration))
      step_reward = collision_reward + exploration_reward + other_reward
      self.rewards[player] += step_reward
      self.step_rewards[player] = step_reward

    # Get observations (based on new agent positions).
    obs = self.get_observation_space()
    # append current observations into the set of observation
    # self.obs_hist.append(self._get_discrete_obs())

    self.current_step += 1
    done = self.current_step >= self.timestep_limit
    if done:
      self.terminateds["__all__"] = True
      self.truncateds["__all__"] = True

    return (
    obs,
    self.rewards,
    self.terminateds,
    self.truncateds,
    self._infos,
    )

  def _move(self, player_i, position, action):
    # This function will return the calculated new position for the specific agent
    # Here it only updates the movement
    # if NOOP, return current position
    if self.ACTION_MAP[action] == [0, 0]:
      return position, None

    # Mapping actions to position changes
    delta = self.ACTION_MAP[action]
    new_position = [
      min(max(position[0] + delta[0], 0), self.height - 1),
      min(max(position[1] + delta[1], 0), self.width - 1),
    ]

    collision_pairs = self.detect_collision(player_i, new_position, action)
    if collision_pairs:
      return position, collision_pairs

    # return the new position of the agent
    return new_position, None

  def detect_collision(self, player_i, pos_i_new, action):
    # case 1: if another agent is already at the new position, remain stationary
    # case 2: if the agent itself occupies the new position, aka., unsuccessful move
    for player_j, pos_j in self.agent_positions.items():
      if (player_j != player_i) and (pos_j == pos_i_new):
        return [player_i, player_j]
    return None

  def check_new_tile(self, player, new_position):
    # Please note, we added one in advance, this should be considered in condition checking
    self.visit_counts[player][new_position[0], new_position[1]] += 1
    # Return True if this is the first visit (count == 1), otherwise False
    return self.visit_counts[player][new_position[0], new_position[1]] == 1

  def calculate_decayed_reward_fraction(self, agent_event_history, decayed_fraction_list):
    explore_count = 0
    collided_found = False

    # Loop backward through the event history
    for event in reversed(agent_event_history):
        # Check for 'collide' event and set the flag
        if 'collide' in event:
            collided_found = True
            break  # Stop the loop once 'collide' is found

        # Count 'explore' events after the last 'collide'
        elif 'explore' in event:
            explore_count += 1

    # Match the count to the decaying rate
    # explore_count - 1 because the counts include the current step
    decayed_reward_rate = decayed_fraction_list[min(explore_count - 1, len(decayed_fraction_list) - 1)]

    return decayed_reward_rate


  def get_observation_space(self):
    if "full" in self.partial_observability_style:
      return self.get_full_space()
    elif "partial" in self.partial_observability_style:
      observation_space = self.get_full_space()
      return self.crop_and_rotate_partial_view(observation_space)

  def crop_and_rotate_partial_view(self, observation_space):
    for idxi, (player_i, pos_i) in enumerate(self.agent_positions.items()):
      agent_grid = np.zeros((self.width, self.height), dtype=np.uint8)
      action_i = self.ACTION_MAP[self.agent_actions[player_i]]
      range_of_view = self.range_of_view[self.roles[player_i]]
      # Verify if range_of_view is dict
      if not isinstance(range_of_view, list):
        raise ValueError('range_of_view is not a list when trying to crop the field of view')
      fov_area = self.calculate_fov_area(pos_i, range_of_view, action_i)
      match action_i:
        case [-1, 0]:
          rot = 0
        case [0, 1]:
          rot = 1
        case [1, 0]:
          rot = 2
        case [0, -1]:
          rot = 3
      rot = self.agent_actions[player_i]
      x_min, y_min, x_max, y_max = fov_area
      x_max += 1
      y_max += 1
      # max_padding = np.max([0 - x_min, 0 - y_min, x_max - self.width, y_max - self.height])
      observation_space_i = observation_space[player_i]
      for key, matrix in observation_space_i.items():
        submatrix = np.full((x_max - x_min, y_max - y_min), 0)

        # Calculate the overlap between the original matrix and the desired submatrix
        x_min_overlap = max(x_min, 0)
        y_min_overlap = max(y_min, 0)
        x_max_overlap = min(x_max, matrix.shape[0])
        y_max_overlap = min(y_max, matrix.shape[1])

        # Copy the overlapping part of the original matrix to the submatrix
        submatrix[x_min_overlap - x_min:x_max_overlap - x_min, y_min_overlap - y_min:y_max_overlap - y_min] = \
          matrix[x_min_overlap:x_max_overlap, y_min_overlap:y_max_overlap]

        submatrix = np.rot90(submatrix, rot)
        observation_space[player_i][key] = submatrix
    return observation_space




  def get_separate_or_shared_space_per_agent(self, agent_grid, idxi, visible_agents, indices, positions):
    # TODO: This function is planned to help with get_full_space and get_partial_observation_space
    PADDING_INDEX = 1

    if not self.separate_grid_representation_for_other_agent:
      # Then, all visible agents will be assigned to a single matrix
      agent_grid[positions[visible_agents, 0], positions[visible_agents, 1]] = visible_agents + PADDING_INDEX
      return {"grid_world": agent_grid, }

    # If we are here, then we are going to create a separate grid for each agent
    # "grid_world" is remained for the observing agent,
    # while other agents will be assigned to a separate grid with their name
    agent_observation = {}
    for idxj in indices: # Loop through all agents
      grid = np.zeros_like(agent_grid, dtype=np.uint8)

      if idxj == idxi: # Handle the current agent
        grid[positions[idxj, 0], positions[idxj, 1]] = PADDING_INDEX
        agent_observation["grid_world"] = grid
        continue
      elif idxj in visible_agents: # If the agent is visible, then we will create a grid with its position
        grid[positions[idxj, 0], positions[idxj, 1]] = PADDING_INDEX
      else: # If the agent is not visible, then we will create a grid with all zeros
         pass

      agent_observation[f'grid_{self.players[idxj]}'] = grid

    return agent_observation

  def calculate_fov_area(self, observer_pos, fov_dims, orientation):
    # param observer_pos: (x, y)
    # param fov_dims: (left, up, right, down)
    # param orientation: 0, 1, 2, 3 for up, right, down, left
    # Unpack the FoV dimensions
    forward, right, back, left = fov_dims

    # Calculate FoV area based on orientation, by the sequence of x_min, y_min, x_max, y_max
    match orientation:
      case [-1, 0]:  # In the direction of x-minus
        return [observer_pos[0] - forward, observer_pos[1] - left, observer_pos[0] + back, observer_pos[1] + right]
      case [0, 1]:  # In the direction of y-plus
        return [observer_pos[0] - left, observer_pos[1] - back, observer_pos[0] + right, observer_pos[1] + forward]
      case [1, 0]:  # In the direction of x-plus
        return [observer_pos[0] - back, observer_pos[1] - right, observer_pos[0] + forward, observer_pos[1] + left]
      case [0, -1]:  # In the direction of y-minus
        return [observer_pos[0] - right, observer_pos[1] - forward, observer_pos[0] + left, observer_pos[1] + back]
      # Below is test only, future implementations should be more careful about the idle condition
      # case [0, 0]:
      #   return [observer_pos[0] - forward, observer_pos[1] - left, observer_pos[0] + back, observer_pos[1] + right]
      case _:
        raise ValueError(f'Invalid orientation {orientation}')
        return None

  def is_in_fov(self, agent_pos, fov_area):
    return (fov_area[0] <= agent_pos[0] <= fov_area[2]) and (fov_area[1] <= agent_pos[1] <= fov_area[3])

  def get_full_space(self):
    """
    Generate full space observations for each agent based on their positions and visibility.
    Handles different visibility ranges and observation styles.
    """
    observations = {}
    positions = np.array(list(self.agent_positions.values()))
    all_agent_indices = np.arange(len(positions))
    PADDING_INDEX = 1 # To differentiate from the default space value 0
    for idxi, (player_i, pos_i) in enumerate(self.agent_positions.items()):
      agent_grid = np.zeros((self.width, self.height), dtype=np.uint8)
      if self.config['range_of_view'] is not None:
        # In the case that we only enter a single int of range of view, it will be symmetrical in 4 orientations
        if isinstance(self.config['range_of_view'], int):
          range_of_view = self.config['range_of_view']
          distances_x, distances_y = abs(positions[:, 0] - pos_i[0]), abs(positions[:, 1] - pos_i[1])
          distances = np.maximum(distances_x, distances_y)
          visible_agents = all_agent_indices[distances <= range_of_view]
          observations[player_i] = self.get_separate_or_shared_space_per_agent(
            agent_grid, idxi, np.array(visible_agents), all_agent_indices, positions)

        # In the case that we enter a list of range of view, range in each orientation will be given in the list
        elif isinstance(self.config['range_of_view'], dict):
          range_of_view = self.range_of_view[self.roles[player_i]]
          fov_area = self.calculate_fov_area(pos_i, range_of_view, self.ACTION_MAP[self.agent_actions[player_i]])
          visible_agents = []
          for idxj, pos_j in enumerate(positions):
            # The function is_in_fov is tolerant to negative values in fov_area
            if (idxj == idxi) or self.is_in_fov(pos_j, fov_area):
              visible_agents.append(idxj)
          observations[player_i] = self.get_separate_or_shared_space_per_agent(
            agent_grid, idxi, np.array(visible_agents), all_agent_indices, positions)

      else: # No range of view limit
        agent_grid[positions[:, 0], positions[:, 1]] = all_agent_indices + PADDING_INDEX
        observations[player_i] = {"grid_world": agent_grid, }

      if self.visit_count_visible:
        observations[player_i]["visit_count"] = self.visit_counts[player_i]

      if 'coordinates' in self.partial_observability_style:
        # TODO: Implement functionality for coordinate based observability
        # coordinates = np.copy(self.coordinates)
        coordinates = np.arange(self.width * self.height).reshape((self.width, self.height)).astype(np.uint16) + 1
        coordinates[pos_i[0], pos_i[1]] = 0
        observations[player_i]['coordinates'] = coordinates
        self.coordinates[player_i] = coordinates
    return observations

  # def get_partial_observation_space(self):
  #   observations = {}
  #   positions = np.array(list(self.agent_positions.values()))
  #   all_agent_indices = np.arange(len(positions))
  #   indices = np.arange(len(positions))
  #   for idxi, (player_i, pos_i) in enumerate(self.agent_positions.items()):
  #     agent_grid = np.zeros((self.width, self.height), dtype=np.uint8)
  #     range_of_view = self.range_of_view[self.roles[player_i]]
  #     # Ensure that range_of_view is dict
  #     if not isinstance(range_of_view, dict):
  #       raise ValueError
  #     # The following 7 lines replicate the other function. Future code optimization may be considered.
  #     fov_area = self.calculate_fov_area(pos_i, range_of_view, self.ACTION_MAP[self.agent_actions[player_i]])
  #     visible_agents = []
  #     for idxj, pos_j in enumerate(positions):
  #       if (idxj == idxi) or self.is_in_fov(pos_j, fov_area):
  #         visible_agents.append(idxj)
  #     observations[player_i] = self.get_separate_or_shared_space_per_agent(
  #       agent_grid, idxi, np.array(visible_agents), all_agent_indices, positions)
  #
  #     if self.visit_count_visible:
  #       observations[player_i]["visit_count"] = self.visit_counts[player_i]
  #
  #     if 'partial' in self.partial_observability_style:
  #       raise NotImplementedError
  #       # TODO: here needs a crop and rotate function
  #     if 'coordinates' in self.partial_observability_style:
  #       raise NotImplementedError
  #   return observations

  def cell_to_str(self, cell):
    if cell == 0:
      return " "
    elif cell >= 10:
      # Convert numbers 10, 11, 12, ... to 'a', 'b', 'c', ...
      return chr(ord('A') + cell - 10)
    else:
      return str(cell)

  def render(self):

    # Using a simple console output for rendering, for more advanced rendering consider using external libraries
    grid = np.zeros((self.width, self.height), dtype=np.uint8)
    for player, pos in self.agent_positions.items():
      grid[pos[0], pos[1]] += self.players.index(player) + 1

    # Display top padding
    print("~~" * (self.width))
    for row in grid:
      row_str = " ".join("{:1}".format(self.cell_to_str(cell)) for cell in row)
      print(f"|{row_str}|")
      # print(" ".join(map(str, row)))
    print("~~" * (self.width))
    print("-------------------------------")
    step_rewards = [f"{reward:.2f}" for reward in self.step_rewards.values()]
    total_rewards = [f"{reward:.2f}" for reward in self.rewards.values()]

    if self.config['using_old_action_map']:
      action_descriptions = {
        0: 'left ',
        1: 'down ',
        2: 'right',
        3: 'up   ',
        4: 'idle ',
      }
    else:
      action_descriptions = {
        0: 'up   ',
        1: 'right',
        2: 'down ',
        3: 'left ',
        4: 'idle ',
      }

    if self.current_step == 0:
      actions = []
      exploring_new_tile = []
    else:
      actions = [
        action_descriptions.get(action[-1], "unknown")
        for action in self.agent_action_history.values()
      ]
      # coliision = [
      #   "yes" if (float(step_reward) > 0) and (action != "idle") else "no"
      #   for step_reward, action in zip(step_rewards, actions)
      # ]

      exploring_new_tile = [
        "yes" if ('explore' in event) else "no" for agent, event in self.agent_events.items()]
    data = [
      ["event", *self.agent_events.values()],
      ["Action", *actions],
      ["Exploring New Tile", *exploring_new_tile],
      ["Step Rewards", *step_rewards],
      ["Total Rewards", *total_rewards],
    ]
    print(tabulate(data, headers=["Agents", *range(1, len(self.players) + 1)]))

  def get_config(self):
    return self.config

  def render_to_image(self):
    ## TODO: Improve this method
    ## Currently, it is temporarily based on Nguyen's implementation with weird adaptors
    # Define some constants for the image
    FONT_PATH = "arial.ttf"  # Replace with the path to your desired font file
    FONT_SIZE = 14
    FONT_COLOR = (0, 0, 0)  # Black color for text
    CELL_SIZE = 20
    BORDER_SIZE = 1
    BORDER_COLOR = (125, 125, 125)  # Black color for borders
    COLORS = [(255,69,0), (0,191,255)]

    # Create a new image with the desired dimensions
    image_width = (self.width * CELL_SIZE) + ((self.width + 1) * BORDER_SIZE)
    image_height = (self.height + 13 * CELL_SIZE) + ((self.height + 13 + 1) * BORDER_SIZE)
    image = Image.new("RGB", (int(image_width * 3.5), image_height), color=(255, 255, 255))

    # Create a drawing context for the image
    draw = ImageDraw.Draw(image)
    font = ImageFont.truetype(FONT_PATH, FONT_SIZE)
    font_text = ImageFont.truetype(FONT_PATH, 12)

    # Add the text labels
    label_x = BORDER_SIZE
    label_y = (self.height * CELL_SIZE) + ((self.height + 1) * BORDER_SIZE) + BORDER_SIZE
    ## Following Nguyen's format, r2 is for chaser and r1 is for excaper
    label_text_r2 = self._infos['agent_0']['event']
    color_text_r2 = (255, 0, 0) if ('collide' in label_text_r2) else (255, 255, 255) if ('explore' in label_text_r2) else (0, 0, 0)
    label_text_r1 = self._infos['agent_1']['event']
    color_text_r1 = (255, 0, 0) if ('collide' in label_text_r1) else (255, 255, 255) if ('explore' in label_text_r1) else (0, 0, 0)
    draw.text((label_x, label_y), label_text_r1, fill=FONT_COLOR, font=font_text)
    draw.text((label_x + 100, label_y), label_text_r2, fill=FONT_COLOR, font=font_text)

    # add rewards
    if self.step_rewards['agent_1'] < 0:
      c1 = 155
    else:
      c1 = 255

    if self.step_rewards['agent_0'] < 0:
      c2 = 155
    else:
      c2 = 255

    draw.text((label_x, label_y + 15), "E={: .1f}".format(self.rewards['agent_1']), fill=FONT_COLOR, font=font_text)
    draw.rectangle((label_x + 115, label_y + 15, label_x + 115 + self.step_rewards['agent_1'] * 20, label_y + 30), fill=(0, 0, c1),
                   outline=BORDER_COLOR)

    draw.text((label_x, label_y + 15 + 15), "C={: .1f}".format(self.rewards['agent_0']), fill=FONT_COLOR, font=font_text)
    draw.rectangle((label_x + 115, label_y + 30, label_x + 115 + self.step_rewards['agent_0'] * 20, label_y + 45), fill=(c2, 0, 0),
                   outline=BORDER_COLOR)

    draw.text((label_x, label_y + 45), "ts={: .1f}".format(self.current_step), fill=FONT_COLOR, font=font_text)
    # add arena name
    draw.text((label_x, label_y + 60), self.__class__.__name__, fill=FONT_COLOR, font=font_text)

    for r in range(self.height):
      for c in range(self.width):
        cell_left = (c * CELL_SIZE) + ((c + 1) * BORDER_SIZE)
        cell_top = (r * CELL_SIZE) + ((r + 1) * BORDER_SIZE)
        cell_right = cell_left + CELL_SIZE
        cell_bottom = cell_top + CELL_SIZE

        # If R1 paint last
        agent1_visited_fields = [(a,b) for a, b in np.array(np.where(self.visit_counts['agent_1'])).T.tolist()]
        agent2_visited_fields = [(a,b) for a, b in np.array(np.where(self.visit_counts['agent_0'])).T.tolist()]
        if ((r, c) in agent1_visited_fields) and ((r, c) in agent2_visited_fields):
          fill_color = (255, 150, 255)  # purple for both painted
        elif (r, c) in agent1_visited_fields and (r, c) not in agent2_visited_fields:
          fill_color = (0, 150, 255)  # blue for R1 painted
        elif (r, c) not in agent1_visited_fields and (r, c) in agent2_visited_fields:
          fill_color = (255, 150, 100)  # red for R2 painted
        else:
          fill_color = (255, 255, 255)  # White color for empty cells

        draw.rectangle((cell_left, cell_top, cell_right, cell_bottom), fill=fill_color, outline=BORDER_COLOR,)

        # Draw the agent symbols
        hist_pos = {
          0: {'agent1': self._infos['agent_1']['position'],
              'agent2': self._infos['agent_0']['position'],
              },
          1: {'agent1': self._infos['agent_1']['old_position'],
               'agent2': self._infos['agent_0']['old_position'],}
        }
        col_prc = [0.4, 0.1]
        for i in range(0, min(len(hist_pos), 4), 1):
          tmp = int(255 * col_prc[i])
          col = (tmp, tmp, tmp)

          r_a1 = int(hist_pos[i]['agent1'][0])
          c_a1 = hist_pos[i]['agent1'][1]

          r_a2 = int(hist_pos[i]['agent2'][0])
          c_a2 = hist_pos[i]['agent2'][1]

          if [r_a1, c_a1] == [r, c]:  # and self.agent2_pos != [r,c]:
            draw.text((cell_left + CELL_SIZE // 2, cell_top + CELL_SIZE // 2), "x", fill=col,
                      font=font_text, anchor="mm")
          if [r_a2, c_a2] == [r, c]:  # and self.agent1_pos != [r,c]:
            draw.text((cell_left + CELL_SIZE // 2, cell_top + CELL_SIZE // 2), "o", fill=col,
                      font=font_text, anchor="mm")

        if self.agent_positions['agent_1'] == [r, c]:
          draw.text((cell_left + CELL_SIZE // 2, cell_top + CELL_SIZE // 2), "X", fill=color_text_r1,
                    font=font, anchor="mm")
        elif self.agent_positions['agent_0'] == [r, c]:
          draw.text((cell_left + CELL_SIZE // 2, cell_top + CELL_SIZE // 2), "O", fill=color_text_r2,
                    font=font, anchor="mm")
    # The following is to add the agent FOV
    for pId, player in enumerate(self.players):
      fov_area = self.calculate_fov_area(self.agent_positions[player], self.range_of_view[self.roles[player]],
                                         self.ACTION_MAP[self.agent_actions[player]])
      x_min, y_min, x_max, y_max = fov_area #
      y_max += 1
      x_max += 1
      FOV_left = (y_min * CELL_SIZE) + ((y_min + 1) * BORDER_SIZE) + 1
      FOV_top = (x_min * CELL_SIZE) + ((x_min + 1) * BORDER_SIZE) + 1
      FOV_right = (y_max * CELL_SIZE) + ((y_max + 1) * BORDER_SIZE) - 1
      FOV_bottom = (x_max * CELL_SIZE) + ((x_max + 1) * BORDER_SIZE) - 1
      # FOV_bottom, FOV_left, FOV_top, FOV_right = np.array(fov_area) * CELL_SIZE
      draw.rectangle((FOV_left, FOV_top, FOV_right, FOV_bottom), fill=None, outline=COLORS[pId%2], width=3)
      #draw.text((label_x, label_y + 15), "E={: .1f}".format(self.rewards['agent_1']), fill=FONT_COLOR, font=font_text
      horizontal_padding = (self.width + 1) * (CELL_SIZE + BORDER_SIZE) * (pId + 1.05)
      for r in range(self.height):
        for c in range(self.width):
          cell_left = (c * CELL_SIZE) + ((c + 1) * BORDER_SIZE) + horizontal_padding
          cell_top = (r * CELL_SIZE) + ((r + 1) * BORDER_SIZE)
          cell_right = cell_left + CELL_SIZE
          cell_bottom = cell_top + CELL_SIZE
          draw.rectangle((cell_left, cell_top, cell_right, cell_bottom), fill=fill_color, outline=BORDER_COLOR)
          if self.visit_counts[player][r,c] > 0:
            draw.text((cell_left + CELL_SIZE // 2, cell_top + CELL_SIZE // 2), str(self.visit_counts[player][r,c]),
                      fill=FONT_COLOR, font=font, anchor="mm")
      draw.rectangle((horizontal_padding + FOV_left, FOV_top, horizontal_padding + FOV_right, FOV_bottom),
                     fill=None, outline=COLORS[pId % 2], width=3)
      draw.text((label_x + horizontal_padding, label_y), self.roles[player] + ' visit history',
                fill=FONT_COLOR, font=font_text)

    return np.array(image)