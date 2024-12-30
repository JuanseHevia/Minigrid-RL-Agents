from minigrid.envs import UnlockPickupEnv
import numpy as np

class RewardShapingUP(UnlockPickupEnv):

    def _gen_grid(self, width, height):
        # Call RoomGrid's _gen_grid method
        super(UnlockPickupEnv, self)._gen_grid(width, height)

        # Add a box to the room on the right
        obj, obj_pos = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, door_pos = self.add_door(0, 0, 0, locked=True)
        
        # add to class attributes
        self.obj = obj
        self.door = door
        self.obj_position = obj_pos
        self.door_position = door_pos

        # Add a key to unlock the door
        key, key_pos = self.add_object(0, 0, "key", door.color)
        self._custom_key = key
        self._custom_key_position = key_pos
        self._custom_key_picked_up = False

        self.place_agent(0, 0)

        self.mission = f"pick up the {obj.color} {obj.type}"

        self.opened_door = False

        # track the state of the door
        self._custom_door_was_locked = True

    def _custom_compute_distance(self, pos1, pos2):
        """
        Compute the Euclidean distance between two positions.
        """
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def step(self, action):
        """
        Add a custom reward shaping to the environment
        based on whether the agent gets closer to the key if the door has not
        been unlocked yet
        """

        obs, reward, terminated, truncated, info = super().step(action)

        # door is locked
        _door_is_locked = self.grid.get(*self.door_position).is_locked

        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                terminated = True
        
        # reward the model for picking up the key
        if (self.carrying == self._custom_key) and (not self._custom_key_picked_up):
            self._custom_key_picked_up = True
            reward = self._reward() / 2

        # reward the model for opening the door
        if (self._custom_door_was_locked) and (not _door_is_locked):
            # if door is not locked, set the custom door_is_locked attribute to False
            self._custom_door_is_locked = False
            
            # reward a slight amount
            reward = self._reward() / 2

        # reward the model for dropping the key after opening the door
        if (not self.carrying) and (not self._custom_door_was_locked):
            reward = self._reward() / 2

        return obs, reward, terminated, truncated, info
    
class RewardShapingUPExtraRewardOnce(UnlockPickupEnv):

    def _gen_grid(self, width, height):
        # Call RoomGrid's _gen_grid method
        super(UnlockPickupEnv, self)._gen_grid(width, height)

        # Add a box to the room on the right
        obj, obj_pos = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, door_pos = self.add_door(0, 0, 0, locked=True)
        
        # add to class attributes
        self.obj = obj
        self.door = door
        self.obj_position = obj_pos
        self.door_position = door_pos

        # Add a key to unlock the door
        key, key_pos = self.add_object(0, 0, "key", door.color)
        self._custom_key = key
        self._custom_key_position = key_pos
        self._custom_key_picked_up = False

        self.place_agent(0, 0)

        self.mission = f"pick up the {obj.color} {obj.type}"

        self.opened_door = False

        # track the state of the door
        self._custom_door_was_locked = True

        # track if the model was rewarded for picking up the key
        self._custom_key_picked_up_rewarded = False

        # track if the model was rewarded for opening the door
        self._custom_door_opened_rewarded = False

        # track if the model was rewarded for dropping the key after opening the door
        self._custom_key_dropped_rewarded = False

        self.EXTRA_REWARD_FACTOR = 0.2 # first version was 0.5

        # initialize placeholder for the object the agent was carrying
        self.was_carrying_obj = None

    def _custom_compute_distance(self, pos1, pos2):
        """
        Compute the Euclidean distance between two positions.
        """
        return np.linalg.norm(np.array(pos1) - np.array(pos2))

    def step(self, action):
        """
        Add a custom reward shaping to the environment
        based on whether the agent gets closer to the key if the door has not
        been unlocked yet
        """

        obs, reward, terminated, truncated, info = super().step(action)

        # door is locked
        _door_is_locked = self.grid.get(*self.door_position).is_locked

        # true reward, for final goal achieved
        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                terminated = True
        
        # reward the model for picking up the key
        elif ((self.front_pos[0] == self._custom_key_position[0]) and (self.front_pos[0] == self._custom_key_position[0])) and (not self._custom_key_picked_up) \
              and (not self._custom_key_picked_up_rewarded):
            
            self._custom_key_picked_up = True
            reward = self._reward() * self.EXTRA_REWARD_FACTOR
            self._custom_key_picked_up_rewarded = True

        # reward the model for opening the door
        if (action == self.actions.toggle) and ((self.front_pos[0] == self.door_position[0]) and \
            (self.front_pos[1] == self.door_position[1])) and (not _door_is_locked) and (not self._custom_door_opened_rewarded):
            # if door is not locked, set the custom door_is_locked attribute to False
            self._custom_door_was_locked = False
            
            # reward a slight amount
            reward = self._reward() * self.EXTRA_REWARD_FACTOR

            self._custom_door_opened_rewarded = True

        # reward the model for dropping the key after opening the door
        if (action == self.actions.drop) and (self.was_carrying_obj == self._custom_key) and \
             (not self._custom_door_was_locked) and (not self._custom_key_dropped_rewarded):
            reward = self._reward() * self.EXTRA_REWARD_FACTOR
            self._custom_key_dropped_rewarded = True

        self.was_carrying_obj = self.carrying

        return obs, reward, terminated, truncated, info

    def reset(self, *, seed = None, options = None):
        obs = super().reset(seed=seed, options=options)

        self._custom_key_picked_up = False
        self._custom_door_was_locked = True
        self._custom_key_picked_up_rewarded = False
        self._custom_door_opened_rewarded = False
        self._custom_key_dropped_rewarded = False
        self.was_carrying_obj = None
        return obs

class RewardShapingUPSimple(RewardShapingUPExtraRewardOnce):

    def step(self, action):
        """
        Simplify intermediate rewards for the agent.
        Just reward the model for toggling the door to open it.
        Keep the original reward for picking up the object.
        """

        obs, reward, terminated, truncated, info = super().step(action)

        # door is locked
        _door_is_locked = self.grid.get(*self.door_position).is_locked

        # true reward, for final goal achieved
        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                terminated = True
        
        # reward the model for opening the door while holding the key
        if (action == self.actions.toggle) \
            and ((self.front_pos[0] == self.door_position[0]) and (self.front_pos[1] == self.door_position[1])) \
            and (not _door_is_locked) and (self.carrying == self._custom_key):
            # if door is not locked, set the custom door_is_locked attribute to False
            self._custom_door_was_locked = False
            
            # reward a slight amount
            reward = self._reward() * 0.1

        # reward the model for dropping the key after opening the door
        if (action == self.actions.drop) and (self.was_carrying_obj == self._custom_key) and \
             (not self._custom_door_was_locked):
            reward = self._reward() * 0.1

        # keep track of the object the agent was carrying
        self.was_carrying_obj = self.carrying

        return obs, reward, terminated, truncated, info

class RewardShapingUPSimpleV1(RewardShapingUPExtraRewardOnce):

    def step(self, action):
        """
        Keep the original reward for picking up the target object.
        Include an intermediate reward for opening the door.
        Include an intermediate reward for dropping the key after opening the door.
        """

        obs, reward, terminated, truncated, info = super(RewardShapingUPExtraRewardOnce, self).step(action)

        # true reward, for final goal achieved
        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                terminated = True
        
        # reward the model for opening the door while holding the key
        if (action == self.actions.toggle):
            if ((self.front_pos[0] == self.door_position[0]) \
                and (self.front_pos[1] == self.door_position[1])) \
                and (self._custom_door_was_locked) \
                and (self.carrying == self._custom_key):
                # if door is not locked, set the custom door_is_locked attribute to False
                self._custom_door_was_locked = False
                
                # reward a slight amount
                reward = self._reward() * 0.25

        # reward the model for dropping the key after opening the door
        if (action == self.actions.drop):
            if (self.was_carrying_obj == self._custom_key) \
                and (not self._custom_key_dropped_rewarded) \
                and (not self._custom_door_was_locked):
                self._custom_key_dropped_rewarded = True
                reward = self._reward() * 0.25

        # keep track of the object the agent was carrying
        self.was_carrying_obj = self.carrying

        return obs, reward, terminated, truncated, info