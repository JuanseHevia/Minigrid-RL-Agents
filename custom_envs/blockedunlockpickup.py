from minigrid.envs import BlockedUnlockPickupEnv
from minigrid.core.world_object import Ball
import numpy as np

class RewardShapingBUP(BlockedUnlockPickupEnv):


    def _gen_grid(self, width, height):
        super(BlockedUnlockPickupEnv, self)._gen_grid(width, height)

        # Add a box to the room on the right
        obj, _ = self.add_object(1, 0, kind="box")
        # Make sure the two rooms are directly connected by a locked door
        door, pos = self.add_door(0, 0, 0, locked=True)
        # Block the door with a ball
        color = self._rand_color()
        self.grid.set(pos[0] - 1, pos[1], Ball(color))

        # get ball object instance
        self.ball = self.grid.get(pos[0] - 1, pos[1])


        # Add a key to unlock the door
        self.add_object(0, 0, "key", door.color)

        self.place_agent(0, 0)

        self.obj = obj
        self.mission = f"pick up the {obj.color} {obj.type}"

        # helper fun
        self.door_pos = pos


        # initialize extra intermediate reward flags
        self._pickup_intermediate_reward = False
        self._dropped_intermediate_reward = False
        self._moved_ball_intermediate_reward = False
        self._was_carrying_ball = False

    def step(self, action):
        
        obs, reward, terminated, truncated, info = super().step(action)

        # reward the model for meeting the task
        if action == self.actions.pickup:
            if self.carrying and self.carrying == self.obj:
                reward = self._reward()
                terminated = True

            # reward the model for picking up the ball
            elif (self.carrying and self.carrying == self.ball):
                self._was_carrying_ball = True
                if (not self._pickup_intermediate_reward):
                    self._pickup_intermediate_reward = True
                    reward = self._reward() * 0.2

        # reward the model for dropping the ball far from the door
        if action == self.actions.drop:
            if (self._was_carrying_ball):
                self._was_carrying_ball = False

                # drop the ball outside a 1 block radius from the door
                if (not self._dropped_intermediate_reward) and \
                    (abs(self.agent_pos[0] - self.door_pos[0]) > 1 or\
                     abs(self.agent_pos[1] - self.door_pos[1]) > 1):

                    self._dropped_intermediate_reward = True
                    reward = self._reward() * 0.25


        return obs, reward, terminated, truncated, info
            

            

        