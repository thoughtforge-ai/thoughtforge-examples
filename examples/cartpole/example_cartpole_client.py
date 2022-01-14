#############
# MIT License
#
# Copyright (C) 2020 ThoughtForge Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a copy 
# of this software and associated documentation files (the "Software"), to deal 
# in the Software without restriction, including without limitation the rights 
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell 
# copies of the Software, and to permit persons to whom the Software is 
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all 
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR 
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, 
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE 
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER 
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, 
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE 
# SOFTWARE.
#############

import gym, os

from thoughtforge_client.thoughtforge_client import BaseThoughtForgeClientSession


# this is just a modification to the cartpole environment to extend it to 500 steps
gym.register(
    id='long-CartPole-v0',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=500,
    reward_threshold=195.0,
)

class ExampleCartpoleSession(BaseThoughtForgeClientSession):

    def _reset_env(self):
        """ local helper function specific for openAI gym environments """
        self.last_observation = self.env.reset()
        self.score = 0

    def sim_started_notification(self):
        """ On sim start, initialize environment """   
        print("Initializing Cartpole...", end='')
        self.env = gym.make('long-CartPole-v0')
        if self.env is not None:
            self._reset_env()
        print("Complete.")

    def sim_ended_notification(self):
        """ On sim start, destroy environment """
        if self.env is not None:
            self.env.close()
            self.env = None
        
    def update(self, motor_dict):
        """ advance the environment sim """
        # render the environment locally
        self.env.render()

        # extract action sent from server
        motor_value = motor_dict['motor']
        cartpole_action = 1 if motor_value >= 0.0 else 0
        
        # step openAI gym env and get updated observation
        env_step_result = self.env.step(cartpole_action)
        self.last_observation, reward, terminal, _ = env_step_result
        self.score += reward
        if terminal:
            print("End of episode. Score =", self.score)
            self._reset_env()
 
        # send updated environment data to server
        sensor_values = {
            'pos_sensor': self.last_observation[0],
            'vel_sensor': self.last_observation[1],
            'angle_sensor1': self.last_observation[2],
            'angle_sensor2': self.last_observation[2],
            'angle_vel_sensor1': self.last_observation[3],
            'angle_vel_sensor2': self.last_observation[3],
        }
        return sensor_values


if __name__ == "__main__": 
    # the basic example doesn't have the best performance, but is simple to follow:
    # session = ExampleCartpoleSession('./examples/cartpole/example_cartpole.params')
    # to see a more advanced solution for cartpole:
    session = ExampleCartpoleSession.from_file('./examples/cartpole/advanced_cartpole.params')
