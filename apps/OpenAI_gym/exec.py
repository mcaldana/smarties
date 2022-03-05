#!/usr/bin/env python3
##
# smarties
# Copyright (c) 2018 CSE-Lab, ETH Zurich, Switzerland. All rights reserved.
# Distributed under the terms of the MIT license.
##
# Created by Guido Novati (novatig@ethz.ch).
##

import numpy as np
from gym.utils import seeding
from gym import spaces, logger
import gym
import math
import smarties as rl
import gym
import sys
import os
import numpy as np
from scipy.integrate import ode

os.environ['MUJOCO_PY_FORCE_CPU'] = '1'

class ContinuousCartPoleEnv(gym.Env):
    def __init__(self):
        self.dt = 0.02
        self.mstep = 0
        self.u = np.asarray([0, 0, 0, 0])
        self.F = 0
        self.t = 0
        self.ODE = ode(self.system).set_integrator('dopri5')

        self.x_threshold = 2.4
        self.theta_threshold_radians = np.pi / 15

        self._max_episode_steps = 500

        high = np.array([
            self.x_threshold * 2,
            np.finfo(np.float64).max,
            self.theta_threshold_radians * 2,
            np.finfo(np.float64).max],
            dtype=np.float64
        )

        self.action_space = spaces.Box(
            low=np.float64(-10),
            high=np.float64(10),
            shape=(1,)
        )
        self.observation_space = spaces.Box(-high, high)

        self.seed()
        self.viewer = None
        self.state = None
        self.steps_beyond_done = None

    def seed(self, seed=None):
        self.np_random, seed = seeding.np_random(seed)
        return [seed]

    @staticmethod
    def system(t, y, act):
        mp, mc, l, g = 0.1, 1, 0.5, 9.81
        x, v, a, w = y[0], y[1], y[2], y[3]
        cosy, siny = np.cos(a), np.sin(a)

        totMass = mp + mc
        fac2 = l*(4./3. - mp*cosy*cosy/totMass)
        F1 = act + mp*l*w*w*siny
        wdot = (g*siny - F1*cosy/totMass)/fac2
        vdot = (F1 - mp*l*wdot*cosy)/totMass
        return [v, vdot, w, wdot]

    def step(self, action):
        self.F = action[0]
        self.ODE.set_initial_value(self.u, self.t).set_f_params(self.F)
        self.u = self.ODE.integrate(self.t + self.dt)
        self.t = self.t + self.dt
        self.mstep = self.mstep + 1
        return self.u, 1.0 - 1.0 * self.isFailed(), self.isOver(), {}

    def reset(self):
        self.u = np.random.uniform(-0.05, 0.05, 4)
        self.mstep = 0
        self.F = 0
        self.t = 0
        return self.u

    def isFailed(self):
        return abs(self.u[0]) > self.x_threshold or \
               abs(self.u[2]) > self.theta_threshold_radians

    def isOver(self):
        return self.mstep >= self._max_episode_steps or self.isFailed()

###############################################################################


def getAction(comm, env):
    buf = comm.recvAction()
    if hasattr(env.action_space, 'n'):
        action = int(buf[0])
    elif hasattr(env.action_space, 'spaces'):
        action = [int(buf[0])]
        for i in range(1, comm.nActions):
            action = action + [int(buf[i])]
    elif hasattr(env.action_space, 'shape'):
        action = buf
    else:
        assert(False)
    return action


def setupSmartiesCommon(comm, task, env=None):
    if env is None:
        env = gym.make(task)

    # setup MDP properties:
    # first figure out dimensionality of state
    dimState = 1
    if hasattr(env.observation_space, 'shape'):
        for i in range(len(env.observation_space.shape)):
            dimState *= env.observation_space.shape[i]
    elif hasattr(env.observation_space, 'n'):
        dimState = 1
    else:
        assert(False)

    # then figure out action dims and details
    if hasattr(env.action_space, 'spaces'):
        dimAction = len(env.action_space.spaces)
        comm.setStateActionDims(dimState, dimAction, 0)  # 1 agent
        control_options = dimAction * [0]
        for i in range(dimAction):
            control_options[i] = env.action_space.spaces[i].n
        comm.setActionOptions(control_options, 0)  # agent 0
    elif hasattr(env.action_space, 'n'):
        dimAction = 1
        comm.setStateActionDims(dimState, dimAction, 0)  # 1 agent
        comm.setActionOptions(env.action_space.n, 0)  # agent 0
    elif hasattr(env.action_space, 'shape'):
        dimAction = env.action_space.shape[0]
        comm.setStateActionDims(dimState, dimAction, 0)  # 1 agent
        isBounded = dimAction * [True]
        comm.setActionScales(env.action_space.high,
                             env.action_space.low, isBounded, 0)
    else:
        assert(False)

    return env


def app_main(comm):
    task = sys.argv[1]
    print("openAI environment: ", task)
    if task == 'ContinuousCartPoleEnv':
        env = setupSmartiesCommon(comm, task, ContinuousCartPoleEnv())
    else:
        env = setupSmartiesCommon(comm, task)

    while True:  # training loop
        observation = env.reset()
        t = 0
        comm.sendInitState(observation)
        while True:  # simulation loop
            action = getAction(comm, env)  # receive action from smarties
            observation, reward, done, info = env.step(action)
            t = t + 1
            if done and t >= env._max_episode_steps:
                comm.sendLastState(observation, reward)
            elif done:
                comm.sendTermState(observation, reward)
            else:
                comm.sendState(observation, reward)
            if done:
                break


if __name__ == '__main__':
    e = rl.Engine(sys.argv)
    if(e.parse()):
        exit()
    e.run(app_main)
