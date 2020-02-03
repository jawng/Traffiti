#!/usr/bin/env python
# Eclipse SUMO, Simulation of Urban MObility; see https://eclipse.org/sumo
# Copyright (C) 2009-2018 German Aerospace Center (DLR) and others.
# This program and the accompanying materials
# are made available under the terms of the Eclipse Public License v2.0
# which accompanies this distribution, and is available at
# http://www.eclipse.org/legal/epl-v20.html
# SPDX-License-Identifier: EPL-2.0

# @file    runner.py
# @author  Lena Kalleske
# @author  Daniel Krajzewicz
# @author  Michael Behrisch
# @author  Jakob Erdmann
# @date    2009-03-26
# @version $Id$

from __future__ import absolute_import
from __future__ import print_function

import os
import sys
import optparse
import random
import numpy as np
from collections import deque
# we need to import python modules from the $SUMO_HOME/tools directory
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(os.environ['SUMO_HOME'], 'tools')
    sys.path.append(tools)
else:
    sys.exit("please declare environment variable 'SUMO_HOME'")

from sumolib import checkBinary  # noqa
import traci  # noqa

lane_length = 200
cell_length = 8
epochs = 100

from agent import Agent

def generate_routefile():
    random.seed(69)  # make tests reproducible
    N = 1500  # number of time steps
    # demand per second from different directions
    pWE = 1. / 10
    pEW = 1. / 11
    pNS = 1. / 30
    pSN = 1. / 31
    with open("data/cross.rou.xml", "w+") as routes:
        print("""<routes>
        <vType id="typeWE" accel="0.8" decel="4.5" sigma="0.5" length="5" minGap="2.5" maxSpeed="16.67" \
guiShape="passenger"/>
        <vType id="typeNS" accel="0.8" decel="4.5" sigma="0.5" length="7" minGap="3" maxSpeed="25" guiShape="bus"/>
        <route id="right" edges="51o 1i 2o 52i" />
        <route id="left" edges="52o 2i 1o 51i" />
        <route id="down" edges="54o 4i 3o 53i" />
        <route id="up" edges="53o 3i 4o 54i" />""", file=routes)
        vehNr = 0
        for i in range(N):
            if random.uniform(0, 1) < pWE:
                print('    <vehicle id="right_%i" type="typeWE" route="right" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pEW:
                print('    <vehicle id="left_%i" type="typeWE" route="left" depart="%i" />' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pNS:
                print('    <vehicle id="down_%i" type="typeNS" route="down" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
            if random.uniform(0, 1) < pSN:
                print('    <vehicle id="up_%i" type="typeNS" route="up" depart="%i" color="1,0,0"/>' % (
                    vehNr, i), file=routes)
                vehNr += 1
        print("</routes>", file=routes)

# The program looks like this
#    <tlLogic id="0" type="static" programID="0" offset="0">
# the locations of the tls are      NESW
#        <phase duration="31" state="GrGr"/>
#        <phase duration="6"  state="yryr"/>
#        <phase duration="31" state="rGrG"/>
#        <phase duration="6"  state="ryry"/>
#    </tlLogic>

def get_state(lights):
    lanes = ['1i_0','2i_0','3i_0','4i_0']
    state = []
    for lane in lanes:
        substate = np.zeros((2,int(lane_length/cell_length)))
        cars = traci.lane.getLastStepVehicleIDs(lane)
        for car in cars:
            pos = 495.25-traci.vehicle.getLanePosition(car)
            if pos>=lane_length:
                continue
            speed = traci.vehicle.getSpeed(car)
            index = int(np.floor(pos/cell_length))
            if substate[0,index] == 0:
                substate[0,index] = 1
            if substate[1,index] == 0:
                substate[1,index] = speed
        state.append(substate)

    # i = 1
    # while i<len(lights) and lights[-i-1] == lights[-1]:
    #     i+=1
    state = np.append(np.array(state).flatten(),lights[-1])
    return state.reshape((1,201))

def get_cost():
    cost = 0
    lanes = ['1i_0','2i_0','3i_0','4i_0']
    for lane in lanes:
        cost += traci.lane.getLastStepHaltingNumber(lane)
    return cost

def get_reward(costs,lights):
    #r = -(costs[-1]-costs[-2])
    r = -costs[-1]

    if lights[-1] != lights[-2]:
        r -= 1

    i = 1
    while i<len(lights) and lights[-i-1] == lights[-1]:
        i+=1
    r -= 5/i

    return r/10

def run():
    """execute the TraCI control loop"""
    step = 0
    instance = []
    costs = []
    lights = []
    rewards = []
    # we start with phase 2 where EW has green
    # a:0 -> NS
    # a:1 -> EW
    light = traci.trafficlight.getPhase("0")
    if light == 0:
        lights.append(0)
    else:
        lights.append(1)

    state = get_state(lights)
    instance.append(state)
    a = agent.predict(state)
    instance.append(a)
    costs.append(get_cost())

    #choose action
    if a == 0:
        traci.trafficlight.setPhase("0", 0)
    else:
        traci.trafficlight.setPhase("0", 2)

    while step < 1500:
        traci.simulationStep()

        light = traci.trafficlight.getPhase("0")
        if light == 0:
            lights.append(0)
        else:
            lights.append(1)

        state = get_state(lights)
        instance.append(state)

        costs.append(get_cost())

        r = get_reward(costs,lights)
        instance.append(r)
        rewards.append(r)

        agent.memorize(instance)
        agent.train()

        instance = []
        instance.append(state)
        a = agent.predict(state)
        instance.append(a)

        if a == 0:
            traci.trafficlight.setPhase("0", 0)
        else:
            traci.trafficlight.setPhase("0", 2)

        step += 1

    print('Average number of cars waiting at any given moment: %s'%(np.mean(costs)))
    print('Average reward: %s'%(np.mean(rewards)))
    f.write('%s: %.3f, %.3f'%(epochs,np.mean(costs),np.mean(rewards)))
    traci.close()
    sys.stdout.flush()


def get_options():
    optParser = optparse.OptionParser()
    optParser.add_option("--nogui", action="store_true",
                         default=False, help="run the commandline version of sumo")
    options, args = optParser.parse_args()
    return options


# this is the main entry point of this script
if __name__ == "__main__":

    agent = Agent()

    options = get_options()

    # this script has been called from the command line. It will start sumo as a
    # server, then connect and run
    if options.nogui:
        sumoBinary = checkBinary('sumo')
    else:
        sumoBinary = checkBinary('sumo-gui')

    with open('results.txt','w+') as f:
    # first, generate the route file for this simulation
        for _ in range(epochs):
            generate_routefile()
            agent.epoch += 1
            # this is the normal way of using traci. sumo is started as a
            # subprocess and then the python script connects and runs
            traci.start([sumoBinary, "-c", "data/cross.sumocfg",
                                     "--tripinfo-output", "tripinfo.xml"])
            run()
            agent.save()