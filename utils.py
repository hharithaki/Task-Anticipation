import ast
import csv
from itertools import permutations
import os
import random
import re
import subprocess
import time
import numpy as np
import pandas as pd
from simulation.unity_simulator import utils_viz, comm_unity
from sklweka.classifiers import Classifier
from sklweka.dataset import to_instance
from openai import OpenAI
import joblib

client = OpenAI()

delimeters = ['(', ')', ',']
human_asp_pre = 'ASP/human_pre.sp'
human_asp = 'ASP/human.sp'
ah_asp_pre = 'ASP/ahagent_pre.sp'
ah_asp_new = 'ASP/ahagent.sp'
display_marker = 'display'
# models = ['human.model', 'ah_agent.model', 'ah_agent.model']
models = ['human.model', 'ahagent1.model', 'ahagent2.model']

food = ['cereal', 'breadslice', 'bananas', 'apple', 'cupcake', 'cutlets', 'chips', 'candybar', 'plum']
drinks = ['milk', 'wine', 'juice', 'coffee']
electricals = ['dishwasher', 'fridge']
appliances = ['computer', 'coffeemaker'] + electricals
plates = ['plate']
glasses = ['waterglass', 'mug']
containers = ['coffeepot'] + plates + glasses
others = ['cellphone', 'book', 'boardgame']
locations = ['kitchentable', 'livingroom_coffeetable', 'bedroom_coffeetable', 'livingroom_desk', 'bedroom_desk', 'counter_one', 'counter_three', 'kitchen_smalltable'] # 'kitchen'
surfaces = locations + appliances
graspable = food + drinks + containers + others
objects = appliances + graspable
coffee_ = False # inertial

def process_state(graph, sub_goal, prev_agent_tasks, prev_agent_actions, flags, id_dict, char_id, num_agents):
    state = []
    # Previous action of the agent
    act = prev_agent_actions[0].split()
    if len(act) == 4:
        state.append('_'.join([act[1][1:-1],act[2][1:-1]]))
    elif len(act) == 6:
        state.append('_'.join([act[1][1:-1],act[2][1:-1],act[4][1:-1]]))
    else:
        state.append(act[0])

    act = prev_agent_actions[1].split()
    if len(act) == 4:
        state.append('_'.join([act[1][1:-1],act[2][1:-1]]))
    elif len(act) == 6:
        state.append('_'.join([act[1][1:-1],act[2][1:-1],act[4][1:-1]]))
    else:
        state.append(act[0])
    
    # Location of the agent
    pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][char_id]
    state.append(pose['position'][0]) # x
    state.append(pose['position'][1]) # y
    state.append(pose['position'][2]) # z
    state.append(pose['rotation'][0]) # x
    state.append(pose['rotation'][1]) # y
    state.append(pose['rotation'][2]) # z
    
    # objects related to the goal
    goal_obj = ','.join(get_goal_obj(sub_goal))
    state.append(goal_obj)

    # get objecs in the ahnd of that agent
    object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == char_id+1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    if len(object_ids) == 2: # obj in both hands
        hand_obj = []
        for item in object_ids: # objects in agents hand
            key_ = list(id_dict.keys())[list(id_dict.values()).index(item)] # performance >_<
            hand_obj.append(key_)
        state.append(','.join(hand_obj))
    elif len(object_ids) == 1:
        hand_obj = []
        hand_obj.append('None')
        for item in object_ids: # objects in human hand
            key_ = list(id_dict.keys())[list(id_dict.values()).index(item)] # performance >_<
            hand_obj.append(key_)
        state.append(','.join(hand_obj))
    else:
        state.append('None,None')

    # objects with other agents
    agent_obj = []
    for agent_id in range(num_agents):
        if agent_id != char_id:
            object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == agent_id+1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
            for item in object_ids: # objects in agents hand
                key_ = list(id_dict.keys())[list(id_dict.values()).index(item)] # performance >_<
                agent_obj.append(key_)
    if agent_obj:
        state.append(','.join(agent_obj))#
    else:
        state.append('None')

    # current task
    state.append(sub_goal)
    # previous task
    state.append(prev_agent_tasks[-1])

    # flags
    if flags['weekday']:
        state.append(1)
        if flags['office']:
            state.append(1)
        else:
            state.append(0)
        state.append(0)
    else:
        state.append(0)
        state.append(0)
        if flags['guests']:
            state.append(1)
        else:
            state.append(0)
    return state

def process_state_new(graph, sub_goal, prev_agent_tasks, prev_agent_actions, flags, id_dict, char_id, num_agents, action):
    state = []
    # Previous action of the agent
    act = prev_agent_actions[0].split()
    if len(act) == 4:
        state.append('_'.join([act[1][1:-1],act[2][1:-1]]))
    elif len(act) == 6:
        state.append('_'.join([act[1][1:-1],act[2][1:-1],act[4][1:-1]]))
    else:
        state.append(act[0])

    act = prev_agent_actions[1].split()
    if len(act) == 4:
        state.append('_'.join([act[1][1:-1],act[2][1:-1]]))
    elif len(act) == 6:
        state.append('_'.join([act[1][1:-1],act[2][1:-1],act[4][1:-1]]))
    else:
        state.append(act[0])
    
    # Location of the agent
    pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][char_id]
    state.append(pose['position'][0]) # x
    state.append(pose['position'][1]) # y
    state.append(pose['position'][2]) # z
    state.append(pose['rotation'][0]) # x
    state.append(pose['rotation'][1]) # y
    state.append(pose['rotation'][2]) # z
    
    # objects related to the goal
    goal_obj = ','.join(get_goal_obj(sub_goal))
    state.append(goal_obj)

    # get objecs in the ahnd of that agent
    object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == char_id+1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    if len(object_ids) == 2: # obj in both hands
        hand_obj = []
        for item in object_ids: # objects in agents hand
            key_ = list(id_dict.keys())[list(id_dict.values()).index(item)] # performance >_<
            hand_obj.append(key_)
        state.append(','.join(hand_obj))
    elif len(object_ids) == 1:
        hand_obj = []
        hand_obj.append('None')
        for item in object_ids: # objects in human hand
            key_ = list(id_dict.keys())[list(id_dict.values()).index(item)] # performance >_<
            hand_obj.append(key_)
        state.append(','.join(hand_obj))
    else:
        state.append('None,None')

    # objects with other agents
    agent_obj = []
    for agent_id in range(num_agents):
        if agent_id != char_id:
            object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == agent_id+1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
            for item in object_ids: # objects in agents hand
                key_ = list(id_dict.keys())[list(id_dict.values()).index(item)] # performance >_<
                agent_obj.append(key_)
    if agent_obj:
        state.append(','.join(agent_obj))#
    else:
        state.append('None')

    # current task
    state.append(sub_goal)
    # previous task
    state.append(prev_agent_tasks[-1])

    # flags
    if flags['weekday']:
        state.append(1)
        if flags['office']:
            state.append(1)
        else:
            state.append(0)
        state.append(0)
    else:
        state.append(0)
        state.append(0)
        if flags['guests']:
            state.append(1)
        else:
            state.append(0)
    action = action.split()
    if len(action) == 4:
        state.append('_'.join([action[1][1:-1],action[2][1:-1]]))
    elif len(action) == 6:
        state.append('_'.join([action[1][1:-1],action[2][1:-1],action[4][1:-1]]))
    return state

def convert_state(graph, ASP_goal_human, id_dict, timestep, human_at_dict, ah_at_dict, add_at_dict):
    human_fluents = []
    ah_fluents = []
    add_fluents = []
    fluents = []

    temp_objects = objects.copy()
    temp_locations = locations.copy()
    
    human_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    ah_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 2 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    add_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 3 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]

    if 1 in human_at_dict.values():
        key_ = list(human_at_dict.keys())[list(human_at_dict.values()).index(1)] # performance >_<
        human_fluents.append('holds(at(human,' + key_ + '),' + timestep + ').')
        ah_fluents.append('holds(agent_at(human,' + key_ + '),' + timestep + ').')

    if 1 in ah_at_dict.values():
        key_ = list(ah_at_dict.keys())[list(ah_at_dict.values()).index(1)] # performance >_<
        ah_fluents.append('holds(at(ahagent1,' + key_ + '),' + timestep + ').')
        human_fluents.append('holds(agent_at(ahagent1,' + key_ + '),' + timestep + ').')

    if 1 in add_at_dict.values():
        key_ = list(add_at_dict.keys())[list(add_at_dict.values()).index(1)] # performance >_<
        add_fluents.append('holds(at(ahagent2,' + key_ + '),' + timestep + ').')
        human_fluents.append('holds(agent_at(ahagent2,' + key_ + '),' + timestep + ').')

    for item in human_object_ids: # objects in human hand
        for key, value in id_dict.items():
            if value == item:
                name_ = key
                human_fluents.append('holds(in_hand(human,' + name_ + '),' + timestep + ').')
                ah_fluents.append('holds(agent_hand(human,' + name_ + '),' + timestep + ').')
                add_fluents.append('holds(agent_hand(human,' + name_ + '),' + timestep + ').')
                temp_objects.remove(name_)

    # if we do not consider that the ad hoc agent will copmplete the task when the objects are with the ad hoc agent-
    # the human will consider and execute unnecessary actions in the first steps so that it can grab whatever in the-
    # hand of the ad hoc agent in the next step 1
    goal_spl = ASP_goal_human.split(',I),')
    goal_spl = [part + ',I)' for part in goal_spl[:-1]] + [goal_spl[-1]]
    for item in ah_object_ids: # objects in ah agent hand
        for key, value in id_dict.items():
            if value == item:
                name_ = key
                ah_fluents.append('holds(in_hand(ahagent1,' + name_ + '),' + timestep + ').')
                human_fluents.append('holds(agent_hand(ahagent1,' + name_ + '),' + timestep + ').')
                add_fluents.append('holds(agent_hand(ahagent1,' + name_ + '),' + timestep + ').')
                temp_objects.remove(name_)
                goal_spl = [item for item in goal_spl if name_ not in item]
    for item in add_object_ids: # objects in add agent hand
        for key, value in id_dict.items():
            if value == item:
                name_ = key
                add_fluents.append('holds(in_hand(ahagent2,' + name_ + '),' + timestep + ').')
                human_fluents.append('holds(agent_hand(ahagent2,' + name_ + '),' + timestep + ').')
                ah_fluents.append('holds(agent_hand(ahagent2,' + name_ + '),' + timestep + ').')
                temp_objects.remove(name_)
                goal_spl = [item for item in goal_spl if name_ not in item]
    ASP_goal_human = ','.join(goal_spl)

    for location in temp_locations:
        location_id = id_dict[location]
        edges_on = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == location_id and edge['relation_type'] == 'ON']
        edges_inside = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == location_id and edge['relation_type'] == 'INSIDE']
        all_edges = edges_on if location == 'counter_three' else edges_on + list(set(edges_inside)-set(edges_on))
        for edge in all_edges:
            for key, value in id_dict.items():
                if value == edge and key != 'microwave':
                    name_ = key
                    if name_ == 'coffeepot' and location == 'counter_three':
                        fluents.append('holds(on(' + name_ + ',coffeemaker),' + timestep + ').')
                    else:
                        fluents.append('holds(on(' + name_ + ','+location+'),' + timestep + ').')
                    temp_objects.remove(name_)

    edges_on = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == id_dict['chair'] and edge['relation_type'] == 'ON']
    for edge in edges_on:
        for key, value in id_dict.items():
            if value == edge:
                name_ = key
                fluents.append('holds(on(' + name_ + ',livingroom_desk),' + timestep + ').')
                temp_objects.remove(name_)

    edges_on = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == id_dict['sofa'] and edge['relation_type'] == 'ON']
    for edge in edges_on:
        for key, value in id_dict.items():
            if value == edge:
                name_ = key
                fluents.append('holds(on(' + name_ + ',livingroom_coffeetable),' + timestep + ').')
                temp_objects.remove(name_)

    edges_on = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == id_dict['microwave'] and edge['relation_type'] == 'ON']
    for edge in edges_on:
        for key, value in id_dict.items():
            if value == edge:
                name_ = key
                fluents.append('holds(on(' + name_ + ',kitchen_smalltable),' + timestep + ').')
                temp_objects.remove(name_)

    edges_inside = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == id_dict['fridge'] and edge['relation_type'] == 'INSIDE']
    for edge in edges_inside:
        for key, value in id_dict.items():
            if value == edge:
                name_ = key
                fluents.append('holds(on(' + name_ + ',fridge),' + timestep + ').')
                temp_objects.remove(name_)

    edges_inside = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == id_dict['dishwasher'] and edge['relation_type'] == 'INSIDE']
    for edge in edges_inside:
        for key, value in id_dict.items():
            if value == edge:
                name_ = key
                fluents.append('holds(on(' + name_ + ',dishwasher),' + timestep + ').')
                temp_objects.remove(name_)
    # default values for objects which have placing issues in virtualhome
    for item in temp_objects:
        if 'bananas' in temp_objects: # on fruit bowl on livingroom_coffeetable
            fluents.append('holds(on(bananas,livingroom_coffeetable),' + timestep + ').')
            temp_objects.remove('bananas')
        if 'plum' in temp_objects: # on fruit bowl on livingroom_coffeetable
            fluents.append('holds(on(plum,livingroom_coffeetable),' + timestep + ').')
            temp_objects.remove('plum')
        if 'breadslice' in temp_objects: # inside toaster
            fluents.append('holds(on(breadslice,counter_three),' + timestep + ').')
            temp_objects.remove('breadslice')
        if 'dishwasher' in temp_objects: # dishwasher == counter_three
            fluents.append('holds(on(dishwasher,counter_three),' + timestep + ').')
            temp_objects.remove('dishwasher')
        if 'fridge' in temp_objects: # on floor
            fluents.append('holds(on(fridge,kitchen),' + timestep + ').')
            temp_objects.remove('fridge')
        if 'cellphone' in temp_objects: # on living room AIR
            fluents.append('holds(on(cellphone,livingroom_coffeetable),' + timestep + ').')
            temp_objects.remove('cellphone')
        if 'cereal' in temp_objects: # on living room floor
            fluents.append('holds(on(cereal,livingroom_desk),' + timestep + ').')
            temp_objects.remove('cereal')
        if 'coffeepot' in temp_objects: # on living room floor
            fluents.append('holds(on(coffeepot,livingroom_desk),' + timestep + ').')
            temp_objects.remove('coffeepot')

    # % --------------- % open/close
    # Status of dishwasher
    dishwasher_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'dishwasher'][0]
    if 'CLOSED' in dishwasher_status:
        fluents.append('-holds(opened(dishwasher),' + timestep + ').')
    elif 'OPEN' in dishwasher_status:
        fluents.append('holds(opened(dishwasher),' + timestep + ').')
    if 'OFF' in dishwasher_status:
        fluents.append('-holds(switchedon(dishwasher),' + timestep + ').')
    elif 'ON' in dishwasher_status:
        fluents.append('holds(switchedon(dishwasher),' + timestep + ').')

    # Status of fridge
    fridge_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'fridge'][0]
    if 'CLOSED' in fridge_status:
        fluents.append('-holds(opened(fridge),' + timestep + ').')
    elif 'OPEN' in fridge_status:
        fluents.append('holds(opened(fridge),' + timestep + ').')
    if 'OFF' in fridge_status:
        fluents.append('-holds(switchedon(fridge),' + timestep + ').')
    elif 'ON' in fridge_status:
        fluents.append('holds(switchedon(fridge),' + timestep + ').')

    # % --------------- % switch on/off
    # Status of computer
    computer_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'computer'][0]
    if 'OFF' in computer_status:
        fluents.append('-holds(switchedon(computer),' + timestep + ').')
    elif 'ON' in computer_status:
        fluents.append('holds(switchedon(computer),' + timestep + ').')

    # Status of coffeemaker
    coffeemaker_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'coffeemaker'][0]
    if 'OFF' in coffeemaker_status:
        fluents.append('-holds(switchedon(coffeemaker),' + timestep + ').')
    elif 'ON' in coffeemaker_status:
        fluents.append('holds(switchedon(coffeemaker),' + timestep + ').')

    global coffee_
    # % --------------- % coffee
    pot_on_maker = [True for edge in graph['edges'] if edge['to_id'] == 'coffeepot' and edge['from_id'] == 'dishwasher' and edge['relation_type'] == 'ON']
    if coffee_ or (all(pot_on_maker) and 'ON' in coffeemaker_status):
        fluents.append('holds(made(coffee),' + timestep + ').')
        coffee_ = True
    else:
        fluents.append('-holds(made(coffee),' + timestep + ').')
    return human_fluents, ah_fluents, add_fluents, fluents, ASP_goal_human

def answer_set_finder(expression, answer):
    if not re.search('[\>\<\=\!]',expression):
        expression = re.sub('I\+1', 'I', expression)
        expression = re.sub('\(', '\(', expression)
        expression = re.sub('\)', '\)', expression)
        expression = re.sub('[A-Z][0-9]?', '[a-z0-9_]+(?:\(.+?\))?', expression)
        literal = re.findall("(?<!-)"+expression, answer)
    else:
        literal = [expression]
    return literal

def process_answerlist(answer,sub_goal_success):
    answer_list = answer_set_finder('occurs(A,I)', answer)
    action_list = []
    for i in range(len(answer_list)):
        for element in answer_list:
            if re.search(rf',{i}\)$',element) != None:
                action_list.insert(i, element)
    action_list_res = [item for item in action_list if 'exo' not in item]
    if not action_list_res: # exo action to complete the task; action_list and 
        sub_goal_success = True
    return action_list_res, sub_goal_success

def get_virtualhome_name(name_):
    item_map = {
        'livingroom_coffeetable':'coffeetable',
        'livingroom_desk':'chair', # floor can be ignored
        'bedroom_coffeetable':'coffeetable',
        'bedroom_desk':'desk',
        'kitchen_smalltable':'tvstand',
        'counter_one':'kitchencounter',
        'counter_three':'dishwasher',
        'kitchen':'fridge'
    }
    if name_ in item_map:
        return item_map[name_]
    else:
        return name_

def generate_script(human_act, ah_act, add_act, id_dict, human_character='<char0>', ah_character='<char1>', add_character='<char2>'):
    script = []
    # human actions
    human_action_split = re.findall(r'\w+',human_act)
    if human_action_split[1] in ['put']:
        if human_action_split[3] in ['milk'] and human_action_split[4] == 'kitchen_smalltable':
            human_script_instruction = human_character + ' [putback] <' + get_virtualhome_name(human_action_split[3]) + '> (' + str(id_dict[human_action_split[3]]) + ') <microwave> (' + str(id_dict['microwave']) + ')'
        elif human_action_split[3] in ['cereal','coffeepot'] and human_action_split[4] == 'livingroom_desk':
            human_script_instruction = human_character + ' [putback] <' + get_virtualhome_name(human_action_split[3]) + '> (' + str(id_dict[human_action_split[3]]) + ') <floor> (' + str(id_dict['floor']) + ')'
        elif human_action_split[3] in ['book'] and human_action_split[4] == 'livingroom_coffeetable':
            human_script_instruction = human_character + ' [putback] <' + get_virtualhome_name(human_action_split[3]) + '> (' + str(id_dict[human_action_split[3]]) + ') <sofa> (' + str(id_dict['sofa']) + ')'
        else:
            human_script_instruction = human_character + ' [putback] <' + get_virtualhome_name(human_action_split[3]) + '> (' + str(id_dict[human_action_split[3]]) + ') <' + get_virtualhome_name(human_action_split[4]) + '> (' + str(id_dict['chair' if human_action_split[4] == 'livingroom_desk' else human_action_split[4]]) + ')'
    elif human_action_split[1] in ['move']:
        human_script_instruction = human_character + ' [find] <' + get_virtualhome_name(human_action_split[3]) + '> (' + str(id_dict[human_action_split[3] if not human_action_split[3] == 'kitchen' else 'fridge']) + ')'
    else:
        human_script_instruction = human_character + ' [' + human_action_split[1].replace('_','') + '] <' + get_virtualhome_name(human_action_split[3]) + '> (' + str(id_dict[human_action_split[3]]) + ')'
    
    # ah agent 1
    if ah_act:
        ah_action_split = re.findall(r'\w+',ah_act)
        if ah_action_split[1] in ['put']:
            if ah_action_split[3] in ['milk'] and ah_action_split[4] == 'kitchen_smalltable':
                ah_script_instruction = ah_character + ' [putback] <' + get_virtualhome_name(ah_action_split[3]) + '> (' + str(id_dict[ah_action_split[3]]) + ') <microwave> (' + str(id_dict['microwave']) + ')'
            elif ah_action_split[3] in ['cereal','coffeepot'] and ah_action_split[4] == 'livingroom_desk':
                ah_script_instruction = ah_character + ' [putback] <' + get_virtualhome_name(ah_action_split[3]) + '> (' + str(id_dict[ah_action_split[3]]) + ') <floor> (' + str(id_dict['floor']) + ')'
            elif ah_action_split[3] in ['book'] and ah_action_split[4] == 'livingroom_coffeetable':
                ah_script_instruction = ah_character + ' [putback] <' + get_virtualhome_name(ah_action_split[3]) + '> (' + str(id_dict[ah_action_split[3]]) + ') <sofa> (' + str(id_dict['sofa']) + ')'
            else:
                ah_script_instruction = ah_character + ' [putback] <' + get_virtualhome_name(ah_action_split[3]) + '> (' + str(id_dict[ah_action_split[3]]) + ') <' + get_virtualhome_name(ah_action_split[4]) + '> (' + str(id_dict['chair' if ah_action_split[4] == 'livingroom_desk' else ah_action_split[4]]) + ')'
        elif ah_action_split[1] in ['move'] and ah_action_split[3] == 'kitchen':
            ah_script_instruction = ah_character + ' [find] <' + get_virtualhome_name(ah_action_split[3]) + '> (' + str(id_dict[get_virtualhome_name(ah_action_split[3])]) + ')'
        elif ah_action_split[1] in ['move']:
            ah_script_instruction = ah_character + ' [find] <' + get_virtualhome_name(ah_action_split[3]) + '> (' + str(id_dict[ah_action_split[3]]) + ')'
        else:
            ah_script_instruction = ah_character + ' [' + ah_action_split[1].replace('_','') + '] <' + get_virtualhome_name(ah_action_split[3]) + '> (' + str(id_dict[ah_action_split[3]]) + ')'
    else:
        ah_script_instruction = None

    # add agent
    if add_act:
        add_action_split = re.findall(r'\w+',add_act)
        if add_action_split[1] in ['put']:
            if add_action_split[3] in ['milk'] and add_action_split[4] == 'kitchen_smalltable':
                add_script_instruction = add_character + ' [putback] <' + get_virtualhome_name(add_action_split[3]) + '> (' + str(id_dict[add_action_split[3]]) + ') <microwave> (' + str(id_dict['microwave']) + ')'
            elif add_action_split[3] in ['cereal','coffeepot'] and add_action_split[4] == 'livingroom_desk':
                add_script_instruction = add_character + ' [putback] <' + get_virtualhome_name(add_action_split[3]) + '> (' + str(id_dict[add_action_split[3]]) + ') <floor> (' + str(id_dict['floor']) + ')'
            elif add_action_split[3] in ['book'] and add_action_split[4] == 'livingroom_coffeetable':
                add_script_instruction = add_character + ' [putback] <' + get_virtualhome_name(add_action_split[3]) + '> (' + str(id_dict[add_action_split[3]]) + ') <sofa> (' + str(id_dict['sofa']) + ')'
            else:
                add_script_instruction = add_character + ' [putback] <' + get_virtualhome_name(add_action_split[3]) + '> (' + str(id_dict[add_action_split[3]]) + ') <' + get_virtualhome_name(add_action_split[4]) + '> (' + str(id_dict['chair' if add_action_split[4] == 'livingroom_desk' else add_action_split[4]]) + ')'
        elif add_action_split[1] in ['move'] and add_action_split[3] == 'kitchen':
            add_script_instruction = add_character + ' [find] <' + get_virtualhome_name(add_action_split[3]) + '> (' + str(id_dict[get_virtualhome_name(add_action_split[3])]) + ')'
        elif add_action_split[1] in ['move']:
            add_script_instruction = add_character + ' [find] <' + get_virtualhome_name(add_action_split[3]) + '> (' + str(id_dict[add_action_split[3]]) + ')'
        else:
            add_script_instruction = add_character + ' [' + add_action_split[1].replace('_','') + '] <' + get_virtualhome_name(add_action_split[3]) + '> (' + str(id_dict[add_action_split[3]]) + ')'
    else:
        add_script_instruction = None

    script_instruction = (human_script_instruction + '|' + ah_script_instruction) if human_script_instruction and ah_script_instruction else (human_script_instruction if human_script_instruction else ah_script_instruction)
    if add_script_instruction:
        script_instruction = script_instruction + '|' + add_script_instruction
    script.append(script_instruction)
    return script

# remove objects
def remove_obj_from_environment(obj, comm, graph):
    ids = [node['id'] for node in graph['nodes']]
    obj_ids = [node['id'] for node in graph['nodes'] if node['class_name'] == obj]
    for obj_id in obj_ids:
        if obj_id in ids:
            edges_to_remove = [edge for edge in graph['edges'] if edge['to_id'] == obj_id or edge['from_id'] == obj_id]
            for edge in edges_to_remove:
                graph['edges'].remove(edge)

            nodes_to_remove = [node for node in graph['nodes'] if node['id'] == obj_id]
            for node in nodes_to_remove:
                graph['nodes'].remove(node)
            success, message = comm.expand_scene(graph)

def clean_graph(comm, graph, objects):
    for obj in objects:
        remove_obj_from_environment(obj, comm, graph)
    utils_viz.clean_graph(graph)
    success1, message = comm.expand_scene(graph)
    success2, graph = comm.environment_graph()
    return success1, message, success2, graph

def preprocess_new_data(new_data, transformer, all_columns, or_columns, nr_columns):
    new_data_df = pd.DataFrame(new_data, columns=all_columns[:-1])
    for col in nr_columns:
        new_data_df[col] = pd.to_numeric(new_data_df[col], errors='coerce')

    new_data_df = new_data_df.replace([np.inf, -np.inf], np.nan).fillna(0)

    transformed_or_columns = transformer.transform(new_data_df[or_columns])
    new_data_df[or_columns] = transformed_or_columns

    return new_data_df

# weka tree
def predict_next_action(graph, current_task, prev_modelagent_tasks, prev_modelagent_actions, flags, id_dict, modelagent_id, num_agents):
    values = process_state(graph, current_task, prev_modelagent_tasks, prev_modelagent_actions, flags, id_dict, modelagent_id, num_agents)
    model, header = Classifier.deserialize(models[modelagent_id]) # 0 - human, 1 - ah_agent1, 2 - ah_agent2
    # create new instance
    inst = to_instance(header,values) # Instance.create_instance(values)
    inst.dataset = header
    # make prediction
    try:
        index = model.classify_instance(inst)
        action = header.class_attribute.value(int(index))
    except:
        print('Null pointer exception')
        action = None
    return action

# other trees
def predict_next_action_new(graph, current_task, prev_modelagent_tasks, prev_modelagent_actions, flags, id_dict, modelagent_id, num_agents):
    values = process_state(graph, current_task, prev_modelagent_tasks, prev_modelagent_actions, flags, id_dict, modelagent_id, num_agents)

    transformer = joblib.load('tree/transformer' + str(modelagent_id) + '.pkl')
    classifier = joblib.load('tree/decision_tree' + str(modelagent_id) + '.pkl')
    metadata = joblib.load('tree/columns_metadata' + str(modelagent_id) + '.pkl')

    all_columns = metadata['all_columns']
    or_columns = metadata['or_columns']
    nr_columns = metadata['nr_columns']

    # pre process data
    processed_values = preprocess_new_data([values], transformer, all_columns, or_columns, nr_columns)
    # make prediction
    action = classifier.predict(processed_values)
    return action[0]

def get_object_locations(fluents):
    # holds(on(cereal,kitchentable),0)
    object_location_dct = {}
    for fluent in fluents:
        if 'holds(on(' in fluent:
            items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)]
            fir_items = [item.split(',')[0] if ',' in item else item for item in items]
            sec_items = [item.split(',')[1] if ',' in item else item for item in items]
            set_objects = ['cereal', 'breadslice', 'bananas', 'plum', 'apple', 'cupcake', 'cutlets', 'chips', 'candybar', 'milk', 'wine', 'juice', 
                           'plate', 'waterglass', 'mug', 'coffeepot', 'cellphone', 'book', 'boardgame']
            set_locations = ['kitchentable', 'livingroom_coffeetable', 'bedroom_coffeetable', 'livingroom_desk', 'bedroom_desk', 'counter_one', 'counter_three', 'kitchen_smalltable']
            objs = [item for item in fir_items if item in set_objects]
            locs = [item for item in sec_items if item in set_locations]
            if objs and locs:
                object_location_dct[objs[0]] = locs[0]
    return object_location_dct

# block in virtualhome: cannot keep a single env since it is impossible to revert the predicted action execution effects and revert back to original script
def get_future_actions(graph, current_task, all_prev_tasks, flags, ah_fluents, common_fluents, ASP_goal, all_prev_actions, env_id, id_dict, current_script, num_agents, current_agent_id):
    future_actions = [[None,None],[None,None]]
    valid_actions = [[None,None],[None,None]]
    graphs = []
    missing_location = False
    actual_predict_time = 0
    future_action_x = [False, False]
    for i in range(2): # for 2 steps to the future
        actual_predict_time_start = time.time()
        predicted_future_actions = []
        for agent_id in range(num_agents):
            if agent_id != current_agent_id:
                step_exception = [agent_id, False]
                try:
                    predicted_future_actions.append(predict_next_action(graph, current_task, all_prev_tasks[agent_id], all_prev_actions[agent_id], flags, id_dict, agent_id, num_agents))
                except Exception as e:
                    step_exception = [agent_id, True]
        actual_predict_time_end = time.time()
        actual_predict_time += (actual_predict_time_end-actual_predict_time_start)
        graphs.append(graph)
        # check the validity of the predicted actions
        for idx, pred_action in enumerate(predicted_future_actions):
            agent_id = idx if idx == 0 else (2 if current_agent_id == 1 else 1) # num_agents = 3
            # if the other agent is predicted to move to a specific place not move there?
            if not pred_action:
                valid_actions[i][idx] = False # valid actions = [[True,True], [True,True]]
                future_actions[i][idx] = None # future actions = [[act1,act2], [act3,act4]]
                continue

            # assume ad hoc agent does nothing; only other agent actions added to script
            duplicate_aspname_dict = {
                'coffeetable': ['livingroom_coffeetable', 'bedroom_coffeetable'],
                'desk': ['livingroom_desk', 'bedroom_desk']
            }
            future_action = pred_action.split('_')
            if len(future_action) == 2:
                if future_action[0] == 'find' and future_action[1] == 'coffeetable':
                    if current_task in ['Breakfast_weekday','Pack_bag','Lunch','Coffee']:
                        loc = str(id_dict['livingroom_coffeetable'])
                        valid_actions[i][idx] = True
                        future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + loc + ')'
                    elif current_task in ['Breakfast_weekend']:
                        loc = str(id_dict['bedroom_coffeetable'])
                        valid_actions[i][idx] = True
                        future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + loc + ')'
                    else:
                        missing_location = True
                        valid_actions[i][idx] = False
                        future_actions[i][idx] = None
                elif future_action[0] == 'find' and future_action[1] == 'desk':
                    if current_task in ['Breakfast_weekday','Lunch','Coffee','Breakfast_weekend']:
                        loc = str(id_dict['livingroom_desk'])
                        valid_actions[i][idx] = True
                        future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + loc + ')'
                    elif current_task in ['Serve_snack']:
                        loc = str(id_dict['bedroom_desk'])
                        valid_actions[i][idx] = True
                        future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + loc + ')'
                    else:
                        missing_location = True
                        valid_actions[i][idx] = False
                        future_actions[i][idx] = None
                else:
                    if future_action[0] == 'grab':
                        obj = future_action[1]
                        running_agent_name = 'ahagent'+str(current_agent_id)
                        # in hand of the current agent who is modelling the other agents
                        for j in range(num_agents):
                            if j == current_agent_id: # curent agent who is modelling the other agents
                                running_agent_positive_inhand = [item for item in ah_fluents if item.startswith('holds(in_hand('+running_agent_name+',')]
                                if ('holds(in_hand('+running_agent_name+','+ obj + '),0).' in running_agent_positive_inhand):
                                    valid_actions[i][idx] = False
                                    future_actions[i][idx] = None
                                    break
                            elif j == 0: # if huma
                                human_positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand(human,')]
                                if ('holds(agent_hand(human,'+ obj + '),0).' in human_positive_inhand): # already in humans hand
                                    valid_actions[i][idx] = False
                                    future_actions[i][idx] = None
                                    break
                            else: # already in the hand of model agent or another ad hoc agent
                                positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand(ahagent'+str(j)+',')]
                                if ('holds(agent_hand(ahagent'+str(j)+','+ obj + '),0).' in positive_inhand):
                                    valid_actions[i][idx] = False
                                    future_actions[i][idx] = None
                                    break
                        if valid_actions[i][idx] == False: # not val will set off for None as well
                            continue
                        object_location_dict = get_object_locations(ah_fluents+common_fluents)
                        if obj in object_location_dict:
                            obj_location = object_location_dict[obj]
                            items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',ASP_goal)]
                            for item in items:
                                if ',' in item:
                                    fir_item = item.split(',')[0]
                                    sec_item = item.split(',')[1]
                                    if fir_item == obj and sec_item == obj_location: # goal achived; do not change the location of that object
                                        valid_actions[i][idx] = False
                                        future_actions[i][idx] = None
                                        break
                            if valid_actions[i][idx] == False: # not val will set off for None as well
                                continue
                    if future_action[1] == 'tvstand':
                        obj = str(id_dict['kitchen_smalltable'])
                    elif future_action[1] == 'kitchencounter':
                        obj = str(id_dict['counter_one'])
                    else:
                        obj = str(id_dict[future_action[1]])
                    if missing_location:
                        future_action_x[idx] = '<char'+str(agent_id)+'> [find] <' + future_action[1] + '> (' + obj + ')'
                    valid_actions[i][idx] = True
                    future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + obj + ')'
            else: # putback or putin
                if future_action[1] in id_dict and future_action[2] in id_dict:
                    # check if the model agent is holding the particular object
                    if agent_id == 0:
                        positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand(human,')]
                        if not('holds(agent_hand(human,'+ future_action[1] + '),0).' in positive_inhand) and not('holds(agent_hand(human,'+ future_action[1] + '),1).' in positive_inhand):
                            valid_actions[i][idx] = False
                            future_actions[i][idx] = None
                            continue
                    else:
                        positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand(ahagent'+str(agent_id)+',')]
                        if not('holds(agent_hand(ahagent'+str(agent_id)+','+ future_action[1] + '),0).' in positive_inhand) and not('holds(agent_hand(ahagent'+str(agent_id)+','+ future_action[1] + '),1).' in positive_inhand):
                            valid_actions[i][idx] = False
                            future_actions[i][idx] = None
                            continue
                    valid_actions[i][idx] = True
                    future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + str(id_dict[future_action[1]]) + ') <' + future_action[2] + '> (' + str(id_dict[future_action[2]]) + ')'
                elif future_action[2] in id_dict:
                    # loc in, obj not in id_dict
                    objs = duplicate_aspname_dict[future_action[1]]
                    keys = []
                    modelagent_object_ids = [edge['to_id'] for edge in graphs[0]['edges'] if edge['from_id'] == agent_id+1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
                    if modelagent_object_ids:
                        for modelagent_obj in modelagent_object_ids:
                            key_ = list(id_dict.keys())[list(id_dict.values()).index(modelagent_obj)] # performance >_<
                            if key_ in objs:
                                keys.append(key_)
                        if keys:
                            obj = keys[0]
                            modelagent_positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand('+ ('human' if agent_id == 0 else 'ahagent'+str(agent_id)) + ',')]
                            if not('holds(agent_hand(' + ('human' if agent_id == 0 else 'ahagent'+str(agent_id)) + ',' + obj + '),0).' in modelagent_positive_inhand) and not('holds(agent_hand(' + ('human' if agent_id == 0 else 'ahagent'+str(agent_id)) + ',' + obj + '),1).' in modelagent_positive_inhand):
                                valid_actions[i][idx] = False
                                future_actions[i][idx] = None
                                continue
                            valid_actions[i][idx] = True
                            future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + str(id_dict[obj]) + ') <' + future_action[2] + '> (' + str(id_dict[future_action[2]]) + ')'
                        else:
                            valid_actions[i][idx] = False
                            future_actions[i][idx] = None
                    else:
                        valid_actions[i][idx] = False
                        future_actions[i][idx] = None
                elif future_action[1] in id_dict:#
                    # obj in, loc not in id_dict
                    modelagent_positive_inhand = [item for item in ah_fluents if item.startswith('holds(agent_hand('+('human' if agent_id == 0 else 'ahagent'+str(agent_id))+',')]
                    if not('holds(agent_hand('+('human' if agent_id == 0 else 'ahagent'+str(agent_id))+','+ future_action[1] + '),0).' in modelagent_positive_inhand) and not('holds(agent_hand('+('human' if agent_id == 0 else 'ahagent'+str(agent_id))+','+ future_action[1] + '),1).' in modelagent_positive_inhand):
                        valid_actions[i][idx] = False
                        future_actions[i][idx] = None
                        continue
                    locs = duplicate_aspname_dict[future_action[2]]
                    loc_proximity = []
                    human_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][agent_id]
                    for loc in locs:
                        id_ = id_dict[loc]
                        loc_pose = [node['obj_transform'] for node in graph['nodes'] if node['id'] == id_][0]
                        prox_human = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(loc_pose['position']))
                        loc_proximity.append(prox_human)
                    proxmin_idx = loc_proximity.index(min(loc_proximity)) # take the less, if equal the first one
                    loc = locs[proxmin_idx]
                    valid_actions[i][idx] = True
                    future_actions[i][idx] = '<char'+str(agent_id)+'> [' + future_action[0] + '] <' + future_action[1] + '> (' + str(id_dict[future_action[1]]) + ') <' + future_action[2] + '> (' + str(id_dict[loc]) + ')'
        
        if future_actions:
            if all(valid_actions[i]) and not step_exception[0] and not step_exception[1]:
                if missing_location and future_action_x[0] and future_action_x[1]:
                    current_script.append(future_action_x[0]+ '|' + future_action_x[1])
                    current_script.append(future_actions[i][0] + '|' + future_actions[i][1])
                elif missing_location and future_action_x[0]:
                    current_script.append(future_action_x[0]+ '|' + future_actions[i][1])
                    current_script.append(future_actions[i][0])
                elif missing_location and future_action_x[1]:
                    current_script.append(future_actions[i][0]+ '|' + future_action_x[1])
                    current_script.append(future_actions[i][1])
                else:
                    current_script.append(future_actions[i][0] + '|' + future_actions[i][1])
            elif valid_actions[0] and not step_exception[0]: # human has an action
                if missing_location and future_action_x[0]:
                    current_script.append(future_action_x[0])
                current_script.append(future_actions[i][0])
            elif valid_actions[1] and not step_exception[1]: # ad hoc agent other than the current agent
                if missing_location and future_action_x[1]:
                    current_script.append(future_action_x[1])
                current_script.append(future_actions[i][1])
            else:
                continue
            
            # initiate a second env
            comm_dummy = comm_unity.UnityCommunication(port='8082')
            comm_dummy.reset(env_id)
            success_dummy, graph_dummy = comm_dummy.environment_graph()
            success1_dummy, message_dummy, success2_dummy, graph_dummy = clean_graph(comm_dummy, graph_dummy, ['chicken'])

            # Add human
            comm_dummy.add_character('Chars/Female1', initial_room='kitchen')
            # Add ad hoc agent
            comm_dummy.add_character('Chars/Male1', initial_room='kitchen')
            # Add ad hoc agent
            comm_dummy.add_character('Chars/Female2', initial_room='kitchen')

            # print('////////////////////////////////////////////////////////////////')
            # print(all_prev_actions)
            # print(future_actions)
            # print(current_script, type(current_script))
            for script_instruction in current_script:
                act_success, success_msgs, message = comm_dummy.render_script([script_instruction], recording=False, skip_animation=True)
                modelagent_id = (2 if current_agent_id == 1 else 1)
                if success_msgs[0] and success_msgs[modelagent_id]:
                    all_prev_actions[0].pop(0)
                    all_prev_actions[0].append(future_actions[i][0])
                    all_prev_actions[modelagent_id].pop(0)
                    all_prev_actions[modelagent_id].append(future_actions[i][1])
                elif success_msgs[0]:
                    all_prev_actions[0].pop(0)
                    all_prev_actions[0].append(future_actions[i][0])
                    script_split = script_instruction.split('|')
                    del current_script[-1]
                    current_script.append(script_split[0])
                elif success_msgs[modelagent_id]:
                    all_prev_actions[modelagent_id].pop(0)
                    all_prev_actions[modelagent_id].append(future_actions[i][1])
                    del current_script[-1]
                    script_split = script_instruction.split('|')
                    if '|' in script_instruction:
                        current_script.append(script_split[1])
                    else:
                        current_script.append(script_split[0])
                else:
                    del current_script[-1]
            # print('////////////////////////////////////////////////////////////////')
            # Get the state observation
            success, graph = comm_dummy.environment_graph()
    return future_actions, graphs, actual_predict_time

# convert action predictions for the other agents to fluent literals of ASP
def get_fluents(ASP_goal, other_actions, graphs, fluents, id_dict, num_agents, current_agent_id):
    future_actions = []
    exo_dict = {
        'grab': 'exo_grab',
        'putback': 'exo_put',
        'putin': 'exo_put',
        'open': 'exo_open',
        'close': 'exo_close',
        'switchon': 'exo_switchon',
        'switchoff': 'exo_switchoff'
    }
    aspname_dict = {
        'cereal': 'cereal',
        'breadslice': 'breadslice',
        'bananas':'bananas',
        'plum':'plum',
        'cupcake': 'cupcake',
        'cutlets': 'cutlets',
        'chips': 'chips',
        'plate': 'plate',
        'candybar': 'candybar',
        'milk': 'milk',
        'wine': 'wine',
        'juice': 'juice',
        'computer': 'computer',
        'coffeemaker': 'coffeemaker',
        'waterglass': 'waterglass',
        'mug': 'mug',
        'coffeepot': 'coffeepot',
        'kitchentable': 'kitchentable',
        'book': 'book',
        'boardgame': 'boardgame',
        'stove': 'stove',
        'cellphone': 'cellphone',
        'apple': 'apple',
        'bookshelf': 'livingroom_desk',
        'microwave': 'kitchen_smalltable',
        'chair':'livingroom_desk',
        'desk':'bedroom_desk',
        'tvstand':'kitchen_smalltable',
        'kitchencounter':'counter_one',
        'dishwasher':'counter_three',
        'floor':'livingroom_desk',
        'sofa':'livingroom_coffeetable',
        'fridge':'fridge' # 'kitchen' find/move low probability
    }
    duplicate_aspname_dict = {
        'coffeetable': ['livingroom_coffeetable', 'bedroom_coffeetable']
    }

    # ASP_goal, other_actions, graphs, fluents, id_dict
    # modify the goal to consider items in the human and other agents hand
    # print(ASP_goal)
    goal_spl = ASP_goal.split(',I),')
    goal_spl = [part + ',I)' for part in goal_spl[:-1]] + [goal_spl[-1]]
    for agent_id in range(num_agents):
        if agent_id != current_agent_id:
            object_ids = [edge['to_id'] for edge in graphs[0]['edges'] if edge['from_id'] == agent_id+1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    if object_ids:
        for human_obj in object_ids:
            name_ = list(id_dict.keys())[list(id_dict.values()).index(human_obj)] # performance >_<
            goal_spl = [item for item in goal_spl if name_ not in item]
        ASP_goal = ','.join(goal_spl)
    # print(ASP_goal)
    if not other_actions:
        return ASP_goal, fluents

    # [['<char0> [find] <coffeetable> (246)', '<char2> [find] <chair> (244)'], 
    #  ['<char0> [grab] <bananas> (251)', '<char2> [grab] <cereal> (195)']]

    for idx, action_list in enumerate(other_actions):
        for agent in range(num_agents-1):
            agent_id = agent if agent == 0 else (2 if current_agent_id == 1 else 1)
            other_act = action_list[agent]
            if other_act:
                # convert to exo
                action_spl = other_act.split(' ')
                act_type = action_spl[1][1:-1]
                act_obj = action_spl[2][1:-1]
                if len(action_spl) == 4 and act_type != 'find': # find has low probability
                    if act_obj in aspname_dict:
                        obj = aspname_dict[act_obj]
                    else:
                        objs = duplicate_aspname_dict[act_obj]
                        obj_proximity = []
                        other_pose = [node['obj_transform'] for node in graphs[idx]['nodes'] if node['class_name'] == 'character'][agent_id]
                        for obj in objs:
                            id_ = id_dict[obj]
                            obj_pose = [node['obj_transform'] for node in graphs[idx]['nodes'] if node['id'] == id_][0]
                            prox_human = np.linalg.norm(np.asarray(other_pose['position'])-np.asarray(obj_pose['position']))
                            obj_proximity.append(prox_human)
                        proxmin_idx = obj_proximity.index(min(obj_proximity)) # take the less, if equal the first one
                        obj = objs[proxmin_idx]           
                    
                    future_action = 'occurs(' + exo_dict[act_type] + '('+ ('human' if agent_id == 0 else 'ahagent'+str(agent_id)) +',' + obj + '),' + str(idx) + ').'
                    future_actions.append(future_action)
                    if act_type == 'grab' and obj in ASP_goal: # ignore move/open/close/switchon/switchoff
                        goal_spl = ASP_goal.split(',I),')
                        goal_spl = [part + ',I)' for part in goal_spl[:-1]] + [goal_spl[-1]]
                        goal_spl = [item for item in goal_spl if obj not in item]
                        ASP_goal = ','.join(goal_spl)
                elif len(action_spl) > 4: # put
                    act_type = action_spl[1][1:-1]
                    act_obj = action_spl[2][1:-1]
                    act_loc = action_spl[4][1:-1]
                    if act_obj in aspname_dict:
                        obj = aspname_dict[act_obj]
                    else:
                        objs = duplicate_aspname_dict[act_obj]
                        # one or two
                        keys = []
                        if object_ids:
                            for human_obj in object_ids:
                                key_ = list(id_dict.keys())[list(id_dict.values()).index(human_obj)] # performance >_<
                                if key_ in objs:
                                    keys.append(key_)
                            obj = keys[0] if keys else None
                        else:
                            obj = None
                    if act_loc in aspname_dict:
                        loc = aspname_dict[act_loc]
                    else:
                        locs = duplicate_aspname_dict[act_loc]
                        loc_proximity = []
                        human_pose = [node['obj_transform'] for node in graphs[idx]['nodes'] if node['class_name'] == 'character'][agent_id]
                        for loc in locs:
                            id_ = id_dict[loc]
                            loc_pose = [node['obj_transform'] for node in graphs[idx]['nodes'] if node['id'] == id_][0]
                            prox_human = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(loc_pose['position']))
                            loc_proximity.append(prox_human)
                        proxmin_idx = loc_proximity.index(min(loc_proximity)) # take the less, if equal the first one
                        loc = locs[proxmin_idx]  
                    if obj:
                        future_action = 'occurs(' + exo_dict[act_type] + '(' +('human' if agent_id == 0 else 'ahagent'+str(agent_id))+ ',' + obj + ',' + loc + '),' + str(idx) + ').'
                        future_actions.append(future_action)
    # print(future_actions)
    return ASP_goal, fluents+future_actions

# map LLM text to goal text
def map_LLM_text_ASP_goal(LLM_goal):
    llm_goal_dict = {
        'Prepare breakfast weekday':'Breakfast_weekday',
        'Prepare coffee':'Coffee',
        'Prepare home work-station':'Workstation',
        'Prepare lunch':'Lunch',
        'Pack bag':'Pack_bag',
        'Clean kitchen':'Clean_kitchen',
        'Prepare breakfast weekend':'Breakfast_weekend',
        'Prepare table for guests':'Make_table',
        'Serve snacks':'Serve_snacks',
        'Prepare activities':'Activities',
        'Clean dishes':'Clean_dishes'
    }
    return llm_goal_dict[LLM_goal]

# map ASP goal to llm text
def map_ASP_goal_LLM_text(LLM_goal):
    llm_goal_dict = {
        'Breakfast_weekday':'Prepare breakfast weekday',
        'Coffee':'Prepare coffee',
        'Workstation':'Prepare home work-station',
        'Lunch':'Prepare lunch',
        'Pack_bag':'Pack bag',
        'Clean_kitchen':'Clean kitchen',
        'Breakfast_weekend':'Prepare breakfast weekend',
        'Make_table':'Prepare table for guests',
        'Serve_snacks':'Serve snacks',
        'Activities':'Prepare activities',
        'Clean_dishes':'Clean dishes'
    }
    return llm_goal_dict[LLM_goal]

# map goal to ASP goal
def map_goal_ASP(goal):
    goal_dict = {
        'Breakfast_weekday': 'holds(on(cereal,kitchentable),I), holds(on(breadslice,kitchentable),I), holds(on(bananas,kitchentable),I)',
        'Coffee': 'holds(made(coffee),I), holds(on(coffeepot,livingroom_desk),I), holds(on(mug,livingroom_desk),I)',
        'Workstation': 'holds(switchedon(computer),I), holds(on(book,livingroom_desk),I), holds(on(cellphone,livingroom_desk),I)',
        'Lunch': 'holds(on(waterglass,kitchentable),I), holds(on(apple,kitchentable),I), holds(on(chips,kitchentable),I)',
        'Pack_bag': 'holds(on(juice,livingroom_desk),I), holds(on(cellphone,livingroom_desk),I)',
        'Clean_kitchen': 'holds(on(cereal,livingroom_desk),I), holds(on(milk,kitchen_smalltable),I), holds(on(cutlets,bedroom_desk),I)',
        'Breakfast_weekend': 'holds(on(cereal,kitchentable),I), holds(on(breadslice,kitchentable),I), holds(on(cupcake,kitchentable),I)',
        'Make_table': 'holds(on(plate,kitchentable),I), holds(on(wine,kitchentable),I)',
        'Serve_snacks': 'holds(on(plum,kitchentable),I), holds(on(candybar,kitchentable),I)',
        'Activities': 'holds(on(book,livingroom_coffeetable),I), holds(on(boardgame,livingroom_coffeetable),I)',
        'Clean_dishes': 'holds(on(plate,dishwasher),I), holds(on(mug,dishwasher),I), holds(switchedon(dishwasher),I)'
    }
    return goal_dict[goal]

# get goal objects
def get_goal_obj(goal):
    goal_dict = {
        'Breakfast_weekday': ['cereal', 'breadslice', 'bananas'],
        'Coffee': ['coffeepot', 'mug'],
        'Workstation': ['computer', 'book', 'cellphone'],
        'Lunch': ['waterglass', 'apple', 'chips'],
        'Pack_bag': ['juice', 'cellphone'],
        'Clean_kitchen': ['cereal', 'milk', 'cutlets'],
        'Breakfast_weekend': ['cereal', 'breadslice, cupcake'],
        'Make_table': ['plate, wine'],
        'Serve_snacks': ['plum', 'candybar'],
        'Activities': ['book', 'boardgame'],
        'Clean_dishes': ['plate', 'mug', 'dishwasher']
    }
    if isinstance(goal,list):
        objs = []
        for task in ah_current_task:
            goal = utils.map_LLM_text_ASP_goal(task)
            objs = objs + goal_dict[goal]
        return objs
    return goal_dict[goal]

def get_const_timeout(goal):
    const_time_dict = {
        'Breakfast_weekday': [16,5],
        'Coffee': [13,5],
        'Workstation': [11,5],
        'Lunch': [16,5],
        'Pack_bag': [12,5],
        'Clean_kitchen': [16,55],
        'Breakfast_weekend': [18,20],
        'Make_table': [14,30],
        'Serve_snacks': [16,5], 
        'Activities': [13,5],
        'Clean_dishes': [18,85]
    }
    return const_time_dict[goal]

def get_location(location_id, id_dict):
    key_ = list(id_dict.keys())[list(id_dict.values()).index(int(location_id))]
    item_map = {
        'dishwasher':'counter_three'
    }
    if key_ in item_map:
        return item_map[key_]
    else:
        return key_

def remove_excess_fluents(ASP_goal):
    goal_spl = ASP_goal.split(',I),')
    goal_spl = [part + ',I)' for part in goal_spl[:-1]] + [goal_spl[-1]]
    if len(goal_spl) > 3:
        goal_spl = goal_spl[:4]
        ASP_goal = ','.join(goal_spl)
    return ASP_goal

def sample_rows(unique_rows):
    print(unique_rows)
    while True:
        rows = random.sample(unique_rows, 2)
        # if eval(rows[0][0])[0] != eval(rows[1][0])[0]:
        return rows

def is_subsequence(sub, main):
    it = iter(main)
    return all(item in it for item in sub)

def find_matching_list(flags, A, nested_list):
    best_match = None
    matching_section = []
    max_match_length = 0
    with_flag_match = None
    # print(flags)
    for lst in nested_list:
        flags_str = str(flags)
        if lst[1] == flags_str: # if there is a matching list then it should be returned first
            lst = eval(lst[0])
            if is_subsequence(A, lst):
                return [item for item in lst if item in A]
            else:
                match_part = [item for item in lst if item in A]
                match_length = len(match_part)
                if match_length > max_match_length:
                    max_match_length = match_length
                    best_match = lst
                    matching_section = match_part
                with_flag_match = match_part.copy()
        else:
            lst = eval(lst[0])
            if is_subsequence(A, lst):
                return [item for item in lst if item in A]
            else:
                match_part = [item for item in lst if item in A]
                match_length = len(match_part)
                if match_length > max_match_length:
                    max_match_length = match_length
                    best_match = lst
                    matching_section = match_part
    if with_flag_match:
        return with_flag_match
    else:
        return matching_section

def get_ordered_tasks(flags, llm_task_list):
    unique_rows = set()
    with open('llm_example_data.csv', mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            unique_rows.add(tuple(row))
    unique_rows = list(unique_rows)
    match_result = find_matching_list(flags, llm_task_list, unique_rows)
    return match_result    

# select and return a goal for the human to execute
def get_human_tasks(weekday):
    task_list = []
    flags = {}
    # to add randomness we can enable the human to select a list of tasks here randomly from the available task list
    if weekday:
        task_list.append('Breakfast_weekday')
        office = np.random.choice([True,False])
        lunch = np.random.choice([True,False], p=[0.8,0.2])
        if office:
            task_list.append('Pack_bag')
            if lunch:
                task_list.append('Lunch')
            task_list.append('Clean_kitchen')
        else:
            task_list.append('Workstation')
            task_list.append('Coffee')
            if lunch:
                task_list.append('Lunch')
            else:
                task_list.append('Serve_snacks')
        flags['weekday'] = weekday
        flags['office'] = office
        flags['lunch'] = lunch
    else:
        task_list.append('Breakfast_weekend')
        guests = np.random.choice([True,False])
        if guests:
            task_list.append('Make_table')
            task_list.append('Lunch')
            task_list.append('Clean_dishes')
        else:
            task_list.append('Activities')
            task_list.append('Serve_snacks')
            task_list.append('Clean_kitchen')
        flags['weekday'] = weekday
        flags['guests'] = guests
    return task_list,flags

def prepare_prompt_msg(flags, completed_tasks):
    if flags['weekday']:
        prompt_msg = 'Complete the following routine on a weekday the human is '
        if flags['office']: # going to office
            prompt_msg = prompt_msg + 'leaving for office'
            if not flags['lunch']: # lunch not needed
                prompt_msg = prompt_msg + ' and take lunch from a resturant'
        else:
            prompt_msg = prompt_msg + 'working from home'
            if not flags['lunch']: # lunch needed
                prompt_msg = prompt_msg + ' and need a snack'
        prompt_msg = prompt_msg + ': ' + str(completed_tasks)
    else:
        prompt_msg = 'Complete the following routine on a weekend'
        prompt_msg = prompt_msg + (' the human is having guests for lunch : ' if flags['guests'] else 'do not have guests: ')
        prompt_msg = prompt_msg + str(completed_tasks)
    return prompt_msg

def prepare_assistant_msg(completed_tasks,future_tasks):
    all_tasks = completed_tasks+future_tasks
    example_dict = {
    "['Prepare breakfast weekday', 'Prepare home work-station', 'Prepare coffee', 'Prepare lunch']": "Since it's a weekday, breakfast should be prepared first to energize the human for the day. Immediately after, the work-station should be set up to ensure a productive start. With the work-station ready, coffee can be prepared next to boost focus during work. Finally, lunch is prepared to provide a midday meal, sustaining energy through the afternoon.",
    "['Prepare breakfast weekday', 'Prepare home work-station', 'Prepare coffee', 'Serve snacks']": "Since it's a weekday, breakfast should be prepared first to energize the human for the day. Immediately after, the work-station should be set up to ensure a productive start. With the work-station ready, coffee can be prepared next to boost focus during work. Finally, since the human asked for a snack, instead of lunch the agent may need to serve snacks after providing coffee.",
    "['Prepare breakfast weekday', 'Pack bag', 'Prepare lunch', 'Clean kitchen']": "Since the human is leaving for the office, it's a busy morning focused on essentials. Breakfast is prepared first to start the day with energy. Next, packing the bag ensures they have everything needed for work. Lunch is then prepared in case they need it later. Finally, after they've left, the kitchen is cleaned to maintain a tidy space.",
    "['Prepare breakfast weekend', 'Prepare table for guests', 'Prepare lunch', 'Clean dishes']": "As everyday on weekends it is important to first prepare breakfast. Since the guests are arriving then the agent must make table to serve lunch to the guests. It is important to start on lunch as soon as the table is ready. Finally the agent may not have time to clean the whole kitchen while having guests but it can clean atleast the dirty dishes from lunch to maintain the hygiene.",
    "['Prepare breakfast weekend', 'Prepare activities', 'Serve snacks', 'Clean kitchen']": "On weekends, preparing breakfast first is essential for a relaxed start to the day. Once breakfast is ready, it's important to prepare activities to plan an enjoyable day. Serving snacks can then follow, as this keeps energy levels up during the activities. Finally, cleaning the kitchen is crucial to maintain a tidy space after all the cooking and snacking, ensuring it's ready for the next use. Cleaning dishes is not necessary since there were no guests and use of plates."
    }
    example_text = str(all_tasks) + ' ' + (example_dict[str(all_tasks)] if str(all_tasks) in example_dict else '')
    # system_msg = '''You are an intelligent assistant designed to help with daily tasks. 
    # Given the following list of tasks, provide a concise chain of thought process explaining 
    # the importance and sequence of each task. The explanation should be brief and to the point, 
    # similar to the example provided.'''
    # prompt_msg = str(answer)
    # user_msg = '''Task list: [Prepare breakfast weekday,Prepare home work-station,Prepare coffee,Prepare lunch]'''
    # assistant_msg = '''Since it is a weekday, breakfast should be prepared first to energize the human 
    # for the day. Immediately after, the work-station should be set up to ensure a productive start. With 
    # the work-station ready, coffee can be prepared next to boost focus during work. Finally, lunch is prepared 
    # to provide a midday meal, sustaining energy through the afternoon. '''

    # completion = client.chat.completions.create(
    #     model="gpt-4o-mini",
    #     messages = [
    #         {"role": "system", "content": system_msg},
    #         {"role": "user", "content": user_msg},
    #         {"role": "assistant", "content": assistant_msg},
    #         {"role": "user", "content": prompt_msg}
    #     ]
    # )
    # example_text = prompt_msg + completion.choices[0].message.content
    # print(example_text)

    return example_text

# get taskpredictions from LLM
def get_llm_next_task(flags, completed_tasks):
    print('---------------------------------------- ^.^')
    system_msg = '''You are an intelligent household assistant. Anticipate future tasks based on 
    previous days' data. You will get a list of tasks that can be executed in the household environment, 
    day of the week, an example of a partially completed routine and the expected output. Without 
    providing an explanation, complete the rest of the tasks for the partially completed routine only 
    using the tasks from the task list, as a Python list, following the examples given. The complete 
    task list should only have 5 tasks. No need to predict more. '''
    task_list = '''Task List: ['Prepare breakfast weekday','Prepare coffee','Prepare home work-station',
    'Prepare lunch','Pack bag','Clean kitchen','Prepare breakfast weekend',
    'Prepare table for guests','Serve snacks','Prepare activities','Clean dishes']'''
    
    unique_rows = set()
    with open('llm_example_data.csv', mode='r', newline='') as file:
        reader = csv.reader(file)
        for row in reader:
            unique_rows.add(tuple(row))
    unique_rows = list(unique_rows)

    if len(unique_rows) < 2: # first one is the real one
        ex1_flags = {'weekday': True, 'office': False, 'lunch': True}
        ex1_completed_tasks = []
        ex1_answer = ['Prepare breakfast weekday','Prepare home work-station','Prepare coffee','Prepare lunch']
        ex2_flags = {'weekday': True, 'office': True, 'lunch': True}
        ex2_completed_tasks = []
        ex2_answer = ['Prepare breakfast weekday','Pack bag','Prepare lunch','Clean kitchen']
    else:
        selected_rows = sample_rows(unique_rows) # random.sample(unique_rows, 2)
        ex1_flags = ast.literal_eval(selected_rows[0][1])
        ex1_completed_tasks = []
        ex1_answer = ast.literal_eval(selected_rows[0][0])
        ex2_flags = ast.literal_eval(selected_rows[1][1])
        ex2_completed_tasks = []
        ex2_answer = ast.literal_eval(selected_rows[1][0])
    
    # examples
    user_msg1 = prepare_prompt_msg(ex1_flags, ex1_completed_tasks)
    assistant_msg1 = str(ex1_answer) # prepare_assistant_msg(ex1_completed_tasks, ex1_answer)
    user_msg2 = prepare_prompt_msg(ex2_flags, ex2_completed_tasks)
    assistant_msg2 = str(ex2_answer) # prepare_assistant_msg(ex2_completed_tasks, ex2_answer)
    
    prompt_msg = prepare_prompt_msg(flags, completed_tasks)

    completion = client.chat.completions.create(
        model="gpt-4o-mini",
        messages = [
            {"role": "system", "content": system_msg+task_list},
            {"role": "user", "content": user_msg1},
            {"role": "assistant", "content": assistant_msg1},
            {"role": "user", "content": user_msg2},
            {"role": "assistant", "content": assistant_msg2},
            {"role": "user", "content": prompt_msg}
        ]
    )

    # print(user_msg1,assistant_msg1)
    # print(user_msg2,assistant_msg2)
    # print(prompt_msg)

    print(completion.choices[0].message.content)
    pattern = r"\[.*?\]"
    match_ = re.search(pattern,completion.choices[0].message.content)
    task_list = ast.literal_eval(match_.group(0))
    task_list = [item for item in task_list if item not in completed_tasks]
    return task_list

def add_unique(value, task_list):
    for val in value:
        if val not in task_list:
            task_list.append(val)
    return task_list

def list_to_sorts(item_list, name):
    items_str = ', '.join(item_list)
    sorts = f'#{name} = {{{items_str}}}'
    return sorts

def join_sorts(variables):
    valid_variable_names = [name for name, item in variables if item]
    temp_graspable = ''
    if valid_variable_names:
        temp_graspable = ' + '.join(valid_variable_names)
    return temp_graspable

def get_sorts_objects(ASP_goal, fluents):
    temp_objects = []
    sorts = ['sorts']
    items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',ASP_goal)]
    for fluent in fluents:
        if fluent.startswith('occurs') or 'in_hand' in fluent or 'agent_hand' in fluent: # exo actions, other aegnts hand or agent hands
            fitems = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)]
            fitems = [item.split(',')[1] if ',' in item else item for item in fitems]
            if fitems[0] not in items:
                items.append(fitems[0])
    fir_items = [item.split(',')[0] if ',' in item else item for item in items]
    sec_items = [item.split(',')[1] if ',' in item else item for item in items]
    items = fir_items + [item for item in sec_items if item not in locations and item not in fir_items]
    # food
    temp_food = [item for item in food if item in items]
    temp_objects += temp_food
    if temp_food:
        temp_food = list_to_sorts(temp_food, 'food')
        sorts.append(temp_food+'.')
    # drinks
    temp_drinks = [item for item in drinks if item in items]+ ['coffee']
    temp_objects += temp_drinks
    if temp_drinks:
        temp_drinks = list_to_sorts(temp_drinks, 'drinks')
        sorts.append(temp_drinks+'.')
    # electricals
    temp_electricals = [item for item in electricals if item in items] + ['fridge']
    temp_objects += temp_electricals
    temp_electricals = list_to_sorts(temp_electricals, 'electricals')
    sorts.append(temp_electricals+'.')
    # appliances
    temp_appliances = [item for item in appliances if item in items and item not in electricals] + ['coffeemaker']
    temp_objects += temp_appliances
    temp_appliances = list_to_sorts(temp_appliances, 'appliances') + ' + #electricals'
    sorts.append(temp_appliances+'.')
    # plates
    temp_plates = [item for item in plates if item in items]
    temp_objects += temp_plates
    if temp_plates:
        temp_plates = list_to_sorts(temp_plates, 'plates')
        sorts.append(temp_plates+'.')
    # glasses
    temp_glasses = [item for item in glasses if item in items]
    temp_objects += temp_glasses
    if temp_glasses:
        temp_glasses = list_to_sorts(temp_glasses, 'glasses')
        sorts.append(temp_glasses+'.')
    # containers
    temp_containers = [item for item in containers if item in items and item not in plates and item not in glasses] + ['coffeepot']
    temp_objects += temp_containers
    # if temp_containers:
    temp_containers = list_to_sorts(temp_containers, 'containers')
    temp_containers += ' + #plates' if temp_plates else ''
    temp_containers += ' + #glasses' if temp_glasses else ''
    sorts.append(temp_containers+'.')
    # elif temp_plates or temp_glasses:
    #     temp_containers = '#plates' if temp_plates else ''
    #     temp_containers += ' + ' if temp_plates and temp_glasses else ''
    #     temp_containers += '#glasses' if temp_glasses else ''
    #     sorts.append(temp_containers+'.')
    # others
    temp_others = [item for item in others if item in items]
    temp_objects += temp_others
    if temp_others:
        temp_others = list_to_sorts(temp_others, 'others')
        sorts.append(temp_others+'.')
    # graspable
    variables = [('#food',temp_food), ('#drinks',temp_drinks), ('#containers',temp_containers), ('#others',temp_others)]
    temp_graspable = join_sorts(variables)
    if temp_graspable:
        sorts.append('#graspable = '+temp_graspable+'.')
        sorts.append('#objects = #appliances + #graspable.')
    else:
        sorts.append('#objects = #appliances.')
    sorts.append(list_to_sorts(locations+['kitchen'],'locations')+'.')
    sorts.append('#surfaces = #locations + #appliances.')
    return temp_objects, sorts

# return filterd fluents according to sorts
def filter_fluents(fluents, temp_objects):
    all_fluents = []
    for fluent in fluents:
        if fluent.startswith('occurs'): # exo actions
            items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)]
            items = [item.split(',')[1] if ',' in item else item for item in items]
            if items[0] in temp_objects:
                all_fluents.append(fluent)
        else:
            items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)]
            items = [item.split(',')[0] if ',' in item else item for item in items]
            if items[0] in temp_objects or items[0] == 'human' or items[0] == 'ahagent1' or items[0] == 'ahagent2': # at
                if items[0] == 'human' or items[0] == 'ahagent1' or items[0] == 'ahagent2':
                    in_items = [item[1:-1] for item in re.findall(r'(\([^\(]*?\))',fluent)]
                    in_items = [item.split(',')[1] if ',' in item else item for item in in_items]
                    if in_items[0] in temp_objects or in_items[0] == 'kitchen' or in_items[0] in locations:
                        all_fluents.append(fluent)
                else:
                    all_fluents.append(fluent)
    return all_fluents

# return answer sets for the new ASP file - human does not consider two tasks at the same time
def run_ASP_human(graph, id_dict, sub_goal, human_at_dict, ah_at_dict, add_at_dict, last_task):
    ASP_goal_human = map_goal_ASP(sub_goal)
    print(ASP_goal_human)
    const_timeout = get_const_timeout(sub_goal)
    sub_goal_success = False
    found_solution = False
    counter = 1
    max_counter = const_timeout[0]
    timeout = const_timeout[1]
    exit_counter = 0
    reader = open(human_asp_pre, 'r')
    pre_asp = reader.read()
    reader.close()
    pre_asp_split = pre_asp.split('\n')
    display_marker_index = pre_asp_split.index(display_marker)
    if last_task:
        human_fluents, ah_fluents, add_fluents, common_fluents, _ = convert_state(graph, ASP_goal_human, id_dict, '0', human_at_dict, ah_at_dict, add_at_dict)
    else:
        human_fluents, ah_fluents, add_fluents, common_fluents, ASP_goal_human = convert_state(graph, ASP_goal_human, id_dict, '0', human_at_dict, ah_at_dict, add_at_dict)

    if not (len(ASP_goal_human) > 0):
        return [], ah_fluents, add_fluents, common_fluents, True # to make the human also do something without waiting for the ad hoc agent to finish
    temp_objects, sorts = get_sorts_objects(ASP_goal_human, human_fluents+common_fluents)
    fluents = filter_fluents(human_fluents+common_fluents, temp_objects)
    while (not found_solution):
        const_term = ['#const n = ' + str(counter) + '.']
        asp_split = const_term + sorts + pre_asp_split[:display_marker_index] + ['goal(I) :- ' + ASP_goal_human + '.'] + fluents + pre_asp_split[display_marker_index:]
        asp = '\n'.join(asp_split)
        f1 = open(human_asp, 'w')
        f1.write(asp)
        f1.close()
        try:
            answer = subprocess.check_output('java -jar ASP/sparc.jar ' +human_asp+' -A -n 1',shell=True, timeout=timeout)
        except subprocess.TimeoutExpired as exec: # timeout exit
            if counter <= max_counter:
                counter = counter+1
            else:
                if exit_counter < 2:
                    exit_counter = exit_counter+1
                    counter = 1
                    ASP_goal_human = remove_excess_fluents(ASP_goal_human)
                    continue
                else:
                    print('reached MAX count!')
                    return None, ah_fluents, add_fluents, common_fluents, False
        answer_split = (answer.decode('ascii'))
        if len(answer_split) > 1:
            found_solution = True
            if len(answer_split) <= 3:
                sub_goal_success = True
        else: # no answer exit const not large enough for an answer for this ASP
            if counter < max_counter:
                counter = counter+1
            else:
                if exit_counter <= 2:
                    exit_counter = exit_counter+1
                    counter = 1
                    ASP_goal_human = remove_excess_fluents(ASP_goal_human)
                    continue
                else:
                    print('reached MAX count!')
                    return None, ah_fluents, add_fluents, common_fluents, False
    actions, _ = process_answerlist(answer_split, sub_goal_success)
    return actions, ah_fluents, add_fluents, common_fluents, sub_goal_success

# return answer sets for the new ASP file
def run_ASP_ahagent(graph, current_task, ASP_goal, const_timeout, all_prev_tasks, flags, ah_fluents, common_fluents, all_prev_actions, env_id, id_dict, current_script, last_task, num_agents, current_agent_id):
    sub_goal_success = False
    found_solution = False
    # answer_split = None
    counter = 1
    max_counter = const_timeout[0]
    timeout = const_timeout[1]
    exit_counter = 0
    reader = open(ah_asp_pre, 'r')
    pre_asp = reader.read()
    reader.close()
    pre_asp_split = pre_asp.split('\n')
    display_marker_index = pre_asp_split.index(display_marker)
    # get future actions for other agents
    other_actions, graphs, _ = get_future_actions(graph, current_task, all_prev_tasks, flags, ah_fluents, common_fluents, ASP_goal, all_prev_actions, env_id, id_dict, current_script, num_agents, current_agent_id)
    if last_task:
        _, fluents = get_fluents(ASP_goal, other_actions, graphs, ah_fluents+common_fluents, id_dict, num_agents, current_agent_id)
    else:
        ASP_goal, fluents = get_fluents(ASP_goal, other_actions, graphs, ah_fluents+common_fluents, id_dict, num_agents, current_agent_id)
    temp_objects, sorts = get_sorts_objects(ASP_goal, fluents)
    sorts.append('#agent = {ahagent' + str(current_agent_id) + '}.')
    sorts.append('#other_agents = {human, ahagent' + ('2' if current_agent_id == 1 else '1') + '}.')
    fluents = filter_fluents(fluents, temp_objects)
    if ASP_goal:
        while (not found_solution):
            const_term = ['#const n = ' + str(counter) + '.']
            asp_split = const_term + sorts + pre_asp_split[:display_marker_index] + ['goal(I) :- ' + ASP_goal + '.'] + fluents + pre_asp_split[display_marker_index:]
            asp = '\n'.join(asp_split)
            f1 = open(ah_asp_new, 'w')
            f1.write(asp)
            f1.close()
            try:
                plan_time_start = time.time()
                answer = subprocess.check_output('java -jar ASP/sparc.jar ' +ah_asp_new+' -A -n 1',shell=True, timeout=timeout)
                plan_time_end = time.time()
            except subprocess.TimeoutExpired as exec:
                print('command timed out')
                if counter <= max_counter:
                    counter = counter+1
                else:
                    if exit_counter < 2:
                        exit_counter = exit_counter+1
                        counter = 1
                        ASP_goal = remove_excess_fluents(ASP_goal)
                        continue
                    else:
                        print('reached MAX count!')
                        return None, False, 0
            answer_split = (answer.decode('ascii'))
            if len(answer_split) > 1:
                found_solution = True
                if len(answer_split) <= 3:
                    sub_goal_success = True
            else: # no answer exit const not large enough for an answer for this ASP
                if counter < max_counter:
                    counter = counter+1
                else:
                    if exit_counter <= 2:
                        exit_counter = exit_counter+1
                        counter = 1
                        ASP_goal = remove_excess_fluents(ASP_goal)
                        continue
                    else:
                        print('reached MAX count!')
                        return None, False, 0
    else:
        return None, False, 0
    plan_time_inside = plan_time_end-plan_time_start
    actions, sub_goal_success = process_answerlist(answer_split,sub_goal_success)
    return actions, sub_goal_success, plan_time_inside

def generate_initialscript(human_act, ah_act, id_dict, human_character, ah_character):
    script = [
        '<char0> [find] <cereal> ({})|<char1> [find] <chips> ({})'.format(id_dict['cereal'], id_dict['chips']),
        '<char0> [grab] <cereal> ({})|<char1> [grab] <chips> ({})'.format(id_dict['cereal'], id_dict['chips']),
        '<char0> [find] <chair> ({})|<char1> [find] <tvstand> ({})'.format(id_dict['chair'], id_dict['kitchen_smalltable']),
        '<char0> [putback] <cereal> ({}) <chair> ({})|<char1> [putback] <chips> ({}) <tvstand> ({})'.format(id_dict['cereal'], id_dict['chair'], id_dict['chips'], id_dict['kitchen_smalltable'])
    ]
    # select the agent with the longest plan
    plan_len = len(human_act) if len(human_act) > len(ah_act) else len(ah_act)
    for action_index in range(plan_len):
        # either of the agents may or may not have an act at the last steps
        if len(human_act) > action_index:
            human_action = human_act[action_index]
            for delimeter in delimeters:
                human_action = " ".join(human_action.split(delimeter))
            human_action_split = human_action.split()
            if human_action_split[1] in ['put']:
                human_script_instruction = human_character + ' [putback] <' + get_virtualhome_name(human_action_split[3]) + '> (' + str(id_dict[human_action_split[3]]) + ') <' + get_virtualhome_name(human_action_split[4]) + '> (' + str(id_dict['chair' if human_action_split[4] == 'livingroom_desk' else human_action_split[4]]) + ')'
            elif human_action_split[1] in ['find']:
                human_script_instruction = human_character + ' [find] <' + get_virtualhome_name(human_action_split[3]) + '> (' + str(id_dict[human_action_split[3] if not human_action_split[3] == 'kitchen' else 'fridge']) + ')'
            else:
                human_script_instruction = human_character + ' [' + human_action_split[1].replace('_','') + '] <' + get_virtualhome_name(human_action_split[3]) + '> (' + str(id_dict[human_action_split[3]]) + ')'
        else:
            human_script_instruction = None
        if len(ah_act) > action_index:
            ah_action = ah_act[action_index]
            for delimeter in delimeters:
                ah_action = " ".join(ah_action.split(delimeter))
            ah_action_split = ah_action.split()
            if ah_action_split[1] in ['put']:
                ah_script_instruction = ah_character + ' [putback] <' + get_virtualhome_name(ah_action_split[3]) + '> (' + str(id_dict[ah_action_split[3]]) + ') <' + get_virtualhome_name(ah_action_split[4]) + '> (' + str(id_dict['chair' if ah_action_split[4] == 'livingroom_desk' else ah_action_split[4]]) + ')'
            elif ah_action_split[1] in ['find']:
                ah_script_instruction = ah_character + ' [find] <' + get_virtualhome_name(ah_action_split[3]) + '> (' + str(id_dict[ah_action_split[3]]) + ')'
            else:
                ah_script_instruction = ah_character + ' [' + ah_action_split[1].replace('_','') + '] <' + get_virtualhome_name(ah_action_split[3]) + '> (' + str(id_dict[ah_action_split[3]]) + ')'
        else:
            ah_script_instruction = None
        script_instruction = (human_script_instruction + '|' + ah_script_instruction) if human_script_instruction and ah_script_instruction else (human_script_instruction if human_script_instruction else ah_script_instruction)
        script.append(script_instruction)
    return script

def select_initialstate(id_dict, human_character, ah_character, add_character): # , row
    human_act = []
    ah_act = []
    day_of_week =  random.choice(['monday','tuesday','wednesday','thursday','friday','saturday','sunday'])
    if day_of_week in ('monday','tuesday','wednesday','thursday','friday'):
        weekday = True
    elif day_of_week in ('saturday', 'sunday'):
        weekday = False
    script = generate_initialscript(human_act, ah_act, id_dict, human_character, ah_character)
    return script, human_act, ah_act, weekday

def process_action(act):
    if len(act) == 3:
        act = act[0][1:-1] + '_' + act[1][1:-1]
    elif len(act) == 5:
        act = act[0][1:-1] + '_' + act[1][1:-1] + '_' + act[3][1:-1]
    return act

def initialize_csv(file_path,header_names):
    if not os.path.exists(file_path):
        with open(file_path, mode='w', newline='') as file:
            writer = csv.writer(file)
            writer.writerow(header_names)

def write_real_action(file_path, action):
    with open(file_path, mode='a', newline='') as file:
        writer = csv.writer(file)
        writer.writerow([action, None])

def write_predict_action(file_path, action):
    rows = []
    with open(file_path, mode='r', newline='') as file:
        reader = csv.reader(file)
        rows = list(reader)
    for index in reversed(range(len(rows))):
        if rows[index][1] == '' or rows[index][1] is None:
            rows[index][1] = action
        break
    with open(file_path, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(rows)

# def save_model_data(graph, sub_goal, prev_human_tasks, script, prev_human_actions, flags, id_dict, char_id):
#     values = None
#     if '|' in script[0]:
#         script_split = script[0].split('|')
#         act = script_split[0]
#         act = act.split()
#         values = process_state(graph, sub_goal, prev_human_tasks, prev_human_actions, flags, id_dict)
#         if len(act) == 4:
#             values.append('_'.join([act[1][1:-1],act[2][1:-1]]))
#         elif len(act) == 6:
#             values.append('_'.join([act[1][1:-1],act[2][1:-1],act[4][1:-1]]))
#     return values

def prepare_data(old_data_file, new_data_file):
    with open(old_data_file, mode='r', newline='') as file:
        reader = csv.reader(file)
        data = list(reader)
    # remove first 100 rows and write back to the file
    data = data[100:]
    with open(old_data_file, mode='w', newline='') as file:
        writer = csv.writer(file)
        writer.writerows(data)
    
    with open(new_data_file, mode='r', newline='') as src:
        reader = csv.reader(src)
        new_data = list(reader)

    with open(old_data_file, mode='a', newline='') as tgt:
        writer = csv.writer(tgt)
        for row in new_data:
            writer.writerow(row)
    
    with open(new_data_file, mode='w', newline='') as file:
        writer = csv.writer(file)
