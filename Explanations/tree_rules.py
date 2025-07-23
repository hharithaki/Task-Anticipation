import re
import subprocess
import trees
import numpy as np
from prettytable import PrettyTable
from sklweka.classifiers import Classifier
from sklweka.dataset import to_instance
from simulation.unity_simulator import utils_viz, comm_unity

delimeters = ['(', ')', ',']
human_model = 'human_model.model'
interacted_items = ['None', 'None']
ah_counter = 28
ah_asp_pre = 'ASP/ahagent_pre.sp'
ah_asp_new = 'ASP/ahagent.sp'
display_marker = 'display'
categorya_food = ['cutlets']
categoryb_food = ['poundcake']
food = ['breadslice'] + categorya_food + categoryb_food
drinks = ['waterglass']
sittable = ['bench']
electricals = ['microwave']
appliances = electricals + ['stove']
containers = ['fryingpan']
graspable = food + drinks + containers
objects = appliances + graspable + sittable
heated_ = [[obj,False] for obj in categoryb_food]
cooked_ = [[obj,False] for obj in categorya_food]

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

def process_graph(graph, prev_actions):

    state = []
    # Previous action of the agent (include multiple actions in their particular order?)
    # state.append(prev_actions) -- TODO
    act = prev_actions[0].split()
    if len(act) == 4:
        state.append('_'.join([act[1][1:-1],act[2][1:-1]]))
    elif len(act) == 6:
        state.append('_'.join([act[1][1:-1],act[2][1:-1],act[4][1:-1]]))
    else:
        state.append('find_watergalss')

    # Item interactions (immediately previous interaction item or multiple items?)
    script_split = prev_actions[1].split()
    if len(script_split) == 4:
        state.append('_'.join([script_split[1][1:-1],script_split[2][1:-1]]))
        interacted_items.pop(0)
        interacted_items.append(script_split[2][1:-1])
    elif len(script_split) == 6:
        state.append('_'.join([script_split[1][1:-1],script_split[2][1:-1],script_split[4][1:-1]]))
        interacted_items.pop(0)
        # interacted_items.pop(0)
        interacted_items.append(script_split[2][1:-1])
        # interacted_items.append(script_split[4][1:-1])
    else:
        state.append('find_waterglass')
        interacted_items.pop(0)
        interacted_items.append('waterglass')
        interacted_items.pop(0)
        interacted_items.append('waterglass')
    state.append(interacted_items[0])
    state.append(interacted_items[1])
    
    # Location of the agent
    human_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
    state.append(human_pose['position'][0]) # x
    state.append(human_pose['position'][1]) # y
    state.append(human_pose['position'][2]) # z
    state.append(human_pose['rotation'][0]) # x
    state.append(human_pose['rotation'][1]) # y
    state.append(human_pose['rotation'][2]) # z

    # Proximity to the kitchen table
    kitchentable_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    prox_kitchentable = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(kitchentable_pose['position']))    
    state.append(prox_kitchentable)

    # Proximity to the kitchen counter
    kitchentable_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'kitchencounter'][0]
    prox_kitchencounter = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(kitchentable_pose['position']))
    state.append(prox_kitchencounter)

    # Status of microwave (on/off/open/closed).
    microwave_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    state.append(microwave_status[0])
    state.append(microwave_status[1])

    #  Items inside microwave
    microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == microwave_id and edge['relation_type'] == 'INSIDE']
    item_name = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        item_name.append(names)
    item_name = [item for idx, item in enumerate(item_name) if item != 'plate']
    state.append(item_name[0] if len(item_name) > 0 else 'None')
    
    # Status of Stove (on/off/open/closed).
    stove_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    state.append(stove_status[0])
    state.append(stove_status[1])

    #  Items on stove (fryingpan) - since it cannot be grabbed
    fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == fryingpan_id and edge['relation_type'] == 'ON']
    item_name = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        item_name.append(names)
    state.append(item_name[0] if len(item_name)>0 else 'None')

    # Items currently on the dinning table
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    edges = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == kitchentable_id and edge['relation_type'] == 'ON']
    table_items = [node['class_name']  for edge in edges for node in graph['nodes'] if node['id'] == edge]

    # Number of items on the dinning table.
    no_table_items = len(table_items)
    state.append(no_table_items)

    return state

def state_all_process(graph):
    state_table = PrettyTable(['Description', 'Value'])
    # Location of the agent
    human_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][0]
    print(human_pose)
    state_table.add_row(['human pose', human_pose['position']])
    state_table.add_row(['human orientation', human_pose['rotation']])

    ah_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'character'][1]
    state_table.add_row(['ad hoc agent pose', ah_pose['position']])
    state_table.add_row(['ad hoc agent orientation', ah_pose['rotation']])

    human_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 1 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    ah_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 2 and (edge['relation_type'] == 'HOLDS_RH' or edge['relation_type'] == 'HOLDS_LH')]
    human_hand_objects = []
    ah_hand_objects = []

    for item in human_object_ids: # objects in human hand
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        human_hand_objects.append(names)
    for item in ah_object_ids: # objects in ah agent hand
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        ah_hand_objects.append(names)

    state_table.add_row(['objects in human hand', human_hand_objects])
    state_table.add_row(['objects in ad hoc agent hand', ah_hand_objects])
    # Proximity to the kitchen table
    kitchentable_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    prox_kitchentable = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(kitchentable_pose['position']))    
    state_table.add_row(['distance to kitchentable', prox_kitchentable])

    # Proximity to the kitchen counter
    kitchentable_pose = [node['obj_transform'] for node in graph['nodes'] if node['class_name'] == 'kitchencounter'][0]
    prox_kitchencounter = np.linalg.norm(np.asarray(human_pose['position'])-np.asarray(kitchentable_pose['position']))
    state_table.add_row(['distance to kitchencounter', prox_kitchencounter])

    # Status of microwave (on/off/open/closed).
    microwave_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    state_table.add_row(['microwave status', microwave_status])

    #  Items inside microwave
    microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == microwave_id and edge['relation_type'] == 'INSIDE']
    item_name = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        item_name.append(names)
    item_name = [item for idx, item in enumerate(item_name) if item != 'plate']
    state_table.add_row(['items inside microwave', item_name])
    
    # Status of Stove (on/off/open/closed).
    stove_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    state_table.add_row(['stove status', stove_status])

    #  Items on stove (fryingpan) - since it cannot be grabbed
    fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == fryingpan_id and edge['relation_type'] == 'ON']
    item_name = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        item_name.append(names)
    state_table.add_row(['items on stove', item_name])

    # Items currently on the dinning table
    # kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    # edges = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == kitchentable_id and edge['relation_type'] == 'ON']
    # table_items = [node['class_name']  for edge in edges for node in graph['nodes'] if node['id'] == edge]
    # state_table.add_row(['items on kitchentable', table_items])
    return state_table

def predict_next_action(graph, prev_human_actions):
    values = process_graph(graph, prev_human_actions)
    if (any(word in values[0] for word in ['bench','eat','drink'])) or (any(word in values[1] for word in ['bench','eat','drink'])):
        # no trianing data
        return None
    model, header = Classifier.deserialize(human_model)
    # create new instance
    inst = to_instance(header,values)
    inst.dataset = header
    # make prediction
    index = model.classify_instance(inst)
    return header.class_attribute.value(int(index))

def convert_state(graph, prev_human_actions, prev_ah_actions, human_success, ah_success, timestep):
    human_fluents = []
    ah_fluents = []
    fluents = []
    human_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 1 and edge['relation_type'] == 'HOLDS_RH']
    ah_object_ids = [edge['to_id'] for edge in graph['edges'] if edge['from_id'] == 2 and edge['relation_type'] == 'HOLDS_RH']
    human_hand_objects = []
    ah_hand_objects = []

    for item in human_object_ids: # objects in human hand
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        human_hand_objects.append(names)
    for item in ah_object_ids: # objects in ah agent hand
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        ah_hand_objects.append(names)

    # % --------------- % find
    if 'find' in prev_human_actions[-1] and human_success:
        action = (prev_human_actions[-1].split())[2][1:-1] # <char0> [find] <poundcake> (248) True
        for obj in objects:
            if obj == action:
                human_fluents.append('holds(found(human,' + action + '),' + timestep + ').')
                ah_fluents.append('holds(agent_found(human,' + action + '),' + timestep + ').')
            else:
                human_fluents.append('-holds(found(human,' + obj + '),' + timestep + ').') 
    else:
        human_found_set = False
        for obj in objects:
            if (obj in appliances or obj in sittable) and prev_human_actions[0] != 'None' and not human_found_set:
                if obj == (prev_human_actions[0].split())[2][1:-1]:
                    human_fluents.append('holds(found(human,' + obj + '),' + timestep + ').')
                    ah_fluents.append('holds(agent_found(human,' + obj + '),' + timestep + ').')
                    human_found_set = True
                else:
                    human_fluents.append('-holds(found(human,' + obj + '),' + timestep + ').')
            elif obj in human_hand_objects and not human_found_set:
                human_fluents.append('holds(found(human,' + obj + '),' + timestep + ').')
                ah_fluents.append('holds(agent_found(human,' + obj + '),' + timestep + ').')
                human_found_set = True
            else:
                human_fluents.append('-holds(found(human,' + obj + '),' + timestep + ').')
    if 'find' in prev_ah_actions[-1] and ah_success:
        action = (prev_ah_actions[-1].split())[2][1:-1] # <char0> [find] <poundcake> (248) True
        for obj in objects:
            if obj == action:
                ah_fluents.append('holds(found(ahagent,' + action + '),' + timestep + ').')
                human_fluents.append('holds(agent_found(ahagent,' + action + '),' + timestep + ').')
            else:
                ah_fluents.append('-holds(found(ahagent,' + obj + '),' + timestep + ').') 
    else:
        ah_found_set = False
        for obj in objects:
            if (obj in appliances or obj in sittable) and prev_ah_actions[0] != 'None' and not ah_found_set:
                if obj == (prev_ah_actions[0].split())[2][1:-1]:
                    ah_fluents.append('holds(found(ahagent,' + obj + '),' + timestep + ').')
                    human_fluents.append('holds(agent_found(ahagent,' + obj + '),' + timestep + ').')
                    ah_found_set = True
                else:
                    ah_fluents.append('-holds(found(ahagent,' + obj + '),' + timestep + ').')
            elif obj in ah_hand_objects and not ah_found_set:
                ah_fluents.append('holds(found(ahagent,' + obj + '),' + timestep + ').')
                human_fluents.append('holds(agent_found(ahagent,' + obj + '),' + timestep + ').')
                ah_found_set = True
            else:
                ah_fluents.append('-holds(found(ahagent,' + obj + '),' + timestep + ').')

    # % --------------- % grab
    if 'grab' in prev_human_actions[-1] and human_success:
        action = (prev_human_actions[-1].split())[2][1:-1]
        for obj in graspable:
            if obj == action:
                human_fluents.append('holds(in_hand(human,' + action + '),' + timestep + ').')
                ah_fluents.append('holds(agent_hand(human,' + action + '),' + timestep + ').')
                new_fluent = 'holds(found(human,' + obj + '),' + timestep + ').'
                human_fluents = [fluent for fluent in human_fluents if fluent != '-'+new_fluent]
                human_fluents.append(new_fluent)
                ah_fluents.append('holds(agent_found(human,' + obj + '),' + timestep + ').')
                human_found_set = True
            else:
                human_fluents.append('-holds(in_hand(human,' + obj + '),' + timestep + ').')
    else:
        for obj in graspable:
            if obj in human_hand_objects:
                human_fluents.append('holds(in_hand(human,' + obj + '),' + timestep + ').')
                ah_fluents.append('holds(agent_hand(human,' + obj + '),' + timestep + ').')
            else:
                human_fluents.append('-holds(in_hand(human,' + obj + '),' + timestep + ').')
    if 'grab' in prev_ah_actions[-1] and ah_success:
        action = (prev_ah_actions[-1].split())[2][1:-1]
        for obj in graspable:
            if obj == action:
                ah_fluents.append('holds(in_hand(ahagent,' + action + '),' + timestep + ').')
                human_fluents.append('holds(agent_hand(ahagent,' + action + '),' + timestep + ').')
                new_fluent = 'holds(found(ahagent,' + obj + '),' + timestep + ').'
                ah_fluents = [fluent for fluent in ah_fluents if fluent != '-'+new_fluent]
                ah_fluents.append(new_fluent)
                human_fluents.append('holds(agent_found(ahagent,' + obj + '),' + timestep + ').')
                ah_found_set = True
            else:
                ah_fluents.append('-holds(in_hand(ahagent,' + obj + '),' + timestep + ').')
    else:
        for obj in graspable:
            if obj in ah_hand_objects:
                ah_fluents.append('holds(in_hand(ahagent,' + obj + '),' + timestep + ').')
                human_fluents.append('holds(agent_hand(ahagent,' + obj + '),' + timestep + ').')
            else:
                ah_fluents.append('-holds(in_hand(ahagent,' + obj + '),' + timestep + ').')
    # % --------------- % put
    # Items on dinning table
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    edges = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == kitchentable_id and edge['relation_type'] == 'ON']
    table_items = [node['class_name']  for edge in edges for node in graph['nodes'] if node['id'] == edge]
    for obj in graspable:
        if obj in table_items:
            fluents.append('holds(on(' + obj + ',kitchentable),' + timestep + ').')
        else:
            fluents.append('-holds(on(' + obj + ',kitchentable),' + timestep + ').')
    # Items inside microwave
    microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == microwave_id and edge['relation_type'] == 'INSIDE']
    microwave_item = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        microwave_item.append(names)
    for obj in graspable:
        if obj in microwave_item:
            fluents.append('holds(on('+ obj + ',microwave),' + timestep + ').')
        else:
            fluents.append('-holds(on('+ obj + ',microwave),' + timestep + ').')
    # Items on stove
    fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
    stove_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    pan_on_stove = [True for edge in graph['edges'] if edge['from_id'] == fryingpan_id and edge['to_id'] == stove_id and edge['relation_type'] == 'ON']
    pan_on_stove = pan_on_stove[0] if pan_on_stove else False
    for obj in graspable:
        if obj == 'fryingpan' and pan_on_stove:
            fluents.append('holds(on('+ obj + ',stove),' + timestep + ').')
        else:
            fluents.append('-holds(on('+ obj + ',stove),' + timestep + ').')
    # temp assumption => nothing on stove
    # default_fluents = ['-holds(on('+ obj + ',stove),0).' for obj in graspable]
    # fluents = fluents + default_fluents

    # % --------------- % open/close
    # Status of microwave
    microwave_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    if 'CLOSED' in microwave_status:
        fluents.append('-holds(opened(microwave),' + timestep + ').')
    elif 'OPEN' in microwave_status:
        fluents.append('holds(opened(microwave),' + timestep + ').')

    # % --------------- % switch on/off
    # Status of microwave
    if 'OFF' in microwave_status:
        fluents.append('-holds(switched_on(microwave),' + timestep + ').')
    elif 'ON' in microwave_status:
        fluents.append('holds(switched_on(microwave),' + timestep + ').')
    # Status of Stove
    stove_status = [node['states'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    if 'OFF' in stove_status:
        fluents.append('-holds(switched_on(stove),' + timestep + ').')
    elif 'ON' in stove_status:
        fluents.append('holds(switched_on(stove),' + timestep + ').')

    # % --------------- % heated
    for obj in categoryb_food: # poundcake
        heated_idx = [idx for idx, item in enumerate(heated_) if item[0] == obj][0]
        if (obj in microwave_item and 'ON' in microwave_status) or heated_[heated_idx][1]:
            fluents.append('holds(heated(' + obj + '),' + timestep + ').')
            heated_[heated_idx][1] = True
        else:
            fluents.append('-holds(heated(' + obj + '),' + timestep + ').')

    # % --------------- % cooked
    # Items on fryingpan
    item_id = [edge['from_id'] for edge in graph['edges'] if edge['to_id'] == fryingpan_id and edge['relation_type'] == 'ON']
    pan_items = []
    for item in item_id:
        names = [node['class_name'] for node in graph['nodes'] if node['id'] == item][0]
        pan_items.append(names)
    for obj in categorya_food: # cutlets
        cooked_idx = [idx for idx, item in enumerate(cooked_) if item[0] == obj][0]
        if (obj in pan_items and 'ON' in stove_status and pan_on_stove) or cooked_[cooked_idx][1]:
            fluents.append('holds(cooked(' + obj + '),' + timestep + ').')
            cooked_[cooked_idx][1] = True
        else:
            fluents.append('-holds(cooked(' + obj + '),' + timestep + ').')
    # % --------------- % ate => once we reach the eat action we will stop running the prg since goal has been reached hence adding some defaults here.
    if 'eat' in prev_human_actions[-1] and human_success:
        obj1 = (prev_human_actions[-1].split())[2][1:-1]
        if 'eat' in prev_human_actions[0]:
            obj2 = (prev_human_actions[0].split())[2][1:-1]
            for obj in food:
                if obj == obj1 or obj == obj2:
                    human_fluents.append('holds(ate(human,' + obj + '),' + timestep + ').')
                else:
                    human_fluents.append('-holds(ate(human,' + obj + '),' + timestep + ').')
        else:
            for obj in food:
                if obj == obj1:
                    human_fluents.append('holds(ate(human,' + obj + '),' + timestep + ').')
                else:
                    human_fluents.append('-holds(ate(human,' + obj + '),' + timestep + ').')
    else:
        default_fluents = ['-holds(ate(human,' + obj + '),' + timestep + ').' for obj in food]
        human_fluents = human_fluents + default_fluents
    # % --------------- % drink => same
    default_fluents = ['-holds(drank(human,' + obj + '),' + timestep + ').' for obj in drinks]
    human_fluents = human_fluents + default_fluents
    # % --------------- % put_in
    for obj in categorya_food:
        if obj in pan_items:
            fluents.append('holds(inside(' + obj + ',fryingpan),' + timestep + ').')
        else:
            fluents.append('-holds(inside(' + obj + ',fryingpan),' + timestep + ').')
    # % --------------- % sit
    if 'sit' in prev_human_actions[-1] and human_success:
        action = (prev_human_actions[-1].split())[2][1:-1]
        for obj in sittable:
            if obj == action:
                human_fluents.append('holds(sat(human,' + action + '),' + timestep + ').')
            else:
                human_fluents.append('-holds(sat(human,' + obj + '),' + timestep + ').')
    else:
        default_fluents = ['-holds(sat(human,' + obj + '),' + timestep + ').' for obj in sittable]
        human_fluents = human_fluents + default_fluents
    return human_fluents, ah_fluents, fluents

def get_future_state(graph, ah_fluents, common_fluents, prev_human_actions, prev_ah_actions, env_id, id_dict, current_script):
    future_action = predict_next_action(graph, prev_human_actions)
    if not future_action:
        return graph, prev_human_actions, prev_ah_actions, current_script, False, False
    if 'grab' in future_action:
        ah_positive_in_hand = [item for item in ah_fluents if item.startswith('holds(in_hand(ahagent,')]
        action_split = future_action.split('_')
        obj = action_split[1]
        if ('holds(in_hand(ahagent,'+ obj + '),0).' in ah_positive_in_hand):
            return graph, prev_human_actions, prev_ah_actions, current_script, False, False
    # assume ad hoc agent does nothing; only human action added to script
    future_action = future_action.split('_')
    if len(future_action) == 2:
        future_action = '<char0> [' + future_action[0] + '] <' + future_action[1] + '> (' + id_dict[future_action[1]] + ')'
    else:
        future_action = '<char0> [' + future_action[0] + '] <' + future_action[1] + '> (' + id_dict[future_action[1]] + ') <' + future_action[2] + '> (' + id_dict[future_action[2]] + ')'
    current_script.append(future_action)
    # initiate a second env
    comm_dummy = comm_unity.UnityCommunication(port='8082')
    comm_dummy.reset(env_id)
    success_dummy, graph_dummy = comm_dummy.environment_graph()
    success1_dummy, message_dummy, success2_dummy, graph_dummy = clean_graph(comm_dummy, graph_dummy, ['chicken'])

    # Add human
    comm_dummy.add_character('Chars/Female1', initial_room='kitchen')
    # Add ad hoc agent
    comm_dummy.add_character('Chars/Male1', initial_room='kitchen')

    for script_instruction in current_script:
        act_success, human_success, ah_success, message = comm_dummy.render_script([script_instruction], recording=False, skip_animation=True)
    # Get the state observation
    success, graph = comm_dummy.environment_graph()
    if human_success:
        prev_human_actions.pop(0)
        prev_human_actions.append(future_action)
    else:
        del current_script[-1]
    return graph, prev_human_actions, prev_ah_actions, current_script, human_success, ah_success

def refine_fluents(graph, prev_human_actions, prev_ah_actions, human_success, ah_success, env_id, id_dict, current_script):
    human_fluents, ah_fluents, common_fluents = convert_state(graph, prev_human_actions, prev_ah_actions, human_success, ah_success, '0')
    all_ah_fluents = ah_fluents
    all_common_fluents = common_fluents
    for i in range(2):
        graph, prev_human_actions, prev_ah_actions, current_script, human_success, ah_success = get_future_state(graph, ah_fluents, common_fluents, prev_human_actions, prev_ah_actions, env_id, id_dict, current_script)
        # process state to fluents
        human_fluents, ah_fluents, common_fluents = convert_state(graph, prev_human_actions, prev_ah_actions, human_success, ah_success, str(i))
        # find and merge human fluents
        if i == 0:
            # remove
            tem_ah_fluents = [item for item in ah_fluents if 'agent_found' in item or 'agent_hand' in item]
            all_ah_fluents = [item for item in all_ah_fluents if item not in tem_ah_fluents and '-'+item not in tem_ah_fluents and item[1:] not in tem_ah_fluents]
            all_ah_fluents = all_ah_fluents + tem_ah_fluents
            tem_common_fluents = [item for item in common_fluents if item not in all_common_fluents]
            all_common_fluents = [item for item in all_common_fluents if item not in tem_common_fluents and '-'+item not in tem_common_fluents and item[1:] not in tem_common_fluents]
            all_common_fluents = all_common_fluents + tem_common_fluents
        else:
            tem_ah_fluents = [item for item in ah_fluents if 'agent_found' in item or 'agent_hand' in item]
            all_ah_fluents = all_ah_fluents + tem_ah_fluents
            tem_common_fluents = [item for item in common_fluents if item.replace('),'+str(i)+').','),0).') not in all_common_fluents]
            all_common_fluents = all_common_fluents + tem_common_fluents
    return all_ah_fluents, all_common_fluents

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

def process_answerlist(answer):
    answer_list = answer_set_finder('occurs(A,I)', answer)
    action_list = []
    for i in range(len(answer_list)):
        for element in answer_list:
            if re.search(rf',{i}\)$',element) != None:
                action_list.insert(i, element)
    return action_list

def run_ASP_ahagent(graph, prev_human_actions, prev_ah_actions, human_success, ah_success, ah_counter, env_id, id_dict, current_script):
    goal = False
    found_solution = False
    answer_split = None
    counter = ah_counter
    ah_fluents, common_fluents = refine_fluents(graph, prev_human_actions, prev_ah_actions, human_success, ah_success, env_id, id_dict, current_script)
    positive_counter = True
    while (not found_solution) or counter == 0:
        const_term = ['#const n = ' + str(counter) + '.']
        reader = open(ah_asp_pre, 'r')
        pre_asp = reader.read()
        reader.close()
        pre_asp_split = pre_asp.split('\n')
        display_marker_index = pre_asp_split.index(display_marker)
        asp_split = const_term + pre_asp_split[:display_marker_index] + ah_fluents + common_fluents + pre_asp_split[display_marker_index:]
        asp = '\n'.join(asp_split)
        f1 = open(ah_asp_new, 'w')
        f1.write(asp)
        f1.close()
        try:
            answer = subprocess.check_output('java -jar ASP/sparc.jar ' +ah_asp_new+' -A -n 1',shell=True, timeout=10)
        except subprocess.TimeoutExpired as exec:
            print('command timed out')
            counter = counter-1
            continue
        answer_split = (answer.decode('ascii'))
        if len(answer_split) > 1:
            found_solution = True
            ah_counter = counter
        if counter > 0 and positive_counter:
            counter = counter-1 # in case
        else:
            counter = counter+1
            positive_counter = False
    actions = process_answerlist(answer_split)
    return actions

def generate_script(human_act, ah_act, id_dict, human_character, ah_character):
    script = []
    # select the agent with the longest plan
    plan_len = len(human_act) if len(human_act) > len(ah_act) else len(ah_act)
    for action_index in range(plan_len):
        # either of the agents may or may not have an act at the last steps
        if len(human_act) > action_index:
            human_action = human_act[action_index]
            for delimeter in delimeters:
                human_action = " ".join(human_action.split(delimeter))
            human_action_split = human_action.split()
            if human_action_split[1] in ['put', 'put_in']:
                if human_action_split[1] == 'put' and human_action_split[4] in ['microwave']:
                    human_script_instruction = human_character + ' [putin] <' + human_action_split[3] + '> (' + id_dict[human_action_split[3]] + ') <' + human_action_split[4] + '> (' + id_dict[human_action_split[4]] + ')'
                else:
                    human_script_instruction = human_character + ' [putback] <' + human_action_split[3] + '> (' + id_dict[human_action_split[3]] + ') <' + human_action_split[4] + '> (' + id_dict[human_action_split[4]] + ')'
            elif human_action_split[1] in ['eat', 'drink']:
                human_script_instruction = None
            else:
                human_script_instruction = human_character + ' [' + human_action_split[1].replace('_','') + '] <' + human_action_split[3] + '> (' + id_dict[human_action_split[3]] + ')'
        else:
            human_script_instruction = None
        if len(ah_act) > action_index:
            ah_action = ah_act[action_index]
            for delimeter in delimeters:
                ah_action = " ".join(ah_action.split(delimeter))
            ah_action_split = ah_action.split()
            if ah_action_split[1] in ['put', 'put_in']:
                if ah_action_split[1] == 'put' and ah_action_split[4] in ['microwave']:
                    ah_script_instruction = ah_character + ' [putin] <' + ah_action_split[3] + '> (' + id_dict[ah_action_split[3]] + ') <' + ah_action_split[4] + '> (' + id_dict[ah_action_split[4]] + ')'
                else:
                    ah_script_instruction = ah_character + ' [putback] <' + ah_action_split[3] + '> (' + id_dict[ah_action_split[3]] + ') <' + ah_action_split[4] + '> (' + id_dict[ah_action_split[4]] + ')'
            else:
                ah_script_instruction = ah_character + ' [' + ah_action_split[1].replace('_','') + '] <' + ah_action_split[3] + '> (' + id_dict[ah_action_split[3]] + ')'
        else:
            ah_script_instruction = None
        script_instruction = (human_script_instruction + '|' + ah_script_instruction) if human_script_instruction and ah_script_instruction else (human_script_instruction if human_script_instruction else ah_script_instruction)
        script.append(script_instruction)
    return script

# look ahend for the given number of steps and return the new script
def get_human_action(current_script, timestep):
    prev_human_actions = ['None', 'None']
    prev_ah_actions = ['None', 'None']
    human_success = False
    ah_success = False
    # initiate env
    env_id = 1
    comm = comm_unity.UnityCommunication(port='8080')
    comm.reset(env_id)
    success, graph = comm.environment_graph()
    success1, message, success2, graph = clean_graph(comm, graph, ['chicken'])

    # Get nodes for differnt objects
    kitchen_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchen'][0]
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    stove_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    bench_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'bench'][0]
    breadslice_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'breadslice'][0]
    cutlets_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cutlets'][0]
    fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
    poundcake_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'poundcake'][0]
    waterglass_id = 103 # bathroom otherwise [node['id'] for node in graph['nodes'] if node['class_name'] == 'waterglass'][0]

    # collect and process data
    id_dict = {
        'kitchentable': str(kitchentable_id),
        'stove': str(stove_id),
        'microwave': str(microwave_id),
        'bench': str(bench_id),
        'breadslice': str(breadslice_id),
        'cutlets': str(cutlets_id),
        'fryingpan': str(fryingpan_id),
        'poundcake': str(poundcake_id),
        'waterglass': str(waterglass_id)
    }

    # Add human
    comm.add_character('Chars/Female1', initial_room='kitchen')
    # Add ad hoc agent
    comm.add_character('Chars/Male1', initial_room='kitchen')

    # success, graph = comm.environment_graph()
    
    for script_instruction in current_script:
        act_success, human_success, ah_success, message = comm.render_script([script_instruction], recording=False, skip_animation=True)
        if '|' in script_instruction:
            script_split = script_instruction.split('|')
            human_act = script_split[0]
            ah_act = script_split[1]
            human_act_split = (human_act.split(' '))[1:]
            ah_act_split = (ah_act.split(' '))[1:]
            if human_act_split == ah_act_split and human_act_split[0] == '[grab]':
                prev_human_actions.pop(0)
                prev_human_actions.append('None')
                if human_success:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append(script_split[0])
                else:
                    prev_human_actions.pop(0)
                    prev_human_actions.append('None')
            else:
                if human_success:
                    prev_human_actions.pop(0)
                    prev_human_actions.append(human_act)
                else:
                    prev_human_actions.pop(0)
                    prev_human_actions.append('None')
                if ah_success:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append(ah_act)
                else:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append('None')
        else:
            if human_success:
                prev_human_actions.pop(0)
                prev_human_actions.append(script_instruction)
            else:
                prev_human_actions.pop(0)
                prev_human_actions.append('None')
            prev_ah_actions.pop(0)
            prev_ah_actions.append('None')
        # Get the state observation
        success, graph = comm.environment_graph()
    
    for i in range(int(timestep)):
        human_action = predict_next_action(graph, prev_human_actions)
        human_action = convert_to_ASp(human_action)
        ah_action = run_ASP_ahagent(graph, prev_human_actions.copy(), prev_ah_actions.copy(), human_success, ah_success, ah_counter, env_id, id_dict, current_script.copy())
        script_instruction = generate_script([human_action], ah_action, id_dict, '<char0>', '<char1>')[0]
        act_success, human_success, ah_success, message = comm.render_script([script_instruction], recording=False, skip_animation=True)
        if '|' in script_instruction:
            script_split = script_instruction.split('|')
            human_act = script_split[0]
            ah_act = script_split[1]
            human_act_split = (human_act.split(' '))[1:]
            ah_act_split = (ah_act.split(' '))[1:]
            if human_act_split == ah_act_split and human_act_split[0] == '[grab]':
                prev_human_actions.pop(0)
                prev_human_actions.append('None')
                if human_success:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append(script_split[0])
                else:
                    prev_human_actions.pop(0)
                    prev_human_actions.append('None')
            else:
                if human_success:
                    prev_human_actions.pop(0)
                    prev_human_actions.append(human_act)
                else:
                    prev_human_actions.pop(0)
                    prev_human_actions.append('None')
                if ah_success:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append(ah_act)
                else:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append('None')
        else:
            if human_success:
                prev_human_actions.pop(0)
                prev_human_actions.append(script_instruction)
            else:
                prev_human_actions.pop(0)
                prev_human_actions.append('None')
            prev_ah_actions.pop(0)
            prev_ah_actions.append('None')
        # Get the state observation
        success, graph = comm.environment_graph()
    human_action = predict_next_action(graph, prev_human_actions)
    return human_action, graph, prev_human_actions

# Use this to return the actions, rules for the time steps except time step 0.
def get_active_rules(current_script, timestep):
    labels = ['previous_action','before_action','immediate_interacted_item','immediate_before_interacted_item','agent_pose_x','agent_pose_y','agent_pose_z',
            'agent_orientation_x','agent_orientation_y','agent_orientation_z','prox_kitchentable','prox_kitchencounter','microwave_status0','microwave_status1',
            'microwave_items','stove_status0','stove_status1','pan_items','no_of_items']
    agent_action, graph, prev_human_actions = get_human_action(current_script, timestep)
    values = process_graph(graph, prev_human_actions)
    # if (any(word in values[0] for word in ['bench','eat','drink'])) or (any(word in values[1] for word in ['bench','eat','drink'])):
    #     # no trianing data
    #     return None
    data_dic = dict(zip(labels,values))
    _, rules = trees.human_tree(data_dic)
    return rules

def do_new_actions(agentname, action, current_script, timestep):
    prev_human_actions = ['None', 'None']
    prev_ah_actions = ['None', 'None']
    human_success = False
    ah_success = False
    # initiate env
    env_id = 1
    comm = comm_unity.UnityCommunication(port='8080')
    comm.reset(env_id)
    success, graph = comm.environment_graph()
    success1, message, success2, graph = clean_graph(comm, graph, ['chicken'])

    # Get nodes for differnt objects
    kitchen_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchen'][0]
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    stove_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0]
    microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    bench_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'bench'][0]
    breadslice_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'breadslice'][0]
    cutlets_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cutlets'][0]
    fryingpan_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fryingpan'][0]
    poundcake_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'poundcake'][0]
    waterglass_id = 103 # bathroom otherwise [node['id'] for node in graph['nodes'] if node['class_name'] == 'waterglass'][0]

    # collect and process data
    id_dict = {
        'kitchentable': str(kitchentable_id),
        'stove': str(stove_id),
        'microwave': str(microwave_id),
        'bench': str(bench_id),
        'breadslice': str(breadslice_id),
        'cutlets': str(cutlets_id),
        'fryingpan': str(fryingpan_id),
        'poundcake': str(poundcake_id),
        'waterglass': str(waterglass_id)
    }

    # Add human
    comm.add_character('Chars/Female1', initial_room='kitchen')
    # Add ad hoc agent
    comm.add_character('Chars/Male1', initial_room='kitchen')

    for script_instruction in current_script:
        act_success, human_success, ah_success, message = comm.render_script([script_instruction], recording=False, skip_animation=True)
        if '|' in script_instruction:
            script_split = script_instruction.split('|')
            human_act = script_split[0]
            ah_act = script_split[1]
            human_act_split = (human_act.split(' '))[1:]
            ah_act_split = (ah_act.split(' '))[1:]
            if human_act_split == ah_act_split and human_act_split[0] == '[grab]':
                prev_human_actions.pop(0)
                prev_human_actions.append('None')
                if human_success:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append(script_split[0])
                else:
                    prev_human_actions.pop(0)
                    prev_human_actions.append('None')
            else:
                if human_success:
                    prev_human_actions.pop(0)
                    prev_human_actions.append(human_act)
                else:
                    prev_human_actions.pop(0)
                    prev_human_actions.append('None')
                if ah_success:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append(ah_act)
                else:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append('None')
        else:
            if human_success:
                prev_human_actions.pop(0)
                prev_human_actions.append(script_instruction)
            else:
                prev_human_actions.pop(0)
                prev_human_actions.append('None')
            prev_ah_actions.pop(0)
            prev_ah_actions.append('None')
        # Get the state observation
        success, graph = comm.environment_graph()
    
    for i in range(int(timestep)+1):
        if i == int(timestep) and agentname == 'human':
            human_action = [action]
        else:
            human_action = predict_next_action(graph, prev_human_actions)
            human_action = convert_to_ASp(human_action)
            if human_action == 'None':
                human_action = []
            else:
                human_action = [human_action]
        if i == int(timestep) and agentname == 'ahagent':
            ah_action = [action]
        else:
            ah_action = run_ASP_ahagent(graph, prev_human_actions.copy(), prev_ah_actions.copy(), human_success, ah_success, ah_counter, env_id, id_dict, current_script.copy())
        script_instruction = generate_script(human_action, ah_action, id_dict, '<char0>', '<char1>')[0]
        act_success, human_success, ah_success, message = comm.render_script([script_instruction], recording=False, skip_animation=True)
        if '|' in script_instruction:
            script_split = script_instruction.split('|')
            human_act = script_split[0]
            ah_act = script_split[1]
            human_act_split = (human_act.split(' '))[1:]
            ah_act_split = (ah_act.split(' '))[1:]
            if human_act_split == ah_act_split and human_act_split[0] == '[grab]':
                prev_human_actions.pop(0)
                prev_human_actions.append('None')
                if human_success:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append(script_split[0])
                else:
                    prev_human_actions.pop(0)
                    prev_human_actions.append('None')
            else:
                if human_success:
                    prev_human_actions.pop(0)
                    prev_human_actions.append(human_act)
                else:
                    prev_human_actions.pop(0)
                    prev_human_actions.append('None')
                if ah_success:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append(ah_act)
                else:
                    prev_ah_actions.pop(0)
                    prev_ah_actions.append('None')
        else:
            if human_success:
                prev_human_actions.pop(0)
                prev_human_actions.append(script_instruction)
            else:
                prev_human_actions.pop(0)
                prev_human_actions.append('None')
            prev_ah_actions.pop(0)
            prev_ah_actions.append('None')
        # Get the state observation
        success, graph = comm.environment_graph()
    return graph

def measure_the_depth(lst):
    max_count = 0
    count = 0
    for char in lst:
        # count brackets
        if char == '(':
            count += 1
        elif char == ')':
            count -= 1
        # update max count
        if count > max_count:
            max_count = count
    return max_count

def convert_to_ASp(human_action):
    if not human_action:
        return 'None'
    human_action = human_action.split('_')
    if len(human_action) == 2:
        human_action = 'occurs('+human_action[0]+'(human,'+human_action[1]+'),0)'
    else:
        if human_action[0] == 'putin' and human_action[2] in ['microwave']:
            human_action = 'occurs('+human_action[0]+'(human,'+human_action[1]+','+human_action[2]+'),0)'
        else:
            human_action = 'occurs(put(human,'+human_action[1]+','+human_action[2]+'),0)'
    return human_action

def convert_state_readable(state):
    state_table = state_all_process(state)
    return state_table