import csv
from simulation.unity_simulator import comm_unity
import utils
import time
import sklweka.jvm as jvm
import numpy as np
from copy import deepcopy

# initiate the simluator
comm = comm_unity.UnityCommunication(port='8080')

jvm.start()

# select the environment
env_id = 6

tasks = [
    ['Breakfast_weekend', 'Activities', 'Serve_snacks', 'Clean_kitchen'],
    ['Breakfast_weekend', 'Make_table', 'Lunch', 'Clean_dishes'],
    ['Breakfast_weekday', 'Pack_bag', 'Lunch', 'Clean_kitchen'],
    ['Breakfast_weekday', 'Workstation', 'Coffee', 'Serve_snacks'],
    ['Breakfast_weekday', 'Pack_bag', 'Lunch', 'Clean_kitchen']
]

all_flags = [
{'weekday': False, 'guests': False},
{'weekday': False, 'guests': True},
{'weekday': True, 'office': True, 'lunch': True},
{'weekday': True, 'office': False, 'lunch': False},
{'weekday': True, 'office': True, 'lunch': True}
]

for iteration_ in range(5):
    time.sleep(5)
    comm.reset(env_id)

    # Get the state observation
    success, graph = comm.environment_graph()

    # remove unnecessary objects and prepare the domain
    success1, message, success2, graph = utils.clean_graph(comm, graph, ['chicken'])

    # Get nodes for differnt objects
    cereal_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cereal'][0]
    breadslice_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'breadslice'][0]
    banana_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'bananas'][0]
    plum_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'plum'][0]
    cupcake_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cupcake'][0]
    cutlets_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cutlets'][0]
    chips_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'chips'][0]
    candybar_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'candybar'][0]
    milk_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'milk'][0]
    wine_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'wine'][0]
    juice_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'juice'][0]
    cellphone_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'cellphone'][1]
    dishwasher_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'dishwasher'][0]
    fridge_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'fridge'][0]
    computer_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'computer'][0]
    coffeemaker_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'coffeemaker'][0]
    plate_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'plate'][0]
    waterglass_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'waterglass'][1]
    mug_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'mug'][0]
    coffeepot_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'coffeepot'][0]
    kitchentable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchentable'][0]
    livingcoffeetable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'coffeetable'][0] # livingroom
    livingdesk_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'desk'][0] # livingroom
    bedcoffeetable_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'coffeetable'][1] # bedroom
    beddesk_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'desk'][2] # bedroom - two desks here
    book_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'book'][0]
    boardgame_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'boardgame'][0]
    tvstand_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'tvstand'][0]
    counterone_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'kitchencounter'][0] # counter_one
    countertwo_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'stove'][0] # counter_two
    counterthree_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'dishwasher'][0] # counter_thre
    apple_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'apple'][11]
    chair_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'chair'][2]
    bookshelf_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'bookshelf'][2]
    microwave_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'microwave'][0]
    sofa_id = [node['id'] for node in graph['nodes'] if node['class_name'] == 'sofa'][0]

    # collect and process data
    id_dict = {
        'cereal': cereal_id,
        'breadslice': breadslice_id,
        'bananas': banana_id,
        'plum': plum_id,
        'cupcake': cupcake_id,
        'cutlets': cutlets_id,
        'chips': chips_id,
        'candybar': candybar_id,
        'milk': milk_id,
        'wine': wine_id,
        'juice': juice_id,
        'dishwasher': dishwasher_id,
        'fridge': fridge_id,
        'computer': computer_id,
        'coffeemaker': coffeemaker_id,
        'plate': plate_id,
        'waterglass': waterglass_id,
        'mug': mug_id,
        'coffeepot': coffeepot_id,
        'kitchentable': kitchentable_id,
        'livingroom_coffeetable': livingcoffeetable_id,
        'livingroom_desk': livingdesk_id,
        'bedroom_coffeetable': bedcoffeetable_id,
        'bedroom_desk': beddesk_id,
        'book': book_id,
        'boardgame': boardgame_id,
        'kitchen_smalltable': tvstand_id,
        'counter_one': counterone_id,
        'stove': countertwo_id,
        'counter_three': counterthree_id,
        'cellphone': cellphone_id,
        'apple': apple_id,
        'chair': chair_id,
        'bookshelf': bookshelf_id,
        'microwave': microwave_id,
        'sofa': sofa_id,
        'floor': 204
    }

    # Add human
    comm.add_character('Chars/Female1', initial_room='kitchen')
    # Add ad hoc agent
    comm.add_character('Chars/Male1', initial_room='kitchen')
    # Add ad hoc agent
    comm.add_character('Chars/Female2', initial_room='kitchen')

    num_agents = 3

    # Select an initial state
    initialscript, _, _, weekday = utils.select_initialstate(id_dict, '<char0>', '<char1>', '<char2>') # , all_rows[iteration_]
    # Prepare domain
    for script_instruction in initialscript:
        act_success, success_msgs, message = comm.render_script([script_instruction], recording=False, skip_animation=True)
        human_success = success_msgs[0]
        ah_success = success_msgs[1]
        add_success = success_msgs[2]
        # only the first two agents are performing the actions in the initialscript
        if not act_success or not human_success or not ah_success:
            exit(1)
        success, graph = comm.environment_graph()

    if not act_success or not human_success or not ah_success:
        exit(1)
    # dependency in virtualhome
    human_at_dict = {
        'kitchentable': 0,
        'livingroom_coffeetable': 0,
        'bedroom_coffeetable': 0,
        'livingroom_desk': 0,
        'bedroom_desk': 0,
        'counter_one': 0,
        'counter_three': 0,
        'kitchen_smalltable': 0,
        'kitchen': 0
    }
    ah_at_dict = {
        'kitchentable': 0,
        'livingroom_coffeetable': 0,
        'bedroom_coffeetable': 0,
        'livingroom_desk': 0,
        'bedroom_desk': 0,
        'counter_one': 0,
        'counter_three': 0,
        'kitchen_smalltable': 0,
        'kitchen': 0
    }
    add_at_dict = {
        'kitchentable': 0,
        'livingroom_coffeetable': 0,
        'bedroom_coffeetable': 0,
        'livingroom_desk': 0,
        'bedroom_desk': 0,
        'counter_one': 0,
        'counter_three': 0,
        'kitchen_smalltable': 0,
        'kitchen': 0
    }

    # ---------------------------------- START: AD HOC TEAMWORK ---------------------------------- #
    step = 0
    task_counter = 0
    prev_actions = []
    for agent in range(num_agents):
        prev_actions.append(['None', 'None'])
    prev_tasks = []
    for agent in range(num_agents):
        prev_tasks.append(['None', 'None'])
    current_script = initialscript # since the ids do not change
    # get the tasks for the human
    # human_tasks, flags = utils.get_human_tasks(weekday)
    all_tasks = tasks[iteration_]
    flags = all_flags[iteration_]
    print(all_tasks, flags)
    # setting last task for the human for using the complete goal
    human_last_task = False
    ah_last_task = False
    add_last_task = False

    # agent number dependent parameters
    # define success parameter for each of the agentsin the env
    human_success = False
    ah_success = False
    add_success = False
    # for storing old actions that can be used in a unsuccessful planning
    old_human_actions = []
    old_ah_actions = []
    old_add_actions = []
    # sucess of the last task
    ah_sub_goal_success = True # for the first time step
    add_sub_goal_success = True # for the first time step
    # oracle tasks for the agents
    human_tasks = deepcopy(all_tasks)
    ah_agent_tasks = deepcopy(all_tasks)
    add_agent_tasks = deepcopy(all_tasks)
    # fail count for ad hoc agent plan fails
    fail_count = 0
    completed_tasks = []
    total_tasks = len(all_tasks)
    ahllm_task_list = []
    addllm_task_list = []

    # write tasks and flags to file so that they can be used as examples for LLM later on
    with open('llm_example_data.csv', 'a', newline='') as f:  
        writer = csv.writer(f)
        temp_tasks = [utils.map_ASP_goal_LLM_text(task) for task in all_tasks]
        writer.writerow([temp_tasks, flags])

    while human_tasks:
        if len(human_tasks) == 1:
            human_last_task = True
        sub_goal = human_tasks.pop(0)
        human_sub_goal_success = False
        while (not human_sub_goal_success):
            # ad hoc agent 1
            if ah_sub_goal_success and (len(completed_tasks)+len(ahllm_task_list) < total_tasks-1) and (not ahllm_task_list):
                ahllm_task_list = utils.get_llm_next_task(flags, completed_tasks)
                ahllm_task_list = utils.get_ordered_tasks(flags, ahllm_task_list.copy())

            if ah_sub_goal_success:
                if len(ah_agent_tasks) == 1:
                    ah_last_task = True
                if ah_agent_tasks:
                    ah_current_task = ah_agent_tasks.pop(0)
                else:
                    ah_current_task = sub_goal
                ah_ASP_goal = utils.map_goal_ASP(ah_current_task)
                ah_const_timeout = np.array(utils.get_const_timeout(ah_current_task))
                ah_current_task = [utils.map_ASP_goal_LLM_text(ah_current_task)]
                # print('ah_ASP_goal1 -', ah_ASP_goal)
                if len(ahllm_task_list) > 1:
                    if ahllm_task_list[0] == ah_current_task[0]:
                        ahllm_task_list.pop(0)
                        future_task = ahllm_task_list.pop(0) # ahllm_task_list[1]
                        ah_current_task = [ah_current_task[0],future_task]
                        ah_ASP_goal = ah_ASP_goal + ',' + utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(future_task))
                        ah_const_timeout = ah_const_timeout + np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(future_task)))
                        # print('ah_ASP_goal12 -', ah_ASP_goal)
                    else:
                        future_task = ahllm_task_list.pop(0)
                        ah_current_task = [ah_current_task[0],future_task]
                        ah_ASP_goal = ah_ASP_goal + ',' + utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(future_task))
                        ah_const_timeout = ah_const_timeout + np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(future_task))) 
                        # print('ah_ASP_goal13 -', ah_ASP_goal)     
                elif ahllm_task_list:
                    if ahllm_task_list[0] != ah_current_task[0]:
                        future_task = ahllm_task_list.pop(0)
                        ah_current_task = [ah_current_task[0],future_task]
                        ah_ASP_goal = ah_ASP_goal + ',' + utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(future_task))
                        ah_const_timeout = ah_const_timeout + np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(future_task)))
                    # print('ah_ASP_goal4 -', ah_ASP_goal)

            # ah hoc agent 2
            if add_sub_goal_success and (len(completed_tasks)+len(addllm_task_list) < total_tasks-1) and (not addllm_task_list):
                addllm_task_list = utils.get_llm_next_task(flags, completed_tasks)
                addllm_task_list = utils.get_ordered_tasks(flags, addllm_task_list.copy())

            if add_sub_goal_success:
                if len(add_agent_tasks) == 1:
                    add_last_task = True
                if add_agent_tasks:
                    add_current_task = add_agent_tasks.pop(0)
                else:
                    add_current_task = sub_goal
                add_ASP_goal = utils.map_goal_ASP(add_current_task)
                add_const_timeout = np.array(utils.get_const_timeout(add_current_task))
                add_current_task = [utils.map_ASP_goal_LLM_text(add_current_task)]

                if len(addllm_task_list) > 1:
                    if addllm_task_list[0] == add_current_task[0]:
                        addllm_task_list.pop(0)
                        future_task = addllm_task_list.pop(0) # addllm_task_list[1]
                        add_current_task = [add_current_task[0],future_task]
                        add_ASP_goal = add_ASP_goal + ',' + utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(future_task))
                        add_const_timeout = add_const_timeout + np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(future_task)))
                    else:
                        future_task = addllm_task_list.pop(0)
                        add_current_task = [add_current_task[0],future_task]
                        add_ASP_goal = add_ASP_goal + ',' + utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(future_task))
                        add_const_timeout = add_const_timeout + np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(future_task)))
                elif addllm_task_list:
                    if addllm_task_list[0] != add_current_task[0]:
                        future_task = addllm_task_list.pop(0)
                        add_current_task = [add_current_task[0],future_task]
                        add_ASP_goal = add_ASP_goal + ',' + utils.map_goal_ASP(utils.map_LLM_text_ASP_goal(future_task))
                        add_const_timeout = add_const_timeout + np.array(utils.get_const_timeout(utils.map_LLM_text_ASP_goal(future_task)))
                
            human_actions, ah_fluents, add_fluents, common_fluents, human_sub_goal_success = utils.run_ASP_human(graph, id_dict, sub_goal, human_at_dict, ah_at_dict, add_at_dict, False)
            if human_actions is None:
                print('cannot find a satisifying solution within the given const n value.')
                human_actions = deepcopy(old_human_actions)

            # check whether sub goal is achived
            if human_sub_goal_success:
                task_counter = task_counter+1
                prev_tasks[0].pop(0)
                prev_tasks[0].append(sub_goal)
                completed_tasks = utils.add_unique([utils.map_ASP_goal_LLM_text(sub_goal)],completed_tasks)
                continue
            
            # add ah agent
            ah_actions, ah_sub_goal_success, _ = utils.run_ASP_ahagent(graph, sub_goal, ah_ASP_goal, ah_const_timeout, deepcopy(prev_tasks), flags, ah_fluents, common_fluents, deepcopy(prev_actions), env_id, id_dict, deepcopy(current_script), False, num_agents, 1)

            # add other agent
            add_actions, add_sub_goal_success, _ = utils.run_ASP_ahagent(graph, sub_goal, add_ASP_goal, add_const_timeout, deepcopy(prev_tasks), flags, add_fluents, common_fluents, deepcopy(prev_actions), env_id, id_dict, deepcopy(current_script), False, num_agents, 2)

            if ah_actions is None:
                print('cannot find a satisifying solution within the given timeoutvalue.')
                if fail_count > 0 or ah_last_task:
                    ah_actions = []
                else:
                    ah_actions = deepcopy(old_ah_actions)

            if ah_sub_goal_success:
                for item in ah_current_task:
                    if utils.map_LLM_text_ASP_goal(item) in all_tasks:
                        completed_tasks = utils.add_unique([item],completed_tasks)
                if not ah_last_task:
                    continue

            if add_actions is None:
                print('cannot find a satisifying solution within the given timeoutvalue.')
                if fail_count > 0 or add_last_task:
                    add_actions = []
                else:
                    add_actions = deepcopy(old_add_actions)

            if add_sub_goal_success:
                for item in add_current_task:
                    if utils.map_LLM_text_ASP_goal(item) in all_tasks:
                        completed_tasks = utils.add_unique([item],completed_tasks)
                if not add_last_task:
                    continue
            
            # print(human_actions[0], ah_actions[0], add_actions[0])
            script = utils.generate_script(human_actions[0], ah_actions[0] if ah_actions else None, add_actions[0] if add_actions else None, id_dict, '<char0>', '<char1>', '<char2>')

            if not script:
                if fail_count > 0:
                    break
                continue
            current_script.append(script[0])

            act1, act2, act3 = None, None, None
            if '|' in script[0]:
                script_split = script[0].split('|')
                if len(script_split) == 3: # all three agents has actions
                    act1, act2, act3 = script_split
                elif len(script_split) == 2:
                    pairs = {
                        ('char0', 'char1'): ('act1', 'act2'),
                        ('char0', 'char2'): ('act1', 'act3'),
                        ('char1', 'char2'): ('act2', 'act3')
                    }
                    for (char_a, char_b), (act_a, act_b) in pairs.items():
                        if char_a in script_split[0] and char_b in script_split[1]:
                            locals()[act_a] = script_split[0]
                            locals()[act_b] = script_split[1]
            else:
                for i, char in enumerate(['char0', 'char1', 'char2']):
                    if char in script[0]:
                        locals()[f'act{i+1}'] = script[0]

            act1_split = (act1.split(' '))[1:] if act1 else None # ['[grab]', '<bananas>', '(251)']
            act2_split = (act2.split(' '))[1:] if act2 else None
            act3_split = (act3.split(' '))[1:] if act3 else None

            # worksround to overcome virtualhome random behaviour issues
            if act1 and act2 and act3 and act1_split == act2_split == act3_split and 'grab' in act1:
                act1 = None
                act1_split = None
                act2 = None
                act2_split = None
                script[0] = act3
            if act1 and act2 and act1_split == act2_split and 'grab' in act1:
                act1 = None
                act1_split = None
                script[0] = act2 + '|' + act3
            if act1 and act3 and act1_split == act3_split and 'grab' in act1:
                act1 = None
                act1_split = None
                script[0] = act2 + '|' + act3
            if act2 and act3 and act2_split == act3_split and 'grab' in act2:
                script[0] = act1 + '|' + act3
                act2 = None
                act2_split = None

            if act1:
                # for human
                values = utils.process_state_new(graph, sub_goal, prev_tasks[0].copy(), prev_actions[0].copy(), flags, id_dict, 0, num_agents, act1)
                if values:
                    with open('state_0.csv', 'a', newline='') as f:  
                        writer = csv.writer(f)
                        writer.writerow(values)
                prev_actions[0].pop(0)
                prev_actions[0].append(act1)
            else:
                prev_actions[0].pop(0)
                prev_actions[0].append('None')
            if act2:
                # for ad hoc agent
                values = utils.process_state_new(graph, sub_goal, prev_tasks[1].copy(), prev_actions[1].copy(), flags, id_dict, 1, num_agents, act2)
                if values:
                    with open('state_1.csv', 'a', newline='') as f:  
                        writer = csv.writer(f)
                        writer.writerow(values)
                prev_actions[1].pop(0)
                prev_actions[1].append(act2)
            else:
                prev_actions[1].pop(0)
                prev_actions[1].append('None')
            if act3:
                # for add agent
                values = utils.process_state_new(graph, sub_goal, prev_tasks[2].copy(), prev_actions[2].copy(), flags, id_dict, 2, num_agents, act3)
                if values:
                    with open('state_2.csv', 'a', newline='') as f:  
                        writer = csv.writer(f)
                        writer.writerow(values)
                prev_actions[2].pop(0)
                prev_actions[2].append(act3)
            else:
                prev_actions[2].pop(0)
                prev_actions[2].append('None')

            print(script[0])

            act_success, success_msgs, message = comm.render_script([script[0]], recording=False, skip_animation=True)
            human_success = success_msgs[0]
            ah_success = success_msgs[1]
            add_success = success_msgs[2]

            if human_success:
                if human_actions and act1 and ('move' in human_actions[0]):
                    location = utils.get_location(act1_split[2][1:-1], id_dict)
                    for key, value in human_at_dict.items():
                        if key == location or (key == 'livingroom_desk' and location == 'bookshelf') or (key == 'kitchen' and location == 'fridge'):
                            human_at_dict[key] = 1
                        else:
                            human_at_dict[key] = 0
            else:
                print('############################################## ACTION FAIL ##############################################')
                fail_count += 1
            if ah_success:
                if ah_actions and act2 and ('move' in ah_actions[0]):
                    location = utils.get_location(act2_split[2][1:-1], id_dict)
                    for key, value in ah_at_dict.items():
                        if key == location or (key == 'livingroom_desk' and location == 'bookshelf') or (key == 'kitchen' and location == 'fridge'):
                            ah_at_dict[key] = 1
                        else:
                            ah_at_dict[key] = 0
            else:
                print('############################################## ACTION FAIL ##############################################')
                fail_count += 1
            if add_success:
                if add_actions and act3 and ('move' in add_actions[0]):
                    location = utils.get_location(act3_split[2][1:-1], id_dict)
                    for key, value in add_at_dict.items():
                        if key == location or (key == 'livingroom_desk' and location == 'bookshelf') or (key == 'kitchen' and location == 'fridge'):
                            add_at_dict[key] = 1
                        else:
                            add_at_dict[key] = 0
            else:
                print('############################################## ACTION FAIL ##############################################')
                fail_count += 1
            if fail_count > 5:
                break
            # get the graph
            success, graph = comm.environment_graph()
            # Previous action of the agent (include multiple actions in their particular order?)
            step = step+1
            if human_actions and len(human_actions) > 1:
                old_human_actions = human_actions[1:]
            if ah_actions and len(ah_actions) > 1:
                old_ah_actions = ah_actions[1:]
            if add_actions and len(add_actions) > 1:
                old_add_actions = add_actions[1:]
    print(step, task_counter)
jvm.stop()
