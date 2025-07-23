def human_tree(data_dic):
    rules = []
    if data_dic['previous_action'] == 'grab_cutlets':
        rules.append('previous_action = grab_cutlets')
        if data_dic['agent_pose_z'] <= 0.391014:
            rules.append('agent_pose_z <= 0.391014')
            if data_dic['prox_kitchentable'] <= 2.463021:
                rules.append('prox_kitchentable <= 2.463021')
                action = 'find_stove'
            elif data_dic['prox_kitchentable'] > 2.463021:
                rules.append('prox_kitchentable > 2.463021')
                if data_dic['agent_pose_x'] <= -7.125843:
                    rules.append('agent_pose_x <= -7.125843')
                    action = 'putback_cutlets_kitchentable'
                elif data_dic['agent_pose_x'] > -7.125843:
                    rules.append('agent_pose_x > -7.125843')
                    if data_dic['agent_pose_x'] <= -6.608572:
                        rules.append('agent_pose_x <= -6.608572')
                        action = 'find_fryingpan'
                    elif data_dic['agent_pose_x'] > -6.608572:
                        rules.append('agent_pose_x > -6.608572')
                        if data_dic['agent_orientation_y'] <= 0.987689:
                            rules.append('agent_orientation_y <= 0.987689')
                            action = 'find_fryingpan'
                        elif data_dic['agent_orientation_y'] > 0.987689:
                            rules.append('agent_orientation_y > 0.987689')
                            action = 'find_stove'
        elif data_dic['agent_pose_z'] > 0.391014:
            rules.append('agent_pose_z > 0.391014')
            action = 'putback_cutlets_kitchentable'
    elif data_dic['previous_action'] == 'find_fryingpan':
        rules.append('previous_action = find_fryingpan')
        action = 'putback_cutlets_fryingpan'
    elif data_dic['previous_action'] == 'putback_cutlets_fryingpan':
        rules.append('previous_action = putback_cutlets_fryingpan')
        if data_dic['agent_pose_z'] <= 2.36045:
            rules.append('agent_pose_z <= 2.36045')
            action = 'find_stove'
        elif data_dic['agent_pose_z'] > 2.36045:
            rules.append('agent_pose_z > 2.36045')
            if data_dic['agent_orientation_y'] <= 0.891007:
                rules.append('agent_orientation_y <= 0.891007')
                action = 'find_poundcake'
            elif data_dic['agent_orientation_y'] > 0.891007:
                rules.append('agent_orientation_y > 0.891007')
                action = 'find_waterglass'
    elif data_dic['previous_action'] == 'find_waterglass':
        rules.append('previous_action = find_waterglass')
        action = 'grab_waterglass'
    elif data_dic['previous_action'] == 'grab_waterglass':
        rules.append('previous_action = grab_waterglass')
        action = 'putback_waterglass_kitchentable'
    elif data_dic['previous_action'] == 'putback_waterglass_kitchentable':
        rules.append('previous_action = putback_waterglass_kitchentable')
        if data_dic['agent_pose_x'] <= -6.098466:
            rules.append('agent_pose_x <= -6.098466')
            if data_dic['agent_orientation_y'] <= 0.891007:
                rules.append('agent_orientation_y <= 0.891007')
                action = 'find_poundcake'
            elif data_dic['agent_orientation_y'] > 0.891007:
                rules.append('agent_orientation_y > 0.891007')
                action = 'find_stove'
        elif data_dic['agent_pose_x'] > -6.098466:
            rules.append('agent_pose_x > -6.098466')
            if data_dic['no_of_items'] <= 13:
                rules.append('no_of_items <= 13')
                action = 'find_cutlets'
            elif data_dic['no_of_items'] > 13:
                rules.append('no_of_items > 13')
                if data_dic['agent_pose_z'] <= 3.186223:
                    rules.append('agent_pose_z <= 3.186223')
                    action = 'find_cutlets'
                elif data_dic['agent_pose_z'] > 3.186223:
                    rules.append('agent_pose_z > 3.186223')
                    if data_dic['agent_orientation_y'] <= 0.891007:
                        rules.append('agent_orientation_y <= 0.891007')
                        action = 'find_breadslice'
                    elif data_dic['agent_orientation_y'] > 0.891007:
                        rules.append('agent_orientation_y > 0.891007')
                        action = 'find_cutlets'
    elif data_dic['previous_action'] == 'find_breadslice':
        rules.append('previous_action = find_breadslice')
        action = 'grab_breadslice'
    elif data_dic['previous_action'] == 'grab_breadslice':
        rules.append('previous_action = grab_breadslice')
        action = 'putback_breadslice_kitchentable'
    elif data_dic['previous_action'] == 'putback_breadslice_kitchentable':
        rules.append('previous_action = putback_breadslice_kitchentable')
        if data_dic['agent_pose_x'] <= -6.249997:
            rules.append('agent_pose_x <= -6.249997')
            action = 'find_bench'
        elif data_dic['agent_pose_x'] > -6.249997:
            rules.append('agent_pose_x > -6.249997')
            if data_dic['prox_kitchentable'] <= 1.620184:
                rules.append('prox_kitchentable <= 1.620184')
                action = 'find_poundcake'
            elif data_dic['prox_kitchentable'] > 1.620184:
                rules.append('prox_kitchentable > 1.620184')
                action = 'find_cutlets'
    elif data_dic['previous_action'] == 'find_poundcake':
        rules.append('previous_action = find_poundcake')
        action = 'grab_poundcake'
    elif data_dic['previous_action'] == 'grab_poundcake':
        rules.append('previous_action = grab_poundcake')
        if data_dic['prox_kitchencounter'] <= 2.51979:
            rules.append('prox_kitchencounter <= 2.51979')
            action = 'find_stove'
        elif data_dic['prox_kitchencounter'] > 2.51979:
            rules.append('prox_kitchencounter > 2.51979')
            if data_dic['agent_orientation_y'] <= 0.891007:
                rules.append('agent_orientation_y <= 0.891007')
                action = 'putback_poundcake_kitchentable'
            elif data_dic['agent_orientation_y'] > 0.891007:
                rules.append('agent_orientation_y > 0.891007')
                action = 'find_microwave'
    elif data_dic['previous_action'] == 'find_microwave': 
        rules.append('previous_action = find_microwave')
        action = 'open_microwave'
    elif data_dic['previous_action'] == 'open_microwave':
        rules.append('previous_action = open_microwave')
        if data_dic['action_before'] in ['find_cutlets','grab_cutlets','find_fryingpan','putback_cutlets_fryingpan','find_waterglass','grab_waterglass','putback_waterglass_kitchentable','find_breadslice','grab_breadslice','putback_breadslice_kitchentable','find_poundcake','grab_poundcake','find_microwave','open_microwave','putin_poundcake_microwave','close_microwave','switchon_microwave']:
            rules.append('action_before='+data_dic['action_before'])
            action = 'putin_poundcake_microwave'
        elif data_dic['action_before'] == 'switchoff_microwave':
            rules.append('action_before = switchoff_microwave')
            if data_dic['agent_orientation_y'] <= 0.999241:
                rules.append('agent_orientation_y <= 0.999241')
                if data_dic['agent_pose_z'] <= 3.47282:
                    rules.append('agent_pose_z <= 3.47282')
                    if data_dic['prox_kitchencounter'] <= 4.155138:
                        rules.append('prox_kitchencounter <= 4.155138')
                        if data_dic['agent_pose_x'] <= -8.370588:
                            rules.append('agent_pose_x <= -8.370588')
                            action = 'find_cutlets'
                        elif data_dic['agent_pose_x'] > -8.370588:
                            rules.append('agent_pose_x > -8.370588')
                            action = 'find_poundcake'
                    elif data_dic['prox_kitchencounter'] > 4.155138:
                        rules.append('prox_kitchencounter > 4.155138')
                        action = 'find_poundcake'
                elif data_dic['agent_pose_z'] > 3.47282:
                    rules.append('agent_pose_z > 3.47282')
                    if data_dic['agent_orientation_y'] <= 0.891007:
                        rules.append('agent_orientation_y <= 0.891007')
                        action = 'find_poundcake'
                    elif data_dic['agent_orientation_y'] > 0.891007:
                        rules.append('agent_orientation_y > 0.891007')
                        action = 'find_breadslice'
            elif data_dic['agent_orientation_y'] > 0.999241:
                rules.append('agent_orientation_y > 0.999241')
                action = 'find_stove'
        elif data_dic['action_before'] == 'putback_poundcake_kitchentable':
            rules.append('action_before = putback_poundcake_kitchentable')
            action = 'putin_poundcake_microwave'
        elif data_dic['action_before'] == 'find_stove':
            rules.append('action_before = find_stove')
            action = 'putin_poundcake_microwave'
        elif data_dic['action_before'] == 'switchon_stove':
            rules.append('action_before = switchon_stove')
            action = 'putin_poundcake_microwave'
        elif data_dic['action_before'] == 'putback_cutlets_kitchentable':
            rules.append('action_before = putback_cutlets_kitchentable')
            action = 'putin_poundcake_microwave'
    elif data_dic['previous_action'] == 'putin_poundcake_microwave':
        rules.append('previous_action = putin_poundcake_microwave')
        action = 'close_microwave'
    elif data_dic['previous_action'] == 'close_microwave':
        rules.append('previous_action = close_microwave')
        action = 'switchon_microwave'
    elif data_dic['previous_action'] == 'switchon_microwave':
        rules.append('previous_action = switchon_microwave')
        action = 'switchoff_microwave'
    elif data_dic['previous_action'] == 'switchoff_microwave':
        rules.append('previous_action = switchoff_microwave')
        action = 'open_microwave'
    elif data_dic['previous_action'] == 'putback_poundcake_kitchentable':
        rules.append('previous_action = putback_poundcake_kitchentable')
        if data_dic['prox_kitchentable'] <= 1.414475:
            rules.append('prox_kitchentable <= 1.414475')
            action = 'find_breadslice'
        elif data_dic['prox_kitchentable'] > 1.414475:
            rules.append('prox_kitchentable > 1.414475')
            if data_dic['agent_orientation_y'] <= 0.987689:
                rules.append('agent_orientation_y <= 0.987689')
                action = 'find_cutlets'
            elif data_dic['agent_orientation_y'] > 0.987689:
                rules.append('agent_orientation_y > 0.987689')
                action = 'find_stove'
    elif data_dic['previous_action'] == 'find_stove':
        rules.append('previous_action = find_stove')
        action = 'switchon_stove'
    elif data_dic['previous_action'] == 'switchon_stove':
        rules.append('previous_action = switchon_stove')
        if data_dic['prox_kitchentable'] <= 3.038775:
            rules.append('prox_kitchentable <= 3.038775')
            if data_dic['prox_kitchentable'] <= 1.451615:
                rules.append('prox_kitchentable <= 1.451615')
                action = 'find_waterglass'
            elif data_dic['prox_kitchentable'] > 1.451615:
                rules.append('prox_kitchentable > 1.451615')
                if data_dic['agent_pose_z'] <= 0.265652:
                    rules.append('agent_pose_z <= 0.265652')
                    action = 'find_cutlets'
                elif data_dic['agent_pose_z'] > 0.265652:
                    rules.append('agent_pose_z > 0.265652')
                    if data_dic['no_of_items'] <= 13:
                        rules.append('no_of_items <= 13')
                        action = 'find_poundcake'
                    elif data_dic['no_of_items'] > 13:
                        rules.append('no_of_items > 13')
                        action = 'find_cutlets'
        elif data_dic['prox_kitchentable'] > 3.038775:
            rules.append('prox_kitchentable > 3.038775')
            if data_dic['agent_orientation_y'] <= 0.951057:
                rules.append('agent_orientation_y <= 0.951057')
                action = 'find_fryingpan'
            elif data_dic['agent_orientation_y'] > 0.951057:
                rules.append('agent_orientation_y > 0.951057')
                if data_dic['agent_orientation_y'] <= 0.999241:
                    rules.append('agent_orientation_y <= 0.999241')
                    action = 'find_microwave'
                elif data_dic['agent_orientation_y'] > 0.999241:
                    rules.append('agent_orientation_y > 0.999241')
                    action = 'find_poundcake'
    elif data_dic['previous_action'] == 'find_cutlets':
        rules.append('previous_action = find_cutlets')
        action = 'grab_cutlets'
    elif data_dic['previous_action'] == 'putback_cutlets_kitchentable':
        rules.append('previous_action = putback_cutlets_kitchentable')
        if data_dic['agent_pose_x'] <= -6.197388:
            rules.append('agent_pose_x <= -6.197388')
            action = 'find_bench'
        elif data_dic['agent_pose_x'] > -6.197388:
            rules.append('agent_pose_x > -6.197388')
            if data_dic['agent_pose_z'] <= 3.381643:
                rules.append('agent_pose_z <= 3.381643')
                action = 'find_breadslice'
            elif data_dic['agent_pose_z'] > 3.381643:
                rules.append('agent_pose_z > 3.381643')
                if data_dic['agent_pose_x'] <= -4.917071:
                    rules.append('agent_pose_x <= -4.917071')
                    action = 'find_breadslice'
                elif data_dic['agent_pose_x'] > -4.917071:
                    rules.append('agent_pose_x > -4.917071')
                    action = 'find_poundcake'
    return action, rules