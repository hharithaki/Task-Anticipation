import ast
import re
import sys
import subprocess
import tree_rules
from sympy import sympify, symbols
import sklweka.jvm as jvm
import contextlib

delimeters = ['(', ')', ',']
vars = ['ahagent', 'human', 'cutlets', 'poundcake', 'breadslice', 'waterglass', 'microwave', 'stove', 'fryingpan', 'kitchentable', 'bench']
# find literals in the Answer Set totally, partially or not grounded
# input --> expression => literal in string format, ASfile - Sprach answer set
# output --> list of matched grounded terms retrived from the ASfile 
#  retrieves planned actions from an Answer set in chronological order ?????
def AnswerSetFinder(Expression, AnswerSetFile):
    if not re.search('[\>\<\=\!]',Expression):
        Expression = re.sub('I\+1', 'I', Expression)
        Expression = re.sub('\(', '\(', Expression)
        Expression = re.sub('\)', '\)', Expression)
        Expression = re.sub('[A-Z][0-9]?', '[a-z0-9_]+(?:\(.+?\))?', Expression)
        # this is added for belief will this cause problems for others?
        Expression = re.sub('not ', '-', Expression)
        literal = re.findall("(?<!-)"+Expression, AnswerSetFile)
    else:
        literal = [Expression]
    return literal

# retrieve axioms in the ASP program that contains a given Literal in its head or body
# inut --> ASPfile = ASP program, opption = head or body (what part of the axiom are you looking for?)
# output --> list of matched rules retrieved from the ASP program
def AxiomsFinder(Literal, ASPfile, option):
    if tree_rules.measure_the_depth(Literal) < 2:
        Literal = re.sub('([0-9]+)', '\\\s?[A-Z]+\+?[0-9]?', Literal)
        Literal = re.sub('\(', '\(', Literal)
        Literal = re.sub('\)', '\)', Literal)
    else:
        Literal = re.sub('([a-zA-Z0-9_]+)(?![a-zA-Z0-9_]*\()', '\\\s?[A-Z]+\+?[0-9]?', Literal)
        Literal = re.sub('\(', '\(', Literal)
        Literal = re.sub('\)', '\)', Literal)
    if option == 'head':
        axiom = re.findall(".*?"+"(?<!-)"+Literal+"\s?:-.*?\.", ASPfile)
    else:
        axiom = re.findall(".*:-.*"+"(?<!-)"+Literal+".*\.", ASPfile)
    return axiom    

# partially/totally ground literals in an axiom based on the ground of other literals from the same axiom
# input --#~> axiom (string), grounded_literal = one geounded term(string), option = head or body
# output --> list of grounded terms
def Grounder(axiom, grounded_literal, option):
    grounds = re.findall('([a-zA-Z0-9_]+)(?![a-zA-Z0-9_]*\()',grounded_literal)
    nonground_finder = re.sub('\(', '\(', grounded_literal)
    nonground_finder = re.sub('\)', '\)', nonground_finder)
    nonground_finder = re.sub('([a-zA-Z0-9_]+)(?![a-zA-Z0-9_]*\\\\\()', '\\\s?([a-zA-Z0-9_\+]+)(?![a-zA-Z0-9_]*\()', nonground_finder)
    variables = re.findall(nonground_finder, axiom) # find nonground_finder(=modified action strings) in axioms
    for idx, variable in enumerate(variables[0]):
        axiom = re.sub(variable+'(?![a-zA-Z0-9])', grounds[idx], axiom)
    if option == 'body':
        axiom = re.findall("(?<=:-)(.*?)\.", axiom)
    else:
        axiom = re.findall("(.*)(?=:-)", axiom)
    axiom = re.sub('\s','',axiom[0]) # remove space here for the next operation
    axiom = [re.sub('\.','',x) for x in re.split(':-|,(?=\s*[-a-zA-Z#0-9_]+?[\(\=\<\>\!])',axiom)]# if not x.startswith('#')]
    axiom = [re.sub('not','not ',x) for x in axiom]
    return axiom

# test if the body of a rule holds based on the set of grounded terms from the answer set
# input --> groundterms = sample list (or string) of grounded terms, rule = axiom(string)
# output --> true or false , mean body holds or not
def validateBody(groundTerms, rule):
    if not any(groundTerms):
        return False
    terms = re.split(':-|,(?=\s*[a-z#A-Z0-9\s_]+?[\(\=\<\>\!])',rule)
    terms = [re.sub(' ','\n?',re.sub('\(','\(',re.sub('\)','\)',term))) for term in terms[1:]]
    terms = [re.sub('[A-Z][0-9]?', '[a-z0-9_]+(?:\(.+?\))?', re.sub('\.','',term)) for term in terms if not re.search('not|[\>\<\=\!]',term)]
    ruleSatisfied = []
    for term in terms:
        groundTermValid = []
        if not (re.search('[\>\<\=\!]',term) or term.startswith('not') or term[2:].startswith('#')):
            for groundTerm in groundTerms:
                if len(groundTerm) > 1:
                    subGroundTermValid = []
                    for insidegroundTerm in groundTerm:
                        if not (re.search('[\>\<\=\!]',insidegroundTerm) or insidegroundTerm.startswith('not') or insidegroundTerm.startswith('#')):
                            subGroundTermValid.append(re.match(term, insidegroundTerm)!=None)
                    groundTermValid.append(any(subGroundTermValid))
                else:
                    if not (re.search('[\>\<\=\!]',groundTerm[0]) or groundTerm[0].startswith('not') or groundTerm[0].startswith('#')):
                        groundTermValid.append(re.match(term, groundTerm[0])!=None)
            ruleSatisfied.append(any(groundTermValid))
    return all(ruleSatisfied)

# # output --> sorted actions by time-step
def process_answerlist(AnswerSet):
    answer_list = AnswerSetFinder('occurs(A,I)', AnswerSet) # return only the actions occured from AnswerSet fi
    action_list = []
    for i in range(len(answer_list)):
        for element in answer_list:
            if re.search(rf',{i}\)$',element) != None:
                action_list.insert(i, element)
    return action_list
    
# get the goal of the agent
def getGoalTerm(ASPprogram):
    goal_axiom = AxiomsFinder('goal(I)', ASPprogram, 'head')
    goal_term = re.findall('holds\('+"(.*)"+',I\).',goal_axiom[0].split(':-',1)[1])[0]
    return goal_term

def whyAction(Filename, AnswerSet, ASPprogram, action, timestep):
    sorted_actions = process_answerlist(AnswerSet)
    action = re.sub("\s","",action)
    state_action = action
    action = re.sub("\(","\(",action)
    action = re.sub("\)","\)",action)
    action_query = [act for act in sorted_actions if re.search(action,act)!=None and re.search(',\s?'+timestep,act)!=None][0]
    exec_conds = [AxiomsFinder('-'+act, ASPprogram, 'head') for act in sorted_actions[sorted_actions.index(action_query)+1:]]
    exec_conds = [[Grounder(exe, sorted_actions[sorted_actions.index(action_query)+1:][ind],'body') for exe in exec_cond] for ind, exec_cond in enumerate(exec_conds)]
    exec_conds = [[item for x in exec_cond for item in x] for exec_cond in exec_conds]
    # replace the not with -
    exec_conds = [[item.replace('not ','-') for item in exec_cond] for exec_cond in exec_conds]
    exec_conds_init = [[re.sub(',\s?([0-9]+)', ','+str(timestep), x) for x in exec_cond] for exec_cond in exec_conds]
    exec_conds = [[re.sub(',\s?([0-9]+)', ','+str(int(timestep)+1), x) for x in exec_cond] for exec_cond in exec_conds]   
    # select only the conditions actually changed in the next timestep
    exec_conds = [[[AnswerSetFinder(x, AnswerSet), indout + int(timestep) + 1] for indin, x in enumerate(exec_cond) if AnswerSetFinder(x, AnswerSet)!=[] and AnswerSetFinder(exec_conds[indout][indin], AnswerSet)==[]] for indout, exec_cond in enumerate(exec_conds_init)]
    exec_conds = [item for cond in exec_conds for item in cond if cond!=[]]
    if len(exec_conds) == 0:
        explanation = [getGoalTerm(ASPprogram)]
        if (len(sorted_actions)-1) == sorted_actions.index(action_query): # last action
            explanation.append(action_query)
        else:
            explanation.append(sorted_actions[sorted_actions.index(action_query)+1])
    else:
        explanation = min(exec_conds, key=lambda x: x[1])
        explanation = [explanation[0][0], sorted_actions[explanation[1]]]
    return explanation

def whyNotAction(Filename, AnswerSet, ASPprogram, action, timestep):
    action = re.sub("\s","",action)
    action_query = "occurs("+re.findall('([a-z]+?\([a-zA-z0-9_,]+?\))', action)[0]+","+timestep+")"
    exec_conds = AxiomsFinder('-'+action_query, ASPprogram, 'head')
    exec_conds = [Grounder(exe, action_query,'body') for exe in exec_conds]
    exec_conds = [item for x in exec_conds for item in x]
    exec_conds = [AnswerSetFinder(x, AnswerSet) for x in exec_conds]
    exec_conds = [item for cond in exec_conds for item in cond if cond!=[] and not item.startswith('#')]
    exec_conds = [exec_cond for exec_cond in exec_conds if 'holds' in exec_cond]
    explanation = [exec_conds]
    return explanation

def whyBelief(AnswerSet, ASPprogram, belief):
    axiom = [[re.sub('-','',belief)] if re.search(r'-',belief) else [belief]]
    axioms = [[AxiomsFinder(ax, ASPprogram, 'head') for ax in axio][0] for axio in axiom]
    axioms_body = [[Grounder(ax, axiom[ind][0],'body') for ax in axi] for ind, axi in enumerate(axioms)]
    axioms_body = [item for x in axioms_body for item in x]
    axioms_body = [[AnswerSetFinder(x, AnswerSet) for x in ax_body] for ax_body in axioms_body]
    # axioms_body = [[AnswerSetFinder(('-'+x), AnswerSet) if (x.startswith('holds') and belief.startswith('-')) else AnswerSetFinder(x, AnswerSet) for x in axiom_body] for axiom_body in axioms_body]
    axioms_body = [[x for x in state if not (x==[] and len(state)>1)] for state in axioms_body] # remove empty
    rules = [item for x in axioms for item in x] # These are the reshaped axioms for being used to body validation
    axioms = [axiom for ind, axiom in enumerate(axioms_body) if validateBody(axiom, rules[ind])]
    axioms = [[item for x in state for item in x] for state in axioms if state!=[]]
    explanation = axioms if axioms!=[] else belief
    # consider the predict actions of the other agents
    # explanation = otherAgentActions('belief', belief, [explanation])
    return explanation

def whatOtherAgent(timestep, script_name):
    t = open(script_name,'r')
    current_script = t.read()
    t.close()
    current_script = ast.literal_eval(current_script) # last two actions in the script are the two actions for the human in step 0 and step 1
    current_script = current_script[:-2]
    agent_action, _, _ = tree_rules.get_human_action(current_script, timestep)
    explanation = agent_action
    return explanation

def whyOtherAgent(timestep, script_name):
    t = open(script_name,'r')
    current_script = t.read()
    t.close()
    current_script = ast.literal_eval(current_script) # last two actions in the script are the two actions for the human in step 0 and step 1
    current_script = current_script[:-2]
    rules = tree_rules.get_active_rules(current_script, timestep)
    return rules

def whatWorld(timestep, script_name):
    t = open(script_name,'r')
    current_script = t.read()
    t.close()
    current_script = ast.literal_eval(current_script) # last two actions in the script are the two actions for the human in step 0 and step 1
    current_script = current_script[:-2]
    _, graph, _ = tree_rules.get_human_action(current_script, timestep)
    explanation = tree_rules.convert_state_readable(graph)
    return explanation

def willAction(agentname, action, timestep, script_name):
    t = open(script_name,'r')
    current_script = t.read()
    t.close()
    current_script = ast.literal_eval(current_script) # last two actions in the script are the two actions for the human in step 0 and step 1
    current_script = current_script[:-2]
    graph = tree_rules.do_new_actions(agentname, action, current_script, timestep)
    explanation = tree_rules.convert_state_readable(graph)
    return explanation

# ................ main ..................

# run ASP file, save answer sets to a file
# answer_set = subprocess.check_output('java -jar sparc.jar set3/learner.sp -A -n 1',shell=True)
# answer_split = (answer_set.decode('ascii'))
# f = open('set3/answer_set','w')
# f.write(answer_split)
# f.close()
# example q: 'why did you move to 5,14 in timestep 3 ?'

question = sys.argv[1]
question = question.split()

pre_fix = 'occurs('
agent = 'ahagent'
question_type = question[1]
sub_action = question[3]
if question_type == 'did':
    if sub_action != 'not' and sub_action != 'believe':
        if sub_action not in ['put', 'putin']: # put, put in and put back
            object_ = question[4]
            timestep = question[7]
            action = pre_fix + sub_action + '(' + agent + ',' + object_ + '),' + timestep + ')'
        elif sub_action in ['put', 'putin']:
            object1_ = question[4]
            object2_ = question[6]
            timestep = question[9]
            action = pre_fix + sub_action + '(' + agent + ',' + object1_ + ',' + object2_ + '),' + timestep + ')'
    elif sub_action == 'not':
        new_sub_action = question[4]
        if new_sub_action not in ['put', 'putin']:
            object_ = question[5]
            timestep = question[8]
            action = pre_fix + new_sub_action + '(' + agent + ',' + object_ + '),' + timestep + ')'
        elif new_sub_action in ['put', 'putin']:
                object1_ = question[5]
                object2_ = question[7]
                timestep = question[10]
                action = pre_fix + new_sub_action + '(' + agent + ',' + object1_ + ',' + object2_ + '),' + timestep + ')'
    elif sub_action == 'believe':
        question = [item for item in question if (item != 'did' and item != 'you')]
elif sub_action == 'think':
    agent_name = question[5]
    if question[0] == 'Why':
        timestep = question[-2]
        new_sub_action = question[7]
        if new_sub_action not in ['put', 'putin']:
            object_ = question[8]
            action = pre_fix + new_sub_action + '(' + agent + ',' + object_ + '),' + timestep + ')' 
        elif new_sub_action in ['put', 'putin']:
            object1_ = question[8]
            object2_ = question[10]
            action = pre_fix + new_sub_action + '(' + agent + ',' + object1_ + ',' + object2_ + '),' + timestep + ')'
    elif question[0] == 'What':
        timestep = question[-2]
elif question_type == 'will':
    if sub_action == 'world':
        timestep = question[-2]
    elif sub_action == 'if':
        timestep = question[-2]
        if question[4] == 'you':
            new_sub_action = question[5]
            if new_sub_action not in ['put', 'putin']:
                object_ = question[6]
                action = pre_fix + new_sub_action + '(' + agent + ',' + object_ + '),' + timestep + ')'
            elif new_sub_action in ['put', 'putin']:
                object1_ = question[6]
                object2_ = question[8]
                action = pre_fix + new_sub_action + '(' + agent + ',' + object1_ + ',' + object2_ + '),' + timestep + ')'
        elif question[5] == 'human':
            agent = question[5]
            new_sub_action = question[6]
            if new_sub_action not in ['put', 'putin']:
                object_ = question[7]
                action = pre_fix + new_sub_action + '(' + agent + ',' + object_ + '),' + timestep + ')'
            elif new_sub_action in ['put', 'putin']:
                object1_ = question[7]
                object2_ = question[9]
                action = pre_fix + new_sub_action + '(' + agent + ',' + object1_ + ',' + object2_ + '),' + timestep + ')'
else:
    print('Please re-phrase the question to match with the template')
# openning Answer set file
Filename = 'asp_149_2.sp'
AnaswerSetName = 'answer_149'
script_name = 'script_149.txt'
t = open(Filename,'r')
ASPprogram = t.read()
t.close()

f1 = open(AnaswerSetName,'r')
AnswerSet = f1.read()
f1.close()

if question_type == 'did':
    if sub_action == 'not':
        explanation = whyNotAction(Filename, AnswerSet, ASPprogram, action, timestep)
        print(explanation)
    elif sub_action == 'believe':
        belief = question[2]
        explanation = whyBelief(AnswerSet, ASPprogram, belief)
        print(explanation) 
    else:
        explanation = whyAction(Filename, AnswerSet, ASPprogram, action, timestep)
        print(explanation)
elif question_type == 'do' and question[0] == 'What':
    jvm.start()
    explanation = whatOtherAgent(timestep, script_name)
    jvm.stop()
    print(explanation)
elif question_type == 'do' and question[0] == 'Why':
    jvm.start()
    explanation = whyOtherAgent(timestep, script_name)
    jvm.stop()
    print(explanation)
elif question_type == 'will':
    if sub_action == 'world':
        jvm.start()
        explanation = whatWorld(timestep, script_name)
        jvm.stop()
        print(explanation)
    elif sub_action == 'if':
        jvm.start()
        explanation = willAction(agent, action, timestep, script_name)
        jvm.stop()
        print(explanation)
else:
    print('Please re-phrase the question to match with the template')