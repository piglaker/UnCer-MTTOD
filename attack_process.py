import json
import random
import re
from tqdm import tqdm

path = "./data/MultiWOZ_2.0/processed/"

names= ["train","test","dev"]
#names= ["test"]

def save_json(obj, save_path, indent=4):
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent)

def random_run(probability):
    """probability% run func(*args)"""
    list = []
    for i in range(probability):
        list.append(1)
    for x in range(100 - probability):
        list.append(0)
    a = random.choice(list)
    return a

#the template for replace user
#template4user =  ['template 1————————','template 2——————————','template 3——————————','template 4——————————','template 5——————————','template 6——————————']
template4user = [""]#["ahhh", "ohnbjj", "route", "uijinob", "me", "what"]
#the template for replace user
template4resp = ['sorry, could you please repeat that XXXXXX ? ',\
    'excuse me, could you tell me XXXXXX.',\
        'i am sorry i do not understand what you just said. please repeat the XXXXXX.', \
        'oh, I am sorry about that. excuse me, What XXXXXX do you mean ?']

#The probability that the same turn will be broken 2/3 times
probability_2 = 15
probability_3 = 5
#the number of broken turns per session
k = 2

def process(all_data):
    data = {}
    for dialog_id, session in all_data.items():
        dial = {'goal': session['goal'], 'log': []}
        single_turn = {}
        turn_nums = 0
        turn_list = [i for i in range(0,len(session['log']))]
        if k <= len(turn_list):
            attack_turn = random.sample(turn_list,k)
        #print(attack_turn)
        attact_count = 0
        for turn_id,turn in enumerate(session['log']):
            user_old = turn['user']
            user_delex_old = turn['user_delex']
            p1 = re.compile(r'[[](.*?)[]]', re.S)
            value_list = re.findall(p1,user_delex_old)

            #if random_run(probability_1) == 1 and len(value_list)!= 0:
            if turn_id in attack_turn and len(value_list)!=0:
                repvalue = '[' + random.choice(value_list) + ']'
                repstr = random.choice(template4user)

                user_delex_old_list = user_delex_old.split(' ')

                #ignore some special ones like "[value_type]s"
                try:
                    if user_delex_old_list.index(repvalue) != 0:
                        prev = user_delex_old_list[user_delex_old_list.index(repvalue)-1]
                    else:
                        prev = ''
                    if user_delex_old_list.index(repvalue)!=len(user_delex_old_list)-1:
                        post = user_delex_old_list[user_delex_old_list.index(repvalue)+1]
                    else:
                        post = ''
                except:
                    dial['log'].append(single_turn_load(turn, turn_nums))
                    turn_nums += 1
                    continue

                #ignore some ( ) or [value] with [value]
                try:
                    if prev == "":
                        user_prerep = re.findall("{}(.*?)\s{}".format(prev,post),user_old)
                    elif post =="":
                        user_prerep = re.findall("{}\s(.*?){}".format(prev, post), user_old)
                    else:
                        user_prerep = re.findall("{}\s(.*?)\s{}".format(prev, post), user_old)
                except:
                    dial['log'].append(single_turn_load(turn, turn_nums))
                    turn_nums += 1
                    continue
                if list(set(user_prerep)) == [''] or list(set(user_prerep)) == []:
                    dial['log'].append(single_turn_load(turn, turn_nums))
                    turn_nums += 1
                    continue
                user_delex_new = user_delex_old.replace(repvalue,'')
                try:
                    user_new = user_old.replace("".join(user_prerep),repstr)
                except:
                    dial['log'].append(single_turn_load(turn, turn_nums))
                    turn_nums += 1
                    continue
                replace4resp = random.choice(template4resp)
                resp = replace4resp.replace('XXXXXX',"".join(re.findall("[_](.*?)[]]",repvalue)))
                nodelx_resp = replace4resp.replace('XXXXXX',repvalue)

                attact_count +=1
                single_turn['user'] = user_new
                single_turn['user_delex'] = user_delex_new
                single_turn['resp'] = resp
                single_turn['nodelx_resp'] = nodelx_resp
                single_turn['pointer'] = turn['pointer']
                single_turn['match'] = turn['match']
                single_turn['constraint'] = turn['constraint']
                single_turn['cons_delex'] = turn['cons_delex']
                single_turn['sys_act'] = turn['sys_act']
                single_turn['turn_num'] = turn_nums
                turn_nums += 1
                single_turn['turn_domain'] = turn['turn_domain']
                dial['log'].append(single_turn)
                single_turn = {}
                if random_run(probability_2) == 1:
                    attact_count += 1
                    user_new_2 = "".join(re.findall("[_](.*?)[]]",repvalue)) +" "+ "is" +" "+ random.choice(template4user) + ' .'
                    user_delex_new_2 = "".join(re.findall("[_](.*?)[]]", repvalue)) +' '+ "is"
                    replace4resp = random.choice(template4resp)
                    resp = replace4resp.replace('XXXXXX',"".join(re.findall("[_](.*?)[]]",repvalue)))
                    nodelx_resp = replace4resp.replace('XXXXXX',repvalue)
                    #
                    single_turn['user'] = user_new_2
                    single_turn['user_delex'] = user_delex_new_2
                    single_turn['resp'] = resp
                    single_turn['nodelx_resp'] = nodelx_resp
                    single_turn['pointer'] = turn['pointer']
                    single_turn['match'] = turn['match']
                    single_turn['constraint'] = turn['constraint']
                    single_turn['cons_delex'] = turn['cons_delex']
                    single_turn['sys_act'] = turn['sys_act']
                    single_turn['turn_num'] = turn_nums
                    turn_nums += 1
                    single_turn['turn_domain'] = turn['turn_domain']
                    dial['log'].append(single_turn)
                    single_turn = {}

                    if random_run(probability_3) == 1:
                        attact_count += 1
                        user_new_3 = "".join(re.findall("[_](.*?)[]]", repvalue)) +" "+"is" + " "+ random.choice(template4user) + ' .'
                        user_delex_new_3 = "".join(re.findall("[_](.*?)[]]", repvalue)) + "is"
                        replace4resp = random.choice(template4resp)
                        resp = replace4resp.replace('XXXXXX', "".join(re.findall("[_](.*?)[]]", repvalue)))
                        nodelx_resp = replace4resp.replace('XXXXXX', repvalue)
                        #
                        single_turn['user'] = user_new_3
                        single_turn['user_delex'] = user_delex_new_3
                        single_turn['resp'] = resp
                        single_turn['nodelx_resp'] = nodelx_resp
                        single_turn['pointer'] = turn['pointer']
                        single_turn['match'] = turn['match']
                        single_turn['constraint'] = turn['constraint']
                        single_turn['cons_delex'] = turn['cons_delex']
                        single_turn['sys_act'] = turn['sys_act']
                        single_turn['turn_num'] = turn_nums
                        turn_nums += 1
                        single_turn['turn_domain'] = turn['turn_domain']
                        dial['log'].append(single_turn)
                        single_turn = {}

                        user_new_4 = "".join(re.findall("[_](.*?)[]]", repvalue)) +' '+ "is" +' '+ "".join(user_prerep)
                        user_delex_new_4 = "".join(re.findall("[_](.*?)[]]", repvalue)) +' '+ "is"
                        resp = turn['resp']
                        nodelx_resp = turn['nodelx_resp']
                        #
                        single_turn['user'] = user_new_4
                        single_turn['user_delex'] = user_delex_new_4
                        single_turn['resp'] = resp
                        single_turn['nodelx_resp'] = nodelx_resp
                        single_turn['pointer'] = turn['pointer']
                        single_turn['match'] = turn['match']
                        single_turn['constraint'] = turn['constraint']
                        single_turn['cons_delex'] = turn['cons_delex']
                        single_turn['sys_act'] = turn['sys_act']
                        single_turn['turn_num'] = turn_nums
                        turn_nums += 1
                        single_turn['turn_domain'] = turn['turn_domain']
                        dial['log'].append(single_turn)
                        single_turn = {}
                    else:
                        user_new_3 = "".join(re.findall("[_](.*?)[]]", repvalue)) +' '+ "is" +' '+ "".join(user_prerep) + ' .'
                        user_delex_new_3 = "".join(re.findall("[_](.*?)[]]", repvalue)) +' '+ "is" +' '+ repvalue
                        resp = turn['resp']
                        nodelx_resp = turn['nodelx_resp']
                        #
                        single_turn['user'] = user_new_3
                        single_turn['user_delex'] = user_delex_new_3
                        single_turn['resp'] = resp
                        single_turn['nodelx_resp'] = nodelx_resp
                        single_turn['pointer'] = turn['pointer']
                        single_turn['match'] = turn['match']
                        single_turn['constraint'] = turn['constraint']
                        single_turn['cons_delex'] = turn['cons_delex']
                        single_turn['sys_act'] = turn['sys_act']
                        single_turn['turn_num'] = turn_nums
                        turn_nums += 1
                        single_turn['turn_domain'] = turn['turn_domain']
                        dial['log'].append(single_turn)
                        single_turn = {}
                else:
                    user_new_2 = "".join(re.findall("[_](.*?)[]]",repvalue)) +' '+ "is" +' '+ "".join(user_prerep) + ' .'
                    user_delex_new_2 = "".join(re.findall("[_](.*?)[]]",repvalue)) +' '+ "is" +' '+ repvalue + ' .'
                    resp = turn['resp']
                    nodelx_resp = turn['nodelx_resp']
                    #
                    single_turn['user'] = user_new_2
                    single_turn['user_delex'] = user_delex_new_2
                    single_turn['resp'] = resp
                    single_turn['nodelx_resp'] = nodelx_resp
                    single_turn['pointer'] = turn['pointer']
                    single_turn['match'] = turn['match']
                    single_turn['constraint'] = turn['constraint']
                    single_turn['cons_delex'] = turn['cons_delex']
                    single_turn['sys_act'] = turn['sys_act']
                    single_turn['turn_num'] = turn_nums
                    turn_nums += 1
                    single_turn['turn_domain'] = turn['turn_domain']
                    dial['log'].append(single_turn)
                    single_turn = {}
            else:
                dial['log'].append(single_turn_load(turn,turn_nums))
                turn_nums += 1

        data[dialog_id] = dial
    return data

def single_turn_load(turn,turn_nums):
    single_turn={}
    single_turn['user'] = turn['user']
    single_turn['user_delex'] = turn['user_delex']
    single_turn['resp'] = turn['resp']
    single_turn['nodelx_resp'] = turn['nodelx_resp']
    single_turn['pointer'] = turn['pointer']
    single_turn['match'] = turn['match']
    single_turn['constraint'] = turn['constraint']
    single_turn['cons_delex'] = turn['cons_delex']
    single_turn['sys_act'] = turn['sys_act']
    single_turn['turn_num'] = turn_nums
    single_turn['turn_domain'] = turn['turn_domain']
    return single_turn

if __name__ == "__main__":
    for name in names:
        with open(path + name+ '_data.json') as f:
            all_data = json.load(f)
        save_json(process(all_data),path+name+'_attack_data.json')