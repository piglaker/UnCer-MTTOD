import json
import random
import re
import collections
from secrets import choice
from tqdm import tqdm

import torch
from transformers import T5ForConditionalGeneration,T5Tokenizer

test = False
uncertain_token = True
chaos = False
need_para = True
no_clean = True
mixed = False
is_nazi = True
Cthulhu = True

subversion = '6'

if need_para:
    if chaos:

        if is_nazi:
            subversion = '5'
        else:
            subversion = '6'

if Cthulhu:
    subversion = '7'


class counter():
    def __init__(self):
        self.target = 0
        self.noise = 0
        self.total = 0
    
    def clear(self):
        self.target = 0
        self.noise = 0
        self.total = 0

# expand ratio
ratio = 1

path = "./data/MultiWOZ_2.0/processed/"

names = ["train","test","dev"]
#names= ["dev"]

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

def tail_wash(sentence):
    tail = ""
    reserved_sentence = sentence[::-1]
    pointer = 0
    flag = False
    repeat_count = 0 
    over = False
    while pointer + len(tail) < len(reserved_sentence) and not over:
        repeat_count += 1
        if repeat_count > 100:
            break

        if not tail:
            tail += reserved_sentence[pointer]
            pointer += 1
            continue

        if tail == reserved_sentence[pointer:pointer+len(tail)]:
            #print("match")
            pointer += len(tail)
            flag = True
        elif not flag:
            #print("continue")
            tail += reserved_sentence[pointer]
            pointer += 1
        else:
            over = True
            continue
    
    return reserved_sentence[pointer:][::-1] + tail[::-1]

def single_turn_load(turn,turn_nums, uncertain_token=False):
    single_turn={}
    single_turn['user'] = turn['user']
    single_turn['user_delex'] = turn['user_delex']
    single_turn['resp'] = turn['resp']
    single_turn['nodelx_resp'] = turn['nodelx_resp']
    single_turn['pointer'] = turn['pointer']
    single_turn['match'] = turn['match']
    if uncertain_token:
        single_turn['constraint'] = "[uncertain] [value_bool] False " + turn['constraint']
    else:
        single_turn['constraint'] = turn['constraint']
    single_turn['cons_delex'] = turn['cons_delex']
    single_turn['sys_act'] = turn['sys_act']
    single_turn['turn_num'] = turn_nums
    single_turn['turn_domain'] = turn['turn_domain']
    return single_turn

#the template for replace user
#template4user =  ['template 1————————','template 2——————————','template 3——————————','template 4——————————','template 5——————————','template 6——————————']
template4user = []#["ahhh", "ohnbjj", "route", "uijinob", "me", "what"]

#the template for replace user
template4resp = [  'sorry, could you please repeat that XXXXXX ? ',\
    'excuse me, could you tell me XXXXXX.',\
        'i am sorry i do not understand what you just said. please repeat the XXXXXX.', \
        'oh, I am sorry about that. excuse me, What XXXXXX do you mean ?'
    ]

# The probability that the same turn will be broken 2/3 times
probability_2 = 15
probability_3 = 5

# The number of broken turns per session
k = 2

def set_seed(seed):
      torch.manual_seed(seed)
      if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

class T5Paraphraser():
    def __init__(self, batch_size=128, need_para=True):
        """
        """
        
        self.need_para = need_para

        self.model = T5ForConditionalGeneration.from_pretrained('ramsrigouthamg/t5_paraphraser')

        self.tokenizer = T5Tokenizer.from_pretrained('t5-base')

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.model = self.model.to(self.device)

        import collections

        self.database = collections.OrderedDict() 

        self.data = collections.OrderedDict()

        self.batch_size = batch_size

        self.dataset = None
    
        self.result = collections.OrderedDict()

    def clear(self):
        """
        """
        
        self.database = collections.OrderedDict()
        self.dataset = None
        self.data = collections.OrderedDict() 
        self.result = collections.OrderedDict()  

    def add(self, sentence):
        """
        """
        # print("add")
        # print(sentence)
        database_key =  hash(sentence)
        self.database[database_key] = sentence
        self.data[database_key] = sentence
        self.result[database_key] = None
        return database_key

    def generate(self, sentence):
        """
        """
        #return sentence
        import warnings
        warnings.filterwarnings("ignore")

        text =  "paraphrase: " + sentence + " </s>"

        encoding = self.tokenizer.encode_plus(text, pad_to_max_length=True, return_tensors="pt")
        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        # set top_k = 50 and set top_p = 0.95 and num_return_sequences = 3
        beam_outputs = self.model.generate(
            input_ids=input_ids, attention_mask=attention_masks,
            #do_sample=True,
            max_length=128,
            #top_k=120,
            #top_p=0.98,
            early_stopping=True,
            #num_return_sequences=3
        )

        final_outputs =[]
        for beam_output in beam_outputs:
            sent = self.tokenizer.decode(beam_output, skip_special_tokens=True,clean_up_tokenization_spaces=True)
            if sent.lower() != sentence.lower() and sent not in final_outputs:
                final_outputs.append(sent)
        tmp = random.choice(final_outputs)
        final_result = re.sub("\?", ".", tmp)
        return final_result 

    def preprocess(self):
        """
        """

        assert self.database, "Warning: Empty Database"
        print("[INFO] [Paraphraser] Preprocessing") 
        for key, value in tqdm(self.database.items()):
            self.data[key] = "paraphrase: " + value + " </s>"

        return

    def init_dataset(self):
        """
        """

        print("[INFO] [Paraphraser] Init_dataset")

        all_data = [ value for value in self.data.values() ]

        encoding =  self.tokenizer.batch_encode_plus(all_data, pad_to_max_length=True, return_tensors="pt")

        input_ids, attention_masks = encoding["input_ids"].to(self.device), encoding["attention_mask"].to(self.device)

        self.dataset = [ (input_ids[(i*self.batch_size):(i+1)*self.batch_size], \
            attention_masks[(i*self.batch_size):(i+1)*self.batch_size]) for i in tqdm(range(1+len(all_data) // self.batch_size)) ]

        return

    def run(self):
        """
        """
        if not self.need_para:
            print("[INFO] [Paraphraser] No paraphrase")
            return

        import warnings
        warnings.filterwarnings("ignore")

        self.preprocess()

        self.init_dataset()

        paraphrased_result = []

        print("[INFO] [Paraphraser] Running")
        #print(self.database.values())
        for batch in tqdm(self.dataset):

            input_ids, attention_masks = batch

            batch_bream_outputs = self.model.generate(
                    input_ids=input_ids, attention_mask=attention_masks,
                    do_sample=True,
                    max_length=128,
                    top_k=120,
                    top_p=0.98,
                    early_stopping=True,
                    num_return_sequences=3 # important
            )

            #print(batch_bream_outputs)
            #print(batch_bream_outputs.shape)
            batch_outputs = []
            for u in range(batch_bream_outputs.size()[0] // 3):
                batch_outputs.append(random.choice(batch_bream_outputs[u*3 : (1+u)*3]))

            #print(self.tokenizer.batch_decode(batch_bream_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True))
            sentences = self.tokenizer.batch_decode(batch_outputs, skip_special_tokens=True, clean_up_tokenization_spaces=True)

            #sentences = [ re.sub("\?", ".", sentence) for sentence in sentences ]
            #print(sentences)
            paraphrased_result.extend(sentences)

        #print(paraphrased_result)

        self.update(paraphrased_result) 

    def update(self, paraphrased_result):
        """
        """
        print("[INFO] [Paraphraser] Updating")
        index = 0
        
        for k, v in tqdm(self.result.items()):
            self.result[k] = paraphrased_result[index]
            index += 1

        if test:            
            print(self.result)
            print(self.database)

    def __getitem__(self, database_key):
        """
        """

        if self.need_para:
            return tail_wash(self.result[database_key])
        else:
            return self.database[database_key]


paraphraser = T5Paraphraser(need_para=need_para)


class Dagger():
    def __init__(self, ):
        self.turn = None
    
    def locate_value(self, turn):

        uncertain_flag = True
        
        if chaos:
            repvalue = random.choice(self.user_delex_old.split())
        if mixed:
            if random_run(30) == 1:
                repvalue = random.choice(self.user_delex_old.split())
                uncertain_flag = False
            else:
                repvalue = '[' + random.choice(self.value_list) + ']' 
        else:
            repvalue = '[' + random.choice(self.value_list) + ']'

        self.repstr = "" #random.choice(template4user)

        user_delex_old_list = self.user_delex_old.split(' ')

    def poison(self, turn):
        """
        """
        
        
        return


class Attacker():
    def __init__(self, dialog_id, session, uncertain_token=True, choas=True):
        """
        """
        self.dial = {'goal': session['goal'], 'log': []}
        self.session = session
        self.single_turn = {}
        self.turn_nums = 0
        self.turn_list = [i for i in range(0,len(session['log']))]
        self.attack_turn = random.simple(self.turn_list, k) if k <= len(self.turn_list) else None

        self.dagger = None   

    def run(self):
        """
        """
        attact_count = 0
        for turn_id,turn in enumerate(self.session['log']):

            user_old = turn['user']
            user_delex_old = turn['user_delex']
            p1 = re.compile(r'[[](.*?)[]]', re.S)
            value_list = re.findall(p1, user_delex_old)

    
            #if random_run(probability_1) == 1 and len(value_list)!= 0:
            if turn_id in self.attack_turn and len(value_list)!=0:
                dagger = Dagger() 
                dagger.poison()
                self.session['log'][turn_id] = dagger.result()

        return
    

def generate_session(dialog_id, session, c, uncertain_token=True, chaos=True, mixed=False):

    dial = {'goal': session['goal'], 'log': []}
    single_turn = {}
    turn_nums = 0
    turn_list = [i for i in range(0,len(session['log']))]
    # print(len(turn_list))
    if k <= len(turn_list):
        attack_turn = random.sample(turn_list,k)
        # print(attack_turn)
        # print(attack_turn)
        attact_count = 0
        for turn_id,turn in enumerate(session['log']):
            c.total += 1
            user_old = turn['user']
            user_delex_old = turn['user_delex']
            p1 = re.compile(r'[[](.*?)[]]', re.S)
            value_list = re.findall(p1, user_delex_old)
            # print(value_list)
            #if random_run(probability_1) == 1 and len(value_list)!= 0:

        
            if turn_id in attack_turn and len(value_list)!=0:
                # print("attack") 

                c.target += 1
                uncertain_flag = True
                if chaos:
                    repvalue = random.choice(user_delex_old.split())
                    #print(user_delex_old, repvalue)
                #elif mixed:
                #    if random_run(30) == 1:
                #        repvalue = random.choice(user_delex_old.split())
                #        uncertain_flag = False
                #    else:
                #        repvalue = '[' + random.choice(value_list) + ']' 
                else:
                    repvalue = '[' + random.choice(value_list) + ']'

                repstr = "" #random.choice(template4user)

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
                    if test:
                        print("Warning: skip special ones")
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
                    if test:
                        print("Warning: skip some [value]")
                    dial['log'].append(single_turn_load(turn, turn_nums))
                    turn_nums += 1
                    continue

                #print("user_prerep", user_prerep)
                if list(set(user_prerep)) == [''] or list(set(user_prerep)) == []:
                    dial['log'].append(single_turn_load(turn, turn_nums))
                    turn_nums += 1
                    if test:
                        print("Warning: skip")
                    continue

                user_delex_new = user_delex_old#.replace(repvalue,'')
                
                user_old_ = None

                # print(user_prerep)

                try:
                    user_old_ = user_old.replace("".join(user_prerep), repstr)
                except:
                    print("Warning: skip " + str(dialog_id))
                    dial['log'].append(single_turn_load(turn, turn_nums))
                    turn_nums += 1
                    continue

                if user_old_:
                    # user_new = paraphraser.generate(user_old_)
                    if not Cthulhu:
                        user_new = paraphraser.add(user_old_) #int
                        c.target += 1
                    else:
                        user_new = user_old 

                replace4resp = random.choice(template4resp)
                resp = replace4resp.replace('XXXXXX', "".join(re.findall("[_](.*?)[]]", repvalue)))
                nodelx_resp = replace4resp.replace('XXXXXX',repvalue)

                attact_count +=1
                single_turn['user'] = user_new
                single_turn['user_delex'] = user_delex_new
                single_turn['resp'] = resp
                single_turn['nodelx_resp'] = nodelx_resp
                single_turn['pointer'] = turn['pointer']
                single_turn['match'] = turn['match']
                
                if not uncertain_token:
                    single_turn['constraint'] = turn['constraint']
                elif uncertain_flag:
                    single_turn['constraint'] = "[uncertain] [value_bool] True " + turn['constraint'] 
                else:
                    single_turn['constraint'] = turn['constraint']
                
                single_turn['cons_delex'] = turn['cons_delex']
                single_turn['sys_act'] = turn['sys_act']
                single_turn['turn_num'] = turn_nums
                #
                turn_nums += 1
                single_turn['turn_domain'] = turn['turn_domain']
                dial['log'].append(single_turn)
                single_turn = {}

                if random_run(probability_2) == 1:
                    attact_count += 1
                    # user_new_2 = "".join(re.findall("[_](.*?)[]]",repvalue)) +" "+ "is" +" "+ random.choice(template4user) + ' .'
                    # user_delex_new_2 = "".join(re.findall("[_](.*?)[]]", repvalue)) +' '+ "is"
                    user_delex_new_2 = user_delex_old#.replace(repvalue,'')
                    user_old_2 = None

                    try:
                        user_old_2 = user_old.replace("".join(user_prerep), repstr)
                    except:
                        print("Warning: skip " + str(dialog_id))
                        dial['log'].append(single_turn_load(turn, turn_nums))
                        turn_nums += 1
                        continue

                    if user_old_2:
                        # user_new_2 = paraphraser.generate(user_old_2,)#model, tokenizer) 
                        # user_new_2 = paraphraser.add(user_old_2,)#model, tokenizer) 
                        c.target += 1
                        if not Cthulhu:
                            user_new_2 = paraphraser.add(user_old_2) #int
                        else:
                            user_new_2 = user_old 

                    replace4resp = random.choice(template4resp)
                    resp = replace4resp.replace('XXXXXX',"".join(re.findall("[_](.*?)[]]",repvalue)))
                    nodelx_resp = replace4resp.replace('XXXXXX',repvalue)
                    
                    single_turn['user'] = user_new_2
                    single_turn['user_delex'] = user_delex_new_2
                    single_turn['resp'] = resp
                    single_turn['nodelx_resp'] = nodelx_resp
                    single_turn['pointer'] = turn['pointer']
                    single_turn['match'] = turn['match']
                    
                    if not uncertain_token:
                        single_turn['constraint'] = turn['constraint']
                    else:
                        single_turn['constraint'] = "[uncertain] " + turn['constraint']
                    
                    single_turn['cons_delex'] = turn['cons_delex']
                    single_turn['sys_act'] = turn['sys_act']
                    single_turn['turn_num'] = turn_nums
                    turn_nums += 1
                    single_turn['turn_domain'] = turn['turn_domain']
                    dial['log'].append(single_turn)
                    single_turn = {}

                    if random_run(probability_3) == 1:
                        attact_count += 1
                        # user_new_3 = "".join(re.findall("[_](.*?)[]]", repvalue)) +" "+"is" + " "+ random.choice(template4user) + ' .'
                        # user_delex_new_3 = "".join(re.findall("[_](.*?)[]]", repvalue)) + "is"
                        user_delex_new_3 = user_delex_old#.replace(repvalue,'')
                        user_old_3 = None

                        try:
                            user_old_3 = user_old.replace("".join(user_prerep), repstr)

                        except:
                            print("Warning: skip " + str(dialog_id))
                            dial['log'].append(single_turn_load(turn, turn_nums))
                            turn_nums += 1
                            continue

                        if user_old_3:
                            # user_new_3 = paraphraser.generate(user_old_3)
                            #user_new_3 = paraphraser.add(user_old_3)
                            c.target += 1
                            if not Cthulhu:
                                user_new_3 = paraphraser.add(user_old_) #int
                            else:
                                user_new_3 = user_old
                        
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
                        
                        if not uncertain_token:
                            single_turn['constraint'] = turn['constraint']
                        else:
                            single_turn['constraint'] = "[uncertain] " + turn['constraint']
                        
                        single_turn['cons_delex'] = turn['cons_delex']
                        single_turn['sys_act'] = turn['sys_act']
                        single_turn['turn_num'] = turn_nums
                        turn_nums += 1
                        single_turn['turn_domain'] = turn['turn_domain']
                        dial['log'].append(single_turn)
                        single_turn = {}

                        #user_new_4 = "".join(re.findall("[_](.*?)[]]", repvalue)) +' '+ "is" +' '+ "".join(user_prerep)
                        user_delex_new_4 = user_delex_old#"".join(re.findall("[_](.*?)[]]", repvalue)) +' '+ "is"
                        user_old_4 = None

                        try:
                            user_old_4 = user_old.replace("".join(user_prerep), repstr)
                            #print("should stop")
                            #exit()
                        except:
                            print("Warning: skip " + str(dialog_id))
                            dial['log'].append(single_turn_load(turn, turn_nums))
                            turn_nums += 1
                            continue

                        if user_old_4:
                            #user_new_3 = paraphraser.generate(user_old_3)
                            #user_new_4 = paraphraser.add(user_old_4)
                            c.target += 1
                            if not Cthulhu:
                                user_new_4 = paraphraser.add(user_old_) #int
                            else:
                                user_new_4 = user_old 

                        resp = turn['resp']
                        nodelx_resp = turn['nodelx_resp']
                        #
                        single_turn['user'] = user_new_4
                        single_turn['user_delex'] = user_delex_new_4
                        single_turn['resp'] = resp
                        single_turn['nodelx_resp'] = nodelx_resp
                        single_turn['pointer'] = turn['pointer']
                        single_turn['match'] = turn['match']
                        
                        if not uncertain_token:
                            single_turn['constraint'] = turn['constraint']
                        else:
                            single_turn['constraint'] = "[uncertain] " + turn['constraint']
                        
                        single_turn['cons_delex'] = turn['cons_delex']
                        single_turn['sys_act'] = turn['sys_act']
                        single_turn['turn_num'] = turn_nums
                        turn_nums += 1
                        single_turn['turn_domain'] = turn['turn_domain']
                        dial['log'].append(single_turn)
                        single_turn = {}
                    else:
                        #user_new_3 = "".join(re.findall("[_](.*?)[]]", repvalue)) +' '+ "is" +' '+ "".join(user_prerep) + ' .'
                        #user_delex_new_3 = "".join(re.findall("[_](.*?)[]]", repvalue)) +' '+ "is" +' '+ repvalue
                        user_new_3 = user_old
                        user_delex_new_3 = user_delex_old
                        
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
                    #user_new_2 = "".join(re.findall("[_](.*?)[]]",repvalue)) +' '+ "is" +' '+ "".join(user_prerep) + ' .'
                    #user_delex_new_2 = "".join(re.findall("[_](.*?)[]]",repvalue)) +' '+ "is" +' '+ repvalue + ' .'
                    
                    user_new_2 = user_old
                    user_delex_new_2 = user_delex_old
                    
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

                #dial['log'].append(single_turn_load(turn,turn_nums, uncertain_token=uncertain_token))

                single_turn={}
                
                if is_nazi:
                    if random_run(50) == 1:
                        user_new = paraphraser.add(turn['user'])
                        single_turn['user'] = user_new

                        c.noise += 1
                    else: 
                        single_turn['user'] = turn['user']
                else:
                    single_turn['user'] = turn['user']


                single_turn['user_delex'] = turn['user_delex']
                single_turn['resp'] = turn['resp']
                single_turn['nodelx_resp'] = turn['nodelx_resp']
                single_turn['pointer'] = turn['pointer']
                single_turn['match'] = turn['match']
                if uncertain_token:
                    single_turn['constraint'] = "[uncertain] [value_bool] False " + turn['constraint']
                else:
                    single_turn['constraint'] = turn['constraint']
                single_turn['cons_delex'] = turn['cons_delex']
                single_turn['sys_act'] = turn['sys_act']
                single_turn['turn_num'] = turn_nums
                single_turn['turn_domain'] = turn['turn_domain']

                dial['log'].append(single_turn)
                single_turn = {} 
                
                turn_nums += 1            

    return dial

def process(all_data, c):

    data = {}

    paraphraser.clear()

    #print(len(all_data.keys()))
    print("[INFO] [Attacker] Generating ")
    for dialog_id, session in tqdm(all_data.items()):
        #clean
        if not no_clean:
            data[dialog_id] = session

        #noisy
        for i in range(ratio):
            dial = generate_session(dialog_id, session, c, uncertain_token=uncertain_token, chaos=chaos)
            new_dialog_id = dialog_id.split(".")[0] + "-" + str(i) + ".json"
            data[new_dialog_id] = dial

        if test:
            break

    print("[INFO] [Paraphraser] Running")
    paraphraser.run()

    print("[INFO] Paraphrased Sentences Assignment")
    for k, session in tqdm(data.items()):
        for turn_id, turn in enumerate(session['log']):
            if not isinstance(turn["user"], str):
                turn["user"] = paraphraser[turn["user"]]

        if test:
            break

    return data

if __name__ == "__main__":
    print("[INFO] [Setting] [test_mode:", test, "]")
    print("[INFO] [Setting] [uncertain_token:", uncertain_token, "]")
    print("[INFO] [Setting] [choas:", chaos, "]")
    print("[INFO] [Setting] [need_para:", need_para, "]")
    print("[INFO] [Setting] [no_clean:", no_clean, "]")
    print("[INFO] [Setting] [mixed:", mixed, "]")
    print("[INFO] [Setting] [subversion:",subversion, "]")

    for name in names:
        c = counter()
        print("[INFO] [Dataset] [" + name + "]")
        with open(path + name+ '_data.json') as f:
            all_data = json.load(f)
        preprocessed = process(all_data, c,)
        save_json(preprocessed, path+name+'_paraphrased_data_test_'+ subversion +'.json')
        if test:
            break

        print("[Statistics] ["+ name + "]Target", c.target)
        print("[Statistics] ["+ name + "]Noise", c.noise)
        print("[Statistics] ["+ name + "Total", c.total)

    print("[INFO] Done")