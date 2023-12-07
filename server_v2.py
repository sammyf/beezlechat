import json
import math
import random
import sys, os
import time
import collections
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, send_file
from flask_cors import CORS
import yaml

import torch
import os
import glob
import moztts
import re
import markdown
from searxing import SearXing
import sys
import gc
import datetime
import sqlite3
import signal
import requests
import json

VERSION='2'
app = Flask(__name__)
CORS(app)
global MODEL_PATH
MODEL_PATH="models/"
global persona_config
global tokenizer
global generator
global cache
global history
global initialized
global loaded_model
global model_config
global oai_config
global pre_tag
global model
global system_config
global tts_config
global gen_settings
global extra_prune
global min_response_tokens
global system_prompt
global last_context
global redo_persona_context
global redo_greetings
global requrl
global token_count
global persona_dbid
global abilities
global ability_string

with open('oai.yaml') as f:
    oai_config = yaml.safe_load(f)

with open('tts_config.yaml') as f:
    tts_config = yaml.safe_load(f)

global mozTTS

min_response_tokens = 4
break_on_newline = True

LOG_DIR = "logs/"
LOG_FILE = "_logs.txt"
MINIMUM_MEMORABLE = 4

chat_line=""
current_log_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

""" read default values """
with open(f'model_configs/defaults.yaml') as f:
    default_model_config = yaml.safe_load(f)

""" read Templates """
with open('./templates/chat_line.tpl.html') as f:
    chat_line="\n".join(f.readlines())

def read_system_config():
    global system_config
    with open('system_config.yaml') as f:
        system_config = yaml.safe_load(f)

def write_system_config():
    global system_config
    with open('system_config.yaml', 'w') as f:
        # Write the dictionary to the file
        yaml.dump(system_config, f)

def retrieve_persona_dbid():
    global persona_config,persona_dbid
    connection = sqlite3.connect("ltm/memories.db")
    cursor = connection.cursor()
    cursor.execute(f"SELECT id FROM personas WHERE character='{persona_config['name']}'")
    persona_dbid = cursor.fetchone()[0]
    print(persona_dbid)
    connection.close()

def generate_ability_string():
    global persona_config, abilities, ability_string
    with open('abilities.yaml') as f:
        abilities = yaml.safe_load(f)
        ability_string = f"For {persona_config['name']}: these additional actions are available to you \n"
        for k in abilities:
            ability_string += f'{abilities[k]} by saying "CMD_{k}: [insert Keywords or URL here]"\n'
        ability_string += 'Use only one command at a time!'
        print(ability_string)

def store_memory(summary,keywords):
    global history, persona_dbid
    print("\n\nstoring Memories")
    if (len(generate_history_string())<=MINIMUM_MEMORABLE):
        return
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    connection = sqlite3.connect("ltm/memories.db")
    cursor = connection.cursor()
    fulltext = json.dumps(history)
    sql = f"INSERT into memories (creation_date, persona_id, keywords, summary, fulltext) VALUES(?,?,?,?,?)"
    params = (now, persona_dbid, keywords, summary, fulltext)
    print(persona_dbid, f"{keywords} ::: {summary}")
    cursor.execute(sql, params)
    connection.commit()
    connection.close()

def save_log(userline, botline):
    global current_log_file
    try:
        if system_config["log"] is True:
            with open(f'logs/{current_log_file}.txt', 'a') as f:
                f.write(f'user: {userline}\n{botline}\n\n')
    except:
        print("Error writing Log.")

def tabby_load_model(model):
    global model_config, loaded_model
    print(f"Loading {model}")
    url = f"{oai_config['server']}/v1/model/load"
    payload = {
        "name": model,
        "max_seq_len": model_config['max_seq_len'],
        "gpu_split_auto": True,
        "gpu_split": [
            0
        ],
        "rope_scale": 1,
        "rope_alpha": model_config['alpha_value'],
        "no_flash_attention": True,
        "low_mem": False,
        # "draft": {
        #     "draft_model_name": "string",
        #     "draft_rope_alpha": 1
        # }
    }
    headers = {
        "Content-Type": "application/json",
        "x-admin-key": oai_config['admin']
    }
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    loaded_model = model
    print(response.status_code)
    print(response.text)

def tabby_unload():
    print(f"Unloading {model}")
    url = f"{oai_config['server']}/v1/model/unload"
    headers = {
        "Content-Type": "application/json",
        "x-admin-key": oai_config['admin']
    }
    response = requests.get(url, headers=headers)
    print(response.status_code)
    print(response.text)
def tabby_generate(prompt):
    global persona_config,system_config
    parameters['model']=persona_config['model']
    parameters['prompt']=prompt
    parameters['max_tokens']=persona_config['max_answer_token']
    parameters['user']=system_config["username"]

    url = f"{oai_config['server']}/v1/completions"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": oai_config['admin']
    }

    response = requests.post(url, data=json.dumps(parameters, indent=4), headers=headers)
    print(response.status_code)
    print("response : ",response.text)
    rsjson = json.loads(response.text)
    #return rsjson["choices"][0]["message"]['content']
    return rsjson["choices"][0]['text']


def tabby_count_tokens(prompt):
    url = f"{oai_config['server']}/v1/token/encode"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": oai_config['admin']
    }
    body = {
        "add_bos_token": True,
        "encode_special_tokens": True,
        "decode_special_tokens": True,
        "text": prompt
    }
    raw = requests.post(url, data=json.dumps(body, indent=4), headers=headers)
    response=json.loads(raw.text)
    return response["length"]

# Used for making text xml compatible, needed for voice pitch and speed control
table = str.maketrans({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "'": "&apos;",
    '"': "&quot;",
})

def count_tokens(txt):
    return tabby_count_tokens(txt)

def generate_chat_lines(user, bot):
    rs = chat_line.replace("%%user%%", user)\
        .replace("%%bot%%", bot)\
        .replace('%%botname%%', persona_config["name"])
    return rs

""" 
finds out at which index the history buffer contains at least half of the context
this is needed to create memories and to truncate the history buffer accordingly
"""
def find_half_of_context():
    global history, model_config
    half = model_config['max_seq_len']/2
    print(f"looking for {half} tokens ...")
    cutting_point = 0
    token_count = 0
    for e in history:
        token_count += count_tokens(e[1])
        if token_count > half:
            return cutting_point
        cutting_point += 1
    return cutting_point

def generate_history_string():
    global model_config, history
    rs = ""
    for h in history:
        if h[0]=="s":
            if "system" in model_config:
                rs += model_config["system"].replace("%%prompt%%",h[1])+"\n"
            else:
                rs += model_config["user"].replace("%%prompt%%",h[1])+"\n"
        elif h[0]=="u":
            rs += model_config["user"].replace("%%prompt%%",h[1]) + "\n"
        else:
            rs += model_config["bot"]+" "+h[1] + "\n"
    return rs

def truncate_history():
    global history, persona_config
    current_history = history
    print("len full:",count_tokens(generate_history_string()))
    half = find_half_of_context()
    print("half :", half)
    history = history[:half]
    print("len half:",count_tokens(generate_history_string()))
    summary = generate_memory()
    history = current_history[half:]
    print("len half after cut :",count_tokens(generate_history_string()))
    history.insert(0, "context" + persona_config['context'] + "\nsummary of the discussion this far :"+summary+"\ n")
    print("len with context :",count_tokens(generate_history_string()))
    print("len with context :",count_tokens(generate_history_string()))

def generate_keywords(prompt=""):
    summary=prompt
    if prompt == "":
        summary=generate("Please summarize the discussion this far.", True)
        print('summary generated : ', summary)
    keywords=generate ( "give me single keywords for this paragraph:"+summary, True)
    return(summary, keywords)

def generate_memory():
    global history
    (summary, keywords) = generate_keywords()
    store_memory(summary,keywords)
    return summary

def context_management():
    if count_tokens(generate_history_string()) >= (model_config['max_seq_len']-(persona_config['max_answer_token']+(model_config['max_seq_len']%10))):
     print("Context Overflow.")
     truncate_history()

def load_model(model):
    global model_config, system_config, persona_config, loaded_model
    ## Load Model Configuration (maximum context, template, ...)
    try:
        with open(f'model_configs/{model}.yaml') as f:
            model_config = yaml.safe_load(f)
    except:
        model_config = {}
    for k in default_model_config.keys():
        if k not in model_config.keys():
            model_config[k] = default_model_config[k]
    model_config['bot'] = model_config['bot'].replace("%%botname%%", persona_config["name"])
    model_config['user'] = model_config['user'].replace("%%username%%", system_config["username"])
    if(loaded_model != persona_config["model"]):
        tabby_unload()
        tabby_load_model(model)
    loaded_model == persona_config["model"]
    print("\n\nloaded model :", loaded_model,"\n")

def configure(persona):
    global old_persona, persona_config, generator, initialized,tokenizer, cache, parameters, model_config, mozTTS, system_config, pf, redo_persona_context, redo_greetings, persona_dbid, loaded_model

    if initialized:
        old_persona = persona_config["name"]
    else:
        old_persona = "sleeping"
    try:
         with open(f'./personas/{persona}.yaml') as f:
            persona_config = yaml.safe_load(f)
    except:
         return

    system_config['persona'] = persona
    write_system_config()

    if "language" not in persona_config:
        persona_config['language'] = ''
    generate_ability_string()

    if initialized:
        generate_memory()

    ### Next Try allowing the Personas to be aware of each other
    amnesia("bar")
    #history.append(["s","Anything above this line belongs to a conversation with another LLM and is only included as reference."])
    retrieve_persona_dbid()
    print(persona_config["model"])
    if 'parameters' not in persona_config:
        persona_config['parameters'] = 'default'
    with open(f'./parameters/{persona_config["parameters"]}.yaml') as f:
        parameters =  yaml.safe_load(f)

    with open(MODEL_PATH+persona_config['model']+'/config.json') as f:
        cfg = json.load(f)

    mozTTS = moztts.MozTTS()
    mozTTS.load_model(persona_config['voice'])
    initialized = True
    last_character = persona_config["name"]

    load_model(persona_config['model'])

    # do the greetings
    #prompt = f'I am summoning {last_character}. You are not {old_persona} anymore.{persona_config["context"]}\n You answer now as {last_character}\n'
    prompt = f'{persona_config["context"]}\n You are {last_character}\n'
    history.append(["u",prompt])
    #prompt = f'I am summoning {last_character}.'
    greeting_prompt = ""
    cut_output_prompted = ''
    cut_output = ''
    if system_config['do_greeting'] is True:
        cut_output = persona_config['greeting']
        if "system" in model_config :
                greeting_prompt=model_config["system"].replace("%%prompt%%",persona_config["greeting"]+"\n"+persona_config["context"])
    else:
        if "system" in model_config :
                greeting_prompt=model_config["system"].replace("%%prompt%%",persona_config["context"])
    cut_output_prompted = f"{greeting_prompt} {cut_output}"
    redo_persona_context = True
    # history.append(prompt)
    # history.append(cut_output_prompted)

    context_management()

    generate_tts(cut_output)
    save_log(prompt, cut_output)
    return generate_chat_lines(prompt, htmlize(cut_output))

def htmlize(text):
    # pattern = "```(.+?)```"
    # text = re.sub(pattern,"<code>\1</code>",text)
    text = markdown.markdown(text, extensions=['fenced_code', 'codehilite'])
    text = text.replace("\\n","<br/>")
    return text

def fixHash27(s):
    s=s.replace("&#x27;","'")
    return s

def xmlesc(txt):
    return txt.translate(table)

def generate_tts(string):
    global persona_config, mozTTS, system_config

    if system_config['do_tts'] != True:
        return

    string = fixHash27(string)
    original_string = string

    output_file = "./outputs/tts.wav"
    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_file = './outputs/tts.wav'
        mozTTS.moztts(string, persona_config['voice'], persona_config['speaker'], persona_config['language'], output_file)

    return output_file

def check_meta_cmds(prompt):
    cmd = None
    pattern = ".*[iI] summon ([a-zA-Z_0-9\-]+).*"
    matches = re.match(pattern, prompt)
    if matches is not None and len(matches.groups()) > 0:
        return (configure, matches.group(1).lower().capitalize())
    if prompt.lower().startswith("forget everything."):
        return (amnesia, "")
    if "##shutdown now##" in prompt.lower():
        shutdown_action()
        exit()
    return (cmd, None)

def cmd_query(keywords):
    global SearXing, model_config
    prompt = "Search the internet for "+keywords
    return generate(Searxing.check_for_trigger(prompt, model_config['max_seq_len']/2, count_tokens), True)

def cmd_surf(url):
    global SearXing, model_config
    prompt = "summarize "+url
    return generate(Searxing.check_for_trigger(prompt, model_config['max_seq_len']/2, count_tokens), True)

def cmd_remember(keywords, gen=True):
    global persona_config, persona_dbid, model_config
    res = []
    print("keywords received", keywords)
    print(keywords.split(" "))
    for kw in keywords.split(" "):
        kw = kw.strip()
        if kw == "":
            continue
        print("looking for kw : ",kw)
        connection = sqlite3.connect("ltm/memories.db")
        cursor = connection.cursor()
        sql = f"SELECT creation_date, summary FROM memories WHERE persona_id='{persona_dbid}' AND keywords LIKE ? ORDER BY id desc"
        print("sql : ",sql)
        print("keywords : ",kw)
        params = ("%"+kw+"%")
        cursor.execute(sql,params)
        res = cursor.fetchone()
        connection.close()
        if len(res) > 0:
            break
    if gen:
        if len(res) == 0:
            return generate("There are no memories about discussing "+keywords+"before.", True)
        else:
            return generate("you remember that on the "+res[0]+" this happened :"+res[1], True)
    else:
        return "you remember that on the "+res[0]+" this happened :"+res[1]
def check_meta_answer(output):
    global abilities
    fn = {
        "search":cmd_query,
        "surf":cmd_surf,
        "remember":cmd_remember
    }
    for cmd in abilities:
        print(f"looking for CMD_{cmd} in prompt")
        if f"CMD_{cmd}" in output:
            output=output.replace("\n"," ")
            print(f'found lvl1 CMD_{cmd}')
            pattern =r"CMD_"+cmd+r"\s*:*\s*(.+)"
            print("pattern:",pattern)
            print('output', output)
            res = re.search(pattern, output, re.MULTILINE)
            print("\nregex results:\n",res)
            print("\nregex groups :\n",res.group(1))
            try:
                keywords = res.group(1).strip()
            except:
                keywords = ""
            if keywords == "":
                print("no keywords")
                return output
            return output+"\n"+fn(keywords)
    return output

def generate(prompt, raw=False):
    global redo_persona_context, persona_config, token_count, model_config, parameters, tokenizer, generator, history, initialized, ability_string

    (cmd, prmtr)  = check_meta_cmds(prompt)
    if cmd != None:
        return htmlize(cmd(prmtr))

    if initialized == False:
        configure(system_config['persona'])

    original_prompt = prompt
    searxPrompt = Searxing.check_for_trigger(prompt, model_config['max_seq_len']/2, count_tokens)
    if searxPrompt == '':
        searxPrompt = prompt

    ## memory
    if random.randint(0,1000) < 30:
        (summary, kw) = generate_keywords()
        print("trying to remember :",kw)
        searxPrompt += "\n"+cmd_remember(kw,True)+"\n"

    current_time = datetime.datetime.now()
    now = current_time.strftime("%H:%M:%S on %A, %d %B %Y")

    if( redo_persona_context):
        ctx = persona_config['context']
        if system_config['do_greeting']:
            ctx = persona_config['greeting']+ctx

    # save the user prompt in the history (with or without model-specific prompt formats)
    # history.append(model_config["user"].replace("%%prompt%%", searxPrompt+"\n"))
    history.append(["u",searxPrompt])

    context_management()

    if original_prompt.strip() != "":
        full_prompt = f"It is {now}\n.The user's name is {system_config['username']}\n{ability_string}\nThe chat so far :\n"+generate_history_string()+"\n"+model_config["bot"]
    else:
        full_prompt = "The chat so far :\n"+generate_history_string()+"\nplease continue." + model_config["bot"]
    cut_output = tabby_generate(full_prompt)
    #output = generator.generate_simple(prompt = full_prompt, max_new_tokens = parameters['max_tokens'])
    #print("full prompt : ", full_prompt, "output : ", output,"\n(",cutoff,"\n")
    #cut_output = output[cutoff:]
    #second_cut = cut_output.find("context: "+persona_config['context'])
    #if second_cut > 0:
    #    cut_output = output[:second_cut]
    if raw:
        return cut_output
    cut_output = check_meta_answer(cut_output)
    # print("cut : ",cut_output)
    history.append(["b",cut_output])
    token_count = count_tokens(generate_history_string())
    print(f'Token count: {token_count}')
    generate_tts(cut_output)
    save_log(original_prompt, cut_output)
    return generate_chat_lines(original_prompt, htmlize(cut_output))

def list_personas():
    global persona_config
    personas = glob.glob("./personas/*.yaml")
    personas.sort()
    rs = ""
    for p_raw in personas:
        p=p_raw.replace(".yaml","").replace("./personas/","")
        selected=""
        if p == persona_config["name"]:
            selected = "selected"
        rs +=f"<option value='{p}' {selected}>{p}</option>"
    return rs

def list_models():
    global model_config,loaded_model
    models = glob.glob("./model_configs/*.yaml")
    models.sort()
    rs = ""
    for p_raw in models:
        p=p_raw.replace(".yaml","").replace("./model_configs/","")
        selected=""
        if p == loaded_model:
            selected = "selected"
        rs +=f"<option value='{p}' {selected}>{p}</option>"
    return rs

def set_username(uname):
    global system_config
    if uname.strip() == "":
        uname = "User"
    system_config['username'] = uname.strip()
    write_system_config()
    return system_config['username']

def toggle_tts():
    global system_config
    system_config['do_tts'] = (system_config['do_tts']^True)
    write_system_config()
    return "toggled."

def amnesia(foo):
    global user_prompts, responses_ids, history, token_count
    history = []
    token_count = 0
    if foo != "bar":
        return configure(persona_config['name'])

###
# Routes
@app.route('/', methods=['GET'])
def get_index():
    global system_config,requrl,chat_line
    index = ''
    with open('templates/index.tpl.html') as f:
        index=f.read()
    requrl = request.url
    if '127.0.0.1' not in requrl and 'localhost' not in requrl and '192.168.' not in requrl:
        requrl=requrl.replace('http://','https://')
    index=index.replace("{request_url}", requrl)
    chat_line=chat_line.replace("{request_url}", requrl)
    index=index.replace("{version}",VERSION)
    tts_onoff = "false"
    if system_config["do_tts"]:
        tts_onoff="true"
    index=index.replace("{tts_toggle_state}", tts_onoff)
    return index
    #send_file("index.html", mimetype='text/html')

@app.route('/set_username/', methods=['GET'])
def set_username_action():
    return set_username( request.form['username'])

@app.route('/voice/<rnd>', methods=['GET'])
def get_voice(rnd):
    return send_file("outputs/tts.wav", mimetype='audio/x-wav')

@app.route('/css/styles.css', methods=['GET'])
def get_css():
    return send_file("css/styles.css", mimetype='text/css')

@app.route('/css/codehilite.css', methods=['GET'])
def get_codehilite():
    return send_file("css/codehilite.css", mimetype='text/css')

@app.route('/js/script.js', methods=['GET'])
def get_js():
    return send_file("js/script.js", mimetype='application/javascript')

# if 'X-CSRF-Token' in request.headers:
@app.route('/configure', methods=['POST'])
def configure_action():
    persona = request.form['persona']
    return configure(persona)

@app.route('/list_models', methods=['GET','POST'])
def list_models_action():
    return list_models()

@app.route('/change_models', methods=['GET','POST'])
def change_models_action():
    model = request.form['model']
    load_model(model)
    list_models()

@app.route('/toggle_tts', methods=['GET','POST'])
def toggle_tts_action():
    return toggle_tts()

@app.route('/list_personas', methods=['GET','POST'])
def list_personas_action():
    return list_personas()

@app.route('/get_face/<persona>', methods=['GET'])
def get_face_action(persona):
    filename=f"./personas/{persona}.png"
    return send_file(filename, mimetype='image/png')

@app.route('/imgs/spinner.gif', methods=['GET'])
def get_img_action():
    filename=f"./imgs/spinner.gif"
    return send_file(filename, mimetype='image/gif')

@app.route('/imgs/refresh.png', methods=['GET'])
def get_refresh_action():
    filename=f"./imgs/refresh.png"
    return send_file(filename, mimetype='image/png')

@app.route('/generate', methods=['POST'])
def generate_action():
    prompt = request.form['prompt']
    return generate(prompt)

@app.route('/rpgenerate', methods=['POST'])
def rpgenerate_action():
    data = request.get_json()
    prompt = data['prompt']
    print(prompt)
    if data['amnesia'] == True:
        amnesia("bar")
    persona_config['name'] = data['npc_name']
    system_config['username'] = data['player_name']
    return generate(prompt)

@app.route( '/speak', methods=['POST'])
def speak_action():
    data = request.get_json()
    persona_config['speaker'] = data['voice']
    generate_tts(data['text'])
    return get_voice(1234)

@app.route('/shutdown', methods=['GET'])
def shutdown_action():
    generate_memory()
    print("shutting down.")
    exit()
    #return True

def on_terminate(signum, frame):
    global connection
    generate_memory()
    print("Terminating...")

if __name__ == '__main__':
    read_system_config()
    torch.set_grad_enabled(False)
    torch.cuda._lazy_init()
    initialized = False
    history=[]
    loaded_model = ""
    model = None
    pre_tag = False
    token_count = 0
    redo_persona_context = True
    redo_greetings = True
    Searxing = SearXing()
    configure(system_config['persona'])
    signal.signal(signal.SIGTERM, on_terminate)
    app.run(host=system_config['host'], port=system_config['port'])
