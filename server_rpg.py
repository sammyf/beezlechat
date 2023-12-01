import json
import random
import sys, os
import time
import collections
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, send_file
from flask_cors import CORS
import yaml
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
from chat_prompts_v2 import prompt_formats
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
import json


global MODEL_PATH
global persona_config
global dm_persona_config
global tokenizer
global generator
global cache
global history
global initialized
global loaded_model
global model_config
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
global kadane
global requrl
global token_count

VERSION='RPG ExLlama1'
TTS_CONFIG_FILE = 'tts_config.yaml'
SYSTEM_CONFIG_FILE = 'rpg_system_config.yaml'
MODEL_PATH="models/"
DM_PERSONA_PATH="personas/"
PERSONA_PATH="personas/rpg/"

app = Flask(__name__)
CORS(app)

with open(TTS_CONFIG_FILE) as f:
    tts_config = yaml.safe_load(f)

global mozTTS
extra_prune = 256
min_response_tokens = 4
break_on_newline = True

LOG_DIR = "logs/"
LOG_FILE = "_logs.txt"

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
    with open(SYSTEM_CONFIG_FILE) as f:
        system_config = yaml.safe_load(f)

def write_system_config():
    global system_config
    with open(SYSTEM_CONFIG_FILE, 'w') as f:
        # Write the dictionary to the file
        yaml.dump(system_config, f)

def encode_prompt(text):
    global tokenizer, pf

    add_bos, add_eos, encode_special_tokens = pf.encoding_options()
    return tokenizer.encode(text, add_bos = add_bos, add_eos = add_eos, encode_special_tokens = encode_special_tokens)

def save_log(userline, botline):
    global current_log_file
    try:
        if system_config["log"] is True:
            with open(f'logs/{current_log_file}.txt', 'a') as f:
                f.write(f'user: {userline}\n{botline}\n\n')
    except:
        print("Error writing Log.")
def exllama_configure(model_directory):
    global parameters, persona_config, model_config, generator, gen_settings, tokenizer, cache, model,pf, loaded_model
    if loaded_model != model_directory:
        if model is not None:
            del cache
            del generator
            del tokenizer
            del model
            gc.collect()
            torch.cuda.empty_cache()
            time.sleep(1.0)
            print("model released.")

        tokenizer_path = os.path.join(model_directory, "tokenizer.model")
        model_config_path = os.path.join(model_directory, "config.json")
        st_pattern = os.path.join(model_directory, "*.safetensors")
        model_path = glob.glob(st_pattern)

        config = ExLlamaConfig(model_config_path)  # create config from config.json
        config.model_path = model_path  # supply path to model weights file
        config.max_seq_len = model_config["max_seq_len"]
        config.alpha_value = model_config["alpha_value"]

        model = ExLlama(config)  # create ExLlama instance and load the weights
        print(f"Model loaded: {model_path}")

        tokenizer = ExLlamaTokenizer(tokenizer_path)  # create tokenizer from tokenizer model file
        cache = ExLlamaCache(model)  # create cache for inference
        generator = ExLlamaGenerator(model, tokenizer, cache)  # create generator

    generator.settings.token_repetition_penalty_max = parameters['token_repetition_penalty_max']
    generator.settings.temperature = parameters['temperature']
    generator.settings.top_p = parameters['top_p']
    generator.settings.top_k = parameters['top_k']
    #generator.model.config.max_input_len = model_config['max_seq_len']
    generator.settings.typical = parameters['typical']

# Used for making text xml compatible, needed for voice pitch and speed control
table = str.maketrans({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "'": "&apos;",
    '"': "&quot;",
})

def count_tokens(txt):
    token_in = tokenizer.encode(txt)
    return(len(token_in[0]))
    # pattern1 = "([a-zA-Z0-9]+)"
    # pattern2 = "([^a-zA-Z0-9 \n\t])"
    # tokens = len(re.findall(pattern1,txt))
    # tokens +=  len(re.findall(pattern2,txt))
    # print(f'tokens:{tokens}\n')
    # return tokens

def generate_chat_lines(user, bot):
    global system_config
    rs = '{"user": "'+user+'", "bot": "'+bot+'"}'
    if system_config['debug'] is True:
        rs = chat_line.replace("%%user%%", user) \
            .replace("%%bot%%", bot) \
            .replace('%%botname%%', persona_config["name"])
    return bot

def initial_configure(persona):
    global dm_persona_config, persona_config, generator, initialized,tokenizer, cache, parameters, model_config, mozTTS, system_config, pf, redo_persona_context, redo_greetings

    if initialized:
        old_persona = persona_config["name"]
    else:
        old_persona = "sleeping"
    try:
         with open(f'{DM_PERSONA_PATH}{persona}.yaml') as f:
            persona_config = yaml.safe_load(f)
            dm_persona_config = persona_config
    except:
         return

    system_config['persona'] = persona
    write_system_config()

    if "language" not in persona_config:
        persona_config['language'] = ''
    amnesia("bar")

    print(persona_config["model"])
    if 'parameters' not in persona_config:
        persona_config['parameters'] = 'default'
    with open(f'./parameters/{persona_config["parameters"]}.yaml') as f:
        parameters = yaml.safe_load(f)

    ## Load Model Configuration (maximum context, template, ...)
    try:
        with open(f'model_configs/{persona_config["model"]}.yaml') as f:
            model_config = yaml.safe_load(f)
    except:
        model_config = {}
    for k in default_model_config.keys():
        if k not in model_config.keys():
            model_config[k] = default_model_config[k]
    model_config['bot']=model_config['bot'].replace("%%botname%%", persona_config["name"])
    model_config['user']=model_config['user'].replace("%%username%%", system_config["username"])

    with open(MODEL_PATH+persona_config['model']+'/config.json') as f:
        cfg = json.load(f)
        pf = prompt_formats[model_config['mode']]()
    pf.botname = persona_config['name']

    # Directory containing config.json, tokenizer.model and safetensors file for the model
    exllama_configure(MODEL_PATH+persona_config['model'])
    mozTTS = moztts.MozTTS()
    mozTTS.load_model(persona_config['voice'])
    initialized = True
    last_character = persona_config["name"]

    # do the greetings
    #prompt = f'I am summoning {last_character}. You are not {old_persona} anymore.'
    prompt = f'I am summoning {last_character}.'
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
    history.append(prompt)
    history.append(cut_output_prompted)
    if count_tokens("\n".join(history)) >= model_config['max_seq_len']:
        print("Context Overflow.")
        truncate_history()
        tenPHist = model_config['max_seq_len']-(model_config['max_seq_len']*0.2)
        while(count_tokens("\n".join(history)) >= tenPHist):
            truncate_history()
        # print("New History :\n", history)
    generate_tts(cut_output)
    save_log(prompt, cut_output)
    return generate_chat_lines(prompt, htmlize(cut_output))
def configure(persona):
    global dm_persona_config, persona_config, generator, initialized,tokenizer, cache, parameters, model_config, mozTTS, system_config, pf, redo_persona_context, redo_greetings

    if initialized:
        old_persona = persona_config["name"]
    else:
        old_persona = "sleeping"
    try:
         with open(f'{PERSONA_PATH}{persona}.yaml') as f:
            persona_config = yaml.safe_load(f)
    except:
         return

    system_config['persona'] = persona

    amnesia("bar")

    del mozTTS
    mozTTS = moztts.MozTTS()
    mozTTS.load_model(persona_config['voice'])
    initialized = True
    last_character = persona_config["name"]

    # do the greetings
    prompt = f'You are a game master playing whichever character you are assigned.'
    history.append()
    greeting_prompt = ""
    cut_output_prompted = ''
    cut_output = ''
    return generate_chat_lines(prompt, cut_output)

def htmlize(text):
    ## nothing to do here as we're returning json
    # text = markdown.markdown(text, extensions=['fenced_code', 'codehilite'])
    return text

def fixHash27(s):
    s=s.replace("&#x27;","'")
    return s

def xmlesc(txt):
    return txt.translate(table)

def truncate_history():
    global history, persona_config
    history = history[3:]
    history.insert(0, "context: " + persona_config['context'])

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

def format_prompt(user_prompt, first):
    global system_prompt, pf, persona_config, last_context, redo_persona_context, system_config, redo_greetings

    if first or redo_persona_context:
        # if redo_greetings:
        #     last_context = f"Context:  It is {now}\n{persona_config['greeting']}\n{persona_config['context']}.The user's name is {system_config['username']}\n"
        # else:
        last_context = f"Context: {persona_config['context']}.\n"
        redo_persona_context = False
        return pf.first_prompt() \
            .replace("<|system_prompt|>", last_context ) \
            .replace("<|user_prompt|>", user_prompt)
    else:
        return pf.subs_prompt() \
            .replace("<|user_prompt|>", f"{user_prompt}")
def check_meta_cmds(prompt):
    cmd = None
    pattern = ".*[iI] summon ([a-zA-Z_0-9\-]+).*"
    matches = re.match(pattern, prompt)
    if matches is not None and len(matches.groups()) > 0:
        return (configure, matches.group(1))
    if prompt.lower().startswith("forget everything."):
        return (amnesia, "")
    return (cmd, None)

def generate(prompt):
    global redo_persona_context, persona_config, token_count, model_config, parameters, tokenizer, generator, history, initialized

    (cmd, prmtr)  = check_meta_cmds(prompt)
    if cmd != None:
        return htmlize(cmd(prmtr))

    if initialized == False:
        configure(system_config['persona'])

    original_prompt = prompt

    searxPrompt = prompt

    stop_conditions = [tokenizer.newline_token_id]

    current_time = datetime.datetime.now()
    now = ""

    if( redo_persona_context):
        ctx = persona_config['context']
        if system_config['do_greeting']:
            ctx = persona_config['greeting']+ctx

    # truncate the history
    history.append(model_config["user"].replace("%%prompt%%", searxPrompt+"\n"))

    ## Unneeded due to Kadane Sliding Window?!
    if count_tokens("\n".join(history)) >= model_config['max_seq_len']:
     print("Context Overflow.")
     truncate_history()
     truncateLength = model_config['max_seq_len']-(parameters['max_tokens']+(parameters['max_tokens']*0.2))
     while(count_tokens("\n".join(history)) >= truncateLength):
         truncate_history()
#     print("New History :\n", history)

    # generate
    #full_prompt = "\nIt is now "+now+"\n"+"The chat so far:"+"\n".join(history)+"\n"+model_config["bot"]

    full_prompt = f"The chat so far :\n"+"\n".join(history)+"\n"+model_config["bot"]

    cutoff = len(full_prompt)
    token_count += count_tokens(full_prompt)
    output = generator.generate_simple(prompt = full_prompt,
                            max_new_tokens = parameters['max_tokens'])
    #print("full prompt : ", full_prompt, "output : ", output,"\n(",cutoff,"\n")
    cut_output = output[cutoff:]
    second_cut = cut_output.find("context: "+persona_config['context'])
    if second_cut > 0:
        cut_output = output[:second_cut]
    # print("cut : ",cut_output)
    history.append(persona_config['name']+": "+cut_output)
    print(f'Token count: {token_count}')
    save_log(original_prompt, cut_output)
    print("\n"+cut_output+"\n")
    return cut_output

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
    return ""

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

@app.route('/set_username', methods=['GET'])
def set_username_action():
    data = request.get_json()
    return set_username( data['name'])

@app.route( '/speak', methods=['POST'])
def speak_action():
    data = request.get_json()
    persona_config['speaker'] = data['voice']
    generate_tts(data['text'])
    return get_voice(1234)

@app.route('/amnesia', methods=['POST'])
def amnesia_action():
    return amnesia("bar")

@app.route('/configure', methods=['POST'])
def configure_action():
    data = request.get_json()
    persona = data['persona']
    return configure(persona)

@app.route('/voice/<rnd>', methods=['GET'])
def get_voice(rnd):
    return send_file("outputs/tts.wav", mimetype='audio/x-wav')

@app.route('/get_face/<persona>', methods=['GET'])
def get_face_action(persona):
    filename=f"./personas/{persona}.png"
    return send_file(filename, mimetype='image/png')

@app.route('/rpgenerate', methods=['POST'])
def generate_action():
    data = request.get_json()
    prompt = data['prompt']
    print(prompt)
    if data['amnesia'] == True:
        amnesia("bar")
    persona_config['name'] = data['npc_name']
    system_config['username'] = data['player_name']
    return generate(prompt)

@app.route('/shutdown', methods=['GET'])
def shutdown_action():
    print("shutting down.")
    exit()
    #return True


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
    redo_greetings = False
    initial_configure(system_config['persona'])
    app.run(host=system_config['host'], port=system_config['port'])