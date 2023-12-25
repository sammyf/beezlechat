import json
import math
import random
import subprocess
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
import whisper

VERSION='2'
app = Flask(__name__)
CORS(app)

MODEL_PATH="models/"

last_loader=""

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
MIN_KW_LENGTH = 4

chat_line=""
current_log_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

""" read default values """
with open(f'model_configs/defaults.yaml') as f:
    default_model_config = yaml.safe_load(f)

""" read Templates """
with open('./templates/chat_line.tpl.html') as f:
    chat_line="\n".join(f.readlines())

def read_system_config():
    """
    Reads the system configuration from the 'system_config.yaml' file.

    :return: None
    """
    global system_config
    with open('system_config.yaml') as f:
        system_config = yaml.safe_load(f)

def write_system_config():
    """
    Write the system_config dictionary to the system_config.yaml file.

    :return: None
    """
    global system_config
    with open('system_config.yaml', 'w') as f:
        # Write the dictionary to the file
        yaml.dump(system_config, f)

def retrieve_persona_dbid():
    """
    Retrieves the database id of the persona.

    :return: The database id of the persona.
    """
    global persona_config,persona_dbid
    connection = sqlite3.connect("ltm/memories.db")
    cursor = connection.cursor()
    cursor.execute(f"SELECT id FROM personas WHERE character='{persona_config['name']}'")
    persona_dbid = cursor.fetchone()[0]
    print(persona_dbid)
    connection.close()

def generate_ability_string():
    """
    Generate ability string. Disabled for now.

    :return: None
    """
    global persona_config, abilities, ability_string
    with open('abilities.yaml') as f:
        abilities = yaml.safe_load(f)
        ability_string = f"For {persona_config['name']}: these additional actions are available to you \n"
        for k in abilities:
            ability_string += f'{abilities[k]} by saying "CMD_{k}: [insert Keywords or URL here]"\n'
        ability_string += 'Use only one command at a time!'
        print(ability_string)
        #### OVERRIDE :
        ability_string = ''

def store_memory(summary,keywords):
    """
    :param summary: A string representing the summary of the memory.
    :param keywords: A string representing the keywords associated with the memory.
    :return: None
    """
    global history, persona_dbid
    print("\n\nstoring Memories")
    if (len(generate_history_string())<=MINIMUM_MEMORABLE):
        return
    now = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    connection = sqlite3.connect("ltm/memories.db")
    cursor = connection.cursor()
    fulltext = json.dumps(history)
    sql = f"INSERT into memories (creation_date, persona_id, keywords, summary, fulltext) VALUES(?,?,?,?,?)"
    params = (now, persona_dbid, "##"+"##".join(keywords)+"##", summary, fulltext)
    print(persona_dbid, f"{keywords} ::: {summary}")
    cursor.execute(sql, params)
    connection.commit()
    connection.close()

def save_log(userline, botline):
    """
    :param userline: The line of text representing the user's input.
    :param botline: The line of text representing the bot's response.
    :return: None

    """
    global current_log_file
    try:
        if system_config["log"] is True:
            with open(f'logs/{current_log_file}.txt', 'a') as f:
                f.write(f'user: {userline}\n{botline}\n\n')
    except:
        print("Error writing Log.")

def tabby_load_model(model):
    """
    Load the specified model.

    :param model: The name of the model to load.
    :return: None

    Example usage:
        >>> tabby_load_model("my_model")
        Loading my_model
        200
        {"message": "Model loaded successfully"}
    """
    global model_config, loaded_model
    print(f"Loading {model}")
    url = f"{oai_config['tby_server']}/v1/model/load"
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
        "x-admin-key": oai_config['tby_admin']
    }
    response = requests.post(url, data=json.dumps(payload), headers=headers)
    loaded_model = model
    print(response.status_code)
    print(response.text)

def tabby_unload():
    """
    Unloads the Tabby model.

    :return: None
    """
    print(f"Unloading {model}")
    url = f"{oai_config['tby_server']}/v1/model/unload"
    headers = {
        "Content-Type": "application/json",
        "x-admin-key": oai_config['tby_admin']
    }
    response = requests.get(url, headers=headers)

def ollama_stop():
    subprocess.run(["/bin/sudo", "/bin/systemctl", "restart", "ollama"])

def ollama_restart():
    subprocess.run(["sudo", "/bin/systemctl", "restart", "ollama"])

def ollama_generate(original_prompt):
    global persona_config,system_config, history
    rs = {
        "model": persona_config["model"]+":"+model_config["tag"],
        "messages": [],
        "stream": True,
        "context": "",
        "options": {}
    }
    with open(f"parameters/{model_config['loader']}_{persona_config['parameters']}.yaml") as f:
        rs['options'] = yaml.safe_load(f)
    seed = random.randint(math.ceil(sys.float_info.min), math.floor(sys.float_info.max))
    print(f"Seed: {seed}")
    rs["options"]["seed"]=seed
    rs["options"]["num_predict"] = persona_config['max_answer_token']
    if 'stop' in model_config.keys():
        rs["options"]["stop"] = model_config["stop"].split(',')

    e = {
        "role": ""
        "content" ""
    }

    current_time = datetime.datetime.now()
    now = current_time.strftime("%H:%M:%S on %A, %d %B %Y")
    rs["context"] = f"It is {now}\n.The user's name is {system_config['username']}.\n{persona_config['context']}"

    if original_prompt.strip() != "":
        rs = generate_history_array(rs)
    else:
        e["role"] = "user"
        e["content"] = "please continue."
        rs = generate_history_array(rs)
        rs["messages"].append(e)

    url = f"{oai_config['ollama_server']}/api/chat"

    headers = {
        "Content-Type": "application/json",
    }

    # response = requests.post(url, data=json.dumps(rs, indent=0))
    rs_stream = requests.post(url, data=json.dumps(rs, indent=0), stream=True)
    response = ""
    js = ""
    if rs_stream.ok:
        for line in rs_stream.iter_lines():
            if line:  # filter out keep-alive new lines
                js = json.loads(line.decode('utf-8'))
                # print("stream",js)
                response += js["message"]["content"]
                if js['done']:
                    break;
    else:
        print('Error:', rs_stream.status_code)


    print("Last response : ",js)
    # rsjson = json.loads(response.text)
    try:
         prompt_speed = "{0:.2f}".format((js['prompt_eval_count']/float(js['prompt_eval_duration']/100000000)))
         answer_speed = "{0:.2f}".format((js['eval_count']/float(js['eval_duration']/100000000)))
         print( f"ollama: token stats:\n    {prompt_speed} t/s)\n    {answer_speed} t/s)\n")
    except:
         print( "Could not compute token stats")
    # return rsjson["message"]["content"]
    return response

def tabby_generate(original_prompt):
    """
    Generate a response using the Tabby API.

    :param prompt: The prompt for generating the response.
    :return: The generated response.
    """
    global persona_config,system_config, history

    with open(f"parameters/{model_config['loader']}_{persona_config['parameters']}.yaml") as f:
        parameters = yaml.safe_load(f)

    current_time = datetime.datetime.now()
    now = current_time.strftime("%H:%M:%S on %A, %d %B %Y")

    if original_prompt.strip() != "":
        prompt = f"It is {now}\n.The user's name is {system_config['username']}\nThe chat so far :\n" + generate_history_string() + "\n" + \
                      model_config["bot"]
    else:
        prompt = "The chat so far :\n" + generate_history_string() + "\nplease continue." + model_config["bot"]

    seed = random.randint(math.ceil(sys.float_info.min), math.floor(sys.float_info.max))
    print(f"Seed: {seed}")
    parameters["seed"] = seed
    parameters['model']=persona_config['model']
    parameters['prompt']=prompt
    parameters['max_tokens']=persona_config['max_answer_token']
    parameters['user']=system_config["username"]
    if 'stop' in model_config.keys():
        parameters["stop"] = model_config["stop"].split(',')

    url = f"{oai_config['tby_server']}/v1/completions"

    headers = {
        "Content-Type": "application/json",
        "x-api-key": oai_config['tby_admin']
    }
    answer = ""
    while answer.strip() == "":
        response = requests.post(url, data=json.dumps(parameters, indent=0), headers=headers)
        print(response.status_code)
        print("response : ",response.text)
        rsjson = json.loads(response.text)
        answer = rsjson["choices"][0]['text']
    return answer

def word_count(prompt):
    return len(prompt.split())

def tabby_count_tokens(prompt):
    """
    :param prompt: The text prompt to be tokenized.
    :return: The number of tokens in the given prompt.
    """
    if model_config['loader'] != 'tabby':
        return word_count(prompt)
    url = f"{oai_config['tby_server']}/v1/token/encode"
    headers = {
        "Content-Type": "application/json",
        "x-api-key": oai_config['tby_admin']
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

generators = {
    'tabby': tabby_generate,
    'ollama': ollama_generate
}

# Used for making text xml compatible, needed for voice pitch and speed control
table = str.maketrans({
    "<": "&lt;",
    ">": "&gt;",
    "&": "&amp;",
    "'": "&apos;",
    '"': "&quot;",
})

def count_tokens(txt):
    """
    :param txt: the string to count tokens from
    :return: the number of tokens in the string
    """
    if model_config['loader'] != 'tabby':
        return word_count(txt)
    return tabby_count_tokens(txt)

def generate_chat_lines(user, bot):
    """
    :param user: The user's message to be replaced in the chat line.
    :param bot: The bot's response to be replaced in the chat line.
    :return: The generated chat line with user and bot messages replaced.
    """
    rs = chat_line.replace("%%user%%", user)\
        .replace("%%bot%%", bot)\
        .replace('%%botname%%', persona_config["name"])
    return rs

""" 
finds out at which index the history buffer contains at least half of the context
this is needed to create memories and to truncate the history buffer accordingly
"""
def find_half_of_context():
    """
    Find the cutting point in the history list that corresponds to half the number of tokens in the model configuration.

    :return: The cutting point index in the history list.
    """
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

def generate_history_array( rs):
    global model_config, history, persona_config

    for h in history:
        e = {
            'role': '',
            'content': ''
        }
        if h[0]=="s":
            e["role"] = "system"
            e["content"] = h[1]
        elif h[0]=="u":
            e["role"] = "user"
            e["content"] = h[1]
        else:
            e["role"] = "assistant"
            e["content"] = h[1]
        rs["messages"].append(e)
    return rs

def generate_history_string():
    """
    Generates a formatted string representing the conversation history.

    :return: A string representing the conversation history.
    """
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
    """
    Truncates the chat history to half its original length and adds a summary of the discussion to the remaining history.

    :return: None
    """
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
    history.insert(0, "\nsummary of the discussion this far :"+summary+"\ n")
    print("len with context :",count_tokens(generate_history_string()))
    print("len with context :",count_tokens(generate_history_string()))

def generate_keywords(prompt=""):
    """
    Generate keywords based on the provided prompt.

    :param prompt: optional prompt string (default: "")
    :return: tuple containing the summary and keywords generated
    """
    summary=prompt
    if prompt == "":
        summary=generate("Please summarize the discussion this far.", True)
        print('summary generated : ', summary)
    _keywords=generate ( "give me single keywords for this paragraph:"+summary, True)
    keywords={}
    for k in _keywords:
        if len(k) < MIN_KW_LENGTH:
            continue
        keywords.append(k)
    return(summary, keywords)

def generate_memory():
    """
    Generate Memory

    This method generates memory based on the retrieved summary and keywords. It stores the memory in the global `history` variable.

    :return: The generated summary of the memory.
    """
    global history
    (summary, keywords) = generate_keywords()
    store_memory(summary,keywords)
    return summary

def context_management():
    """
    This method is used for context management. It checks if the number of tokens in the history string generated is greater than or equal to the maximum allowed sequence length minus the
    * maximum number of answer tokens and the modulo of the maximum sequence length with 10. If the condition is true, it prints "Context Overflow" and truncates the history.

    :return: None
    """
    if count_tokens(generate_history_string()) >= (model_config['max_seq_len']-(persona_config['max_answer_token']+(model_config['max_seq_len']%10))):
     print("Context Overflow.")
     truncate_history()

def load_model(model):
    """
    :param model: The name of the model to load.
    :return: None

    This method is used to load a specific model using the provided name. It first loads the model configuration from a YAML file based on the given model name. If the file does not exist
    * or cannot be loaded, an empty dictionary is used as the model configuration.

    Next, it checks if any keys in the default model configuration are missing in the loaded model configuration and adds them if necessary. It also replaces placeholders in the 'bot' and
    * 'user' values of the model configuration with the appropriate values from the persona and system configurations, respectively.

    If the loaded model is different from the model currently stored in the persona configuration, the current model is unloaded and the specified model is loaded using the 'tabby_load_model
    *' function.

    Finally, the loaded model is printed for verification.

    Example usage:
    load_model("model_name")
    """
    global model_config, system_config, persona_config, loaded_model, last_loader
    ## Load Model Configuration (maximum context, template, ...)
    try:
        with open(f'model_configs/{model.replace(":","_")}.yaml') as f:
            model_config = yaml.safe_load(f)
    except:
        model_config = {}
        return

    # just unload any potentially loaded models from tabby if we are using ollama
    if model_config["loader"] == "tabby" and last_loader == "ollama":
        last_loader = "tabby"
        ollama_stop()
    if model_config["loader"] == "ollama":
        if last_loader == "tabby":
            ollama_restart()
            try:
                tabby_unload()
            except:
                pass
        loaded_model = persona_config["model"]
        print("\n\nloaded model :", loaded_model, "\n")
        return

    for k in default_model_config.keys():
        if k not in model_config.keys():
            model_config[k] = default_model_config[k]
    model_config['bot'] = model_config['bot'].replace("%%botname%%", persona_config["name"])
    model_config['user'] = model_config['user'].replace("%%username%%", system_config["username"])
    if(loaded_model != persona_config["model"]):
        tabby_unload()
        tabby_load_model(model)
    loaded_model = persona_config["model"]
    print("\n\nloaded model :", loaded_model,"\n")

def configure(persona):
    """
    :param persona: The name of the persona to configure. This is used to load the corresponding persona configuration file located in the "personas" directory.
    :return: The result of calling the generate_chat_lines method with the configured prompt and cut_output.
    """
    global last_loader, old_persona, persona_config, generator, initialized,tokenizer, cache, parameters, model_config, mozTTS, system_config, pf, redo_persona_context, redo_greetings, persona_dbid, loaded_model

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
    #generate_ability_string()

    # if initialized:
    #     generate_memory()

    amnesia("bar")
    retrieve_persona_dbid()
    print(persona_config["model"])
    if 'parameters' not in persona_config:
        persona_config['parameters'] = 'default'

    #with open(MODEL_PATH+persona_config['model']+'/config.json') as f:
    #    cfg = json.load(f)

    mozTTS = moztts.MozTTS()
    mozTTS.load_model(persona_config['voice'])
    initialized = True
    last_character = persona_config["name"]

    load_model(persona_config['model'])

    # do the greetings
    prompt = f'{persona_config["context"]}\n You are {last_character}\n'
    history.append(["s",prompt])
    if system_config['do_greeting'] is True:
        cut_output = persona_config['greeting']
    else:
        cut_output = f"{persona_config['name']} appears."
    redo_persona_context = True

    #context_management()

    generate_tts(cut_output)
    save_log(prompt, cut_output)
    return generate_chat_lines(prompt, htmlize(cut_output))

def htmlize(text):
    """
    Convert the plaintext `text` into HTML format.

    :param text: The plaintext to be converted.
    :return: The converted HTML string.
    """
    # pattern = "```(.+?)```"
    # text = re.sub(pattern,"<code>\1</code>",text)
    text = markdown.markdown(text, extensions=['fenced_code', 'codehilite'])
    text = text.replace("\\n","<br/>")
    return text

def fixHash27(s):
    """
    Fixes the hash symbol encoded as "&#x27;" and replaces it with the character "'".

    :param s: the input string with encoded hash symbol "&#x27;"
    :return: the fixed string with hash symbol replaced
    """
    s=s.replace("&#x27;","'")
    return s

def xmlesc(txt):
    """
    :param txt: The input text to be escaped
    :return: The escaped text

    """
    return txt.translate(table)

def generate_tts(string):
    """
    :param string: The text to be converted to speech.
    :return: The path of the generated audio file.
    """
    global persona_config, mozTTS, system_config

    if system_config['do_tts'] != True:
        return

    string = fixHash27(string)
    original_string = string

    output_file = "./audio/tts.wav"
    if string == '':
        string = '*Empty reply, try regenerating*'
    else:
        output_file = './audio/tts.wav'
        mozTTS.moztts(string, persona_config['voice'], persona_config['speaker'], persona_config['language'], output_file)

    return output_file

def check_meta_cmds(prompt):
    """
    Check if prompt matches any meta commands and return a corresponding action.

    :param prompt: the input prompt to be checked
    :return: a tuple containing the command and additional parameters if applicable
    """
    cmd = None
    news_sources = [
    'https://news.yahoo.com',
    'https://news.bbc.co.uk',
    'https://www.aljazeera.com',
    'https://www.npr.org',
    'https://www.tageschau.de'
    ]
    pattern = ".*[iI] summon ([a-zA-Z_0-9\-]+).*"
    matches = re.match(pattern, prompt)
    if matches is not None and len(matches.groups()) > 0:
        return (configure, matches.group(1).lower().capitalize())
        return (configure, matches.group(1).lower().capitalize())
    if prompt.lower().startswith("forget everything"):
        return (amnesia, "")
    if prompt.lower().startswith("check the news"):
        source = random.choice(news_sources)
        print(f"\nNews Source selected : {source}\n")
        return (None,f"Read {source}. tell me about the headlines you found there. Translate them to English if they aren't in English")
    if "##shutdown now##" in prompt.lower():
        shutdown_action()
        exit()
    return (cmd, None)

def generate(prompt, raw=False):
    global generators, redo_persona_context, persona_config, token_count, model_config, parameters, tokenizer, generator, history, initialized, ability_string

    (cmd, prmtr)  = check_meta_cmds(prompt)
    if cmd != None:
        return htmlize(cmd(prmtr))
    elif prmtr != None:
        prompt = prmtr

    if initialized == False:
        configure(system_config['persona'])

    original_prompt = prompt
    searxPrompt = Searxing.check_for_trigger(prompt, model_config['max_seq_len']/2, count_tokens)
    if searxPrompt == '':
        searxPrompt = prompt

    if( redo_persona_context):
        ctx = persona_config['context']
        if system_config['do_greeting']:
            ctx = persona_config['greeting']+ctx

    # save the user prompt in the history (with or without model-specific prompt formats)
    # history.append(model_config["user"].replace("%%prompt%%", searxPrompt+"\n"))
    history.append(["u",searxPrompt])

    if model_config['loader'] == 'tabby':
        context_management()

    cut_output = generators[model_config['loader']](original_prompt)

    if raw:
        return cut_output
    # disabling meta answers, as it tends to fall in weird loops
    # cut_output = check_meta_answer(cut_output)
    history.append(["b",cut_output])
    if model_config['loader'] == 'tabby':
        token_count = count_tokens(generate_history_string())
        print(f'Token count: {token_count}')
    generate_tts(cut_output)
    save_log(original_prompt, cut_output)
    return generate_chat_lines(original_prompt, htmlize(cut_output))

def list_personas():
    """
    Return a formatted string containing HTML code for a dropdown list of personas.

    :return: A string containing HTML code for the dropdown list of personas.
    """
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
    """
    Return a string containing HTML options for selecting model configurations.

    :return: A string containing HTML options for selecting model configurations.
    """
    global model_config,loaded_model
    models_bin = glob.glob("./models/*")
    models_config_raw = glob.glob("./model_configs/*.yaml")
    models_config = []
    models = []
    print('\nconfigs available:')
    for p_raw in models_config_raw:
        p = p_raw.replace(".yaml", "").replace("./model_configs/", "")
        models_config.append(p)

    print('\nmodels available:')
    for b_raw in models_bin:
        b = b_raw.replace("./models/", "")
        if b in models_config:
            models.append(f"./model_configs/{b}.yaml")
        else:
            print(f"\n{b} (tabby) doesn't have a valid config")

    # get the ollama models
    url = f"{oai_config['ollama_server']}/api/tags"
    response = requests.get(url)
    rsjson = json.loads(response.text)
    for model in rsjson['models']:
        p_raw = model['name']
        p = p_raw[0:p_raw.find(':')]
        if p in models_config:
            models.append(f"./model_configs/{p}.yaml")
        else:
            print(f"\n{p} (ollama) doesn't have a valid config")

    models.sort()
    rs = ""
    for p_raw in models:
        with open(p_raw) as f:
            tmp=yaml.safe_load(f)
            ldr = tmp['loader']

        loader = ldr[0].upper()
        p=p_raw.replace(".yaml","").replace("./model_configs/","")
        selected=""
        if p == loaded_model:
            selected = "selected"
        rs +=f"<option value='{p}' {selected} class='tabby'>({loader}) {p}</option>"



    return rs

def set_username(uname):
    """
    ``set_username(uname)``
    -----------------------

    Sets the username for the system configuration.

    Parameters:
        uname (str): The username to be set.

    Returns:
        str: The updated username.

    """
    global system_config
    if uname.strip() == "":
        uname = "User"
    system_config['username'] = uname.strip()
    write_system_config()
    return system_config['username']

def toggle_tts():
    """
    Toggle TTS Mode

    This method toggles the TTS (Text To Speech) mode in the system configuration.

    :return: A string indicating the status of the TTS mode after toggling.
    """
    global system_config
    system_config['do_tts'] = (system_config['do_tts']^True)
    write_system_config()
    return "toggled."

def amnesia(foo):
    """
    Clear the history and reset token count if foo is not equal to "bar".

    :param foo: A string value.
    :return: None.
    """
    global user_prompts, responses_ids, history, token_count
    history = []
    token_count = 0
    if foo != "bar":
        return configure(persona_config['name'])

###
# Routes
@app.route('/', methods=['GET'])
def get_index():
    """
    Get the contents of the index.html file, replace the placeholders with appropriate values,
    and return the modified content.

    :return: The modified content of the index.html file.
    """
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
    autonomy = "false"
    if system_config["do_tts"]:
        tts_onoff="true"
    if system_config["autonomy"]:
        autonomy="true"
    index=index.replace("{tts_toggle_state}", tts_onoff)
    index=index.replace("{autonomy}", autonomy)
    return index
    #send_file("index.html", mimetype='text/html')

@app.route('/set_username/', methods=['GET'])
def set_username_action():
    """
    Perform the action of setting the username.

    :return: None
    """
    return set_username( request.form['username'])

@app.route('/voice/<rnd>', methods=['GET'])
def get_voice(rnd):
    """
    :param rnd: A random number used to generate the voice.
    :type rnd: str

    :return: The voice file in WAV format.
    :rtype: file

    :raises: FileNotFoundError if the voice file is not found.

    This method generates a voice file using the given `rnd` parameter
    and returns it as a WAV file.
    """
    return send_file("audio/tts.wav", mimetype='audio/x-wav')

@app.route('/css/styles.css', methods=['GET'])
def get_css():
    """
    Returns the styles.css file.

    :return: The styles.css file as a response.
    """
    return send_file("css/styles.css", mimetype='text/css')

@app.route('/css/codehilite.css', methods=['GET'])
def get_codehilite():
    """
    This method returns the codehilite CSS file.

    :return: The codehilite CSS file.
    """
    return send_file("css/codehilite.css", mimetype='text/css')

@app.route('/js/script.js', methods=['GET'])
def get_js():
    """
    Returns the content of the 'js/script.js' file as a response to a GET request made to '/js/script.js'.

    :return: The content of the 'js/script.js' file as a response object with the mimetype set to 'application/javascript'.
    """
    return send_file("js/script.js", mimetype='application/javascript')

# if 'X-CSRF-Token' in request.headers:
@app.route('/configure', methods=['POST'])
def configure_action():
    """
    Configure action method.

    :return: None
    """
    persona = request.form['persona']
    return configure(persona)

@app.route('/list_models', methods=['GET','POST'])
def list_models_action():
    """
    Returns a list of models.

    :return: a list of models.
    """
    return list_models()

@app.route('/load_model', methods=['GET','POST'])
def load_model_action():
    """
    Change the current model and load it into memory.

    :return: None
    """
    model = request.form['model']
    load_model(model)
    list_models()

@app.route('/toggle_tts', methods=['GET','POST'])
def toggle_tts_action():
    """
    Toggles the text-to-speech (TTS) action.

    :return: None
    """
    return toggle_tts()

@app.route('/list_personas', methods=['GET','POST'])
def list_personas_action():
    """
    Returns a list of personas.

    :return: A list of personas.
    :rtype: list
    """
    return list_personas()

@app.route('/get_face/<persona>', methods=['GET'])
def get_face_action(persona):
    """
        :param persona: The persona of the face image to retrieve. It is a string representing the persona.
        :return: The face image associated with the specified persona in PNG format.
    """
    filename=f"./personas/{persona}.png"
    return send_file(filename, mimetype='image/png')

@app.route('/imgs/spinner.gif', methods=['GET'])
def get_img_action():
    """
    Returns the image file specified by the URL route '/imgs/spinner.gif' with the GET method.

    :return: The image file specified by the URL route.
    """
    filename=f"./imgs/spinner.gif"
    return send_file(filename, mimetype='image/gif')

@app.route('/imgs/refresh.png', methods=['GET'])
def get_refresh_action():
    """
    This method handles the GET request to retrieve the refresh action image.

    :return: The refresh action image file.
    """
    filename=f"./imgs/refresh.png"
    return send_file(filename, mimetype='image/png')

@app.route('/generate', methods=['POST'])
def generate_action():
    """
    Generate an action based on user input.

    :return: None
    """
    prompt = request.form['prompt']
    return generate(prompt)

@app.route('/whisper', methods=['POST'])
def whisper_action():
    global whisper_model
    audio = request.files['audio']
    audio.save('audio/input.ogg')
    result = whisper_model.transcribe("audio/input.ogg")
    return result["text"]

@app.route('/rpgenerate', methods=['POST'])
def rpgenerate_action():
    """
    Generates an action based on the provided prompt.

    :return: The generated action.

    :raises: `KeyError` if the required fields are missing in the request data.

    POST Parameters:
    - `prompt` (string): The prompt for generating the action.
    - `amnesia` (boolean): Whether to trigger amnesia or not.
    - `npc_name` (string): The name of the NPC.
    - `player_name` (string): The name of the player.

    Example Usage:
    ```python
    import requests

    url = "http://example.com/rpgenerate"
    data = {
        'prompt': 'Go to the village and search for clues.',
        'amnesia': True,
        'npc_name': 'John Doe',
        'player_name': 'Jane Doe'
    }
    response = requests.post(url, json=data)
    action = response.text
    print(action)
    ```
    """
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
    """
    Sends a text to be spoken by the server's configured speaker.

    :return: The generated spoken voice.
    """
    data = request.get_json()
    persona_config['speaker'] = data['voice']
    generate_tts(data['text'])
    return get_voice(1234)

@app.route('/shutdown', methods=['GET'])
def shutdown_action():
    """
    Performs shutdown actions.

    :return: None
    """
    generate_memory()
    print("shutting down.")
    exit()
    #return True


def on_terminate(signum, frame):
    """
    Handle termination signal.

    :param signum: the signal number
    :param frame: the current stack frame
    :return: None
    """
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
    whisper_model = whisper.load_model("base")
    configure(system_config['persona'])
    signal.signal(signal.SIGTERM, on_terminate)
    app.run(host=system_config['host'], port=system_config['port'])
