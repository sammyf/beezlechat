import base64
import hashlib
import math
import random
import shutil
import subprocess
import sys, os
import urllib
from urllib.parse import quote

import ffmpeg
from langchain_community.llms.ollama import Ollama

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from flask import Flask, request, send_file, jsonify, render_template, send_from_directory
from flask_cors import CORS
import yaml
import torch
import glob
import moztts
import re
import markdown
from searxing import SearXing
from ollama_langchain import OllamaLangchain
import sys
import datetime
import signal
import requests
import json
import whisper


VERSION='2'
app = Flask(__name__)
CORS(app)

MODEL_PATH="models/"

last_loader=""
requrl="http://127.0.0.1"

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


current_log_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

""" read default values """
with open(f'model_configs/defaults.yaml') as f:
    default_model_config = yaml.safe_load(f)

with open('system_config.yaml') as f:
    system_config = yaml.safe_load(f)

""" read Templates """
# if system_config['do_wav2lip']:
#     with open('./templates/chat_line_animated.tpl.html') as f:
#         chat_line="\n".join(f.readlines())
# else:
with open('./templates/chat_line.tpl.html') as f:
     chat_line="\n".join(f.readlines())

def write_system_config():
    """
    Write the system_config dictionary to the system_config.yaml file.

    :return: None
    """
    global system_config
    with open('system_config.yaml', 'w') as f:
        # Write the dictionary to the file
        yaml.dump(system_config, f)

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
    print(url)
    headers = {
        "Content-Type": "application/json",
        "x-admin-key": oai_config['tby_admin']
    }
    response = requests.post(url, headers=headers)

def ollama_restart():
    subprocess.run(["sudo", "/bin/systemctl", "restart", "ollama"])

def ollama_generate(original_prompt, client):
    global persona_config,system_config, historyMP, usernames
    rs = {
        "model": persona_config["model"]+":"+persona_config["tag"],
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
    rs["context"] = f"It is {now}\n.The user's name is {usernames[client]}.\n{persona_config['context']}"

    if original_prompt.strip() != "":
        rs = generate_history_array(rs,client)
    else:
        e["role"] = "user"
        e["content"] = "please continue."
        rs = generate_history_array(rs, client)
        rs["messages"].append(e)

    url = f"{oai_config['ollama_server']}/api/chat"

    headers = {
        "Content-Type": "application/json",
    }

    # print( "\n:::::::::\n")
    # print(json.dumps(rs, indent=4))
    # print( "\n:::::::::\n")

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
                    break
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

def tabby_generate(original_prompt,client):
    """
    Generate a response using the Tabby API.

    :param original_prompt: The prompt for generating the response.
    :return: The generated response.
    """
    global persona_config,system_config, historyMP, usernames

    with open(f"parameters/{model_config['loader']}_{persona_config['parameters']}.yaml") as f:
        parameters = yaml.safe_load(f)

    current_time = datetime.datetime.now()
    now = current_time.strftime("%H:%M:%S on %A, %d %B %Y")

    if original_prompt.strip() != "":
        prompt = f"It is {now}\n.The user's name is {usernames[client]}\nThe chat so far :\n" + generate_history_string(client) + "\n" + \
                      model_config["bot"]
    else:
        prompt = "The chat so far :\n" + generate_history_string(client) + "\nplease continue." + model_config["bot"]

    seed = random.randint(math.ceil(sys.float_info.min), math.floor(sys.float_info.max))
    print(f"Seed: {seed}")
    parameters["seed"] = seed
    parameters['model']=persona_config['model']
    parameters['prompt']=prompt
    parameters['max_tokens']=persona_config['max_answer_token']
    parameters['user']=usernames[client]
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

def generate_chat_lines(user, bot, upimg_name=None, genimg_name=None):
    """
    :param user: The user's message to be replaced in the chat line.
    :param bot: The bot's response to be replaced in the chat line.
    :return: The generated chat line with user and bot messages replaced.
    """
    ## moving the animation to the top of the screen and keeping the portrait
    ## in the chatlines
    if upimg_name is not None:
        with open('templates/upload.tpl.html', "r") as f:
            uploadTpl = f.read()
        user = (uploadTpl.replace("%%imgname%%", upimg_name.replace("/","").replace("\\","")).
                replace("%%user%%", user)).replace("%%rnd%%",str(random.randint(0,99999999999)))
    if genimg_name is not None:
        with open('templates/genimg.tpl.html', "r") as f:
            uploadTpl = f.read()
        bot = (uploadTpl.replace("%%imgname%%", genimg_name.replace("/","").replace("\\","").replace("generated","generated/")).
                replace("%%bot%%", bot)).replace("%%rnd%%",str(random.randint(0,99999999999)))
    rs = (chat_line.replace("%%user%%", user)\
        .replace("%%bot%%", bot)\
        .replace('%%botname%%', persona_config["name"])
          .replace("%%rnd%%", str(random.randint(0,99999999999))))
    return rs

""" 
finds out at which index the history buffer contains at least half of the context
this is needed to create memories and to truncate the history buffer accordingly
"""
def find_half_of_context(client):
    """
    Find the cutting point in the history list that corresponds to half the number of tokens in the model configuration.

    :return: The cutting point index in the history list.
    """
    global historyMP, model_config
    half = model_config['max_seq_len']/2
    print(f"looking for {half} tokens ...")
    cutting_point = 0
    token_count = 0
    for e in historyMP[client]:
        token_count += count_tokens(e[1])
        if token_count > half:
            return cutting_point
        cutting_point += 1
    return cutting_point

def generate_history_array( rs, client):
    global model_config, historyMP, persona_config

    for h in historyMP[client]:
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

def generate_history_string(client):
    """
    Generates a formatted string representing the conversation history.

    :return: A string representing the conversation history.
    """
    global model_config, historyMP
    rs = ""
    for h in historyMP[client]:
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

def truncate_history(client):
    """
    Truncates the chat history to half its original length and adds a summary of the discussion to the remaining history.

    :return: None
    """
    global historyMP, persona_config
    current_history = historyMP[client]
    print("len full:",count_tokens(generate_history_string(client)))
    half = find_half_of_context(client)
    print("half :", half)
    historyMP[client] = historyMP[client][:half]
    print("len half:",count_tokens(generate_history_string(client)))
    summary = generate_memory(client)
    historyMP[client] = current_history[half:]
    print("len half after cut :",count_tokens(generate_history_string(client)))
    historyMP[client].insert(0, "\nsummary of the discussion this far :"+summary+"\ n")
    print("len with context :",count_tokens(generate_history_string(client)))
    print("len with context :",count_tokens(generate_history_string(client)))

def generate_keywords(client, prompt=""):
    """
    Generate keywords based on the provided prompt.

    :param prompt: optional prompt string (default: "")
    :return: tuple containing the summary and keywords generated
    """
    summary=prompt
    _keywords = []
    if prompt == "":
        summary=generate("Please summarize the discussion this far.", client,True)
        print('summary generated : ', summary)
    _keywords=generate ( "give me single keywords for this paragraph:"+summary, client,True)
    keywords={}
    for k in _keywords:
        if len(k) < MIN_KW_LENGTH:
            continue
        keywords.append(k)
    return(summary, keywords)

def generate_memory(client):
    """
    Generate Memory

    This method generates memory based on the retrieved summary and keywords. It stores the memory in the global `history` variable.

    :return: The generated summary of the memory.
    """
    global historyMP
    (summary, keywords) = generate_keywords(client)
    return summary

def context_management(client):
    """
    This method is used for context management. It checks if the number of tokens in the history string generated is greater than or equal to the maximum allowed sequence length minus the
    * maximum number of answer tokens and the modulo of the maximum sequence length with 10. If the condition is true, it prints "Context Overflow" and truncates the history.

    :return: None
    """
    if count_tokens(generate_history_string( client)) >= (model_config['max_seq_len']-(persona_config['max_answer_token']+(model_config['max_seq_len']%10))):
     print("Context Overflow.")
     truncate_history(client)

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
        ollama_restart()
    elif model_config["loader"] == "ollama":
        try:
            tabby_unload()
        except:
            pass
        last_loader = "ollama"
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

def configure(persona, client=None):
    """
    :param persona: The name of the persona to configure. This is used to load the corresponding persona configuration file located in the "personas" directory.
    :return: The result of calling the generate_chat_lines method with the configured prompt and cut_output.
    """
    global last_loader, old_persona, persona_config, generator, initialized,tokenizer, \
        cache, parameters, model_config, mozTTS, system_config, pf, redo_persona_context, \
        redo_greetings, persona_dbid, loaded_model

    try:
         with open(f'./personas/{persona}.yaml') as f:
            persona_config = yaml.safe_load(f)
    except:
         if not initialized:
             with open(f'./personas/{system_config["fallback"]}.yaml') as f:
                 persona_config = yaml.safe_load(f)
         else:
            generate_tts("Sorry. There is no persona with that name.")
            return generate_chat_lines(f"I summon {persona}.","Sorry. There is no persona with that name.")

    if initialized:
        old_persona = persona_config["name"]
    else:
        old_persona = "sleeping"


    system_config['persona'] = persona
    write_system_config()

    if "language" not in persona_config:
        persona_config['language'] = ''
    #generate_ability_string()

    # if initialized:
    #     generate_memory()

    amnesia( None)
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


def run_w2l_blocking( target):
    command = "./w2l.sh"
    subprocess.call([command, target])

def run_w2l_api( target):
    global persona_config
    if "disable_w2l" in persona_config and persona_config["disable_w2l"] == True:
        print("\n\nbypassing W2L\n\n")
        try:
            os.remove("/media/GINTONIC/AIArtists/beezlechat/video/result.mp4")
        except:
            pass
        video = ffmpeg.input(f"video/{target}_idle.mp4").video
        audio = ffmpeg.input("/media/GINTONIC/AIArtists/beezlechat/audio/tts.wav").audio
        out = ffmpeg.output(video, audio, "/media/GINTONIC/AIArtists/beezlechat/video/result.mp4", vcodec='copy',
                            acodec='aac', strict='experimental')
        out.run()
        return
    rs={
        "checkpoint_path": "checkpoints/wav2lip.pth",
        "face": f"/media/GINTONIC/AIArtists/Wav2Lip/targets/{target}_talk_long.mp4",
        "audio": "/media/GINTONIC/AIArtists/beezlechat/audio/tts.wav",
        "outfile": "/media/GINTONIC/AIArtists/beezlechat/video/result.mp4"
    }
    requests.post(oai_config['w2l_endpoint'], data=json.dumps(rs, indent=0))

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
        if os.path.exists(output_file):
            os.unlink(output_file)
        mozTTS.moztts(string, persona_config['voice'], persona_config['speaker'], persona_config['language'], output_file)
        if system_config['do_wav2lip']:
            while not os.path.exists(output_file):
                pass
            try:
                run_w2l_api(persona_config['name'])
            except:
                print("w2l failed.")
                system_config['do_wav2lip'] = False
    return output_file

def generate_image(prompt, client):
    global persona_config, system_config, oai_config
    if persona_config["generate_img"] == False:
        generate_tts("I'm sorry, but there is no space left to load an image generator.")
        return generate_chat_lines(f"{prompt}", f"I'm sorry, but there is no space left to load an image generator.")
    pattern = ".*generate an image of (.+)$"
    matches = re.match(pattern, prompt)
    if matches is not None:
        clean_prompt = matches.group(1)
    else:
        pattern = ".*\/imagine (.+)$"
        matches = re.match(pattern, prompt)
        if matches is not None:
            clean_prompt = matches.group(1)
        else:
            print("Looking for a topic.")
            topic = generate("what is the topic of this page? summarize the content of this random wikipedia page https://en.wikipedia.org/wiki/Special:Random", client, True)
            clean_prompt = f" something about the topic mentioned in \"{topic}\""
    sdprompt = generate(f"write a stable diffusion prompt to generate a high quality image of {clean_prompt}. Add keywords about the medium (for example: photography, drawing, painting, 3D, ...)"
                        f"style (for example: photo realistic, cartoon, anime, surrealistic, pointillism, etc), detail level and lighting at the end of the prompt.", client, True)
    print(f"prompt : '{clean_prompt}'")
    with (open("invoke/t2i-generation.json", "r") as f):
        invokePrompt = json.load(f)
    sdprmpt = sdprompt.replace("\n","").replace("\"","'")
    invokePrompt["prompt"] = invokePrompt["prompt"].replace("%%PROMPT%%", sdprmpt)
    invokePrompt["checkpoint"] = invokePrompt["checkpoint"].replace("%%MODEL%%", persona_config["checkpoint"])
    invokePrompt["firstphase_width"] =  persona_config["image_size"]
    invokePrompt["firstphase_height"] = persona_config["image_size"]
    print(f"\n:::::::::::\n{json.dumps(invokePrompt, indent=4)}\n::::::::::::::\n")
    rs_raw = requests.post(f'{oai_config["a1111"]}sdapi/v1/txt2img', json=invokePrompt, headers={"Content-Type": "application/json"})
    rs = json.loads(rs_raw.text)

    # Get current date as a string
    current_date = str(datetime.datetime.now())
    # Combine data with current date
    combined_data = f"{sdprmpt}{current_date}"
    # Generate SHA256 hash
    hash_object = hashlib.sha256(combined_data.encode())
    hex_dig = hash_object.hexdigest()
    fname = f"generated/{hex_dig}.png"
    img = base64.b64decode(rs['images'][0])
    try:
        with open(f"uploads/{fname}", 'wb') as f:
            f.write(img)
    except:
        generate_tts("Something went wrong!")
        return generate_chat_lines(prompt, "Something went wrong.", client)

    generate_tts(f"Here is what I came up with : '{sdprompt}'")
    genlines = generate_chat_lines(f"{prompt}", f"Here is what I came up with : '{sdprompt}'", None, fname)
    if persona_config['look_at_gen'] == True:
        analysis = image_analysis(rs['images'][0], fname, client, "This is the image that was generated based on your prompt.")
    generate_tts(f"Here is what I came up with : '{sdprompt}'")
    return genlines
def check_meta_cmds(prompt, client):
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
    'https://www.tageschau.de',
    'https://news.google.com',
    'https://en.wikinews.org/wiki/Main_Page'
    ]
    display_personas = [
        'who can i talk to?'
    ]
    clean_prompt = prompt.lower().strip()
    print("\n\nClean Prompt :", clean_prompt, "\n\n")
    if clean_prompt == ("forget everything"):
        return (amnesia, None)
    if clean_prompt in display_personas:
        print("display_personas")
        return (personas_table, None)
    if clean_prompt.startswith("check the news."):
        source = random.choice(news_sources)
        print(f"\nNews Source selected : {source}\n")
        return (None,f"Read {source}. summarize and give me your thoughts about the five first headlines. Translate them to English if they aren't in English")
    if "generate an image of" in clean_prompt or "/imagine" in clean_prompt:
        print("imagine found.")
        return (generate_image, clean_prompt)
    if "ask wikipedia about " in clean_prompt:
        pattern = ".*ask wikipedia about [\'\"]([ a-zA-Z0-9_\-\']+)[\'\"].*"
        matches = re.match(pattern, clean_prompt)
        if matches is not None:
            query =  matches.group(1)
            return (None, Searxing.ask_wikipedia_search(query))
        else:
            pattern = ".*ask wikipedia about ([a-zA-Z0-9_\-\']+)[^\w]*"
            matches = re.match(pattern, clean_prompt)
            if matches is not None:
                query =  matches.group(1)
                return (None, Searxing.ask_wikipedia_search(query))
    pattern = ".*[iI] summon ([a-zA-Z_0-9\-]+).*"
    matches = re.match(pattern, clean_prompt)
    if matches is not None and len(matches.groups()) > 0:
        return (configure, matches.group(1).lower().capitalize())
    pattern = ".*my name is ([a-zA-Z_0-9]+).*"
    matches = re.match(pattern, prompt)
    if matches is not None and len(matches.groups()) > 0:
        set_username(matches.group(1).lower().capitalize(), client)
        return (None,None)
    if "##shutdown now##" in prompt.lower():
        shutdown_action()
        exit()
    return (cmd, None)

def generate(prompt, client, raw=False):
    global generators, redo_persona_context, persona_config, token_count, model_config, parameters, tokenizer, generator, initialized, ability_string, historyMP, usernames

    if initialized == False:
        configure(system_config['persona'])

    if client not in historyMP:
        initialize_historyMP(client)

    (cmd, prmtr) = check_meta_cmds(prompt, client)
    if cmd != None:
        if prmtr != None:
            return htmlize(cmd(prmtr, client))
        else:
            return htmlize(cmd(client))
    elif prmtr != None:
        prompt = prmtr

    original_prompt = prompt

    if model_config['loader'] == 'ollama' and prompt.startswith("(langchain)"):
        olc = OllamaLangchain()
        fn = Searxing.extract_file_name(prompt)
        url = Searxing.extract_url(prompt)
        q = Searxing.extract_query(prompt)
        vector = None
        do_langchain = True
        if url != "":
            vector = olc.open_url(url=url, ollama_server=oai_config['ollama_server'],model_tag=f"{persona_config['model']}:{persona_config['tag']}")
        elif fn != "":
            vector = olc.open_pdf(path=fn, ollama_server=oai_config['ollama_server'],model_tag=f"{persona_config['model']}:{persona_config['tag']}")
        elif q != "":
            url = f"{Searxing.config['searx_server']}?q={q}&format=json"
            vector = olc.open_url(url=url, ollama_server=oai_config['ollama_server'],
                              model_tag=f"{persona_config['model']}:{persona_config['tag']}")
        else:
            do_langchain = False
        if do_langchain:
            ctx = ""
            # if (redo_persona_context):
            #     ctx = persona_config['context']
            #     if system_config['do_greeting']:
            #         ctx = persona_config['greeting'] + ctx
            rs_raw = olc.query(vectorstore=vector,prompt=ctx+"\n"+prompt, ollama_server=oai_config['ollama_server'],model_tag=f"{persona_config['model']}:{persona_config['tag']}")
            print("\n\n___________________\n",rs_raw,"\n___________________\n\n")
            rs = rs_raw["result"]

            generate_tts(rs)
            historyMP[client].append(["u", prompt])
            historyMP[client].append(["b", rs])
            return generate_chat_lines(prompt, htmlize(rs))

    searxPrompt = Searxing.check_for_trigger(prompt, model_config['max_seq_len']/2, count_tokens)
    if searxPrompt == '':
        searxPrompt = prompt

    if( redo_persona_context):
        ctx = persona_config['context']
        if system_config['do_greeting']:
            ctx = persona_config['greeting']+ctx

    # save the user prompt in the history (with or without model-specific prompt formats)
    # history.append(model_config["user"].replace("%%prompt%%", searxPrompt+"\n"))
    historyMP[client].append(["u",searxPrompt])
    if model_config['loader'] == 'tabby':
        context_management(client)

    cut_output = generators[model_config['loader']](original_prompt, client)
    print(cut_output)

    if raw:
        return cut_output

    historyMP[client].append(["b",cut_output])
    if model_config['loader'] == 'tabby':
        token_count = count_tokens(generate_history_string(client))
        print(f'Token count: {token_count}')
    generate_tts(cut_output)
    save_log(original_prompt, cut_output)
    return generate_chat_lines(original_prompt, htmlize(cut_output))

def image_analysis(img, img_name, client, user_prompt=None):
    global persona_config, system_config, historyMP, usernames
    url = f"{oai_config['ollama_server']}/api/chat"
    print("len: ", len(img))
    headers = {
        "Content-Type": "application/json",
    }

    rs = {
        "model": system_config['multimodal'],
        "stream": False,
        "messages": [{
            "role":"user",
            "content": "describe this image in great details, please.",
            "images": [img]
        }],
        "options": {}
    }
    with open(f"parameters/ollama_{persona_config['parameters']}.yaml") as f:
         rs['options'] = yaml.safe_load(f)
    seed = random.randint(math.ceil(sys.float_info.min), math.floor(sys.float_info.max))
    rs["options"]["seed"] = seed
    rs["options"]["stop"] = ['USER:','</s>']

    #rsai = requests.post(url, headers=headers, data=json.dumps(rs, indent=0))
    # print(rsai.json())
    # response = rsai.json()['message']['content']

    jsdata = json.dumps(rs, indent=2)
    rs_stream = requests.post(url,  headers=headers, data=jsdata, stream=False)
    response = rs_stream.json()['message']['content']

    if client not in historyMP:
        initialize_historyMP(client)
    user_output = user_prompt
    if user_prompt == None:
        user_prompt = "[System: This is your analysis of a file the user just uploaded.]"
        user_output ="describe this image in great details, please."

    historyMP[client].append(["u", user_prompt])
    historyMP[client].append(["b",response])
    generate_tts(response)

    return generate_chat_lines(user_output, response, urllib.parse.quote(img_name))

def initialize_historyMP(client):
    global system_config, historyMP
    historyMP[client] = []
    usernames[client] = system_config['username']

def list_personas():
    """
    Return a formatted string containing HTML code for a dropdown list of personas.

    :return: A string containing HTML code for the dropdown list of personas.
    """
    global persona_config
    personas = glob.glob("./personas/*.yaml")
    personas.sort()
    rs = ""
    persona = ""
    for p_raw in personas:
        p=p_raw.replace(".yaml","").replace("./personas/","")
        selected=""
        if p == persona_config["name"]:
            selected = "selected"
            persona = p
        rs +=f"<option value='{p}' {selected}>{p}</option>"
    return persona

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

def list_model():
    global model_config,loaded_model,persona_config
    loader = '(?)'
    if model_config['loader'] == 'tabby':
        loader = '(t)'
    elif model_config['loader'] == "ollama":
        loader = '(o)'

    output = f"{loader} {persona_config['model']}"
    return f"<span id='persona_name'>{persona_config['name']}</span><br/><span id='model_name'>{output}</span>"

def set_username(uname, client):
    """
    ``set_username(uname)``
    -----------------------

    Sets the username for the system configuration.

    Parameters:
        uname (str): The username to be set.

    Returns:
        str: The updated username.

    """
    global system_config, usernames
    if uname.strip() == "":
        uname = system_config["username"]
    usernames['client'] = uname
    return usernames['client']

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

def personas_table(client=None):
    global persona_config, requrl
    print("generating persona's table")
    personas = glob.glob("./personas/*.yaml")
    personas.sort()
    rs = ""
    with open('templates/persona_cell.tpl.html', 'r') as f:
        tpl = f.read()

    for p_raw in personas:
        with open(p_raw) as f:
            prsna = yaml.safe_load(f)
        with open(f"model_configs/{prsna['model']}.yaml") as f:
            mdl = yaml.safe_load(f)
        if mdl["loader"] == "ollama":
            m = f"(o) {prsna['model']}"
        else:
            m = f"(t) {prsna['model']}"
        p=p_raw.replace(".yaml","").replace("./personas/","")
        pdiv=tpl.replace("{path}",f"{requrl}/get_face/{p}").replace("{name}",p).replace("{model}",m).replace("{role}",prsna['role'])
        rs += pdiv
    generate_tts("Those are the personas currently available, along with their respective L L M.")
    return generate_chat_lines("who can I talk to?", rs)

def amnesia(client = None):
    """
    Clear the history and reset token count if foo is not equal to "bar".

    :param foo: A string value.
    :return: None.
    """
    global user_prompts, responses_ids, historyMP, token_count
    if client != None:
        historyMP[client] = []
    else:
        historyMP={}
    token_count = 0


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
    print("\n CLIENT IP :")
    print(request.headers.get('CF-Connecting-IP'))
    print("\n CLIENT IP :")
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
    extras = 0
    print(system_config)
    if system_config['do_wav2lip'] == True:
        extras = 2
        index = index.replace("{select_w2l}", 'checked')
    elif system_config['do_tts'] == True:
        index = index.replace("{select_tts}", 'checked')
        extras = 1
    else:
        index = index.replace("{select_noav}", 'checked')

    index = index.replace("{select_noav}", '')
    index = index.replace("{select_w2l}", '')
    index = index.replace("{select_tts}", '')

    index=index.replace("{extra_option}",str(extras))
    return index
    #send_file("index.html", mimetype='text/html')

@app.route('/set_av/<option>', methods=['GET'])
def set_av_action(option):
    if option == "noav":
        system_config['do_tts'] = False
        system_config['do_wav2lip'] = False
    elif option == "w2l":
        system_config['do_tts'] = True
        system_config['do_wav2lip'] = True
    elif option == "tts":
        system_config['do_tts'] = True
        system_config['do_wav2lip'] = False
    write_system_config()
    return "ok"

@app.route('/set_username/', methods=['GET'])
def set_username_action():
    """
    Perform the action of setting the username.

    :return: None
    """
    client = request.headers.get('CF-Connecting-IP')
    return set_username( request.form['username'], client)

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

@app.route('/video/<rnd>', methods=['GET'])
def get_video(rnd):
    return send_file("video/result.mp4", mimetype='video/mp4')

@app.route('/idle/<rnd>', methods=['GET'])
def get_idle(rnd):
    global persona_config
    return send_file(f"video/{persona_config['name']}_idle.mp4", mimetype='video/mp4')

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
    client = request.headers.get('CF-Connecting-IP')
    persona = request.form['persona']
    return configure(persona)

@app.route('/list_models', methods=['GET','POST'])
def list_models_action():
    """
    Returns a list of models.

    :return: a list of models.
    """
    return list_model()

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

@app.route('/uploads/<img>', methods=['GET'])
def get_uploads_action(img):
    """
        :param persona: The persona of the face image to retrieve. It is a string representing the persona.
        :return: The face image associated with the specified persona in PNG format.
    """
    filename=f"./uploads/{img}"
    return send_file(filename, mimetype='image')

@app.route('/uploads/generated/<img>', methods=['GET'])
def get_generated_action(img):
    """
        :param persona: The persona of the face image to retrieve. It is a string representing the persona.
        :return: The face image associated with the specified persona in PNG format.
    """
    filename=f"./uploads/generated/{img}"
    return send_file(filename, mimetype='image')

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
    client = request.headers.get('CF-Connecting-IP')
    print("\n\n\n*****")
    print("\n",client,"\n")
    print("prompt : ",prompt)
    print("\n\n\n*****")
    return generate(prompt, client,False)

@app.route('/whisper', methods=['POST'])
def whisper_action():
    global whisper_model
    audio = request.files['audio']
    audio.save('audio/input.ogg')
    result = whisper_model.transcribe("audio/input.ogg")
    return result["text"]

@app.route('/rpgenerate', methods=['POST'])
def rpgenerate_action(client):
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
        amnesia(client)
    persona_config['name'] = data['npc_name']
    system_config['username'] = data['player_name']
    return generate(prompt, client)

@app.route("/upload", methods=["POST"])
def upload():
    print("\nUpload\n")
    print(request.files.keys())
    if "file" not in request.files:
        return jsonify({"error": "Please select a file."}), 425
    client = request.headers.get('CF-Connecting-IP')
    file = request.files["file"]
    file_in_bytes = file.read()
    base64_encoded = base64.b64encode(file_in_bytes).decode('Utf-8')
    # with open(f"uploads/{file.filename}.txt", "w") as f:
    #     f.write(base64_encoded)
    with open(f"uploads/{file.filename}", "wb") as f:
        f.write(file_in_bytes)
    print("done uploading.")
    return image_analysis(base64_encoded, file.filename, client)

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
    client = request.headers.get('CF-Connecting-IP')
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
    client = request.headers.get('CF-Connecting-IP')
    generate_memory(client)
    print("Terminating...")

if __name__ == '__main__':
    ollama_restart()
    torch.set_grad_enabled(False)
    torch.cuda._lazy_init()
    initialized = False
    usernames = {}
    historyMP = {}
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
