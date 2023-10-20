import datetime
from flask import Flask, request, send_file
from flask_cors import CORS
import yaml
from model import ExLlama, ExLlamaCache, ExLlamaConfig
from tokenizer import ExLlamaTokenizer
from generator import ExLlamaGenerator
import torch
import os
import glob
import moztts
import re
import markdown
from searxing import SearXing

VERSION='1'
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
global pre_tag
global model
global system_config
global tts_config
with open('tts_config.yaml') as f:
    tts_config = yaml.safe_load(f)

global mozTTS

LOG_DIR = "logs/"
LOG_FILE = "_logs.txt"

chat_line=""
current_log_file = datetime.datetime.now().strftime("%Y%m%d-%H%M%S")

with open('./templates/chat_line.tpl.html') as f:
    chat_line="\n".join(f.readlines())

def read_system_config():
    with open('system_config.yaml') as f:
        global system_config
        system_config = yaml.safe_load(f)

def write_system_config():
    with open('system_config.yaml', 'w') as f:
        # Write the dictionary to the file
        yaml.dump(system_config, f)

def save_log(userline, botline):
    global current_log_file
    if system_config["log"] is True:
        with open(f'logs/{current_log_file}.txt', 'a') as f:
            f.write(f'user: {userline}\n{botline}\n\n')
def exllama_configure(model_directory):
    global parameters, persona_config, model_config, generator, tokenizer, cache, model
    if loaded_model != model_directory:
        if model is not None:
            ExLlama.free_unmanaged(model)
            del model
            del cache
            del tokenizer
            del generator
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
    generator.settings.typical = 1.0  # Disabled

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
    rs = chat_line.replace("%%user%%", user)\
        .replace("%%bot%%", bot)\
        .replace('%%botname%%', persona_config["name"])
    return rs

def configure(persona):
    global persona_config, generator, initialized,tokenizer, cache, parameters, model_config, mozTTS, system_config

    with open(f'./personas/{persona}.yaml') as f:
        persona_config = yaml.safe_load(f)

    system_config['persona'] = persona
    write_system_config()

    if "language" not in persona_config:
        persona_config['language'] = ''
    # history.clear()
    history.append("context: "+persona_config['context'])
    print(persona_config["model"])
    if 'parameters' not in persona_config:
        persona_config['parameters'] = 'default'
    with open(f'./parameters/{persona_config["parameters"]}.yaml') as f:
        parameters = yaml.safe_load(f)
    with open(f'model_configs/{persona_config["model"]}.yaml') as f:
        model_config = yaml.safe_load(f)

    # Directory containing config.json, tokenizer.model and safetensors file for the model
    exllama_configure(MODEL_PATH+persona_config['model'])
    mozTTS = moztts.MozTTS()
    mozTTS.load_model(persona_config['voice'])
    initialized = True
    last_character = persona_config["name"]

    # do the greetings
    prompt = f'I am summoning {last_character}'
    greeting_prompt = ""
    cut_output_prompted = ''
    cut_output = ''
    if system_config['do_greeting'] is True:
        cut_output = persona_config['greeting']
        if "system" in model_config :
                greeting_prompt=model_config["system"]
        cut_output_prompted = f"{greeting_prompt} {cut_output}"
    history.append(prompt)
    history.append(cut_output_prompted)
    if count_tokens("\n".join(history)) >= model_config['max_seq_len']:
        print("Context Overflow.")
        truncate_history()
        tenPHist = model_config['max_seq_len']-(model_config['max_seq_len']*0.2)
        while(count_tokens("\n".join(history)) >= tenPHist):
            truncate_history()
        print("New History :\n", history)
    generate_tts(cut_output)
    save_log(prompt, cut_output)
    return generate_chat_lines(prompt, htmlize(cut_output))

def htmlize(text):
    # pattern = "```(.+?)```"
    # text = re.sub(pattern,"<code>\1</code>",text)
    text = markdown.markdown(text, extensions=['fenced_code', 'codehilite'])
    return text

def fixHash27(s):
    s=s.replace("&#x27;","'");
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

def generate(prompt):
    global persona_config, model_config, parameters, tokenizer, generator, history, initialized

    if initialized == False:
        configure(system_config['persona'])

    searxPrompt = Searxing.check_for_trigger(prompt)
    if searxPrompt == '':
        searxPrompt = prompt

    stop_conditions = [tokenizer.newline_token_id]

    current_time = datetime.datetime.now()
    now = current_time.strftime("%H:%M:%S on %A, %d %B %Y")

    # truncate the history
    history.append(model_config["user"]+searxPrompt+"\n")
    if count_tokens("\n".join(history)) >= model_config['max_seq_len']:
        print("Context Overflow.")
        truncate_history()
        truncateLength = model_config['max_seq_len']-(parameters['max_tokens']+(parameters['max_tokens']*0.2))
        while(count_tokens("\n".join(history)) >= truncateLength):
            truncate_history()
        print("New History :\n", history)

    # generate
    full_prompt = "\nIt is now "+now+"\n"+"The chat so far:"+"\n".join(history)+"\n"+model_config["bot"]
    cutoff = len(full_prompt)
    output = generator.generate_simple(prompt = full_prompt,
                            max_new_tokens = parameters['max_tokens'])
    print("full prompt : ", full_prompt, "output : ", output,"\n(",cutoff,"\n")
    cut_output = output[cutoff:]
    second_cut = cut_output.find("context: "+persona_config['context'])
    if second_cut > 0:
        cut_output = output[:second_cut]
    # print("cut : ",cut_output)
    history.append(persona_config['name']+": "+cut_output)
    generate_tts(cut_output)
    save_log(prompt, cut_output)
    return generate_chat_lines(prompt, htmlize(cut_output))

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

###
# Routes

@app.route('/', methods=['GET'])
def get_index():
    index = ''
    with open('html/index.html.tpl') as f:
        index=f.read()
    requrl = request.url
    if '127.0.0.1' not in requrl and 'localhost' not in requrl and '192.168.' not in requrl:
        requrl=requrl.replace('http://','https://')
    index=index.replace("{request_url}", requrl)
    index=index.replace("{version}",VERSION)
    return index
    #send_file("index.html", mimetype='text/html')

@app.route('/voice/<rnd>', methods=['GET'])
def get_voice(rnd):
    return send_file("outputs/tts.wav", mimetype='audio/x-wav')

@app.route('/css/styles.css', methods=['GET'])
def get_css():
    return send_file("css/styles.css", mimetype='text/css')

@app.route('/js/script.js', methods=['GET'])
def get_js():
    return send_file("js/script.js", mimetype='application/javascript')

# if 'X-CSRF-Token' in request.headers:
@app.route('/configure', methods=['POST'])
def configure_action():
    persona = request.form['persona']
    return configure(persona)

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

@app.route('/shutdown', methods=['POST'])
def shutdown_action():
    print("shutting down.")
    return True
    exit()

if __name__ == '__main__':
    torch.set_grad_enabled(False)
    torch.cuda._lazy_init()
    initialized = False
    history=[]
    loaded_model = ""
    model = None
    pre_tag = False
    Searxing = SearXing()
    read_system_config()
    configure(system_config['persona'])
    app.run(host=system_config['host'], port=system_config['port'])