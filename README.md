# beezlechat
### Persona-centric Frontend with TTS and internet access for basically any openAI-compatible LLM API Server

(but written with the excellent [Tabby](https://github.com/theroyallab/tabbyAPI/) in mind.)

This is a personal project, but I figured someone might get a kick out of it or use it to show bad coding practices. 
Anyway, create a virtual environment with 

`python -m venv bchat`

enter the environment with 

`source bchat/bin/activate`

then install the required packages with 

`pip install -r requirements.txt`

Copy `system_config.yaml.dist` and searxing_config.yaml.dist to `system_config.yaml` and `searxing_config.yaml` and edit them to reflect your configuration

Create a symlink to your models directory if you're using tabby or another API server. If you haven't got any models yet there is a handy `download_models.py` script to ... well download models 

next execute `./start.sh` and open your browser to the host and port you set in `system_config.yaml` (http://127.0.0.1:9090 by default)

### hidden features :
not really hidden (just check the Searxing.py file for details. It implements actually more than just calling Searx): ask the LLM nicely to 
`search the internet for` something if you want to call the Searx instance, include a fully formed URL in your prompt (that means : including `https://`) 
to force the LLM to read the page. Tell it to `forget everything.` to reset the context and 
say `I summon ` followed by a persona name to clear the context history, load whichever model is 
associated with a persona and then load said persona.

### model and persona setup
check the yaml files in personas and model_config to see how to add new models and personas. I highly recommend
using a StableDiffusion frontend to generate the avatars ([invokeAI](https://invoke-ai.github.io/InvokeAI/) is particularly good IMO)

### Disclaimer : 
This is my playground, and as such there are many undocumented features and definitely quite a few bugs. You can always ask for help, 
but I might have already removed whatever broke for you (or more probably made it worse). I'll try to help if I can, but you are 
encouraged to look at the code (it's really not complex), shake you head in disbelief and write your own much shinier and better version.

Cheers
Sammy