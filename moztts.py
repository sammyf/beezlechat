import torch
from TTS.api import TTS
import pydub

class MozTTS:
    tts = None
    devide = None

    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

    def load_model( self, model_name):
        self.tts = TTS(model_name).to(self.device)

    def moztts(self, sentence, bot_voice, bot_speaker, bot_language, outpath):
        if self.tts is None:
            self.load_model(bot_voice)
        if bot_speaker != '' and bot_language !='':
            self.tts.tts_to_file(text=sentence, speaker=bot_speaker, language=bot_language, file_path=outpath)
        elif bot_speaker != '':
            self.tts.tts_to_file(text=sentence, speaker=bot_speaker, file_path=outpath)
        elif bot_language != '':
            self.tts.tts_to_file(text=sentence, language=bot_language , file_path=outpath)
        else:
            self.tts.tts_to_file(text=sentence, file_path=outpath)

