#!/bin/bash
cd /media/GINTONIC/AIArtists/Wav2Lip/
#echo /media/GINTONIC/AIArtists/Wav2Lip/targets/$2
source w2l/bin/activate && python inference.py --checkpoint_path  checkpoints/wav2lip.pth --face /media/GINTONIC/AIArtists/Wav2Lip/targets/$1_idle.mp4 --audio /media/GINTONIC/AIArtists/beezlechat/audio/tts.wav --outfile /media/GINTONIC/AIArtists/beezlechat/video/result.mp4


