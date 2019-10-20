#!usr/bin/env python  
#coding=utf-8  

import wave  

#define stream chunk   
chunk = 1024  

#open a wav format music  
audio_obj = wave.open(r"implode.wav","rb")  

print("number of audio channels: " + str(audio_obj.getnchannels()))
print("sample width: " + str(audio_obj.getsampwidth()))
print("framerate: " + str(audio_obj.getframerate()))
print("parameters: " + str(audio_obj.getparams()))
print("read 1 frame: " + str(audio_obj.readframes(1)))
print("read 1 frame: " + str(audio_obj.readframes(1)))

audio_obj.rewind()

print("read 1 frame: " + str(audio_obj.readframes(1)))
