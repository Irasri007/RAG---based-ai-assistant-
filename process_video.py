import os 
import subprocess

files= os.listdir("vdeos")
print(files)
for file in files :
    tutorial_number =file.split("[")[0].split("#")[1]
    file_name = file.split(" ｜ ")[0]
    print(tutorial_number, file_name)
    subprocess.run(["ffmpeg" , "-i" , f"vdeos/{file}" , f"audios/{tutorial_number}_{file_name}.mp3"])