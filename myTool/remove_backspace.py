def remove_backspace(filepath:str,newfile_name:str):
    lines = []
    with open(filepath,'r') as fs:
        for line in fs.readlines():
            lines.append(line.replace(" ",""))
    
    with open(newfile_name,'w') as fs:
        fs.writelines(lines)


remove_backspace("IllegalWebsite/title/data/train.txt","train.txt")
remove_backspace("IllegalWebsite/title/data/test.txt","test.txt")
remove_backspace("IllegalWebsite/title/data/dev.txt","dev.txt")
