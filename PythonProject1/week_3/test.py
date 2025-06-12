import chardet
with open("../data/beijing_tianqi_2019.csv", "rb") as f:
    result = chardet.detect(f.read())
    print("检测到的编码:", result['encoding'])