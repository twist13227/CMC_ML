def check(s, filename):
    dict = {}
    for word in sorted(s.lower().split()):
        dict[word] = dict.get(word,0) + 1
    f = open(filename, "w")
    for words, count in dict.items():
        f.write(f"{words} {count}\n")
    f.close()
