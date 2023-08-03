import re
def find_shortest(l):
    m = [len(i) for i in re.findall('[a-z A-Z]+', l)]
    return min(m) if m else 0
