import math

number = max(1,2)
text = f"Det største tallet av 1 og 2 er {number}.\n"

logOfEight = math.log(8)
text += "Logartimen til åtte er {logOfEight}"

text += "Forøvrig er de første 100000 tallene som følger:\n"

for i in range(100000):
  text += f"{i+1}, "

with open('output.txt', 'w') as file:
    file.write(text)
