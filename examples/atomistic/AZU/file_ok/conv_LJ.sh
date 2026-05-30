# Extract 
# Pairs from input toml
with open('options.toml', "rb") as in_file:
    toml_content = tomli.load(in_file)

pairs = toml_content['field']['LJ_param']

# Pairs from ff
# this should contain only the list of pair parameters.
with open('martini.itp') as file:
    martini_pairs = [ ]
    for line in file:
        line = line.rstrip().split()
        martini_pairs.append(line)

for pair in pairs:
    for mpair in martini_pairs:
            if (pair[0] == mpair[0] and pair[1] == mpair[1]) or (pair[0] == mpair[1] and pair[1] == mpair[0]):
                print(f'[ \"{mpair[0]}\", \"{mpair[1]}\", {mpair[3]}, {mpair[4]}]')
