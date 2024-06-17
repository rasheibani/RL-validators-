import textworld.generator.inform7.world2inform7 as compiler

# read the 'aaatest.ni' files in a way that \n and indentation are preserved
with open('data/Environments/aaatest.ni', 'r') as file:
    data = file.readlines()

a = '\n'.join(data)


compiler.compile_inform7_game(a, 'data/Environments/aaatest.z8')