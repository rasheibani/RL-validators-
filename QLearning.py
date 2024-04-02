# this program merge route instruction and Environment together then unified them for a unified state representation
# to implement Q-learning algorithm ff

import textworld
import re
from Read_Route_Instructions import read_route_instructions
import xml.etree.ElementTree as ET
import creating_ni_2
import creating_ni_3
import openpyxl
import multiprocessing
import os


# Extracting the x and y coordinates from the sentence
def extract_coordinates(game_state):
    # Regular expression pattern to match the X and Y values
    pattern = r"X:\s*([\d.]+)\s*\nY:\s*([\d.]+)"

    matches = re.search(pattern, game_state)
    if matches:
        x = float(matches.group(1))
        y = float(matches.group(2))
        return x, y
    else:
        return None


def process_letter(letter):
    lettertext = letter.get('name')
    results = []
    if True:
        print(lettertext)
        for route in letter.findall('route'):
            x_origin = route.get('x_origin')
            y_origin = route.get('y_origin')
            x_destination = route.get('x_destination')
            y_destination = route.get('y_destination')
            primary_orientation = 6
            # check if the game is already created
            game_address = 'games/random/' + lettertext + '_' + x_origin + '_' + y_origin + '.z8'

            for style in route.findall('style'):
                if style.get('name') == "Turn-based":
                    for grammar in style.findall('grammar'):
                        sentence = grammar.find('route_instruction').text

                        # Split the sentence into a list of sentences
                        sentences_list = sentence.split(". ")
                        # calculate the size of the list
                        size_of_list = len(sentences_list)

                        if size_of_list < 2:
                            continue

                        try:
                            env = textworld.start(game_address)
                        except Exception as e:
                            print(f"Failed to start game for letter {lettertext}: {e}")
                            continue

                        game_state = env.reset()
                        # env.render()

                        for sentence in sentences_list:
                            if sentence != 'Arrive at destination.':
                                # print (sentence)

                                if False:
                                    # sentence in ['Make a sharp right', 'Make a sharp left']:
                                    # 'Turn slightly left', 'Turn slightly right']:
                                    print('un-understandable grammar')
                                    break
                                else:
                                    # print('\x1b[6;30;43m' + sentence + '\x1b[0m')
                                    game_state, reward, done = env.step(sentence)
                                    # env.render()

                            else:
                                x, y = extract_coordinates(game_state.feedback)

                        if x == float(x_destination) and y == float(y_destination):
                            valid_invalid = "Valid"
                            # print in green a VALID message
                            # print('\x1b[6;30;42m' + "||||||||| Valid Route Instruction, Intended Destination!! ||||||| " + '\x1b[0m')

                        else:
                            valid_invalid = "Invalid"
                            # print in red an INVALID message
                            # print('\x1b[6;30;41m' + "||||||||| Invalid Route Instruction, Wrong Destination!! ||||||| " + '\x1b[0m')

                        env.close()
                        results.append(
                            [lettertext, x_origin, y_origin, x_destination, y_destination, grammar.get("name"),
                             valid_invalid])
    return results


if __name__ == '__main__':
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)  # Create a process pool

    # Create a new workbook and select the active sheet
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    # Add headers to the columns
    sheet.append(["Letter", "Origin_X", "Origin_Y", "Destination_X", "Destination_Y", "Grammar", "Valid/Invalid"])
    lock = multiprocessing.Lock()

    tree = ET.parse('data/RI/Route_Instructions_LongestShortestV10.xml')
    # Get the root element
    root = tree.getroot()

    letters = root.findall('letter')
    # letters = letters[0:5]

    result = pool.map(process_letter, letters)

    for r in result:
        for rr in r:
            sheet.append(rr)
    # Save the workbook to a file
    workbook.save("test.xlsx")

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()