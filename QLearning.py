# this program merge route instruction and Environment together then unified them for a unified state representation
# to implement Q-learning algorithm ff

import textworld
import re
import xml.etree.ElementTree as ET
import openpyxl
import multiprocessing
import numpy as np
from Pretraining import Pretraining


# Extracting the x and y coordinates from the sentence
def extract_coordinates(game_state):
    # Regular expression pattern to match the X and Y values
    pattern = r"X:\s*([\d.]+)\s*\nY:\s*([\d.]+)"

    matches = re.search(pattern, game_state)
    if matches:
        x = float(matches.group(1))
        y = float(matches.group(2))
        return np.array([x]), np.array([y])
    else:

        return np.array([0]), np.array([0])



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
            game_address = 'data/Environments/' + lettertext + '_' + x_origin + '_' + y_origin + '.z8'

            for style in route.findall('style'):
                if style.get('name') == "Turn-based":
                    for grammar in style.findall('grammar'):
                        sentence = grammar.find('route_instruction').text

                        sentences_list = sentence.split(". ")
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
                                    game_state, reward, done = env.step(sentence)
                                    done = False
                                    reward = step_reward
                                    print(game_state.feedback)
                                    # env.render()
                            else:
                                done = True
                                reward = final_reward
                                x, y = extract_coordinates(game_state.feedback)

                        if x == float(x_destination) and y == float(y_destination):
                            valid_invalid = "Valid"

                        else:
                            valid_invalid = "Invalid"

                        env.close()
                        results.append(
                            [lettertext, x_origin, y_origin, x_destination, y_destination, grammar.get("name"),
                             valid_invalid])
    return results


if __name__ == '__main__':
    step_reward = -1
    final_reward = 100
    num_cores = multiprocessing.cpu_count()
    pool = multiprocessing.Pool(num_cores)  # Create a process pool

    # Create a new workbook and select the active sheet
    workbook = openpyxl.Workbook()
    sheet = workbook.active

    # Add headers to the columns
    sheet.append(["Letter", "Origin_X", "Origin_Y", "Destination_X", "Destination_Y", "Grammar", "Valid/Invalid"])
    lock = multiprocessing.Lock()

    tree = ET.parse('data/RouteInstructions/Route_Instructions_LongestShortestV8.xml')
    # Get the root element
    root = tree.getroot()

    letters = root.findall('letter')
    letters = letters[0:1]

    result = pool.map(process_letter, letters)

    for r in result:
        for rr in r:
            sheet.append(rr)
    # Save the workbook to a file
    workbook.save("test.xlsx")

    # Close the pool and wait for all processes to finish
    pool.close()
    pool.join()


