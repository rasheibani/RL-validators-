import textworld
import re
from collections import deque
import pprint
# Test address for the game environment
testaddress = 'data/Environments/A_Average-Regular_Approach2_525768.0_999991.0.z8'

def extract_area_id(feedback):
    pattern = r"An area \((\d+)\) in r(\d+)"
    matches = re.search(pattern, feedback)
    if matches:
        area_id = matches.group(1)
        room_id = matches.group(2)
        return f"a{area_id}r{room_id}"
    else:
        # Debugging: Print feedback when extraction fails
        print("Failed to extract area ID from feedback:")
        print(feedback)
        print("-" * 50)
        return None

def get_admissible_actions(feedback):
    directions = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']
    admissible_actions = []

    for direction in directions:
        pattern = r'going ' + direction + ' '
        if re.search(pattern, feedback, re.IGNORECASE):
            admissible_actions.append('go ' + direction)

    return admissible_actions

def reverse_action(action):
    reverse_mapping = {
        'go north': 'go south', 'go south': 'go north',
        'go east': 'go west', 'go west': 'go east',
        'go northeast': 'go southwest', 'go northwest': 'go southeast',
        'go southeast': 'go northwest', 'go southwest': 'go northeast'
    }
    return reverse_mapping.get(action, None)

def extract_coordinates(feedback):
    pattern = r"X:\s*([\d.]+)\s*Y:\s*([\d.]+)"
    matches = re.search(pattern, feedback)
    if matches:
        x_coord = float(matches.group(1))
        y_coord = float(matches.group(2))
        return x_coord, y_coord
    else:
        return None, None

def z8file_to_dictionaries(gameaddress):
    env = textworld.start(gameaddress)
    env.reset()

    game_dict = {}
    room_positions = {}
    visited_rooms = set()  # To keep track of visited rooms
    room_queue = deque([[]])  # Store paths to reach rooms

    # Helper function to explore a room and add its neighbors to the queue
    def explore_room(path):
        # Rebuild the environment to the current state
        env.reset()
        game_state = env.step("look")[0]
        for a in path:
            game_state, _, _ = env.step(a)

        feedback = game_state.feedback
        room_id = extract_area_id(feedback)
        x_coord, y_coord = extract_coordinates(feedback)

        if not room_id:
            return  # Skip if we cannot extract room ID

        # Add room to the dictionary if not already present
        if room_id not in game_dict:
            game_dict[room_id] = {}
            room_positions[room_id] = (x_coord, y_coord)

        # Mark the room as visited
        visited_rooms.add(room_id)

        # Get all admissible directions (actions) for the current room
        actions = get_admissible_actions(feedback)

        # Iterate over each action and explore the resulting room
        for action in actions:
            # If the direction is already explored, skip it
            if action in game_dict[room_id]:
                continue

            # Copy the environment
            env_copy = env.copy()
            # Perform the action in the copied environment
            new_state, _, done = env_copy.step(action)
            new_feedback = new_state.feedback
            new_room_id = extract_area_id(new_feedback)

            # Add neighbor room ID to the current room's entry in the dictionary
            game_dict[room_id][action] = new_room_id

            if new_room_id and new_room_id not in visited_rooms:
                # If the new room hasn't been visited, add its path to the queue
                room_queue.append(path + [action])

    # Start BFS exploration
    while room_queue:
        current_path = room_queue.popleft()
        explore_room(current_path)

    return game_dict, room_positions

# Run the function and print the resulting dictionary
if __name__ == "__main__":
    game_dict, r = z8file_to_dictionaries(testaddress)
    pprint.pprint(game_dict)
    pprint.pprint(r)
