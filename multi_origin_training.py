import os
import numpy as np
import torch
import random
import pandas as pd
from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import EvalCallback, StopTrainingOnRewardThreshold, BaseCallback
from stable_baselines3.common.monitor import Monitor
import gymnasium as gym
from z8file_to_dictionaries import z8file_to_dictionaries
from environment import TextWorldEnv, normalize, admissible_actions_to_observation
from utils import determine_complexity, determine_n_instructions

class MultiOriginTrainingData:
    """Class to generate and manage training data with multiple origins for the same destination"""
    
    def __init__(self, game_dict, room_positions, num_origins_per_destination=5):
        self.game_dict = game_dict
        self.room_positions = room_positions
        self.num_origins_per_destination = num_origins_per_destination
        self.room_ids = list(game_dict.keys())
        self.destinations = {}  # Map destinations to list of origins
        self.training_origins = {}  # Track origins used during training
        self.evaluation_origins = {}  # Origins reserved for evaluation
        
    def generate_training_data(self, n_destinations=10):
        """Generate training data with multiple origins per destination"""
        training_data = []
        
        # Select random destinations
        potential_destinations = random.sample(self.room_ids, min(n_destinations, len(self.room_ids)))
        
        for dest_room_id in potential_destinations:
            dest_x, dest_y = self.room_positions[dest_room_id]
            
            # Find rooms that can reach this destination (have a path to the destination)
            reachable_origins = self._find_reachable_origins(dest_room_id)
            
            if len(reachable_origins) < self.num_origins_per_destination + 2:  # +2 for evaluation
                continue  # Skip if not enough origins can reach this destination
                
            # Select origins for training
            training_origins = random.sample(reachable_origins, self.num_origins_per_destination)
            
            # Reserve some origins for evaluation (not used in training)
            remaining_origins = [o for o in reachable_origins if o not in training_origins]
            eval_origins = random.sample(remaining_origins, min(2, len(remaining_origins)))
            
            # Store for tracking
            self.destinations[dest_room_id] = (dest_x, dest_y)
            self.training_origins[dest_room_id] = training_origins
            self.evaluation_origins[dest_room_id] = eval_origins
            
            # Generate route instructions from each training origin to the destination
            for origin_id in training_origins:
                route_instructions = self._generate_route_from_origin_to_dest(origin_id, dest_room_id)
                
                if route_instructions:  # Only add if we have valid instructions
                    origin_x, origin_y = self.room_positions[origin_id]
                    training_data.append({
                        'origin_id': origin_id,
                        'origin_x': origin_x,
                        'origin_y': origin_y,
                        'destination_id': dest_room_id,
                        'destination_x': dest_x,
                        'destination_y': dest_y,
                        'route_instructions': route_instructions
                    })
        
        return training_data
    
    def _find_reachable_origins(self, dest_room_id):
        """Find all rooms that can reach the destination using BFS"""
        reachable = []
        visited = set()
        
        # Create a reversed graph
        reversed_graph = {}
        for room_id in self.game_dict:
            for action, next_room in self.game_dict[room_id].items():
                if next_room is not None:
                    if next_room not in reversed_graph:
                        reversed_graph[next_room] = []
                    reversed_graph[next_room].append(room_id)
        
        # BFS from destination backwards
        queue = [dest_room_id]
        visited.add(dest_room_id)
        
        while queue:
            room = queue.pop(0)
            
            if room != dest_room_id:  # Don't include destination as an origin
                reachable.append(room)
                
            if room in reversed_graph:
                for prev_room in reversed_graph[room]:
                    if prev_room not in visited:
                        visited.add(prev_room)
                        queue.append(prev_room)
        
        return reachable
    
    def _generate_route_from_origin_to_dest(self, origin_id, dest_room_id):
        """Generate route instructions from origin to destination using BFS"""
        # Find shortest path first
        path = self._find_shortest_path(origin_id, dest_room_id)
        
        if not path:
            return None
            
        # Convert path to actions
        route_instructions = []
        for i in range(len(path) - 1):
            current = path[i]
            next_room = path[i + 1]
            
            # Find action that leads from current to next
            for action, target in self.game_dict[current].items():
                if target == next_room:
                    # Extract direction from action text (e.g., "go north" -> "north")
                    direction = action.split()[-1]
                    
                    # Convert to action index based on environment's direction mapping
                    # This is a bit hardcoded for the directions in GRAMMAR_DIRECTIONS[8]
                    directions = ['north', 'south', 'east', 'west', 'northeast', 'northwest', 'southeast', 'southwest']
                    if direction in directions:
                        route_instructions.append(directions.index(direction))
                    break
        
        return route_instructions
        
    def _find_shortest_path(self, start_id, end_id):
        """Find shortest path from start to end using BFS"""
        if start_id == end_id:
            return [start_id]
            
        queue = [(start_id, [start_id])]
        visited = set([start_id])
        
        while queue:
            (node, path) = queue.pop(0)
            
            for action, next_room in self.game_dict[node].items():
                if next_room is not None and next_room not in visited:
                    if next_room == end_id:
                        return path + [next_room]
                    visited.add(next_room)
                    queue.append((next_room, path + [next_room]))
        
        return None  # No path found
    
    def get_evaluation_scenarios(self):
        """Get evaluation scenarios: new origins to familiar destinations"""
        eval_scenarios = []
        
        for dest_id, eval_origins in self.evaluation_origins.items():
            dest_x, dest_y = self.destinations[dest_id]
            
            for origin_id in eval_origins:
                route_instructions = self._generate_route_from_origin_to_dest(origin_id, dest_id)
                
                if route_instructions:
                    origin_x, origin_y = self.room_positions[origin_id]
                    eval_scenarios.append({
                        'origin_id': origin_id,
                        'origin_x': origin_x,
                        'origin_y': origin_y,
                        'destination_id': dest_id,
                        'destination_x': dest_x,
                        'destination_y': dest_y,
                        'route_instructions': route_instructions
                    })
        
        return eval_scenarios

def train_multi_origin_agent(env_info, num_origins=5, max_iterations=10000):
    """Train an agent using multiple origins for each destination"""
    env_name = env_info['env']
    gameaddress = f'data/Environments/{env_name}'
    
    # Create directories for saving
    model_dir = f'data/trained/multi_origin/{env_name}'
    os.makedirs(model_dir, exist_ok=True)
    
    # Load environment data
    game_dict, room_positions = z8file_to_dictionaries(gameaddress)
    
    # Create the multi-origin training data generator
    data_generator = MultiOriginTrainingData(
        game_dict=game_dict,
        room_positions=room_positions,
        num_origins_per_destination=num_origins
    )
    
    # Generate training data
    training_data = data_generator.generate_training_data(n_destinations=10)
    
    if not training_data:
        print(f"Could not generate any valid training data for {env_name}")
        return None
    
    # Save the training data for reference
    training_df = pd.DataFrame(training_data)
    training_df.to_csv(f'{model_dir}/training_data.csv', index=False)
    
    # Create the training environment
    env = MultiOriginEnv(
        game_dict=game_dict,
        room_positions=room_positions,
        training_data=training_data,
        grammar=8,
        reward_type='step_cost'  # You could make this a parameter
    )
    
    env = gym.wrappers.TimeLimit(env, max_episode_steps=100)
    env = Monitor(env, filename=f'{model_dir}/monitor.log')
    
    # Initialize the model
    model = PPO(
        'MlpPolicy', 
        env=env,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        verbose=1,
        policy_kwargs=dict(net_arch=[64, 64]),
        learning_rate=0.001,
        tensorboard_log='data/tensorboard'
    )
    
    # Setup callbacks
    stop_callback = StopTrainingOnRewardThreshold(reward_threshold=24.9, verbose=1)
    eval_callback = EvalCallback(
        env,
        best_model_save_path=model_dir,
        eval_freq=5000,
        callback_on_new_best=stop_callback,
        verbose=1
    )
    
    # Train the model
    model.learn(
        total_timesteps=max_iterations,
        callback=eval_callback,
        tb_log_name=f"PPO_multi_origin",
        reset_num_timesteps=False
    )
    
    # Save the final model
    model_path = os.path.join(model_dir, 'final_model.zip')
    model.save(model_path)
    print(f"Saved trained model to {model_path}")
    
    # Save the evaluation scenarios
    eval_scenarios = data_generator.get_evaluation_scenarios()
    eval_df = pd.DataFrame(eval_scenarios)
    eval_df.to_csv(f'{model_dir}/evaluation_scenarios.csv', index=False)
    
    return model, data_generator

class MultiOriginEnv(TextWorldEnv):
    """Extended TextWorldEnv for training with multiple origins to the same destination"""
    
    def __init__(self, game_dict, room_positions, training_data, grammar=8, reward_type='step_cost'):
        super().__init__(
            game_dict=game_dict,
            room_positions=room_positions,
            grammar=grammar,
            reward_type=reward_type
        )
        self.training_data = training_data
        self.current_scenario_index = 0
    
    def reset(self, **kwargs):
        """Reset environment with a random scenario from training data"""
        self.counter = 0
        
        # Select a random scenario from training data
        self.current_scenario_index = random.randint(0, len(self.training_data) - 1)
        scenario = self.training_data[self.current_scenario_index]
        
        # Set origin
        self.current_room_id = scenario['origin_id']
        self.x, self.y = scenario['origin_x'], scenario['origin_y']
        self.x_origin, self.y_origin = self.x, self.y
        
        # Set destination
        self.x_destination = scenario['destination_x']
        self.y_destination = scenario['destination_y'] 
        
        # Set route instructions
        self.route_instructions = scenario['route_instructions']
        
        self.visited_states_actions.clear()
        self.instruction_index = 0
        
        self.dist_from_origin_to_destination = np.sqrt(
            (self.x_destination - self.x_origin) ** 2 + (self.y_destination - self.y_origin) ** 2
        )
        
        admissible_actions = self.get_admissible_actions()
        observation = np.concatenate((
            admissible_actions_to_observation(admissible_actions, self.directions),
            self.pad_instructions(),
            np.array([self.instruction_index, self.route_instructions[self.instruction_index] 
                     if self.instruction_index < len(self.route_instructions) else self.max_directions]),
        ))
        
        assert observation.shape[0] == self.observation_space.shape[0], \
            f"Observation shape mismatch. Expected {self.observation_space.shape[0]}, got {observation.shape[0]}"
        
        observation = normalize(observation)
        return observation, {}

def evaluate_multi_origin_generalization(env_info, model=None, baseline_model=None):
    """Evaluate the agent's ability to navigate from new origins to familiar destinations"""
    env_name = env_info['env']
    gameaddress = f'data/Environments/{env_name}'
    
    # Load environment data
    game_dict, room_positions = z8file_to_dictionaries(gameaddress)
    
    # Create the multi-origin training data generator
    data_generator = MultiOriginTrainingData(
        game_dict=game_dict,
        room_positions=room_positions,
        num_origins_per_destination=5
    )
    
    # Generate training data (this will also setup evaluation origins)
    data_generator.generate_training_data(n_destinations=10)
    
    # Get evaluation scenarios
    eval_scenarios = data_generator.get_evaluation_scenarios()
    
    if not eval_scenarios:
        print(f"No evaluation scenarios available for {env_name}")
        return None
    
    # Load the model if not provided
    if model is None:
        model_path = f'data/trained/multi_origin/{env_name}/best_model.zip'
        if os.path.exists(model_path):
            from stable_baselines3 import PPO
            model = PPO.load(model_path)
        else:
            print(f"No trained model found at {model_path}")
            return None
    
    # Create results container
    results = {
        'multi_origin': {'success': 0, 'episodes': 0, 'path_efficiency': []},
        'baseline': {'success': 0, 'episodes': 0, 'path_efficiency': []}
    }
    
    # Create evaluation environment
    eval_env = TextWorldEnv(
        game_dict=game_dict,
        room_positions=room_positions,
        grammar=8,
        reward_type='step_cost'  # Use same as training for consistency
    )
    
    # Evaluate on each scenario
    for scenario in eval_scenarios:
        # Set up environment for this scenario
        eval_env.current_room_id = scenario['origin_id']
        eval_env.x, eval_env.y = scenario['origin_x'], scenario['origin_y']
        eval_env.x_origin, eval_env.y_origin = eval_env.x, eval_env.y
        eval_env.x_destination = scenario['destination_x']
        eval_env.y_destination = scenario['destination_y']
        eval_env.route_instructions = scenario['route_instructions']
        
        # Calculate optimal path length
        optimal_length = len(scenario['route_instructions'])
        
        # Evaluate multi-origin model
        observation, _ = eval_env.reset()
        done = False
        truncated = False
        steps = 0
        path_efficiency = 0
        
        while not done and not truncated and steps < 100:
            action, _ = model.predict(observation, deterministic=True)
            observation, reward, done, truncated, _ = eval_env.step(action)
            steps += 1
            
            # Check for success
            if done and reward > 0:
                results['multi_origin']['success'] += 1
                path_efficiency = optimal_length / steps
                results['multi_origin']['path_efficiency'].append(path_efficiency)
                
        results['multi_origin']['episodes'] += 1
        
        # If baseline model is provided, evaluate it too
        if baseline_model is not None:
            # Reset environment for baseline evaluation
            eval_env.current_room_id = scenario['origin_id']
            eval_env.x, eval_env.y = scenario['origin_x'], scenario['origin_y']
            eval_env.x_origin, eval_env.y_origin = eval_env.x, eval_env.y
            
            observation, _ = eval_env.reset()
            done = False
            truncated = False
            steps = 0
            
            while not done and not truncated and steps < 100:
                action, _ = baseline_model.predict(observation, deterministic=True)
                observation, reward, done, truncated, _ = eval_env.step(action)
                steps += 1
                
                # Check for success
                if done and reward > 0:
                    results['baseline']['success'] += 1
                    path_efficiency = optimal_length / steps
                    results['baseline']['path_efficiency'].append(path_efficiency)
                    
            results['baseline']['episodes'] += 1
    
    # Calculate success rates
    results['multi_origin']['success_rate'] = (
        results['multi_origin']['success'] / results['multi_origin']['episodes'] 
        if results['multi_origin']['episodes'] > 0 else 0
    )
    
    if baseline_model is not None:
        results['baseline']['success_rate'] = (
            results['baseline']['success'] / results['baseline']['episodes']
            if results['baseline']['episodes'] > 0 else 0
        )
    
    # Calculate average path efficiency
    results['multi_origin']['avg_path_efficiency'] = (
        sum(results['multi_origin']['path_efficiency']) / len(results['multi_origin']['path_efficiency'])
        if results['multi_origin']['path_efficiency'] else 0
    )
    
    if baseline_model is not None:
        results['baseline']['avg_path_efficiency'] = (
            sum(results['baseline']['path_efficiency']) / len(results['baseline']['path_efficiency'])
            if results['baseline']['path_efficiency'] else 0
        )
    
    return results

def run_multi_origin_experiment(env_infos, num_origins=5, max_iterations=50000):
    """Run the complete multi-origin experiment workflow"""
    results = []
    
    for env_info in env_infos:
        env_name = env_info['env']
        print(f"\n=== Training multi-origin agent for {env_name} ===")
        
        # Train multi-origin agent
        multi_origin_model, data_generator = train_multi_origin_agent(
            env_info, 
            num_origins=num_origins,
            max_iterations=max_iterations
        )
        
        if multi_origin_model is None:
            print(f"Skipping evaluation for {env_name} due to training failure")
            continue
        
        # Train baseline model (single origin per destination)
        print(f"\n=== Training baseline agent for {env_name} ===")
        baseline_model_dir = f'data/trained/baseline/{env_name}'
        os.makedirs(baseline_model_dir, exist_ok=True)
        
        # For baseline, use only the first origin for each destination
        baseline_training_data = []
        for dest_id, origins in data_generator.training_origins.items():
            if origins:
                origin_id = origins[0]  # Just use the first origin
                dest_x, dest_y = data_generator.destinations[dest_id]
                origin_x, origin_y = data_generator.room_positions[origin_id]
                route_instructions = data_generator._generate_route_from_origin_to_dest(origin_id, dest_id)
                
                if route_instructions:
                    baseline_training_data.append({
                        'origin_id': origin_id,
                        'origin_x': origin_x,
                        'origin_y': origin_y,
                        'destination_id': dest_id,
                        'destination_x': dest_x,
                        'destination_y': dest_y,
                        'route_instructions': route_instructions
                    })
        
        # Create baseline environment and train
        game_dict, room_positions = z8file_to_dictionaries(f'data/Environments/{env_name}')
        baseline_env = MultiOriginEnv(
            game_dict=game_dict,
            room_positions=room_positions,
            training_data=baseline_training_data,
            grammar=8,
            reward_type='step_cost'
        )
        
        baseline_env = gym.wrappers.TimeLimit(baseline_env, max_episode_steps=100)
        baseline_env = Monitor(baseline_env, filename=f'{baseline_model_dir}/monitor.log')
        
        # Initialize the baseline model
        baseline_model = PPO(
            'MlpPolicy', 
            env=baseline_env,
            device='cuda' if torch.cuda.is_available() else 'cpu',
            verbose=1,
            policy_kwargs=dict(net_arch=[64, 64]),
            learning_rate=0.001,
            tensorboard_log='data/tensorboard'
        )
        
        # Train the baseline model
        baseline_model.learn(
            total_timesteps=max_iterations,
            tb_log_name=f"PPO_baseline",
            reset_num_timesteps=False
        )
        
        # Save the baseline model
        baseline_model_path = os.path.join(baseline_model_dir, 'final_model.zip')
        baseline_model.save(baseline_model_path)
        
        # Evaluate both models
        print(f"\n=== Evaluating generalization for {env_name} ===")
        eval_results = evaluate_multi_origin_generalization(
            env_info, 
            model=multi_origin_model,
            baseline_model=baseline_model
        )
        
        if eval_results:
            eval_results['env_name'] = env_name
            results.append(eval_results)
            
            # Print results
            print(f"\nResults for {env_name}:")
            print(f"Multi-origin success rate: {eval_results['multi_origin']['success_rate']:.2f}")
            print(f"Multi-origin avg path efficiency: {eval_results['multi_origin']['avg_path_efficiency']:.2f}")
            print(f"Baseline success rate: {eval_results['baseline']['success_rate']:.2f}")
            print(f"Baseline avg path efficiency: {eval_results['baseline']['avg_path_efficiency']:.2f}")
            
            # Save results to CSV
            results_df = pd.DataFrame([{
                'env_name': env_name,
                'multi_origin_success_rate': eval_results['multi_origin']['success_rate'],
                'multi_origin_path_efficiency': eval_results['multi_origin']['avg_path_efficiency'],
                'baseline_success_rate': eval_results['baseline']['success_rate'],
                'baseline_path_efficiency': eval_results['baseline']['avg_path_efficiency']
            }])
            
            results_df.to_csv(f'data/multi_origin_results_{env_name}.csv', index=False)
    
    # Aggregate and save overall results
    if results:
        all_results_df = pd.DataFrame([{
            'env_name': r['env_name'],
            'multi_origin_success_rate': r['multi_origin']['success_rate'],
            'multi_origin_path_efficiency': r['multi_origin']['avg_path_efficiency'],
            'baseline_success_rate': r['baseline']['success_rate'],
            'baseline_path_efficiency': r['baseline']['avg_path_efficiency']
        } for r in results])
        
        all_results_df.to_csv('data/multi_origin_results_all.csv', index=False)
        
        # Calculate average improvement
        avg_success_improvement = (
            all_results_df['multi_origin_success_rate'].mean() - all_results_df['baseline_success_rate'].mean()
        )
        
        avg_efficiency_improvement = (
            all_results_df['multi_origin_path_efficiency'].mean() - all_results_df['baseline_path_efficiency'].mean()
        )
        
        print("\n=== Overall Results ===")
        print(f"Average success rate improvement: {avg_success_improvement:.4f}")
        print(f"Average path efficiency improvement: {avg_efficiency_improvement:.4f}")
    
    return results

if __name__ == "__main__":
    from utils import load_envs
    
    # Load a subset of environments for the experiment
    all_envs = load_envs()
    selected_envs = all_envs[:5]  # Use the first 5 environments or change as needed
    
    # Run the experiment
    run_multi_origin_experiment(selected_envs, num_origins=5, max_iterations=50000)