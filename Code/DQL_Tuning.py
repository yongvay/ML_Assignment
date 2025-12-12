# ==============================================================================
# DQN_Full_Architecture_Tuner.py - Comprehensive Grid Search (8 Total Runs)
# GOAL: Find optimal Layer 1 size, Layer 2 size, and Activation Function, 
#       and generate 4 graphs for the best model to compare with Q_Learning.
# ==============================================================================

# Step 1: Add modules 
import os
import sys
import random
import numpy as np
import matplotlib.pyplot as plt
import datetime
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import traci

# --- Hyperparameters to Test ---
LAYER_SIZES = [16, 32] 
ACTIVATION_FUNCTIONS = ['relu', 'tanh']
# -------------------------------

# --- Fixed RL and Learning Parameters ---
FIXED_LEARNING_RATE = 0.001 
FIXED_EPSILON = 0.01
FIXED_ALPHA = 0.1
FIXED_GAMMA = 0.9
# FAIRNESS CONSTRAINT: Set to 100 for comparison against best Q_Learning run
FIXED_MIN_GREEN_STEPS = 100 
# ------------------------------------------


# Step 2 & 3: SUMO Setup 
SUMO_BASE_PATH = r'C:\Program Files (x86)\Eclipse\Sumo'
os.environ['SUMO_HOME'] = SUMO_BASE_PATH
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(SUMO_BASE_PATH, 'tools')
    sys.path.append(tools)
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Define the model build function 
def build_model(state_size, action_size, size_1, size_2, activation_fn, learning_rate):
    """Builds the DQN model with independently tunable layer sizes and activation."""
    model = keras.Sequential() 
    model.add(layers.Input(shape=(state_size,))) 
    model.add(layers.Dense(size_1, activation=activation_fn)) 
    model.add(layers.Dense(size_2, activation=activation_fn)) 
    model.add(layers.Dense(action_size, activation='linear')) 
    model.compile(
        loss='mse',
        optimizer=keras.optimizers.Adam(learning_rate=learning_rate)
    )
    return model

# Utility functions (simplified for the tuner)
def to_array(state_tuple):
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))
def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)
def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)
def get_reward(state):
    total_queue = sum(state[:-1])
    return -float(total_queue)
def get_waiting_time(detector_id):
    return traci.lane.getWaitingTime(detector_id) # Assumes ID is lane ID


# --- Waiting Time Collection for Full History ---
def get_waitingTimeState_local():
    # This function collects the data needed for the 3rd and 4th plots.
    wt_ids = ["Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2", 
              "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"]
    wts = [get_waiting_time(id) for id in wt_ids]
    return tuple(wts)
# -----------------------------------------------


# ==============================================================================
# WRAPPER FUNCTION: Runs one full simulation and saves the model (COLLECTS ALL 4 METRICS)
# ==============================================================================
def run_simulation(size_1, size_2, activation_fn):
    
    tf.keras.backend.clear_session()
    
    current_simulation_step = 0
    TOTAL_STEPS = 10000 
    
    # 1. Instantiate the DQN model
    state_size = 7
    action_size = 2
    dqn_model = build_model(state_size, action_size, size_1, size_2, activation_fn, FIXED_LEARNING_RATE)

    # 2. SUMO Setup (Headless for speed)
    Sumo_config = [
        'sumo',
        '-c', 'RL.sumocfg',
        '--step-length', '0.10',
        '--delay', '0',
        '--lateral-resolution', '0'
    ]
    
    # Lists to capture data for visualization (ALL 4 METRICS)
    step_history = []
    reward_history = []
    queue_history = []
    cumulative_waitingtime_history = []  # Metric 3: Instantaneous WT
    cumulative_waitingtime_total_history = [] # Metric 4: Cumulative WTSum
    
    cumulative_waitingtime_total = 0.0
    cumulative_reward = 0.0
    
    try:
        traci.start(Sumo_config)
        last_switch_step = -FIXED_MIN_GREEN_STEPS

        # --- Local function definitions (uses fixed RL parameters) ---
        def get_state_local():
            q_ids = ["Node1_2_EB_0", "Node1_2_EB_1", "Node1_2_EB_2", 
                     "Node2_7_SB_0", "Node2_7_SB_1", "Node2_7_SB_2"]
            queues = [get_queue_length(id) for id in q_ids]
            current_phase = get_current_phase("Node2")
            return tuple(queues + [current_phase])

        def apply_action_local(action, tls_id="Node2"):
            nonlocal last_switch_step, current_simulation_step
            if action == 1:
                if current_simulation_step - last_switch_step >= FIXED_MIN_GREEN_STEPS:
                    program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
                    num_phases = len(program.phases)
                    next_phase = (get_current_phase(tls_id) + 1) % num_phases
                    traci.trafficlight.setPhase(tls_id, next_phase)
                    last_switch_step = current_simulation_step

        def get_action_from_policy_local(state):
            if random.random() < FIXED_EPSILON:
                return random.choice(ACTIONS)
            else:
                state_array = to_array(state)
                Q_values = dqn_model.predict(state_array, verbose=0)[0]
                return int(np.argmax(Q_values))

        def update_Q_table_local(old_state, action, reward, new_state):
            Q_values_old = dqn_model.predict(to_array(old_state), verbose=0)[0]
            Q_values_new = dqn_model.predict(to_array(new_state), verbose=0)[0]
            best_future_q = np.max(Q_values_new)
            
            Q_values_old[action] = Q_values_old[action] + FIXED_ALPHA * (reward + FIXED_GAMMA * best_future_q - Q_values_old[action])
            dqn_model.fit(to_array(old_state), np.array([Q_values_old]), verbose=0)
        # --- End Local Functions ---


        # 3. Start Simulation Loop
        for step in range(TOTAL_STEPS):
            current_simulation_step = step
            state = get_state_local()
            action = get_action_from_policy_local(state)
            apply_action_local(action)
            traci.simulationStep()
            new_state = get_state_local()
            new_wt = get_waitingTimeState_local() # New: Collect waiting time for plotting
            
            reward = get_reward(new_state)
            cumulative_reward += reward
            
            # Metric 4 Update: Total Waiting Time Sum (if phase changed)
            if (state[-1] != new_state[-1]):
                cumulative_waitingtime_total += sum(new_wt)

            update_Q_table_local(state, action, reward, new_state)
            
            # Record data every 100 steps for visualization
            if step % 100 == 0:
                step_history.append(step)
                reward_history.append(cumulative_reward)
                queue_history.append(sum(new_state[:-1])) 
                cumulative_waitingtime_history.append(sum(new_wt)) # Metric 3: Instantaneous WT
                cumulative_waitingtime_total_history.append(cumulative_waitingtime_total) # Metric 4: Cumulative WTSum
            
        traci.close()
        
        # --- SAVE MODEL WEIGHTS AFTER SUCCESSFUL RUN ---
        MODEL_SAVE_PATH = f'weights/dqn_L1-{size_1}_L2-{size_2}_Act-{activation_fn}.h5'
        os.makedirs('weights', exist_ok=True)
        dqn_model.save_weights(MODEL_SAVE_PATH)
        print(f"Model saved to: {MODEL_SAVE_PATH}")
        # -----------------------------------------------
        
        # Return the final cumulative reward AND all history lists
        return cumulative_reward, step_history, reward_history, queue_history, cumulative_waitingtime_history, cumulative_waitingtime_total_history

    except traci.TraCIException as e:
        print(f"SUMO Error during run (L1: {size_1}, L2: {size_2}, Act: {activation_fn}): {e}")
        try: traci.close() 
        except: pass
        # Return negative infinity for failed run
        return -float('inf'), [], [], [], [], []
    except Exception as e:
        print(f"General Python Error: {e}")
        return -float('inf'), [], [], [], [], []


# ==============================================================================
# STEP 9: Execute the Grid Search and Store Results
# ==============================================================================

print("Starting DQN Full Architecture Grid Search (8 Total Runs)...")
best_reward = -float('inf')
best_params = {}
all_results = []
all_reward_histories = {} 

# Loop through all 8 combinations
for size_1 in LAYER_SIZES:
    for size_2 in LAYER_SIZES:
        for activation_fn in ACTIVATION_FUNCTIONS:
            
            run_key = f"L1-{size_1}_L2-{size_2}_Act-{activation_fn}"
            print(f"\n--- Testing {run_key} ---")
            
            # Run the simulation and unpack the results (NOW 6 RETURN VALUES)
            final_reward, s_hist, r_hist, q_hist, wt_hist, wtsum_hist = run_simulation(size_1, size_2, activation_fn)
            
            # Store final score result
            all_results.append({
                'L1_size': size_1,
                'L2_size': size_2,
                'activation': activation_fn,
                'cumulative_reward': final_reward
            })
            
            # Store history for later plotting 
            all_reward_histories[run_key] = {
                'step': s_hist, 
                'reward': r_hist, 
                'queue': q_hist,
                'wt': wt_hist,
                'wtsum': wtsum_hist
            }
            
            # Update best parameters found so far (maximizing reward)
            if final_reward > best_reward:
                best_reward = final_reward
                best_params = {'L1_size': size_1, 'L2_size': size_2, 'activation': activation_fn}
                
            print(f"Result: Cumulative Reward = {final_reward:.2f}")


print("\n=============================================")
print("GRID SEARCH COMPLETE")
print("=============================================")

# 1. Print All Results Table
print("All Test Results (Lower Penalty is Better):")
for res in all_results:
    penalty = -res['cumulative_reward']
    print(f"L1: {res['L1_size']}, L2: {res['L2_size']}, Act: {res['activation']} -> Penalty: {penalty:.2f}")

print("\n---------------------------------------------")
print("BEST PERFORMING MODEL (Highest Reward / Lowest Penalty):")
print(f"Best L1 Size: {best_params.get('L1_size')}")
print(f"Best L2 Size: {best_params.get('L2_size')}")
print(f"Best Activation: {best_params.get('activation')}")
print(f"Best Cumulative Penalty: {-best_reward:.2f}")
print("---------------------------------------------")


# ==============================================================================
# 2. FINAL VISUALIZATION (Plotting the 4 graphs for the best model)
# ==============================================================================
best_run_key = f"L1-{best_params['L1_size']}_L2-{best_params['L2_size']}_Act-{best_params['activation']}"
best_hist = all_reward_histories[best_run_key]
plot_title = f"DQN Best Model (L1:{best_params['L1_size']}, L2:{best_params['L2_size']}, Act:{best_params['activation']})"

if best_hist['step']:
    print(f"\nGenerating 4-Graph Visualization for the BEST MODEL: {best_run_key}")
    
    # --- Graph 1: Cumulative Reward over Steps ---
    plt.figure(figsize=(10, 6))
    plt.plot(best_hist['step'], best_hist['reward'], marker='o', linestyle='-')
    plt.xlabel("Simulation Step")
    plt.ylabel("Cumulative Reward")
    plt.title(f"{plot_title}: Cumulative Reward over Steps")
    plt.grid(True)
    plt.show()

    # --- Graph 2: Total Queue Length over Steps ---
    plt.figure(figsize=(10, 6))
    plt.plot(best_hist['step'], best_hist['queue'], marker='o', linestyle='-')
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Queue Length")
    plt.title(f"{plot_title}: Total Queue Length over Steps")
    plt.grid(True)
    plt.show()


    # --- Graph 3: Total Waiting Time at a certain step (Instantaneous) ---
    plt.figure(figsize=(10, 6))
    plt.plot(best_hist['step'], best_hist['wt'], marker='o', linestyle='-')
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Waiting Time at a certain step (sec)")
    plt.title(f"{plot_title}: Total Waiting Time (Instantaneous)")
    plt.grid(True)
    plt.show()

    # --- Graph 4: Total Waiting Time sum in sec (Cumulative Delay) ---
    plt.figure(figsize=(10, 6))
    plt.plot(best_hist['step'], best_hist['wtsum'], marker='o', linestyle='-')
    plt.xlabel("Simulation Step")
    plt.ylabel("Total Waiting Time Sum (sec)")
    plt.title(f"{plot_title}: Total Waiting Time Sum (Cumulative Delay)")
    plt.grid(True)
    plt.show()
else:
    print("\nError: Best run history is empty. Cannot generate plots.")