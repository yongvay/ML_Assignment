# =========================================================================
# Deep Q-Learning (DQN) for Traffic Signal Control - Architecture Grid Search
# =========================================================================

# Step 1: Add modules to provide access to specific libraries and functions
import os 
import sys 
import random
import numpy as np
import matplotlib.pyplot as plt 
import datetime
# Step 1.1: (Additional) Imports for Deep Q-Learning
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras.optimizers import Adam
# Clear the Keras backend session for clean runs between trials
tf.keras.backend.clear_session()

# Step 2: Establish path to SUMO (SUMO_HOME)
# NOTE: Update this path if necessary
SUMO_BASE_PATH = r'C:\Program Files (x86)\Eclipse\Sumo'
os.environ['SUMO_HOME'] = SUMO_BASE_PATH
if 'SUMO_HOME' in os.environ:
    tools = os.path.join(SUMO_BASE_PATH, 'tools')
    sys.path.append(tools)

# Insert this line to guarantee CWD is correct:
os.chdir(os.path.dirname(os.path.abspath(__file__)))

# Step 3: Add Traci module to provide access to specific libraries and functions
import traci

# Step 4: Define Sumo configuration
Sumo_config = [
    'sumo',
    '-c', 'RL.sumocfg',
    '--step-length', '0.10',
    '--delay', '0',
    '--lateral-resolution', '0'
]

# -------------------------
# Step 5: Define Global Hyperparameters
# -------------------------

# ---- Reinforcement Learning Hyperparameters (NOT TUNED) ----
TOTAL_STEPS = 10000     # The total number of simulation steps for continuous (online) training.
ALPHA = 0.1             # Learning rate (α)
GAMMA = 0.9             # Discount factor (γ)
EPSILON = 0.01          # Exploration rate 
ACTIONS = [0, 1]        # The discrete action space (0 = keep phase, 1 = switch phase)
state_size = 7          # (q_EB_0, ..., q_SB_2, current_phase)
action_size = len(ACTIONS)

# ---- Additional Stability Parameters ----
MIN_GREEN_STEPS = 100
# last_switch_step will be initialized inside run_training_trial

# --- ARCHITECTURE GRID DEFINITION ---
# The list of architectures to test: (L1_UNITS, L2_UNITS, ACTIVATION_FN, NAME)
ARCH_GRID = [
    (16, 16, 'relu', "Test 1: 16 relu, 16 relu"),
    (32, 16, 'relu', "Test 2: 32 relu, 16 relu"),
    (16, 32, 'relu', "Test 3: 16 relu, 32 relu"),
    (32, 32, 'relu', "Test 4: 32 relu, 32 relu"),
    (16, 16, 'tanh', "Test 5: 16 tanh, 16 tanh"),
    (16, 32, 'tanh', "Test 6: 16 tanh, 32 tanh"),
    (32, 16, 'tanh', "Test 7: 32 tanh, 16 tanh"),
    (32, 32, 'tanh', "Test 8: 32 tanh, 32 tanh"),
]

# -------------------------
# Step 6: Define Functions (Updated)
# -------------------------

# --- DQN Model Construction (Updated) ---
def build_model(state_size, action_size, L1_UNITS, L2_UNITS, ACTIVATION_FN):
    """
    Build a simple feedforward neural network that approximates Q-values,
    using parameterized architecture size and activation function.
    """
    model = keras.Sequential()
    model.add(layers.Input(shape=(state_size,)))

    # Parameterized Hidden Layers
    model.add(layers.Dense(L1_UNITS, activation=ACTIVATION_FN))
    model.add(layers.Dense(L2_UNITS, activation=ACTIVATION_FN))

    model.add(layers.Dense(action_size, activation='linear'))
    model.compile(
        loss='mse',
        optimizer=Adam(learning_rate=0.001)
    )
    return model

def to_array(state_tuple):
    """
    Convert the state tuple into a NumPy array for neural network input.
    """
    # Uses reshape((1, -1)) to ensure the input is (1, state_size)
    return np.array(state_tuple, dtype=np.float32).reshape((1, -1))

# --- Environment Interaction Functions (Unchanged) ---

def get_reward(state):
    """
    Simple reward function: Negative of total queue length.
    """
    total_queue = sum(state[:-1])  # Exclude the current_phase element
    reward = -float(total_queue)
    return reward

def get_queue_length(detector_id):
    return traci.lanearea.getLastStepVehicleNumber(detector_id)

def get_waiting_time(detector_id):
    # NOTE: traci.lane.getWaitingTime is usually vehicle-specific, 
    # but based on your original use, we keep the detector-based call.
    # SUMO typically provides last-step information for area detectors.
    try:
        return traci.lane.getWaitingTime(detector_id)
    except traci.exceptions.TraCIException:
        # Fallback if detector_id is a lane area detector and doesn't support getWaitingTime
        return 0.0

def get_current_phase(tls_id):
    return traci.trafficlight.getPhase(tls_id)


def get_state():
    """
    Collects the current queue lengths and the current traffic light phase.
    """
    
    # Detector IDs
    detector_Node1_2_EB_0 = "Node1_2_EB_0"
    detector_Node1_2_EB_1 = "Node1_2_EB_1"
    detector_Node1_2_EB_2 = "Node1_2_EB_2"
    detector_Node2_7_SB_0 = "Node2_7_SB_0"
    detector_Node2_7_SB_1 = "Node2_7_SB_1"
    detector_Node2_7_SB_2 = "Node2_7_SB_2"
    
    # Traffic light ID
    traffic_light_id = "Node2"
    
    # Get queue lengths from each detector
    q_EB_0 = get_queue_length(detector_Node1_2_EB_0)
    q_EB_1 = get_queue_length(detector_Node1_2_EB_1)
    q_EB_2 = get_queue_length(detector_Node1_2_EB_2)
    q_SB_0 = get_queue_length(detector_Node2_7_SB_0)
    q_SB_1 = get_queue_length(detector_Node2_7_SB_1)
    q_SB_2 = get_queue_length(detector_Node2_7_SB_2)

    # Get current phase index
    current_phase = get_current_phase(traffic_light_id)

    return (q_EB_0, q_EB_1, q_EB_2, q_SB_0, q_SB_1, q_SB_2, current_phase)


def get_waitingTimeState():
    """
    Collects the current waiting times.
    """
    
    detector_Node1_2_EB_0 = "Node1_2_EB_0"
    detector_Node1_2_EB_1 = "Node1_2_EB_1"
    detector_Node1_2_EB_2 = "Node1_2_EB_2"
    detector_Node2_7_SB_0 = "Node2_7_SB_0"
    detector_Node2_7_SB_1 = "Node2_7_SB_1"
    detector_Node2_7_SB_2 = "Node2_7_SB_2"
    
    wt_EB_0 = get_waiting_time(detector_Node1_2_EB_0)
    wt_EB_1 = get_waiting_time(detector_Node1_2_EB_1)
    wt_EB_2 = get_waiting_time(detector_Node1_2_EB_2)
    wt_SB_0 = get_waiting_time(detector_Node2_7_SB_0)
    wt_SB_1 = get_waiting_time(detector_Node2_7_SB_1)
    wt_SB_2 = get_waiting_time(detector_Node2_7_SB_2)

    return (wt_EB_0, wt_EB_1, wt_EB_2, wt_SB_0, wt_SB_1, wt_SB_2)


# --- Policy & Update Functions (Modified for encapsulation) ---

def apply_action(action, current_simulation_step, tls_id="Node2"):
    """
    Executes the chosen action on the traffic light, incorporating MIN_GREEN_STEPS logic.
    """
    global last_switch_step
    
    if action == 0:
        # Do nothing (keep current phase)
        return
    elif action == 1:
        # Check if minimum green time has passed before switching
        if current_simulation_step - last_switch_step >= MIN_GREEN_STEPS:
            program = traci.trafficlight.getAllProgramLogics(tls_id)[0]
            num_phases = len(program.phases)
            next_phase = (get_current_phase(tls_id) + 1) % num_phases
            traci.trafficlight.setPhase(tls_id, next_phase)
            # Record when the switch happened
            last_switch_step = current_simulation_step


def update_Q_table(dqn_model, old_state, action, reward, new_state):
    """
    Performs a single-step gradient update for the DQN model (Q-Learning update rule).
    """
    # 1) Predict current Q-values from old_state (current state)
    old_state_array = to_array(old_state)
    Q_values_old = dqn_model.predict(old_state_array, verbose=0)[0]
    
    # 2) Predict Q-values for new_state to get max future Q (new state)
    new_state_array = to_array(new_state)
    Q_values_new = dqn_model.predict(new_state_array, verbose=0)[0]
    best_future_q = np.max(Q_values_new)
        
    # 3) Calculate the Target Q-value using the Q-Learning equation:
    # Q(s, a) = Q(s, a) + α * [r + γ * max Q(s', a') - Q(s, a)]
    target_q_value = reward + GAMMA * best_future_q
    
    # Incorporate ALPHA to partially update the Q-value
    Q_values_old[action] = Q_values_old[action] + ALPHA * (target_q_value - Q_values_old[action])
    
    # 4) Train (fit) the DQN on this single sample
    # The target array is [Q_values_old], which now holds the updated Q-value for the action taken.
    dqn_model.fit(old_state_array, np.array([Q_values_old]), verbose=0)


def get_action_from_policy(dqn_model, state):
    """
    Epsilon-greedy strategy using the DQN's predicted Q-values.
    """
    if random.random() < EPSILON:
        return random.choice(ACTIONS)
    else:
        state_array = to_array(state)
        Q_values = dqn_model.predict(state_array, verbose=0)[0]
        return int(np.argmax(Q_values))


# -------------------------
# Step 7: Training Trial Function (New)
# -------------------------

# Global variables must be re-initialized for each trial
last_switch_step = -MIN_GREEN_STEPS

def run_training_trial(L1_UNITS, L2_UNITS, ACTIVATION_FN, TRIAL_NAME):
    """
    Runs the full online continuous learning loop for one set of hyperparameters.
    """
    global last_switch_step
    
    # Initialize variables for the specific trial
    cumulative_reward = 0.0
    cumulative_waitingtime_total = 0.0
    last_switch_step = -MIN_GREEN_STEPS # Reset for the new run

    # Lists for logging (only inside this function)
    step_history = []
    reward_history = []
    queue_history = []
    
    # 1. Initialize the DQN model for this specific trial
    dqn_model = build_model(state_size, action_size, L1_UNITS, L2_UNITS, ACTIVATION_FN)

    print(f"\n--- Starting Trial: {TRIAL_NAME} | L1={L1_UNITS}, L2={L2_UNITS}, Act={ACTIVATION_FN} ---")
    
    # 2. Training Loop (Online Continuous Learning)
    for step in range(TOTAL_STEPS):
        current_simulation_step = step
        
        state = get_state()
        action = get_action_from_policy(dqn_model, state)
        apply_action(action, current_simulation_step)
        
        traci.simulationStep()  # Advance simulation by one step
        
        new_state = get_state()
        new_wt = get_waitingTimeState()

        # Update cumulative waiting time upon phase change
        if (state[-1] != new_state[-1]):
            cumulative_waitingtime_total += sum(new_wt)
            # print("Traffic light just changed phase, adding extra cumulative time:", cumulative_waitingtime_total) # Optional logging

        reward = get_reward(new_state)
        cumulative_reward += reward

        update_Q_table(dqn_model, state, action, reward, new_state)
        
        # Record data every 100 steps
        if step % 100 == 0:
            updated_q_vals = dqn_model.predict(to_array(state), verbose=0)[0]
            cumulative_waitingtime_dateTime = str(datetime.timedelta(seconds=sum(new_wt)))
            
            print(f"Step {step}, Action: {action}, Reward: {reward:.2f}, Cum. Reward: {cumulative_reward:.2f}, Q-values: {updated_q_vals}, Current Waiting Time: {cumulative_waitingtime_dateTime}")
            step_history.append(step)
            reward_history.append(cumulative_reward)
            queue_history.append(sum(new_state[:-1]))

    print(f"Trial Finished. Final Cumulative Reward: {cumulative_reward:.2f}")

    # 3. Clean up the model memory after the trial
    del dqn_model
    tf.keras.backend.clear_session()
    
    # Return the metric we care about for the grid search comparison
    return cumulative_reward, step_history, reward_history, queue_history, cumulative_waitingtime_total


# -------------------------
# Step 8: Main Grid Search Execution
# -------------------------
if __name__ == "__main__":
    
    ALL_RESULTS = {}
    
    # Variables for plotting the best run
    best_reward = -np.inf
    best_plot_data = None
    best_test_name = ""

    for l1, l2, activation, name in ARCH_GRID:
        
        print(f"\n=======================================================")
        print(f"🚀 STARTING GRID SEARCH TRIAL: {name}")
        print(f"=======================================================")

        # A. Start TraCI for the new trial
        try:
            traci.start(Sumo_config) 
        except traci.exceptions.TraCIException as e:
            if "connection already established" in str(e):
                traci.close()
                traci.start(Sumo_config)
            else:
                raise e

        # B. Run the full simulation and training
        final_cumulative_reward, steps, rewards, queues, final_waiting_time = run_training_trial(
            L1_UNITS=l1, 
            L2_UNITS=l2, 
            ACTIVATION_FN=activation, 
            TRIAL_NAME=name
        )

        # C. Store the results
        ALL_RESULTS[name] = {
            'L1_Units': l1,
            'L2_Units': l2,
            'Activation': activation,
            'Final_Reward': final_cumulative_reward,
            'Final_Waiting_Time_Sum': final_waiting_time
        }
        
        # D. Track the best model for final visualization
        if final_cumulative_reward > best_reward:
            best_reward = final_cumulative_reward
            best_plot_data = (steps, rewards, queues)
            best_test_name = name

        # E. Stop TraCI to reset the simulation state for the next trial
        traci.close()

    # -------------------------
    # Step 9: Final Results Summary
    # -------------------------
    print("\n=============================================")
    print("🏆 ARCHITECTURE GRID SEARCH SUMMARY 🏆")
    print("=============================================")
    
    # Print a summary table
    print(f"{'Test Name':<28} | {'L1':<4} | {'L2':<4} | {'Act':<4} | {'Final Reward':<12} | {'Final Waiting Sum (s)':<22}")
    print("-" * 80)
    for name, data in ALL_RESULTS.items():
        print(f"{name:<28} | {data['L1_Units']:<4} | {data['L2_Units']:<4} | {data['Activation']:<4} | {data['Final_Reward']:.2f} | {data['Final_Waiting_Time_Sum']:.2f}")

    print("\n---------------------------------------------")
    print(f"**BEST MODEL: {best_test_name}** with a Cumulative Reward of {best_reward:.2f}")
    print("---------------------------------------------")

    # -------------------------
    # Step 10: Visualization of Best Result
    # -------------------------
    if best_plot_data:
        steps, rewards, queues = best_plot_data
        
        plt.figure(figsize=(12, 6))
        plt.plot(steps, rewards, marker='o', linestyle='-', label=f"Cumulative Reward ({best_test_name})")
        plt.xlabel("Simulation Step")
        plt.ylabel("Cumulative Reward")
        plt.title(f"Best RL Architecture Training: Cumulative Reward over Steps (Test: {best_test_name})")
        plt.legend()
        plt.grid(True)
        plt.show()

        plt.figure(figsize=(12, 6))
        plt.plot(steps, queues, marker='o', linestyle='-', label=f"Total Queue Length ({best_test_name})")
        plt.xlabel("Simulation Step")
        plt.ylabel("Total Queue Length")
        plt.title(f"Best RL Architecture Training: Queue Length over Steps (Test: {best_test_name})")
        plt.legend()
        plt.grid(True)
        plt.show()

# NOTE: The original plotting code has been replaced by the visualization of the single best run.