import pandas as pd
import matplotlib.pyplot as plt
import sys
sys.path.append('C:/Users/emanu/THESIS_BJ_POKER_REINFORCEMENTLEARNING/')

def plot_function(csv_file):
    data = pd.read_csv(csv_file)
    # Check because it had some problem reading csv
    required_columns = ['loss_0', 'mean_episode_return_0', 'frames']
    if not all(col in data.columns for col in required_columns):
        print("Required columns not found in the CSV file.")
        return
    
    # Convert columns to numeric
    for col in ['loss_0', 'mean_episode_return_0', 'frames']:
        if data[col].dtype == 'object':
            data[col] = pd.to_numeric(data[col], errors='coerce')
    
    frames = data['frames']
    mean_loss = data['loss_0']
    mean_reward = data['mean_episode_return_0']
    
    # Filter mean loss values less than 10
    filtered_loss = mean_loss[mean_loss < 2]
    filtered_frames_for_loss = frames[mean_loss < 2]

    # Plot mean loss per frame for values < 2
    plt.figure(figsize=(10, 5))
    plt.plot(filtered_frames_for_loss, filtered_loss, label='Mean Loss')
    plt.xlabel('Frames')
    plt.ylabel('Mean Loss')
    plt.title('Mean Loss per Frame')
    plt.legend()
    plt.grid(True)
    plt.savefig('_mean_loss_filtered.png')
    plt.show()
    
    # Plot mean reward per frame
    plt.figure(figsize=(10, 5))
    plt.plot(frames, mean_reward, label='Mean Reward')
    plt.xlabel('Frames')
    plt.ylabel('Mean Reward')
    plt.title('Mean Reward per Frame')
    plt.legend()
    plt.grid(True)
    plt.savefig('_mean_reward.png')
    plt.show()



plot_function('experiments/dmc_result_final/blackjack/logs.csv')

