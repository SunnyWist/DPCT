from typing import List, Tuple

import numpy as np
import os


def generate_raw_map(file_path: str, number_of_agents: int = -1) -> List[np.ndarray]:
    # Generate a random raw map
    raw_map = np.loadtxt(file_path, delimiter=",", dtype=int)
    obstacle_map = np.where(raw_map == -1, -1, 0)
    ep_map = np.where(raw_map == -9, 1, 0)
    state_map = obstacle_map.copy()
    goals_map = obstacle_map.copy()

    # エージェントの位置をランダムに選択
    one_indices = np.where(raw_map == 1)
    one_coords = [(x, y) for x, y in zip(one_indices[0], one_indices[1])]
    selected_agents_idx = np.random.choice(len(one_coords), number_of_agents, replace=False)
    for i, idx in enumerate(selected_agents_idx):
        state_map[one_coords[idx]] = i + 1
    print(state_map)
    np.savetxt("state.csv", state_map, delimiter=",", fmt="%5d")

    # エージェントのゴールをランダムに選択
    ep_indices = np.where(raw_map == -9)
    ep_coords = [(x, y) for x, y in zip(ep_indices[0], ep_indices[1])]
    selected_goals_idx = np.random.choice(len(ep_coords), number_of_agents, replace=False)
    for i, idx in enumerate(selected_goals_idx):
        goals_map[ep_coords[idx]] = i + 1
    print(goals_map)
    np.savetxt("goals.csv", goals_map, delimiter=",", fmt="%5d")

    np.savetxt("ep.csv", ep_map, delimiter=",", fmt="%5d")

    return [state_map, goals_map, ep_map]


def main():
    file_path = "warehouse_with_ep2.csv"
    save_parent_folder = "warehouse1_ep2"
    number_of_envs = 20
    # number_of_agents = 192
    agents_list = [4, 8, 16, 32, 64, 128, 256]
    # agents_list = [8]
    for number_of_agents in agents_list:
        for i in range(number_of_envs):
            raw_map = generate_raw_map(file_path, number_of_agents)
            save_folder = os.path.join(save_parent_folder, str(number_of_agents))
            os.makedirs(save_folder, exist_ok=True)
            save_file_name = str(number_of_agents) + "agents_48size_0.0density_0wall_id" + str(i)
            save_file_path = os.path.join(save_folder, save_file_name)
            np.save(save_file_path, raw_map)


if __name__ == "__main__":
    main()
