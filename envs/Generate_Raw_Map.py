from typing import List, Tuple

import numpy as np


def generate_raw_map(file_path: str, create_agents: bool = False, number_of_agents: int = -1) -> List[np.ndarray]:
    # Generate a random raw map
    raw_state_map = np.loadtxt(file_path, delimiter=",", dtype=int)
    raw_obstacle_map = np.where(raw_state_map == -1, -1, 0)
    if create_agents:
        zero_indices = np.where(raw_state_map == 0)
        zero_coords = [(x, y) for x, y in zip(zero_indices[0], zero_indices[1])]
        selected_agents_idx = np.random.choice(len(zero_coords), number_of_agents, replace=False)
        for i, idx in enumerate(selected_agents_idx):
            raw_state_map[zero_coords[idx]] = i + 1
    else:
        one_indices = np.where(raw_state_map >= 1)
        replacement_values = np.arange(1, len(one_indices[0]) + 1)
        raw_state_map[one_indices] = replacement_values
    print(raw_state_map)
    np.savetxt("state.csv", raw_state_map, delimiter=",", fmt="%5d")
    if not create_agents:
        number_of_agents = len(one_indices[0])
    zero_indices = np.where(raw_state_map == 0)
    # 座標の形に変換
    zero_coords = [(x, y) for x, y in zip(zero_indices[0], zero_indices[1])]
    # Randomly select goals for each agent
    selected_goals_idx = np.random.choice(len(zero_coords), number_of_agents, replace=False)
    raw_goals_map = raw_obstacle_map.copy()
    for i, idx in enumerate(selected_goals_idx):
        raw_goals_map[zero_coords[idx]] = i + 1
    print(raw_goals_map)
    np.savetxt("goals.csv", raw_goals_map, delimiter=",", fmt="%5d")
    return [raw_state_map, raw_goals_map]


def main():
    file_path = "warehouse_with_ep.csv"
    number_of_envs = 20
    number_of_agents = 128
    for i in range(number_of_envs):
        raw_map = generate_raw_map(file_path, True, number_of_agents)
        # raw_map = generate_raw_map(file_path, False)
        np.save("warehouse2/100agents_161size_0density_0wall_id" + str(i), raw_map)


if __name__ == "__main__":
    main()
