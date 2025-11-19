import numpy as np
import sys

seed = int(sys.argv[1])
n_agents = int(sys.argv[2])
n_episodes = int(sys.argv[3])

rng: np.random.Generator = np.random.default_rng(seed=seed)


v_pref = 0.6
radius = 0.15
min_dist = radius * 6
circle_radius_x = 1.5
circle_radius_y = 1.5
states = []

for _ in range(n_episodes // 2 + 1):
    positions = []
    for i in range(n_agents):
        ct=0
        while True:
            angle = rng.random() * np.pi * 2
            # add some noise to simulate all the possible cases robot could meet with human
            px_noise = (rng.random() - 0.5) * v_pref
            py_noise = (rng.random() - 0.5) * v_pref
            px = circle_radius_x * np.cos(angle) + px_noise
            py = circle_radius_y * np.sin(angle) + py_noise
            collide = False
    
            for j in range(i):
                # keep human at least 3 meters away from robot
                if np.linalg.norm((px - positions[j][0], py - positions[j][1])) < min_dist:
                    collide = True
                    break
            if not collide:
                break
            ct+=1
            if ct >= 1000:
                print("Too many regeneration")
                exit(1)
    
        positions.append(np.array([px,py]))
    positions = np.array(positions)
    states.append(positions)
    states.append(-positions)

np.save(sys.argv[4], np.array(states))
