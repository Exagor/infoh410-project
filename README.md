# Info-h410 project

We will do reinforcement learning on space invaders (normally).

The project is realised with the gymnasium library.
The space invaders environment is provided by the gym library at this [link](https://gym.openai.com/envs/SpaceInvaders-v0/).

## Installation

To install the required libraries, run the following command:

```bash
pip install -r requirements.txt
```

## Environnement

Your objective is to destroy the space invaders by shooting your laser cannon at them before they reach the Earth. The game ends when all your lives are lost after taking enemy fire, or when they reach the earth.

### Action space
SpaceInvaders has the action space of Discrete(6) with the table below listing the meaning of each action’s meanings.

| Value | Meaning   |
|-------|-----------|
| 0     | NOOP      |
| 1     | FIRE      |
| 2     | RIGHT     |
| 3     | LEFT      |
| 4     | RIGHTFIRE |
| 5     | LEFTFIRE  |

### Observation
Atari environments have three possible observation types: "rgb", "grayscale" and "ram".

obs_type="rgb" -> observation_space=Box(0, 255, (210, 160, 3), np.uint8)

obs_type="ram" -> observation_space=Box(0, 255, (128,), np.uint8)

obs_type="grayscale" -> Box(0, 255, (210, 160), np.uint8), a grayscale version of the “rgb” type

### Reward
You gain points for destroying space invaders. The invaders in the back rows are worth more points. For a more detailed documentation