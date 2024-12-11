# f1tenth-reinforcement-learning
- Code for Autonomous Driving(M3621.000300) class
```bash
 ros2_ws
    ├── build
    ├── install
    ├── log
    ├── src
        ├── f1tenth
        │    ├── * algorithm
        │    ├── configs
        │    ├── installation
        │    ├── leaderboard.py
        │    ├── main_il.py
        │    ├── main_rl.py
        │    ├── README.md
        │    ├── requirements.txt
        │    ├── * utils
        │    └── visualize.py
        └── f1tenth_ros
             ├── ackermann_mux
             ├── * f1tenth_stack
             ├── LICENSE
             ├── README.md
             ├── scripts
             ├── teleop_tools
             └── vesc

```

## Installation
0. Before colcon build, ```pip install setuptools==58.2.0 empy==3.3.4 lark```
1. ```cd ros2_ws```
2. ```colcon build --symlink-install```
3. ```cd src/f1tenth/installation && pip install -e .```
