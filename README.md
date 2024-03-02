
# NEAT Self Driving Car with Pygame

This project implements a simple car simulation game using the NEAT (NeuroEvolution of Augmenting Topologies) algorithm and Pygame library in Python.

![Screenshot 2024-03-02 111014](https://github.com/amalsam/Self-Driving-Car-Simulation/assets/47856775/5b0b998a-a88e-490a-8cf1-cddb45ff355b)

## Overview

In this game, cars are controlled by neural networks trained using NEAT. The objective is to navigate the cars through a track without colliding with obstacles. The cars' behavior is controlled by neural networks trained through a genetic algorithm provided by the NEAT library.

## Features

- NEAT algorithm for training neural networks.
- Pygame for graphics rendering and game implementation.
- Cars with radar sensors for obstacle detection.
- Genetic algorithm for evolving neural networks.

## Prerequisites

Before running the program, make sure you have the following installed:

- Python 3.x
- Pygame library
- NEAT library

You can install Pygame and NEAT via pip:

```bash pip install pygame```
```pip install neat-python```


``` git clone https://github.com/amalsam/Self-Driving-Car-Simulation.git```
```cd Self-Driving-Car-Simulation ```
```python main.py```



## Customization
You can customize various aspects of the game, such as track layout, car appearance, and neural network configuration:

Modify the track image in the Assets folder to change the track layout.
Adjust parameters in the config.txt file to modify the NEAT algorithm's behavior.
Replace the car image in the Assets folder to change the car's appearance.

## Acknowledgements
This project was inspired by various tutorials and resources on NEAT and Pygame.
Special thanks to OpenAI for developing the NEAT library.
