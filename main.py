import pygame
import os
import math
import sys
import neat
import random

pygame.init()
SCREEN_WIDTH = 1658
SCREEN_HEIGHT = 795
SCREEN = pygame.display.set_mode((SCREEN_WIDTH, SCREEN_HEIGHT))

TRACK = pygame.image.load(os.path.join("Assets", "mytrack orginal2.png"))


class Car(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.original_image = pygame.image.load(os.path.join("Assets", "car.png"))
        self.image = self.original_image
        self.rect = self.image.get_rect(center=(random.randrange(150, 1200), 170))
        self.velocity = pygame.math.Vector2(0.8, 0)
        self.angle = 0
        self.rotation_velocity = 5
        self.direction = 0
        self.alive = True
        self.radars = []

    def update(self):
        self.radars.clear()
        self.drive()
        self.rotate()
        for radar_angle in (-60, -30, 0, 30, 60):
            self.detect_obstacles(radar_angle)
        self.check_collision()
        self.collect_data()

    def drive(self):
        self.rect.center += self.velocity * 6

    def check_collision(self):
        length = 40
        collision_point_right = [int(self.rect.center[0] + math.cos(math.radians(self.angle + 18)) * length),
                                 int(self.rect.center[1] - math.sin(math.radians(self.angle + 18)) * length)]
        collision_point_left = [int(self.rect.center[0] + math.cos(math.radians(self.angle - 18)) * length),
                                int(self.rect.center[1] - math.sin(math.radians(self.angle - 18)) * length)]

        # Check collision with track boundary
        if SCREEN.get_at(collision_point_right) == pygame.Color(34, 177, 76, 255) \
                or SCREEN.get_at(collision_point_left) == pygame.Color(34, 177, 76, 255):
            self.alive = False

        # Draw collision points
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_right, 4)
        pygame.draw.circle(SCREEN, (0, 255, 255, 0), collision_point_left, 4)

    def rotate(self):
        if self.direction == 1:
            self.angle -= self.rotation_velocity
            self.velocity.rotate_ip(self.rotation_velocity)
        if self.direction == -1:
            self.angle += self.rotation_velocity
            self.velocity.rotate_ip(-self.rotation_velocity)

        self.image = pygame.transform.rotozoom(self.original_image, self.angle, 0.1)
        self.rect = self.image.get_rect(center=self.rect.center)

    def detect_obstacles(self, radar_angle):
        length = 0
        x = int(self.rect.center[0])
        y = int(self.rect.center[1])

        # Check for obstacles along radar direction
        while not SCREEN.get_at((x, y)) == pygame.Color(34, 177, 76, 255) and length < 100:
            length += 1
            x = int(self.rect.center[0] + math.cos(math.radians(self.angle + radar_angle)) * length)
            y = int(self.rect.center[1] - math.sin(math.radians(self.angle + radar_angle)) * length)

        # Draw radar
        pygame.draw.line(SCREEN, (255, 255, 255, 255), self.rect.center, (x, y), 1)
        pygame.draw.circle(SCREEN, (0, 255, 0, 0), (x, y), 3)

        distance = int(math.sqrt(math.pow(self.rect.center[0] - x, 2)
                                 + math.pow(self.rect.center[1] - y, 2)))

        self.radars.append([radar_angle, distance])

    def collect_data(self):
        input_data = [0, 0, 0, 0, 0]
        for i, radar in enumerate(self.radars):
            input_data[i] = int(radar[1])
        return input_data


def remove_car(index):
    cars.pop(index)
    genome.pop(index)
    neural_nets.pop(index)


def draw_text(surface, text, size, x, y):
    font = pygame.font.Font("freesansbold.ttf", size)
    text_surface = font.render(text, True, "black")
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surface.blit(text_surface, text_rect)


generation = 0


def evaluate_genomes(genomes, config,b):

    global cars, genome, neural_nets, generation

    cars = []
    genome = []
    neural_nets = []
    generation += 1

    for genome_id, genome_data in genomes:
        cars.append(pygame.sprite.GroupSingle(Car()))
        genome.append(genome_data)
        neural_net = neat.nn.FeedForwardNetwork.create(genome_data, config)
        neural_nets.append(neural_net)
        genome_data.fitness = 0

    run_simulation = True
    while run_simulation:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

        SCREEN.blit(TRACK, (0, 0))

        if len(cars) == 0:
            break

        for i, car in enumerate(cars):
            genome[i].fitness += 1
            if not car.sprite.alive:
                remove_car(i)

        for i, car in enumerate(cars):
            output = neural_nets[i].activate(car.sprite.collect_data())
            if output[0] > 0.7:
                car.sprite.direction = 1
            if output[1] > 0.7:
                car.sprite.direction = -1
            if output[0] <= 0.7 and output[1] <= 0.7:
                car.sprite.direction = 0

        # Update display
        draw_text(SCREEN, "GENERATION :" + str(generation), 30, 700, 30)
        draw_text(SCREEN, "ALIVE :" + str(len(cars)), 30, 900, 30)
        for car in cars:
            car.draw(SCREEN)
            car.update()
        pygame.display.update()


# NEAT Neural Network Setup
def run_neat(config_path):
    global population
    config = neat.config.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        config_path
    )

    population = neat.Population(config)

    population.add_reporter(neat.StdOutReporter(True))
    stats = neat.StatisticsReporter()
    population.add_reporter(stats)

    population.run(evaluate_genomes, 50)


if __name__ == '__main__':
    local_dir = os.path.dirname(__file__)
    config_path = os.path.join(local_dir, 'config.txt')
    run_neat(config_path)
