import pygame
from ple.games import base
import sys

from car import Car
from map import Map
from sensor import Sensors

# Display Screen
SCREEN_WIDTH = 1000
SCREEN_HEIGHT = 653
CAPTION = "Self Driving Car in 2D"

# Source Car
CAR_SRC = "img/car-1.png"
CAR_LENGTH = 10
CAR_WIDTH = 10

# Map
MAP_SRC = "img/map-2.png"


# Colors
WHITE = (255, 255, 255)
BLACK = (0, 0, 0)

# Show sensors
SHOW_SENSOR = True

class CarRacingEnv(base.PyGameWrapper):
    def __init__(self, width=SCREEN_WIDTH, height=SCREEN_HEIGHT):

        actions = {
            0: pygame.K_LEFT,
            1: pygame.K_RIGHT,
            2: None
            # 2: pygame.K_UP
            # 3: pygame.K_DOWN
        }

        base.PyGameWrapper.__init__(self, width, height, actions=actions)

    def init(self):
        self.score = 0
        self.game_over_flag = False
        # self.car = Car(CAR_SRC, 0, SCREEN_WIDTH / 2 - CAR_WIDTH / 2, SCREEN_HEIGHT - 60)
        self.car = Car(CAR_SRC, 0, 200, 535)
        self.car.speed = 5
        self.car.max_speed = 10
        self.car.acceleration = 2
        self.map = Map(MAP_SRC)
        self.sensors = Sensors(self.car, self.map, self.screen)
        self.measurement = []

    def getScore(self):
        return self.score

    def getGameState(self):

        state = {
            "sensors": self.measurement
        }
        return state

    def game_over(self):
        return self.game_over_flag

    def _handle_player_events(self):
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                sys.exit()

            if event.type == pygame.KEYDOWN:
                key = event.key
                if key == self.actions[0]:
                    self.car.wheel -= 0.5
                if key == self.actions[1]:
                    self.car.wheel += 0.5
                # if key == self.actions[2]:
                #     self.car.speed = min(self.car.speed + self.car.acceleration, self.car.max_speed)
                # if key == self.actions[3]:
                #     self.car.speed = max(self.car.speed - self.car.acceleration, -self.car.max_speed/2)

    def step(self, dt):
        self.screen.fill(WHITE)
        self._handle_player_events()
        # self.score += self.rewards["tick"]
        self.car.blit(self.screen, dt)
        self.map.blit(self.screen)

        # sensors.measure()
        self.measurement = self.sensors.measure(SHOW_SENSOR)

        if self.map.img_mask.overlap(self.car.img_mask, (int(self.car.x), int(self.car.y))) is not None:
            self.game_over_flag = True
            self.score += self.rewards["loss"]
            self.score += -20
        else:
            # self.score += 1
            self.score += self.rewards["positive"]
            pass


if __name__ == "__main__":

    pygame.init()
    game = CarRacingEnv()
    game.screen = pygame.display.set_mode(game.getScreenDims(), 0, 32)
    game.clock = pygame.time.Clock()
    game.init()

    while True:
        dt = game.clock.tick_busy_loop(30)
        if game.game_over():
            game.reset()
            print("Over:")

        game.step(dt)
        pygame.display.update()
        print("\nScore: ", game.score)
