import pygame
import math

class Sensors:
    def __init__(self, car, circuit, screen=None):
        self.screen = screen
        self.car = car
        self.circuit = circuit
        self.nr_of_sensors = 7
        self.gap = 5
        self.sensors = self.init_sensors() # [] Sensor angle list

    def init_sensors(self):
        sensors = []
        angle_range = (self.nr_of_sensors - 1) * self.gap
        start_angle = -angle_range // 2

        for deg in range(start_angle + angle_range, start_angle - 1, -self.gap):
            sensors.append((deg))
        return sensors

    def measure(self, show_sensor):
        measurements = []
        for sensor_deg in self.sensors:
            car_center_x = self.car.x + self.car.img_rect.center[0]
            car_center_y = self.car.y + self.car.img_rect.center[1]

            deg_x = deg_x = 0
            len = 0
            for i in range(1500):
                deg_x = car_center_x + math.cos(math.radians(sensor_deg + self.car.dir)) * len
                deg_y = car_center_y - math.sin(math.radians(sensor_deg + self.car.dir)) * len

                line_surface = pygame.Surface((2, 2), pygame.SRCALPHA)
                line_rect = line_surface.get_rect()
                line_rect.topleft = deg_x, deg_y
                line_surface.fill((255,0,0))
                # https://stackoverflow.com/questions/62008457/overlap-between-mask-and-fired-beams-in-pygame-ai-car-model-vision
                if self.circuit.img_mask.overlap(pygame.mask.from_surface(line_surface),
                                                (int(line_rect[0]), int(line_rect[1]))) is not None:
                    break
                len += 1
            measurements.append(len/1000)
            if show_sensor:
                pygame.draw.line(self.screen, (255,255, 0), (car_center_x, car_center_y), (deg_x, deg_y), 2)
            
        return measurements
