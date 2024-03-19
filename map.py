import pygame 
import os

class Map:
    def __init__(self, src):
        self.img = pygame.image.load(src)
        self.img_mask = None
        self.img_rect = None

    def blit(self, screen):
        self.img_mask = pygame.mask.from_surface(self.img)
        self.img_rect = self.img.get_rect()
        screen.blit(pygame.transform.scale(self.img,(screen.get_size())), (0, 0))

        # win.blit(pygame.transform.scale(screen, win.get_rect().size), (0, 0))
