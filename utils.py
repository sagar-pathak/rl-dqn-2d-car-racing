import pygame
import numpy as np

# Rotate an image with given angle while keeping its center and size
def rot_center(image, angle):
    orig_rect = image.get_rect()
    rot_image = pygame.transform.rotate(image, angle)
    rot_rect = orig_rect.copy()
    rot_rect.center = rot_image.get_rect().center
    rot_image = rot_image.subsurface(rot_rect).copy()
    return rot_image

# PLE SPECIFIC
def process_state(state):
    return np.array(list(state.values())).flatten()