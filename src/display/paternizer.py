import pygame
import time
import random
import numpy as np


class Note:
    def __init__(self, name, dt=-100):
        self.name = name
        self.dt = dt

def add(u, v): return (u[0]+v[0], u[1]+v[1])

def dist(p1, p2):
    z_to_pos = {'BL': (-0.5, -0.5),
              'UL': (-0.5, +0.5),
              'C': (0, 0),
              'UR': (+0.5, +0.5),
              'BR': (+0.5, -0.5)}
    o_to_pos = [(0, 0), (2, 0)]
    
    p1_pos = add(z_to_pos[p1[0]], o_to_pos[p1[1]-1])
    p2_pos = add(z_to_pos[p2[0]], o_to_pos[p2[1]-1])
    return (p1_pos[0]-p2_pos[0])**2 + (p1_pos[1]-p2_pos[1])**2

class Paternizer:
    def __init__(self):
        self.screen = pygame.display.set_mode((1200, 900))
        self.arrows = pygame.image.load("./data/arrows.PNG")
        self.arrows_pos = {'C': (0, 0, 78, 78),
                           'UR': (78, 0, 76, 76),
                           'UL': (76*2+2, 0, 76, 76),
                           'BR': (76*3+2, 0, 76, 76),
                           'BL': (76*4+2, 0, 76, 76)}
        
        self.decalage = {('BL', 1): 0,
                         ('UL', 1): 1,
                         ('C', 1): 2,
                         ('UR', 1): 3,
                         ('BR', 1): 4,
                         ('BL', 2): 5,
                         ('UL', 2): 6,
                         ('C', 2): 7,
                         ('UR', 2): 8,
                         ('BR', 2): 9}

        self.notes = [Note(('C', 1), -100)]
        self.speed = 170
    
    def run(self):
        dt = 0
        t0 = time.perf_counter()
        t1 = t0

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return
                
                if event.type == pygame.KEYDOWN:
                    if event.key == pygame.K_SPACE:
                        self.speed -= 10
                    elif event.key == pygame.K_RETURN:
                        self.speed += 10
            
            # draw
            for note in self.notes:
                if -200 <= note.dt <= 50:
                    dx = self.decalage[note.name]
                    self.screen.blit(self.arrows, (100+dx*100, 50-int(note.dt*10)), self.arrows_pos[note.name[0]])

            for pos in self.decalage:
                dx = self.decalage[pos]
                self.screen.blit(self.arrows, (100+dx*100, 50), self.arrows_pos[pos[0]])


            dt = time.perf_counter() - t1
            t1 = time.perf_counter()
            # update
            for note in self.notes:
                note.dt += self.speed * dt
                if note.dt > 50:
                    self.notes.remove(note)

            if len(self.notes) < 5 and not(self.notes) or self.notes[-1].dt > -90:
                if np.cos((time.perf_counter() - t0)*np.pi*4) >= 0.99:
                    p = random.choice(list(self.decalage.keys())[3:-3])
                    if self.notes:
                        lp = self.notes[-1].name
                    if dist(p, lp) < 3:    
                        self.notes.append(Note(p))

            pygame.display.flip()
            self.screen.fill((37, 67, 97))
            