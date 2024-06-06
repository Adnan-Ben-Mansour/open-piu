import pygame
import numpy as np
import time


def load_images(path, format=(1,1), factor=(1,1)):
    img = pygame.image.load(path)
    img = pygame.transform.smoothscale(img, (int(img.get_width()*factor[0]), int(img.get_height()*factor[1])))
    size = img.get_size()
    sx, sy = int(size[0]/format[0]), int(size[1]/format[1])
    res = {}

    for i in range(format[0]):
        for j in range(format[1]):
            res[(i,j)] = img.subsurface((i*sx, j*sy, sx, sy))
    
    return res


def greyscale(surface: pygame.Surface, newalpha=86):
    surface_copy = surface.copy()
    arr = pygame.surfarray.pixels3d(surface_copy)
    mean_arr = np.dot(arr, [0.216, 0.587, 0.144])
    arr[:, :, 0] = mean_arr
    arr[:, :, 1] = mean_arr
    arr[:, :, 2] = mean_arr
    surface_copy.set_alpha(newalpha)
    return surface_copy


def sigmoid(x):
  if x < -10: return 0
  return 1 / (1 + np.exp(-x))


class Displayer:
    def __init__(self):
        self.dim = (1200, 900)
        self.screen = pygame.display.set_mode(self.dim, pygame.RESIZABLE) # | pygame.SCALED)
        self.skin = "./data/HD/"

        pygame.display.set_caption("OpenPIU")
        pygame.display.set_icon(pygame.image.load("./data/icon.png"))

        self.load_surfaces(skin=self.skin, dim=self.dim)

        self.mode = None # un seul mode pour l'instant: lecture de running_chart
        
        self.running_chart = None
        self.reader = None
        self.stop_reading = False
        self.read_notes = []
    
    def load_surfaces(self, skin, dim):
        fx, fy = dim[0]/1200, dim[1]/900

        self.tap = load_images(skin+"Tap 5x2.PNG", (5,2), (fx, fy))

        self.tapnotes = [load_images(skin+"DownLeft TapNote 3x2.PNG", (3,2), (fx, fy)),
                         load_images(skin+"UpLeft TapNote 3x2.PNG", (3,2), (fx, fy)),
                         load_images(skin+"Center TapNote 3x2.PNG", (3,2), (fx, fy)),
                         load_images(skin+"UpRight TapNote 3x2.PNG", (3,2), (fx, fy)),
                         load_images(skin+"DownRight TapNote 3x2.PNG", (3,2), (fx, fy))]

        self.holds = [load_images(skin+"DownLeft Hold 6x1.PNG", (6,2), (fx, fy)),
                      load_images(skin+"UpLeft Hold 6x1.PNG", (6,2), (fx, fy)),
                      load_images(skin+"Center Hold 6x1.PNG", (6,2), (fx, fy)),
                      load_images(skin+"UpRight Hold 6x1.PNG", (6,2), (fx, fy)),
                      load_images(skin+"DownRight Hold 6x1.PNG", (6,2), (fx, fy))]
        
        self.stepfx = load_images(skin+"StepFX 5x1.PNG", (5, 1), (fx, fy))
        

        for i in range(5):
            self.tap[(i,0)].blit(greyscale(self.tapnotes[i][(0, 0)]), (0, 0))

    def convert_pos(self, column, row, size=None):
        row0 = int(self.dim[1]//15)
        column0 = int(self.dim[0]//6)
        x = int(column0 + column * (self.dim[0] - 2*column0)/9)
        y = row0 - row * (660*2)
        if size:
            return (x-size[0]//2, y-size[1]//2)
        return (x, y)
    
    def run(self):
        t0 = time.perf_counter()

        while True:
            for event in pygame.event.get():
                if event.type == pygame.QUIT:
                    return

                if event.type == pygame.VIDEORESIZE:
                    self.dim = self.screen.get_size()
                    self.load_surfaces(self.skin, self.dim)

                if event.type == pygame.KEYDOWN and event.key == pygame.K_q:
                    return
            
            t = time.perf_counter() - t0


            for i in range(10):
                surf = self.tap[(i%5,0)]
                self.screen.blit(surf, self.convert_pos(i, 0, surf.get_size()))


            if self.running_chart:
                to_remove = []
                for note in self.read_notes:
                    column = note[1]
                    pos = note[2]*(16/167)
                    dt = t - pos

                    if note[0] == 1: # TAPNOTE
                        arrow = self.tapnotes[column%5][(0, 0)]
                        alpha = int(sigmoid(-dt*50+3)*255)
                        s2 = None
                        if 0 < alpha < 254:
                            step = alpha//51
                            s2 = self.stepfx[(step, 0)]
                            s2.set_alpha(60)

                        arrow = arrow.copy()
                        arrow.set_alpha(alpha)
                        self.screen.blit(arrow, self.convert_pos(column, dt, arrow.get_size()))
                        if s2:
                            self.screen.blit(s2, self.convert_pos(column, 0, s2.get_size()), special_flags=pygame.BLEND_RGB_ADD)
                    
                    elif note[0] == 2: # HOLDHEAD
                        end_pos = min([n[2] for n in self.read_notes if (n[0] == 3) and (n[1] == column) and (n[2] >= note[2])])*(16/167)
                        end_dt = t - end_pos

                        holding = self.holds[column%5][(int(t*2)%6, 0)]
                        arrow = self.tapnotes[column%5][(0, 0)]

                        p1 = self.convert_pos(column, dt, arrow.get_size())
                        p2 = self.convert_pos(column, end_dt, arrow.get_size())
                        holding = pygame.transform.smoothscale(holding, (holding.get_width(), p2[1]-p1[1]))
                        self.screen.blit(holding, (p1[0], p1[1]+arrow.get_height()//2))

                        alpha = int(sigmoid(-dt*50+3)*255)
                        s2 = None
                        if 0 < alpha < 254:
                            step = alpha//51
                            s2 = self.stepfx[(step, 0)]
                            s2.set_alpha(60)

                        arrow = arrow.copy()
                        arrow.set_alpha(alpha)
                        self.screen.blit(arrow, self.convert_pos(column, dt, arrow.get_size()))
                        if s2:
                            self.screen.blit(s2, self.convert_pos(column, 0, s2.get_size()), special_flags=pygame.BLEND_RGB_ADD)

                    elif note[0] == 3: # HOLDTAIL
                        arrow = self.tapnotes[column%5][(0, 0)]
                        alpha = int(sigmoid(-dt*50+3)*255)
                        s2 = None
                        if 0 < alpha < 254:
                            step = alpha//51
                            s2 = self.stepfx[(step, 0)]
                            s2.set_alpha(60)

                        arrow = arrow.copy()
                        arrow.set_alpha(alpha)
                        self.screen.blit(arrow, self.convert_pos(column, dt, arrow.get_size()))
                        if s2:
                            self.screen.blit(s2, self.convert_pos(column, 0, s2.get_size()), special_flags=pygame.BLEND_RGB_ADD)
                    
                    if (dt > 3):
                        to_remove.append(note)
                    
                to_clean = []
                for note in to_remove:
                    if note[0] == 1:
                        self.read_notes.remove(note)
                    elif note[0] == 3:
                        self.read_notes.remove(note)
                        to_clean.append(note[1])
                for note in to_remove:
                    if note[0] == 2 and (note[1] in to_clean):
                        self.read_notes.remove(note)
                        

                # print(f"Notes affichées: {len(self.read_notes)}")

                
                while not(self.stop_reading) and ((len(self.read_notes) < 50) or (len([n for n in self.read_notes if n[0]==2])>len([n for n in self.read_notes if n[0]==3]))):
                    new_note = self.reader.__next__()
                    print(new_note)
                    if new_note is not None:
                        self.read_notes.append(tuple(new_note))
                    else:
                        self.stop_reading = True


            pygame.display.flip()
            self.screen.fill((7, 17, 27))