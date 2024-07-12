import pygame
import pygame_widgets
from pygame_widgets.button import Button
import math

WHITE = (255, 255, 255)

colors = [[0] * 28 for _ in range(28)]
drawing_rect = pygame.Rect(40, 40, 380, 380)
drawing_pane = pygame.Surface((380, 380))
drawing_pane.fill(WHITE)

def main():

    # pygame setup
    pygame.init()

    width, height = 720, 500
    screen = pygame.display.set_mode((width, height))

    reset = Button(
        screen,
        60,
        430,
        160,
        40,

        text='Clear',
        fontSize=25,
        inactiveColour=(255,255,255),
        hoverColour=(200,200,200),
        pressedColour=(100,100,100),
        radius=5,
        onClick=lambda: clear(drawing_pane)
    )

    classify = Button(
        screen,
        240,
        430,
        160,
        40,

        text='Classify',
        fontSize=25,
        inactiveColour=(255,255,255),
        hoverColour=(200,200,200),
        pressedColour=(100,100,100),
        radius=5,
        onClick=lambda: classify()
    )

    screen.fill((100, 100, 255))

    running = True

    while running:
        
        eventList = pygame.event.get()
        for event in eventList:
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEMOTION:
                if pygame.mouse.get_pressed()[0]:
                    last_pos = (event.pos[0]-event.rel[0], event.pos[1]-event.rel[1])
                    draw_line(drawing_pane, (0, 0, 0), last_pos, pygame.mouse.get_pos())
        
        pygame_widgets.update(eventList)
        screen.blit(drawing_pane, (40, 40))
        pygame.display.flip()

    pygame.quit()

# adapted from https://github.com/drewvlaz/draw-mnist/blob/master/main.py
def draw_line(surface, color, start, end):
    dy = end[1] - start[1]
    dx = end[0] - start[0]

    distance = round(math.sqrt(dy**2 + dx**2))
    for i in range(distance):
        x, y = start[0]+i/distance*dx, start[1]+i/distance*dy
        if drawing_rect.collidepoint(x, y):
            pygame.draw.circle(surface, color, (x - 40, y - 40), 9)
    

def clear(surface):
    surface.fill(WHITE)

def classify():
    pass

if __name__=="__main__":
    main()