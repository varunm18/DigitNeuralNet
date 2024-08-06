import pygame
import pygame_widgets
from pygame_widgets.button import Button
import math
from NeuralNet import MLP
import numpy as np

WHITE = (255, 255, 255)
BLACK = (0, 0, 0)
BACKGROUND = (100, 100, 255)
UPDATE = 0.01
SLEEP = 0.5

pygame.init()

pygame.font.init()
label_font = pygame.font.SysFont('Comic Sans MS', 25)
prob_font = pygame.font.SysFont('Comic Sans MS', 15)

clock = pygame.time.Clock()

width, height = 720, 500
screen = pygame.display.set_mode((width, height))
screen.fill(BACKGROUND)

drawing_rect = pygame.Rect(40, 40, 380, 380)
drawing_pane = pygame.Surface((380, 380))
drawing_pane.fill(WHITE)

pixel_pane = pygame.Surface((380, 380))
pixel_pane.fill(WHITE)
pixelated = False

predictions = pygame.Surface((220, 380))
predictions.fill(WHITE)

probabilities = pygame.Surface((40, 380))
probabilities.fill(BACKGROUND)

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
    onClick=lambda: clear()
)
toggle = Button(
    screen,
    240,
    430,
    160,
    40,

    text='Pixelated',
    fontSize=25,
    inactiveColour=(255,255,255),
    hoverColour=(200,200,200),
    pressedColour=(100,100,100),
    radius=5,
    onClick=lambda: toggle_view()
)

net = MLP.load("Models/98.31%")

def main():

    running = True
    dt = 0
    stop_time = 0

    draw_labels()

    while running:
        
        eventList = pygame.event.get()
        for event in eventList:
            if event.type == pygame.QUIT:
                running = False
                
            if event.type == pygame.MOUSEMOTION and not pixelated:
                if pygame.mouse.get_pressed()[0]:
                    last_pos = (event.pos[0]-event.rel[0], event.pos[1]-event.rel[1])
                    draw_line(last_pos, pygame.mouse.get_pos())
                    stop_time = 0
        
        if len(eventList) == 0:
            stop_time += dt
        
        if dt > UPDATE and stop_time < SLEEP and not pixelated:
            draw_chart(net.run(classify()))
            dt = 0

        if not pixelated:
            screen.blit(drawing_pane, (40, 40))
        else:
            screen.blit(pixel_pane, (40, 40))
        screen.blit(predictions, (460, 40))
        screen.blit(probabilities, (680, 40))

        pygame_widgets.update(eventList)
        pygame.display.flip()

        dt += clock.tick(60) / 1000

    pygame.quit()

def draw_labels():
    for i in range(10):
        screen.blit(label_font.render(str(i), True, BLACK), (440, 40+i*38))

def draw_chart(probs):
    predictions.fill(WHITE)
    probabilities.fill(BACKGROUND)
    pred = np.argmax(probs, axis=1)[0]
    for i in range(10):
        pygame.draw.rect(predictions, (50, 50, 200), (0, 8+i*38, probs[0][i]*220, 20))
        if i==pred:
            probabilities.blit(prob_font.render(percent(probs[0][i]), True, WHITE), (5, 2+i*38))
        else:
            probabilities.blit(prob_font.render(percent(probs[0][i]), True, BLACK), (5, 2+i*38))

def percent(prob):
    return str(round(prob*100)) + "%"

# adapted from https://github.com/drewvlaz/draw-mnist/blob/master/main.py
def draw_line(start, end):
    dy = end[1] - start[1]
    dx = end[0] - start[0]

    distance = round(math.sqrt(dy**2 + dx**2))
    for i in range(distance):
        x, y = start[0]+i/distance*dx, start[1]+i/distance*dy
        if drawing_rect.collidepoint(x, y):
            pygame.draw.circle(drawing_pane, BLACK, (x - 40, y - 40), 12)

def classify():
    scaledBackground = pygame.transform.smoothscale(drawing_pane, (28, 28))
    image = pygame.surfarray.array3d(scaledBackground)
    image = abs(1-image/253)
    image = np.mean(image, axis=2)

    image = image.transpose()
    image = image.ravel()
    return image

def clear():
    drawing_pane.fill(WHITE)
    global pixelated
    if pixelated:
        pixelated = False
        toggle.setText("Pixelated")

def toggle_view():
    global pixelated
    if toggle.string=="Pixelated":
        toggle.setText("Draw")
        pixelated = True

        image = classify().reshape(28, 28)
        image = 255 - image * 255
        for i in range(len(image)):
            for j in range(len(image[i])):
                pygame.draw.rect(pixel_pane, (image[i][j], image[i][j], image[i][j]), (int(380/28*j), int(380/28*i), int(380/28*j), int(380/28*i)))
    else:
        toggle.setText("Pixelated")
        pixelated = False

if __name__=="__main__":
    main()