import pygame
from pygame_widgets.button import Button

colors = [[255] * 28 for _ in range(28)]

def main():

    # pygame setup
    pygame.init()

    width, height = 720, 500
    screen = pygame.display.set_mode((width, height))

    running = True
    drawing = False

    reset = Button(
        screen,
        600,
        40,
        80,
        40,

        text='Reset',
        fontSize=50,
        inactiveColour=(117,150,86),
        hoverColour=(125,166,79),
        pressedColour=(128,182,76),
        radius=20,
        onClick=lambda: resetColors()
    )

    while running:
        # poll for events
        # pygame.QUIT event means the user clicked X to close your window
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            if event.type == pygame.MOUSEBUTTONDOWN:
                drawing = True
            if event.type == pygame.MOUSEBUTTONUP:
                drawing = False
                    
        if drawing:
            x, y = pygame.mouse.get_pos()
            i, j = (int)((x-40)/15), (int)((y-40)/15)
            if i >= 0 and i < 28 and j >= 0 and j < 28:
                colors[i][j] = 0
                if i+1<28:
                    colors[i+1][j] = 40
                if j+1<28:
                    colors[i][j+1] = 40
                if i+1<28 and j+1<28:
                    colors[i+1][j+1] = 80
                
        
        for i in range(28):
            for j in range(28):
                squareRect = pygame.Rect(40+i*15, 40+j*15, 12, 12)
                pygame.draw.rect(screen, (colors[i][j], colors[i][j], colors[i][j]), squareRect)

        # flip() the display to put your work on screen
        pygame.display.flip()

    pygame.quit()

def resetColors():
    colors = [[255] * 28 for _ in range(28)]

if __name__=="__main__":
    main()