import pygame, random
import pandas as pd
import time
import cv2
from matplotlib import pyplot as plt
import numpy as np
import threading
#from calibracion_camara import *

#-----------------Seguimiento de video-----------------------------------------------------

# Iniciamos camara a utilizar
imag = cv2.VideoCapture(0)

# Se selecciona el rango de color en formato HSV de 0 a 255 (0% a 100%) a detectar
'''Detectara un color rojo'''
RangoBajo1 = np.array([25, 100, 20], np.uint8)
RangoAlto1 = np.array([0, 255, 255], np.uint8)

'Valores a utlizar en Kalman'
dt = 0.01  # Periodo de muestreo del sensor (segundos)

pos_0 = 0  # Posición inicial del sensor (m)
vel_final = 0  # Velocidad final del sensor (m/s)
acel_0 = 0  # Aceleracion inicial del sensor

Q = np.array([[2.7473e+05, 4.0006e+04, 9.0248e-12], [4.0006e+04, 8.0010e+03, 1.7881e-08], [9.0248e-12, 1.7881e-08, 0.0038]])

R = 0.0038

'''
Definimos el modelo a utilizar para Kalman

F=[1 dt (dt^2)/2
   0  1  dt
   0  0  1]

F = [P
     V
     A]
'''
F = np.array([[1, dt, (dt * dt) / 2], [0, 1, dt], [0, 0, 1]])

F_trans = np.transpose(F)  # Transpuesta de estados

# Modelo de observación
H = np.array([[1, 0, 0]])

H_trans = np.array([[1], [0], [0]])

# Vector de estados inicial
x_nn = np.array([[pos_0], [vel_final], [acel_0]])

# Matriz de covarianza a-posteriori
P = np.array([[0.8, 0.0, 0.0], [0.0, 0.8, 0.0], [0.0, 0.0, 0.8]])

eye_3 = np.array([[1, 0, 0], [0, 1, 0], [0, 0, 1]])

'''Se elimina este comentario si se desea ver las graficas
tf = 10
t = np.arange(0, tf, dt)

M = t.size
print(M)
sizeInput = 2
Ñ = np.zeros((sizeInput, M))'''

k = 0
val_y = 0
val_x = 500

kernel = np.ones((5, 5), np.uint8)

'''Funcion de kalman para el desplazamiento en x'''

def Kalman_x(z, name1):
    global x_nn, F, P, F_trans, H, H_trans, y_m, R, S, eye_3, k, val_x

    # PREDICCION
    x_nn = np.dot(F, x_nn)
    # print(x_nn)
    P = np.add(np.dot(np.dot(F, P), F_trans), Q)
    # print(P)

    # CORRECCIÓN
    y_m = np.subtract(z, (np.dot(H, x_nn)))
    S = np.add(np.dot(np.dot(H, P), H_trans), R)
    K = np.dot(np.dot(P, H_trans), 1 / S)
    x_nn = np.add(x_nn, np.dot(K, y_m))
    P = np.dot(eye_3 - np.dot(K, H), P)

    k = k + 1
    val_x = x_nn[0]


'''Funcion de kalman para el desplazamiento en y'''


def Kalman_y(z, name2):
    global x_nn, F, P, F_trans, H, H_trans, y_m, R, S, eye_3, k, val_y

    # PREDICCION
    x_nn = np.dot(F, x_nn)
    # print(x_nn)
    P = np.add(np.dot(np.dot(F, P), F_trans), Q)

    # CORRECCIÓN
    y_m = np.subtract(z, (np.dot(H, x_nn)))
    S = np.add(np.dot(np.dot(H, P), H_trans), R)
    K = np.dot(np.dot(P, H_trans), 1 / S)
    x_nn = np.add(x_nn, np.dot(K, y_m))
    P = np.dot(eye_3 - np.dot(K, H), P)

    k = k + 1
    val_y = x_nn[0]


#-------------------Juego-------------------------------------------------------------------

WIDTH = 800 #Dimensiones de la ventana
HEIGHT = 600
BLACK = (0, 0, 0)
WHITE = (255, 255, 255)
GREEN = (0, 187, 45)
RED = (255, 36, 0)

pygame.init()
pygame.mixer.init()
screen = pygame.display.set_mode((WIDTH, HEIGHT)) #Se asignan las dimensiones
pygame.display.set_caption("Bichito que le pega a los otros bichitos") #Nombre de la ventana
clock = pygame.time.Clock()

def draw_text(surface, text, size, x, y, color):
    font = pygame.font.SysFont("serif", size)
    text_surface = font.render(text, True, color)
    text_rect = text_surface.get_rect()
    text_rect.midtop = (x, y)
    surface.blit(text_surface, text_rect)

def draw_shield_bar(surface, x, y, percentage):
    BAR_LENGHT = 180
    BAR_HEIGTH = 15
    fill = (percentage/100)* BAR_LENGHT
    border = pygame.Rect(x, y, BAR_LENGHT, BAR_HEIGTH)
    fill = pygame.Rect(x, y, fill, BAR_HEIGTH)
    if percentage >= 50: #Condicion cambio de color de barra
        pygame.draw.rect(surface, GREEN, fill)
    else:
        pygame.draw.rect(surface, RED, fill)
    pygame.draw.rect(surface, WHITE, border, 2)

def show_go_screen():
    screen.blit(background, [0,0])
    draw_text(screen, "Bichito que le pega a los otros bichitos", 40, WIDTH // 2, HEIGHT // 4, WHITE)
    draw_text(screen, "Instrucciones (Están en veremos)", 27, WIDTH // 2, HEIGHT // 2, WHITE)
    draw_text(screen, "Presiona una tecla", 20, WIDTH // 2, HEIGHT * 3/4, RED)
    pygame.display.flip()
    waiting = True
    while waiting:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
            if event.type == pygame.KEYUP:
                waiting = False

def show_game_over():
    screen.blit(background, [0,0])
    draw_text(screen, "GAME OVER", 100, WIDTH // 2, HEIGHT // 4, WHITE)
    draw_text(screen, "Presiona una tecla para reiniciar", 20, WIDTH // 2, HEIGHT * 3/5, RED)
    pygame.display.flip()
    waiting = True
    time.sleep(2)
    while waiting:
        clock.tick(60)
        for event in pygame.event.get():
            if event.type == pygame.KEYUP:
                waiting = False
#---------------------Clase del jugador--------------------------------------------
class Player(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = pygame.image.load("Recursos/player.png").convert() #Se carga la imagen de la nave
        self.image.set_colorkey(BLACK) #Se retira el borde negro de la imagen
        self.rect = self.image.get_rect()
        self.rect.centerx = WIDTH // 2 #Se ubica el centro de la imagen
        self.rect.bottom = HEIGHT - 10
        self.speed_x = 0
        self.shield = 100

    def update(self):
        self.speed_x = 0

        self.rect.update(XXX, 590, 100, 100)

        if self.rect.right > WIDTH: #Se verifica la posición para evitar que salga de la pantalla
            self.rect.right = WIDTH
        if self.rect.left < 0:
            self.rect.left = 0
        if self.rect.top < 0:
            self.rect.top = 0
        if self.rect.bottom > HEIGHT:
            self.rect.bottom = HEIGHT

    def shoot(self):
        bullet = Bullet(self.rect.centerx, self.rect.top) #Se asigna la ubicación de la nave al elemento bala
        all_sprites.add(bullet)
        bullets.add(bullet)
        laser_sound.play()

#-----------------Clase de los meteoritos-----------------------------------------------
class Meteor(pygame.sprite.Sprite):
    def __init__(self):
        super().__init__()
        self.image = random.choice(meteor_images) #Se carga la imagen de los meteoritos
        self.image.set_colorkey(BLACK) #Se retira el borde negro de la imagen
        self.rect = self.image.get_rect()
        self.rect.x = random.randrange(WIDTH - self.rect.width) #Se asignan velocidades aleatorias en X y Y
        self.rect.y = random.randrange(-140, -100)
        self.speedy = random.randrange(1, 10)
        self.speedx = random.randrange(-5, 5)

    def update(self):
        self.rect.y += self.speedy
        self.rect.x += self.speedx
        if self.rect.top > HEIGHT + 10 or self.rect.left < -40 or self.rect.right > WIDTH + 25: #Si un elemento se sale de la ventana se vuelve a ingresar
            self.rect.x = random.randrange(WIDTH - self.rect.width)
            self.rect.y = random.randrange(-140, -100)
            self.speedy = random.randrange(1, 10)

#----------Clase de las balas---------------------------------------
class Bullet(pygame.sprite.Sprite):
    def __init__(self, x, y):
        super().__init__()
        self.image = pygame.image.load("Recursos/laser1.png") # se trae la imagen de la bala
        self.image.set_colorkey(BLACK)
        self.rect = self.image.get_rect()
        self.rect.y = y
        self.rect.centerx = x
        self.speedy = -10

    def update(self):
        self.rect.y += self.speedy
        if self.rect.bottom < 0: #Si se sale de la ventana se eimina el elemento
            self.kill()

#-----------Clase explosión-----------------------------
class Explosion(pygame.sprite.Sprite):
    def __init__(self, center, vel):
        super().__init__()
        self.image = explosion_anim[0]
        self.rect = self.image.get_rect()
        self.rect.center = center
        self.frame = 0
        self.last_update = pygame.time.get_ticks()
        self.frame_rate = vel #Velocidad de la explosión

    def update(self):
        now = pygame.time.get_ticks()
        if now - self.last_update > self.frame_rate:
            self.last_update = now
            self.frame += 1
            if self.frame == len(explosion_anim):
                self.kill()
            else:
                center = self.rect.center
                self.image = explosion_anim[self.frame]
                self.rect = self.image.get_rect()
                self.rect.center = center

#Lista de meteoritos
meteor_images = []
meteor_list = ["Recursos/meteorGrey_big1.png", "Recursos/meteorGrey_big2.png", "Recursos/meteorGrey_big3.png",
               "Recursos/meteorGrey_big4.png", "Recursos/meteorGrey_med1.png", "Recursos/meteorGrey_med2.png",
               "Recursos/meteorGrey_small1.png", "Recursos/meteorGrey_small2.png", "Recursos/meteorGrey_tiny1.png",
               "Recursos/meteorGrey_tiny2.png"]
for img in meteor_list:
    meteor_images.append(pygame.image.load(img).convert())

#Imagenes explosión
explosion_anim = []
for i in range(9):
    file = "Recursos/regularExplosion0{}.png".format(i)
    img = pygame.image.load(file).convert()
    img.set_colorkey(BLACK)
    img_scale = pygame.transform.scale(img, (70,70))
    explosion_anim.append(img_scale)

#Cargar la imagen de fondo
background = pygame.image.load("Recursos/background.png").convert()

#Cargar sonidos
laser_sound = pygame.mixer.Sound("Recursos/laser5.ogg")
explosion_sound = pygame.mixer.Sound("Recursos/explosion.wav")
pygame.mixer.music.load("Recursos/music.ogg")
pygame.mixer.music.set_volume(0.2)

pygame.mixer.music.play(loops=-1) #Se reproduce la musica de fondo

#----------Game Over-------------------------------------------
game_over = False
Start = True

running = True
while running:
    # Lectura de imagen
    ret, frame = imag.read()

    # Se pasa a escala de colores la imagen obtenida
    frameHSV = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    # Se le aplica un filtrado de color para detectar el que se desa
    mascara = cv2.inRange(frameHSV, RangoBajo1, RangoAlto1)

    # Se alica un mascara de mas para ver el color y el fondo negro a detectar
    '''Opcional'''
    maskRedivs = cv2.bitwise_and(frame, frame, mask=mascara)

    # Se aplica la morfologia para obtener los datos a analizar
    cnts = cv2.morphologyEx(mascara, cv2.MORPH_OPEN, kernel)

    # Obtencion de datos analizar
    x, y, w, h = cv2.boundingRect(cnts)

    '''Visualizacion en rectangulos y el centroide con un punto al objeto detectado'''
    cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
    cv2.circle(frame, (int(x + w / 2), int(y + h / 2)), 1, (255, 0, 0), 4)

    # Imagenes a observar
    cv2.imshow('Camara', cv2.flip(frame,1 ))
    cv2.imshow('mascara', mascara)
    '''cv2.imshow('Inv', maskRedivs)'''

    # Calculo de variables para introducir a Kalman
    arduinoSerial1 = (x + w / 2)
    arduinoSerial2 = (y + h / 2)

    # Toma de tiempo por si se desea ver cuanto se demora
    tic = time.time()

    # Se crean los hilos para Kalman en los ejes x y y
    Kalman_1 = threading.Thread(target=Kalman_x, args=(arduinoSerial1, 'Kal_1'))
    Kalman_2 = threading.Thread(target=Kalman_y, args=(arduinoSerial2, 'Kal_2'))
    Kalman_1.start()
    Kalman_2.start()

    X_ant = val_x

    val_x = np.round(val_x, 2)
    val_y = np.round(val_y, 2)

    XXX = int(630-((val_x*800)/630))

    # Se imprimen los valores salientes de los hilos
    print(((val_x*187)/630),((val_y*147)/475))

    '''Cuando se desee detener el codigo se oprime la tecla Q'''
    if cv2.waitKey(1) == ord('q'):
        break



    if game_over:
        show_game_over()
        Start = True
        game_over = False

    if Start:
        show_go_screen()

        Start = False

        all_sprites = pygame.sprite.Group()
        meteor_list = pygame.sprite.Group()
        bullets = pygame.sprite.Group()

        player = Player()  # Se crea el jugador
        all_sprites.add(player)

        for i in range(8):  # Se crean los objetos meteoritos
            meteor = Meteor()
            all_sprites.add(meteor)
            meteor_list.add(meteor)

        score = 0


    clock.tick(60) #60 frames por segundo
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            running = False

        elif event.type == pygame.KEYDOWN:
            if event.key == pygame.K_SPACE:
                player.shoot()

    if val_y < 200:
        player.shoot()

    all_sprites.update()

    #Verificar colisiones bala - meteorito
    hits = pygame.sprite.groupcollide(meteor_list, bullets, True, True)
    for hit in hits:
        score += 10
        explosion_sound.play()
        explosion = Explosion(hit.rect.center, 50)
        all_sprites.add(explosion)
        meteor = Meteor()
        all_sprites.add(meteor)
        meteor_list.add(meteor)

    #Verificar colisiones nave - meteorito
    hits = pygame.sprite.spritecollide(player, meteor_list, True)
    for hit in hits:
        player.shield -= 20

        explosion_sound.play()
        explosion = Explosion(hit.rect.center, 50)
        all_sprites.add(explosion)
        meteor = Meteor()
        all_sprites.add(meteor)
        meteor_list.add(meteor)

        if player.shield <= 0:
            explosion_sound.play()
            explosion = Explosion(hit.rect.center, 1000000)
            all_sprites.add(explosion)
            all_sprites.update()
            game_over = True


    screen.blit(background, [0,0]) #Se pone el fondo


    #Mensajitos
    #if score >= 500:
     #   draw_text(screen, str("Eres muy basto mk"), 25, WIDTH // 2, 50, WHITE)
    if score >= 600:
        screen.blit(background, [0, 0])
        draw_text(screen, str("Sigue así"), 25, WIDTH // 2, 50, WHITE)
    if score >= 800:
        screen.blit(background, [0, 0])
        draw_text(screen, str("GO GO GO"), 25, WIDTH // 2, 50, WHITE)
    if score >= 1000:
        screen.blit(background, [0, 0])
        draw_text(screen, str("Estás a otro nivel"), 25, WIDTH // 2, 50, WHITE)

    # Marcador
    draw_text(screen, str(score), 25, WIDTH // 2, 10, WHITE)

    #Barra de salud
    draw_shield_bar(screen, 10, 10, player.shield)

    all_sprites.draw(screen) #Se dibuja tod en la ventana

    pygame.display.flip()

imag.release()
cv2.destroyAllWindows()
pygame.quit()
