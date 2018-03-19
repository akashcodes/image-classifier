from tkinter import *
from PIL import Image
import numpy as np


imgmat = np.zeros([400, 400])


win = Tk()
win.geometry("400x400")
canvas = Canvas(win, width=400, height=400, bg="#ffffff")

def callbackDraw( event=None ):
        canvas.create_rectangle(event.x-10,event.y-10,event.x+10,event.y+10,fill='black', outline='black')
        for i in range(event.x-15, event.x+16):
            for j in range(event.y-15, event.y+16):
                try:
                    imgmat[i][j] = 1
                except:
                    pass

def callbackErase( event=None ):
        canvas.create_rectangle(event.x-15,event.y-15,event.x+15,event.y+15,fill='#ffffff', outline='white')
        for i in range(event.x-30, event.x+31):
            for j in range(event.y-30, event.y+31):
                try:
                    imgmat[i][j] = 0
                except:
                    pass

canvas.bind('<B1-Motion>',callbackDraw)
canvas.bind('<B3-Motion>',callbackErase)

canvas.grid()

win.mainloop()

print(imgmat)

img = (((imgmat - imgmat.min()) / (imgmat.max() - imgmat.min())) * 255.9).astype(np.uint8)
im = Image.fromarray(img)

im.thumbnail([20, 20])

arr = np.transpose(np.array(im.getdata()))

np.savetxt("test_x.txt", arr)