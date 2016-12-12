import os

def get_pixel(x,y):
    """ Virtual function. """
    pass

if os.name == 'nt': # if windows
    from ctypes import windll
    dc = windll.user32.GetDC(0)

    def get_pixel(x,y):
        rgb = windll.gdi32.GetPixel(dc, x, y)
        r = rgb%256
        g = (rgb>>8)%256
        b = (rgb>>16)%256
        return (r, g, b)

if __name__ == '__main__':
    print(get_pixel(300,300))