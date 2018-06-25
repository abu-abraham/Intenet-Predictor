def getImageArray():
    img = Image.open('blum.png').convert('RGBA')
    if black_white:
        img = img.convert('1')
    return str(np.array(img))
