import cv2

def binary_filter(image):
    w,h = image.shape

    pfilter = 0

    for x in range(w):
        for y in range(h):
            pfilter += image[x][y]

    pfilter //= (h * w)

    if pfilter > 150:
        clahe = cv2.createCLAHE(clipLimit=(pfilter // 8), tileGridSize=(8, 8))
        image = clahe.apply(image)


    filter = 0

    for x in range(w):
        for y in range(h):
            filter += image[x][y]

    filter //= h * w
    filter = filter ** 0.92

    ret, bin_image = cv2.threshold(image, filter, 255, cv2.THRESH_BINARY)

    white_ctr = 0
    black_ctr = 0
    for x in range(w):
        for y in range(h):
            if bin_image[x][y] == 255:
                white_ctr += 1
            elif bin_image[x][y] == 0:
                black_ctr += 1

    if white_ctr < black_ctr:
        bin_image = cv2.bitwise_not(bin_image)

    kernel = cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))
    dst2 = cv2.morphologyEx(bin_image, cv2.MORPH_CLOSE, kernel, iterations=2)

    dst2 = dst2[0:h - 50, 10:w - 10].copy()

    return dst2
