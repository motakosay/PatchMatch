import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

def cal_distance(a, b, A_padding, B, p_size):
    p = p_size // 2
    patch_a = A_padding[a[0]:a[0]+p_size, a[1]:a[1]+p_size, :]
    patch_b = B[b[0]-p:b[0]+p+1, b[1]-p:b[1]+p+1, :]
    temp = patch_b - patch_a
    num = np.sum(1 - np.int32(np.isnan(temp)))
    dist = np.sum(np.square(np.nan_to_num(temp))) / num
    return dist

def reconstruction(f, A, B):
    A_h, A_w = A.shape[:2]
    temp = np.zeros_like(A)

    for i in range(A_h):
        for j in range(A_w):
            if f[i, j][0] < B.shape[0] and f[i, j][1] < B.shape[1]:  # Ensure valid index
                temp[i, j, :] = B[f[i, j][0], f[i, j][1], :]

    return temp  # Return reconstructed image instead of showing it directly

def initialization(A, B, p_size):
    A_h, A_w = A.shape[:2]
    B_h, B_w = B.shape[:2]
    p = p_size // 2

    random_B_r = np.random.randint(p, B_h-p, [A_h, A_w])
    random_B_c = np.random.randint(p, B_w-p, [A_h, A_w])

    A_padding = np.ones([A_h+p*2, A_w+p*2, 3]) * np.nan
    A_padding[p:A_h+p, p:A_w+p, :] = A

    f = np.zeros([A_h, A_w], dtype=object)
    dist = np.zeros([A_h, A_w])

    for i in range(A_h):
        for j in range(A_w):
            a = np.array([i, j])
            b = np.array([random_B_r[i, j], random_B_c[i, j]], dtype=np.int32)
            f[i, j] = b
            dist[i, j] = cal_distance(a, b, A_padding, B, p_size)

    return f, dist, A_padding

def NNS(img, ref, p_size, itr):
    A_h, A_w = img.shape[:2]
    f, dist, img_padding = initialization(img, ref, p_size)

    for iteration in range(1, itr+1):
        for i in range(A_h):
            for j in range(A_w):
                a = np.array([i, j])
                if iteration % 2 == 1:  # Odd iteration
                    propagation(f, a, dist, img_padding, ref, p_size, True)
                else:  # Even iteration
                    propagation(f, a, dist, img_padding, ref, p_size, False)
                random_search(f, a, dist, img_padding, ref, p_size)
        
        print(f"Iteration: {iteration}")

    return f

if __name__ == "__main__":
    img = np.array(Image.open("./cup_a.jpg").convert("RGB"))  # Ensure RGB mode
    ref = np.array(Image.open("./cup_b.jpg").convert("RGB"))

    p_size = 3
    itr = 5

    import time
    start = time.time()
    f = NNS(img, ref, p_size, itr)
    end = time.time()
    
    print(f"Time taken: {end - start:.2f} seconds")
    
    result = reconstruction(f, img, ref)  # Now returns an array
    plt.imshow(result.astype(np.uint8))
    plt.axis("off")
    plt.show()
