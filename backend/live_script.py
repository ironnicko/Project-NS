import cv2
import numpy as np
from matplotlib import pyplot as plt
from keras import backend as K
from script import connectedComponents, model, input_image, SIZE_OF_GRID

cap = cv2.VideoCapture(0)

while True:

    _, frame = cap.read()

    resized_frame = input_image(frame)

    segmented = model.predict(
        resized_frame[int(resized_frame.shape[0]*0.9):])

    seg = np.squeeze((segmented > 0.5).astype(np.uint8))

    visited = [[0 for _ in range(SIZE_OF_GRID)]
               for __ in range(SIZE_OF_GRID)]

    _2Darray = seg

    nuclei_count, adj_nuclei_count = connectedComponents()

    print(f"Adjusted Nuclei Count: {adj_nuclei_count}")

    font = cv2.FONT_HERSHEY_SIMPLEX

    org = [50, 50]

    fontScale = 0.5

    color = (255, 255, 255)

    thickness = 2

    item = f'Adjusted Nuclei Count: {adj_nuclei_count}'

    frame = cv2.putText(frame, item, org, font,
                        fontScale, color, thickness, cv2.LINE_AA)

    item = "Press 'F' to freeze and graph"

    org[1] += 100

    frame1 = cv2.putText(frame, item, org, font, fontScale,
                         color, thickness, cv2.LINE_AA)

    cv2.imshow("Window", frame1)

    key = cv2.waitKey(1)
    if key == (ord('q')):
        break

    elif key == ord('f'):
        cv2.imshow(seg)
        plt.show()
        plt.savefig("test.png")


cv2.destroyAllWindows()
cap.release()
