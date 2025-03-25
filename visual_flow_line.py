from GUI import GUI
from HAL import HAL
import cv2
import numpy as np

Kp = 0.005
Ki = 0.0001
Kd = 0.0001

# previous error and error some
prev_err = 0
err_sum = 0

# color detection range
lower_red = np.array([0, 100, 100])
upper_red = np.array([10, 255, 255])

while True:
    frame = HAL.getImage()
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, lower_red, upper_red)

    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if contours:
        cnt = max(contours, key=cv2.contourArea)
        M = cv2.moments(cnt)

        if M["m00"] != 0:
            # position variables
            x = int(M["m10"] / M["m00"])
            y = int(M["m01"] / M["m00"])

            # error calculation
            center = frame.shape[1] // 2
            err = x - center

            # PID adjustments
            P = Kp * err
            err_sum += err
            I = Ki * err_sum
            D = Kd * (err - prev_err)
            prev_err = err

            ang_vel = -(P + I + D)

            HAL.setV(1)
            HAL.setW(ang_vel)

        cv2.drawContours(frame, [cnt], -1, (0, 255, 0), 2)
        cv2.circle(frame, (x, y), 5, (0, 255, 0), -1)

    GUI.showImage(frame)