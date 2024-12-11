'''checking if the ip web cam runs properly or not with the system'''

import cv2

# Set up IP Webcam video stream URL
ip_webcam_url = 'http://<ur ip address of ip webcam>/video'  # Replace with your IP webcam URL

# Capture video from IP webcam
cap = cv2.VideoCapture(ip_webcam_url)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    # Display the live feed (optional)
    cv2.imshow("IP Webcam", frame)

    # Break the loop when 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
