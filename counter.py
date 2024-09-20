import cv2
import datetime

video_src = 'data/output_en_pers_greyscale_20230319-141825.avi'
cap = cv2.VideoCapture(video_src)
fgbg = cv2.createBackgroundSubtractorMOG2()
in_count = 1
out_count = 0
prev_direction = None
tracked_person = None
current_count = 1

while True:
    # Initialize video writer for output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('telle_status_2703.mp4',fourcc, 30.0, (2000,2000))
    
    ret, frame = cap.read()

    if not ret:
        break

    fgmask = fgbg.apply(frame)
    fgmask = cv2.erode(fgmask, None, iterations=15)
    fgmask = cv2.dilate(fgmask, None, iterations=15)

    contours, hierarchy = cv2.findContours(fgmask.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    direction = None

    for contour in contours:
        if cv2.contourArea(contour) < 85000:
            continue

        (x, y, w, h) = cv2.boundingRect(contour)
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

        # Check if tracked person is still in the same location
        if tracked_person is not None and (x, y, w, h) == tracked_person:
            continue

        # Check if person is going in or out
        if prev_direction is None:
            prev_direction = "in" if x < frame.shape[1] // 2 else "out"
        elif x < frame.shape[1] // 2:
            direction = "in"
        else:
            direction = "out"

        # Check if contour has minimum number of consecutive white pixels
        if direction is not None:
            if direction == "in":
                if prev_direction != "in":
                    if cv2.countNonZero(fgmask[y:y+h, x:x+w]) >= 10:
                        in_count += 1
                        current_count += 1
                        print("Person entered the room at {}.".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        tracked_person = (x, y, w, h)
            elif direction == "out":
                if prev_direction != "out":
                    if cv2.countNonZero(fgmask[y:y+h, x:x+w]) >= 10:
                        out_count += 1
                        current_count -= 1
                        print("Person left the room at {}.".format(datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")))
                        tracked_person = (x, y, w, h)
            if current_count == 0:
                cv2.putText(frame, "No one in the room, turning of the heat and lights", (10, 220), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 0, 255), 2)
                cv2.waitKey(250) # wait for 3 seconds


            prev_direction = direction
    cv2.putText(frame, "In: {}".format(str(in_count)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    cv2.putText(frame, "Out: {}".format(str(out_count)), (10, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    cv2.putText(frame, "Number of people currently in the room: {}".format(str(current_count)), (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 0, 0), 2)
    cv2.imshow("Frame", frame)
    cv2.imshow("FG Mask", fgmask)
    cv2.waitKey(60)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
