import cv2
import sys
import matplotlib.pyplot as plt

video = cv2.VideoCapture(0)

if video.isOpened() == False:
    print("Error reading video file")


stats = {}

while True:
    ret, frame = video.read()


    # for index, person in enumerate(result):

    #     if not stats.__contains__(index):
    #         stats[index] = []
    #     stats[index].append(person['emotions'])
    #     bounding_box = person['box']
    #     cv2.rectangle(frame, (bounding_box[0], bounding_box[1]), (bounding_box[0] + bounding_box[2], bounding_box[1] + bounding_box[3]), (255, 0, 0), 4)

    #     max_value = 0
    #     for value in person['emotions'].values():
    #         if value > max_value:
    #             max_value = value

    #     cv2.putText(frame, f"Person {index}", (bounding_box[0], bounding_box[1] - TEXT_SPACING), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)
    #     for i, (emotion, value) in enumerate(person['emotions'].items()):
    #         if value == max_value:
    #             color = (0, 255, 0)
    #         else:
    #             color = (0, 0, 255)
    #         cv2.putText(frame, f"{emotion}: {value}", (bounding_box[0], bounding_box[1] + bounding_box[3] + TEXT_SPACING * (i + 1)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1, cv2.LINE_AA)

    if ret == True:
        cv2.imshow('Press q to close', frame)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    else:
        break

video.release()
cv2.destroyAllWindows()

