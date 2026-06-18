import cv2

from database_manager import *
from face_detector import detect_face
from face_encoder import get_embedding

create_database()

name = input("Enter Name: ")
age = int(input("Enter Age: "))

camera = cv2.VideoCapture(0)

while True:

    ret, frame = camera.read()

    cv2.imshow(
        "Press S To Save Face",
        frame
    )

    key = cv2.waitKey(1)

    if key == ord("s"):

        face = detect_face(frame)

        if face is None:
            print("No Face Found")
            continue

        embedding = get_embedding(face)

        person_id = add_person(
            name,
            age
        )

        save_embedding(
            person_id,
            embedding
        )

        print("Saved Successfully")

        break

camera.release()
cv2.destroyAllWindows()