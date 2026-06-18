import cv2
import json
import numpy as np

from database_manager import (
    create_database,
    add_person,
    save_embedding,
    get_all_embeddings
)

from face_detector import detect_face
from face_encoder import get_embedding

create_database()


def cosine_similarity(a, b):
    return np.dot(a, b) / (
        np.linalg.norm(a) * np.linalg.norm(b)
    )


camera = cv2.VideoCapture(0)

print("\nFace Recognition Started...")
print("Press:")
print("  r = Register New Face")
print("  q = Quit\n")

while True:

    ret, frame = camera.read()

    if not ret:
        break

    face = detect_face(frame)

    if face is not None:

        current_embedding = get_embedding(face)

        records = get_all_embeddings()

        best_score = -1
        best_person = None

        for row in records:

            name = row[1]
            age = row[2]

            stored_embedding = np.array(
                json.loads(row[3])
            )

            score = cosine_similarity(
                current_embedding,
                stored_embedding
            )

            if score > best_score:
                best_score = score
                best_person = (name, age)

        if best_score > 0.75:

            cv2.putText(
                frame,
                f"{best_person[0]} ({best_person[1]})",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2
            )

        else:

            cv2.putText(
                frame,
                "Unknown Person",
                (30, 40),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 0, 255),
                2
            )

    cv2.imshow("Face Recognition System", frame)

    key = cv2.waitKey(1)

    # Register New Person
    if key == ord("r"):

        if face is None:
            print("No face detected")
            continue

        name = input("Enter Name: ")
        age = int(input("Enter Age: "))

        person_id = add_person(name, age)

        embedding = get_embedding(face)

        save_embedding(
            person_id,
            embedding
        )

        print(f"{name} Registered Successfully")

    # Quit
    if key == ord("q"):
        break


camera.release()
cv2.destroyAllWindows()