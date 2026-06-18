import cv2
import json
import numpy as np

from database_manager import *
from face_detector import detect_face
from face_encoder import get_embedding


def cosine_similarity(a, b):

    return np.dot(a, b) / (
        np.linalg.norm(a)
        * np.linalg.norm(b)
    )


camera = cv2.VideoCapture(0)

while True:

    ret, frame = camera.read()

    face = detect_face(frame)

    if face is not None:

        current_embedding = get_embedding(
            face
        )

        records = get_all_embeddings()

        best_score = -1
        best_person = None

        for row in records:

            person_id = row[0]
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
                best_person = (
                    name,
                    age
                )

        if best_score > 0.75:

            cv2.putText(
                frame,
                f"{best_person[0]} Age:{best_person[1]}",
                (50, 50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,255,0),
                2
            )

        else:

            cv2.putText(
                frame,
                "Unknown Person",
                (50,50),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0,0,255),
                2
            )

    cv2.imshow(
        "Recognition",
        frame
    )

    if cv2.waitKey(1) == 27:
        break

camera.release()
cv2.destroyAllWindows()