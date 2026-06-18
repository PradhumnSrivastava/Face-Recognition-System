import torch
import numpy as np

from PIL import Image
from facenet_pytorch import InceptionResnetV1

model = InceptionResnetV1(
    pretrained="vggface2"
).eval()


def get_embedding(face_image):

    face_image = cv2.cvtColor(
        face_image,
        cv2.COLOR_BGR2RGB
    )

    image = Image.fromarray(face_image)

    image = image.resize((160, 160))

    image = np.array(image)

    image = torch.tensor(
        image
    ).permute(2, 0, 1).float()

    image = image.unsqueeze(0)

    image = image / 255.0

    with torch.no_grad():
        embedding = model(image)

    return embedding[0].numpy()