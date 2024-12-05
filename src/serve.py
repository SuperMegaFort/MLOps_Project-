from __future__ import annotations
from bentoml.validators import ContentType
from typing import Annotated
from PIL.Image import Image
from pydantic import Field
import service
import json


@service.service(name="celestial_bodies_classifier")
class CelestialBodiesClassifierService:
    bento_model = service.keras.get("celestial_bodies_classifier_model")

    def __init__(self) -> None:
        self.preprocess = self.bento_model.custom_objects["preprocess"]
        self.postprocess = self.bento_model.custom_objects["postprocess"]
        self.model = self.bento_model.load_model()

    @service.api()
    def predict(
            self,
            image: Annotated[Image, ContentType("image/jpeg")] = Field(description="Planet image to analyze"),
    ) -> Annotated[str, ContentType("application/json")]:
        image = self.preprocess(image)

        predictions = self.model.predict(image)

        return json.dumps(self.postprocess(predictions))