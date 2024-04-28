from __future__ import annotations

import os
import reprlib
from typing import TYPE_CHECKING, Any, Dict, List, Optional

from google.api_core.client_options import ClientOptions
from google.cloud import speech_v2 as google_speech
from google.protobuf import field_mask_pb2
from langchain_core._api.deprecation import deprecated
from langchain_core.documents import Document
from langchain_community.document_loaders.base import BaseLoader
from langchain_community.utilities.vertexai import get_client_info

if TYPE_CHECKING:
    from google.cloud.speech_v2 import (
        AutoDetectDecodingConfig,
        RecognitionConfig,
        RecognitionFeatures,
        RecognizeRequest,
        SpeechClient,
        SpeechRecognitionAlternative,
        SpeechRecognitionResult,
    )
    from google.protobuf.field_mask_pb2 import FieldMask

class GoogleSpeechToTextLoader(BaseLoader):
    """
    Loader for Google Cloud Speech-to-Text audio transcripts.

    It uses the Google Cloud Speech-to-Text API to transcribe audio files
    and loads the transcribed text into one or more Documents,
    depending on the specified format.

    To use, you should have the ``google-cloud-speech`` python package installed.

    Audio files can be specified via a Google Cloud Storage uri or a local file path.

    For a detailed explanation of Google Cloud Speech-to-Text, refer to the product
    documentation.
    https://cloud.google.com/speech-to-text
    """

    __version__ = "0.0.32"

    def __init__(
        self,
        project_id: str,
        file_path: str,
        location: str = "us-central1",
        recognizer_id: str = "_",
        config: Optional[RecognitionConfig] = None,
        config_mask: Optional[FieldMask] = None,
    ):
        """
        Initializes the GoogleSpeechToTextLoader.

        Args:
            project_id: Google Cloud Project ID.
            file_path: A Google Cloud Storage URI or a local file path.
            location: Speech-to-Text recognizer location.
            recognizer_id: Speech-to-Text recognizer id.
            config: Recognition options and features.
                For more information:
                https://cloud.google.com/python/docs/reference/speech/latest/google.cloud.speech_v2.types.RecognitionConfig
            config_mask: The list of fields in config that override the values in the
                ``default_recognition_config`` of the recognizer during this
                recognition request.
                For more information:
                https://cloud.google.com/python/docs/reference/speech/latest/google.cloud.speech_v2.types.RecognizeRequest
        """
        self.project_id = project_id
        self.file_path = file_path
        self.location = location
        self.recognizer_id = recognizer_id
        self.config = config or RecognitionConfig(
            auto_decoding_config=AutoDetectDecodingConfig(),
            language_codes=["en-US"],
            model="chirp",
            features=RecognitionFeatures(
                # Automatic punctuation could be useful for language applications
                enable_automatic_punctuation=True,
            ),
        )
        self.config_mask = config_mask

        self._client = SpeechClient(
            client_info=get_client_info(module="speech-to-text"),
            client_options=(
                ClientOptions(api_endpoint=f"{location}-speech.googleapis.com")
                if location != "global"
                else None
            ),
        )
        self._recognizer_path = self._client.recognizer_path(
            project_id, location, recognizer_id
        )

    def __repr__(self) -> str:
        """
        Returns a string representation of the class.
        """
        return (
            f"{self.__class__.__name__}("
            f"project_id={self.project_id!r}, "
            f"file_path={self.file_path!r}, "
            f"location={self.location!r}, "
            f"recognizer_id={self.recognizer_id!r}, "
            f"config={self.config!r}, "
            f"config_mask={self.config_mask!r})"
        )

    def _validate_inputs(self) -> None:
        """
        Validates the inputs.
        """
        if not isinstance(self.project_id, str):
            raise ValueError("`project_id` must be a string.")

        if not isinstance(self.file_path, str):
            raise ValueError("`file_path` must be a string.")

        if not isinstance(self.location, str):
            raise ValueError("`location` must be a string.")

        if not isinstance(self.recognizer_id, str):
            raise ValueError("`recognizer_id` must be a string.")

        if self.config is not None and not isinstance(self.config, RecognitionConfig):
            raise ValueError("`config` must be a RecognitionConfig object.")

        if self.config_mask is not None and not isinstance(self.config_mask, FieldMask):
            raise ValueError("`config_mask` must be a FieldMask object.")

    def _transcribe_audio(self) -> List[SpeechRecognitionResult]:
        """
        Transcribes the audio file.

        Returns:
            A list of SpeechRecognitionResult objects.
        """
        request = RecognizeRequest(
            recognizer=self._recognizer_path,
            config=self.config,
            config_mask=self.config_mask,
        )

        if "gs://" in self.file_path:
            request.uri = self.file_path
        else:
            with open(self.file_path, "rb") as f:
                request.content = f.read()

        response = self._client.recognize(request=request)

        return response.results

    def _parse_response(self, results: List[SpeechRecognitionResult]) -> List[Document]:
        """
        Parses the response from the API.

        Args:
            results: A list of SpeechRecognitionResult objects.

        Returns:
            A list of Document objects.
        """
        documents = []

        for result in results:
            alternatives = result.alternatives
            if not alternatives:
                continue

            document = Document(
                page_content=alternatives[0].transcript,
                metadata={
                    "language_code": result.language_code,
                    "result_end_offset": result.result_end_offset,
                },
            )

            documents.append(document)

        return documents

    def load(self) -> List[Document]:
        """
        Transcribes the audio file and loads the transcript into documents.

        It uses the Google Cloud Speech-to-Text API to transcribe the audio file
        and blocks until the transcription is finished.

        Returns:
            A list of Document objects.
        """
        self._validate_inputs()

        results = self._transcribe_audio()

        return self._parse_response(results)
