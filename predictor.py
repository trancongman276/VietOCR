from collections import defaultdict

import numpy as np
import yaml
from openvino.runtime import Core
from PIL import Image

from model.beam import Beam
from model.utils import log_softmax_batch
from model.vocab import Vocab
from tool.translate import process_input


class OpenVINOModelWrapper:
    """OpenVINO Model Wrapper for VietOCR"""

    def __init__(self, encoder_path, decoder_path, beam_size, vocab_path) -> None:
        ie = Core()
        ie.set_property({"CACHE_DIR": "../cache"})

        # Load Encoder and Decoder
        encoder = ie.read_model(encoder_path)
        self.encoder = ie.compile_model(model=encoder)

        decoder = ie.read_model(decoder_path)
        self.decoder = ie.compile_model(model=decoder)

        # Load Vocab
        with open(vocab_path, encoding="utf8") as f:
            config = yaml.load(f, Loader=yaml.FullLoader)
        self.vocab = Vocab(config["vocab"])
        self.beam_size = beam_size

    def predict(self, imgs: list):
        """Predict text from images

        Args:
            imgs (list): List of images or paths to images

        Raises:
            TypeError: Images must be numpy array or PIL image or path

        Returns:
            list: List of predicted texts
        """
        # Using bucket to speed up inference by gather images with same width into a batch
        bucket = defaultdict(list)
        bucket_idx = defaultdict(list)
        bucket_pred = {}

        preds = [0] * len(imgs)

        # Preprocess images
        for i, img in enumerate(imgs):
            if isinstance(img, np.ndarray):
                img = Image.fromarray(img)
            elif isinstance(img, str):
                img = Image.open(img)
            elif isinstance(img, Image.Image):
                pass
            else:
                raise TypeError("Images must be numpy array or PIL image or path")
            # Default preprocess
            img = process_input(
                img, image_height=32, image_min_width=32, image_max_width=512
            )
            img = np.array(img, dtype=np.float32)
            bucket[img.shape[-1]].append(img)
            bucket_idx[img.shape[-1]].append(i)

        # Predict
        for k, batch in bucket.items():
            batch = np.concatenate(batch, axis=0)
            memories = self.forward(batch)
            # return memories
            _preds = []
            for i in range(batch.shape[0]):
                memory = self.get_memory(memories, i)
                sent = self.search(memory)
                _preds.append(sent)
            bucket_pred[k] = self.vocab.batch_decode(_preds)

        # Reorder
        for k in bucket_pred:
            idx = bucket_idx[k]
            pred = bucket_pred[k]
            for i, j in enumerate(idx):
                preds[j] = pred[i]

        return preds

    def get_memory(self, memory: np.ndarray, i: int):
        """Get memory of a specific image in batch

        Args:
            memory (np.ndarray): Batched memory
            i (int): Index of image in batch

        Returns:
            np.ndarray: Memory of image
        """
        return memory[:, i * 4 : (i + 1) * 4, :]

    def forward(self, img: np.ndarray) -> np.ndarray:
        """Forward image through encoder

        Args:
            img (np.ndarray): Image

        Returns:
            np.ndarray: Memory
        """
        return self.encoder((img, np.array(self.beam_size).astype(np.int64)))[
            self.encoder.outputs[0]
        ]

    def search(self, memory: np.ndarray) -> list:
        """Search for text in image using beam search

        Args:
            memory (np.ndarray): Memory of image

        Returns:
            list: List of predicted text
        """
        beam = Beam(
            beam_size=self.beam_size,
            min_length=0,
            n_top=1,
            ranker=None,
            start_token_id=1,
            end_token_id=2,
        )
        for _ in range(128):
            tgt_inp_np = beam.get_current_state().T
            decoder_outputs, memory = self.decoder((tgt_inp_np, memory)).values()

            log_prob = log_softmax_batch(decoder_outputs[:, -1, :])
            beam.advance(log_prob)

            if beam.done():
                break

        _, ks = beam.sort_finished(minimum=1)

        hypothesises = []
        for _, (times, k) in enumerate(ks[:1]):
            hypothesis = beam.get_hypothesis(times, k)
            hypothesises.append(hypothesis)

        return [1] + [int(i) for i in hypothesises[0][:-1]]


if __name__ == "__main__":
    model = OpenVINOModelWrapper(
        encoder_path="./weights/int8/encoder.xml",
        decoder_path="./weights/int8/decoder.xml",
        beam_size=4,
        vocab_path="./config/vocab.yml",
    )
    images = ["./sample/sample2.png", "./sample/sample1.png"]
    # Calculate time
    import time

    start = time.time()
    preds = model.predict(images)
    print(f"Time (int8 model): {time.time() - start}")
    print(f"Predictions: {preds}")

    model = OpenVINOModelWrapper(
        encoder_path="./weights/encoder_sim.xml",
        decoder_path="./weights/decoder_sim.xml",
        beam_size=4,
        vocab_path="./config/vocab.yml",
    )
    start = time.time()
    preds = model.predict(images)
    print(f"Time (float32 model): {time.time() - start}")
    print(f"Predictions: {preds}")
