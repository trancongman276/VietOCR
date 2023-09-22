import os

from predictor import OpenVINOModelWrapper


def parse():
    import argparse

    parser = argparse.ArgumentParser(description="VietOCR on OpenVINO")

    parser.add_argument(
        "--encoder",
        type=str,
        default="./weights/encoder_sim.xml",
        help="path to encoder.xml",
    )
    parser.add_argument(
        "--decoder",
        type=str,
        default="./weights/decoder_sim.xml",
        help="path to decoder.xml",
    )
    parser.add_argument("--beamsize", type=int, default=4, help="beam search width")
    parser.add_argument(
        "--vocabs", type=str, default="./config/vocab.yml", help="path to vocabs.yml"
    )
    parser.add_argument(
        "--input",
        type=str,
        default="./sample",
        help="path to input image or folder of images",
    )

    args = parser.parse_args()
    return args


if __name__ == "__main__":
    args = parse()
    model = OpenVINOModelWrapper(args.encoder, args.decoder, args.beamsize, args.vocabs)

    if os.path.isdir(args.input):
        imgs = []
        for file in os.listdir(args.input):
            if file.endswith(".jpg") or file.endswith(".png"):
                imgs.append(os.path.join(args.input, file))
    else:
        imgs = [args.input]

    preds = model.predict(imgs)
    for img, pred in zip(imgs, preds):
        print(f"{img}: \t{pred}")
