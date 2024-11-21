from functools import partial
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import torchvision.transforms as transforms
import typer
from natsort import natsorted
from PIL import Image, ImageFile
from tqdm import tqdm


ImageFile.LOAD_TRUNCATED_IMAGES = True


class CenterCropLongEdge(object):
    """
    this code is borrowed from https://github.com/ajbrock/BigGAN-PyTorch
    MIT License
    Copyright (c) 2019 Andy Brock
    """

    def __call__(self, img):
        return transforms.functional.center_crop(img, min(img.size))

    def __repr__(self):
        return self.__class__.__name__


center_crop_trsf = CenterCropLongEdge()


def resize_and_center_crop(image_np, resize_size):
    image_pil = Image.fromarray(image_np)
    image_pil = center_crop_trsf(image_pil)

    if resize_size is not None:
        image_pil = image_pil.resize((resize_size, resize_size), Image.Resampling.LANCZOS)
    return image_pil


def process_image(file_path, input_dir, output_dir, resize_size, suffix):
    relative_path = file_path.relative_to(input_dir)
    output_path = output_dir / relative_path
    output_path = output_path.with_suffix(suffix)
    output_path.parent.mkdir(exist_ok=True, parents=True)
    image_np = np.array(Image.open(file_path.as_posix()))
    if image_np.shape[-1] == 1:
        image_np = np.tile(image_np, (1, 1, 3))
    processed_image = resize_and_center_crop(image_np, resize_size)

    if processed_image.mode in ("RGBA", "P"):
        processed_image = processed_image.convert("RGB")

    processed_image.save(output_path.as_posix())


def compress_(file_path, input_dir, output_dir):
    relative_path = file_path.relative_to(input_dir)
    output_path = output_dir / relative_path
    output_path = output_path.with_suffix(".jpg")
    output_path.parent.mkdir(exist_ok=True, parents=True)
    image = Image.open(file_path.as_posix())
    image.save(output_path.as_posix())


app = typer.Typer(pretty_exceptions_show_locals=False)


@app.command()
def main(
    input_dir: Path = typer.Argument(..., help="inputdir", dir_okay=True, exists=True),
    output_dir: Path = typer.Argument(..., help="outputdir", dir_okay=True),
    resize_size: int = typer.Option(256, help="img sz"),
    num_processes: int = typer.Option(16, help="num_processes"),
    nsamples: int = typer.Option(-1, help="nsamples"),
    suffix: str = typer.Option(".jpg", help="image output"),
):
    if nsamples < 0:
        # files = natsorted(list(input_dir.rglob("**/*.[JPEG][jpg][png]")))
        files = natsorted(list(input_dir.rglob("*.JPEG")))
    else:
        files = natsorted(list(input_dir.rglob("*.JPEG")))[:nsamples]

    process_partial = partial(
        process_image,
        input_dir=input_dir,
        output_dir=output_dir,
        resize_size=resize_size,
        suffix=suffix,
    )

    with Pool(num_processes) as p:
        list(tqdm(p.imap(process_partial, files), total=len(files)))


@app.command()
def compress(
    input_dir: Path = typer.Argument(..., help="inputdir", dir_okay=True, exists=True),
    output_dir: Path = typer.Argument(..., help="outputdir", dir_okay=True),
    num_processes: int = typer.Option(32, help="num_processes"),
    nsamples: int = typer.Option(30000, help="nsamples"),
):
    if nsamples < 0:
        files = natsorted(list(input_dir.rglob("*.[jp][pn]g")))
    else:
        files = natsorted(list(input_dir.rglob("*.[jp][pn]g")))[:nsamples]
    process_partial = partial(
        compress_,
        input_dir=input_dir,
        output_dir=output_dir,
    )

    with Pool(num_processes) as p:
        list(tqdm(p.imap(process_partial, files), total=len(files)))


if __name__ == "__main__":
    app()
