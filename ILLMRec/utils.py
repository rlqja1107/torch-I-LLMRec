import os
import logging
from PIL import Image

def is_main():
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        return True if rank ==0 else False
    else:
        return True

def is_gemma_tokenizer(tokenizer):
    return "gemma" in tokenizer.__class__.__name__.lower()


def process_image(image_file, data_args, image_folder=None):
    processor = data_args.image_processor
    if isinstance(image_file, str):
        if image_folder is not None:
            image = Image.open(os.path.join(image_folder, image_file)).convert("RGB")
        else:
            image = Image.open(image_file).convert("RGB")
    else:
        # image is stored in bytearray
        image = image_file
    if data_args.image_aspect_ratio == "resize":
        if hasattr(data_args.image_processor, "crop_size"):
            # CLIP vision tower
            crop_size = data_args.image_processor.crop_size
        else:
            # SIGLIP vision tower
            assert hasattr(data_args.image_processor, "size")
            crop_size = data_args.image_processor.size
        image = image.resize((crop_size["height"], crop_size["width"]))
    if data_args.image_aspect_ratio == "pad":

        def expand2square(pil_img, background_color):
            width, height = pil_img.size
            if width == height:
                return pil_img
            elif width > height:
                result = Image.new(pil_img.mode, (width, width), background_color)
                result.paste(pil_img, (0, (width - height) // 2))
                return result
            else:
                result = Image.new(pil_img.mode, (height, height), background_color)
                result.paste(pil_img, ((height - width) // 2, 0))
                return result

        image = expand2square(image, tuple(int(x * 255) for x in processor.image_mean))
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    else:
        image = processor.preprocess(image, return_tensors="pt")["pixel_values"][0]
    return image


def mlog(logger, *args, **kwargs):
    rank = int(os.environ.get("RANK", 0))
    world_size = int(os.environ.get("WORLD_SIZE", 1))
    if world_size > 1:
        if rank == 0:
            return logger.info(f"[dist-{rank}-of-{world_size}]", *args, **kwargs)
        else:
            return
    else:
        return logger.info(*args, **kwargs)


def setup_logging(log_file, level = logging.INFO, include_host=False, print_log = True):
    os.makedirs(f"{'/'.join(log_file.split('/')[:-1])}", exist_ok=True)
    
    if include_host:
        import socket
        hostname = socket.gethostname()
        formatter = logging.Formatter(
            f'%(asctime)s |  {hostname} | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')
    else:
        formatter = logging.Formatter('%(asctime)s | %(levelname)s | %(message)s', datefmt='%Y-%m-%d,%H:%M:%S')

    logging.root.setLevel(level)
    loggers = [logging.getLogger(name) for name in logging.root.manager.loggerDict]
    logger = logging.getLogger()
    if print_log:
        for logger in loggers:
            logger.setLevel(level)

        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logging.root.addHandler(stream_handler)

    if log_file:
        file_handler = logging.FileHandler(filename=log_file)
        #file_handler.setFormatter(formatter)
        logging.root.addHandler(file_handler)
    return logger
