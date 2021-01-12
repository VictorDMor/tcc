import io
from .classifier import identify_event
from PIL import Image

def handle_uploaded_file(f, name):
    image = Image.open(io.BytesIO(bytearray(f.read())))
    image_path = 'scout_generator/static/scout_generator/img/{}'.format(name)
    image.save(image_path)
    event = identify_event(image, name)
    return {
        'image_path': image_path,
        'image_name': name,
        'event_name': event
    }
        