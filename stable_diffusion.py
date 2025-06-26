import asyncio
from functools import lru_cache
import threading
from typing import Any, Literal, Optional
import enum
import base64
from io import BytesIO

import numpy as np
import httpx
from PIL import Image


class Direction(enum.Enum):
    LEFT  = '←'
    RIGHT = '→'
    UP    = '↑'
    DOWN  = '↓'

    @property
    def is_horizontal(self) -> bool:
        return self in (Direction.LEFT, Direction.RIGHT)

    @property
    def x_vector(self) -> int:
        match self:
            case Direction.LEFT:  return -1
            case Direction.RIGHT: return  1
            case others:          return  0

    @property
    def y_vector(self) -> int:
        match self:
            case Direction.UP:    return -1
            case Direction.DOWN:  return  1
            case others:          return  0


Status = Literal['idle', 'generating', 'cancelling'] # TODO: error


def image_to_base64(img:Image.Image) -> str:
    bio = BytesIO()
    img.save(bio, 'png')
    return base64.b64encode(bio.getbuffer()).decode('ascii')

def base64_to_image(b64:str) -> Image.Image:
    data = base64.b64decode(b64)
    return Image.open(BytesIO(data), formats=['png'])

@lru_cache(1)
def generate_mask(mask_width:int, dir:Direction, image_size:int) -> Image.Image:
    data = np.full((image_size, image_size), True, np.bool_)
    match dir:
        case Direction.LEFT:  data[:, -mask_width:] = False
        case Direction.RIGHT: data[:, :mask_width ] = False
        case Direction.UP:    data[-mask_width:, :] = False
        case Direction.DOWN:  data[:mask_width,  :] = False
        case invalid: raise ValueError(invalid)
    return Image.fromarray(data)

def pad_image(img:Image.Image, dir:Direction, image_size:int) -> Image.Image:
    array = np.array(img)
    out = np.zeros_like(array, shape=(image_size, image_size, 3))
    match dir:
        case Direction.LEFT:  out[:, -array.shape[1]:] = array
        case Direction.RIGHT: out[:, :array.shape[1] ] = array
        case Direction.UP:    out[-array.shape[0]:, :] = array
        case Direction.DOWN:  out[:array.shape[0],  :] = array
        case invalid: raise ValueError(invalid)
    return Image.fromarray(out)

def concat_images(original:Image.Image, generated:Image.Image, generate_width:int, dir:Direction) -> Image.Image:
    expansion = generate_width
    output = Image.new(original.mode, (original.width + expansion, original.height) if dir.is_horizontal else (original.width, original.height + expansion))
    match dir:
        case Direction.LEFT:  positions = ((generate_width, 0), (0, 0))
        case Direction.RIGHT: positions = ((0, 0), (original.width + generate_width - generated.width, 0))
        case Direction.UP:    positions = ((0, generate_width), (0, 0))
        case Direction.DOWN:  positions = ((0, 0), (0, original.height + generate_width - generated.height))
    output.paste(original, positions[0])
    output.paste(generated, positions[1])
    return output


mask_blur = 8

class StableDiffusion:
    client:     httpx.AsyncClient
    event_loop: asyncio.AbstractEventLoop

    def __init__(self, base_url:str='http://127.0.0.1:7860/sdapi/v1/', *client_args, **client_kwargs):
        self.client = httpx.AsyncClient(base_url=base_url, *client_args, **client_kwargs)
        self.event_loop = asyncio.get_event_loop()

        def _async_thread():
            "注意: イベントループ内で発生した例外は表示されない！"
            asyncio.set_event_loop(self.event_loop)
            self.event_loop.run_forever()
        event_loop_thread = threading.Thread(target=_async_thread, daemon=True, name='HTTP')
        event_loop_thread.start()
        asyncio.set_event_loop(self.event_loop)


    async def expand_generatively(self, original:Image.Image, generate_width:int, direction:Direction, image_size:int, generate_kwargs:dict[str, Any]) -> Image.Image:
        "Stable Diffusionによる画像の拡張を実行する"
        w, h = original.size
        match direction:
            case Direction.LEFT:  box = (0, 0, image_size-generate_width, h)
            case Direction.RIGHT: box = (w-image_size+generate_width, 0, w, h)
            case Direction.UP:    box = (0, 0, w, image_size-generate_width)
            case Direction.DOWN:  box = (0, h-image_size+generate_width, w, h)
            case invalid: raise ValueError(invalid)
        source = original.crop(box)
        output = await self._generate(source, direction, image_size=image_size, **generate_kwargs)
        return concat_images(original, output, generate_width, direction)


    async def _generate(self, __img:Image.Image, __dir:Direction, mask_blur:int, image_size:int, **kwargs) -> Image.Image:
        response = await self.client.post('img2img', timeout=60*30, json=dict(
            restore_faces=False,
            tiling=False,
            denoising_strength=1,
            init_images=[image_to_base64(pad_image(__img, __dir, image_size))],
            mask=image_to_base64(generate_mask((__img.width if __dir.is_horizontal else __img.height) - mask_blur*2, __dir, image_size)),
            inpainting_fill=0,
            mask_blur=mask_blur,
            width=image_size,
            height=image_size,
        ) | kwargs)
        response.raise_for_status()

        return base64_to_image(response.json()['images'][0])

    async def interrupt_generation(self):
        "APIに生成の中止をリクエストする"
        response = await self.client.post('interrupt')
        response.raise_for_status()

    async def get_generation_progress(self) -> Optional[float]:
        "生成の進行度合い(0-1)を返す。生成中でなければNone"
        response = await self.client.get('progress?skip_current_image=true')
        response.raise_for_status()
        progress = response.json()['progress']
        if progress:
            return progress

    async def get_sampler_or_scheduler_names(self, kind:Literal['samplers', 'schedulers']) -> list[str]:
        response = await self.client.get(kind)
        print(response)
        if response.is_success:
            root = response.json()
            return [ item['name'] for item in root ]
        else:
            return []


DEFAULT_OPTIONS = {
    'prompt': '',
    'negative_prompt': 'blurry, blur, up close',
    'steps': 30,
    'cfg_scale': 7,
    'mask_blur': 8,
    'sampler_name': 'Heun',
    'scheduler': 'Automatic'
}
