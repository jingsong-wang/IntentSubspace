from __future__ import annotations

import os
import random
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont


class ImageFactory:
    def __init__(self, font_path: str | None = None, font_size: int = 40, seed: int | None = None):
        self.max_font_size = font_size
        self.font_path = font_path
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self.default_font = self._get_font_instance(self.max_font_size)

    def _get_font_instance(self, size: int):
        candidates = []
        if self.font_path:
            candidates.append(self.font_path)
        candidates.extend(
            [
                "arial.ttf",
                "DejaVuSans.ttf",
                "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
                "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
                "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
            ]
        )
        for candidate in candidates:
            try:
                if os.path.exists(candidate) or candidate.endswith(".ttf"):
                    return ImageFont.truetype(candidate, size)
            except OSError:
                continue
        return ImageFont.load_default()

    def _wrap_long_word(self, word: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
        chunks = []
        current = ""
        for char in word:
            candidate = current + char
            bbox = draw.textbbox((0, 0), candidate, font=font)
            if bbox[2] - bbox[0] <= max_width or not current:
                current = candidate
            else:
                chunks.append(current)
                current = char
        if current:
            chunks.append(current)
        return chunks

    def _wrap_text(self, text: str, font, max_width: int, draw: ImageDraw.ImageDraw) -> list[str]:
        final_lines: list[str] = []
        for paragraph in text.split("\n"):
            if not paragraph.strip():
                final_lines.append("")
                continue

            current_line: list[str] = []
            for word in paragraph.split():
                test_line = " ".join(current_line + [word])
                bbox = draw.textbbox((0, 0), test_line, font=font)
                if bbox[2] - bbox[0] <= max_width:
                    current_line.append(word)
                    continue

                if current_line:
                    final_lines.append(" ".join(current_line))
                    current_line = []

                word_bbox = draw.textbbox((0, 0), word, font=font)
                if word_bbox[2] - word_bbox[0] > max_width:
                    final_lines.extend(self._wrap_long_word(word, font, max_width, draw))
                else:
                    current_line = [word]

            if current_line:
                final_lines.append(" ".join(current_line))
        return final_lines

    def _random_color(self, is_text: bool = True) -> tuple[int, int, int]:
        if is_text:
            return (self.rng.randint(0, 150), self.rng.randint(0, 150), self.rng.randint(0, 150))
        return (self.rng.randint(0, 255), self.rng.randint(0, 255), self.rng.randint(0, 255))

    def _add_salt_pepper_noise(self, image: Image.Image, prob: float = 0.01) -> Image.Image:
        img_arr = np.array(image)
        rnd = self.np_rng.random(img_arr.shape[:2])
        img_arr[rnd < prob] = 0
        img_arr[rnd > 1 - prob] = 255
        return Image.fromarray(img_arr)

    def _add_random_lines(self, image: Image.Image, max_lines: int = 8) -> Image.Image:
        draw = ImageDraw.Draw(image)
        width, height = image.size
        for _ in range(self.rng.randint(3, max_lines)):
            draw.line(
                [
                    (self.rng.randint(0, width), self.rng.randint(0, height)),
                    (self.rng.randint(0, width), self.rng.randint(0, height)),
                ],
                fill=self._random_color(is_text=False),
                width=self.rng.randint(1, 3),
            )
        return image

    def create_ocr_image(
        self,
        text: str,
        width: int = 1000,
        height: int = 1000,
        bg_color: tuple[int, int, int] = (255, 255, 255),
        text_color: tuple[int, int, int] = (0, 0, 0),
        perturb: bool = False,
        margin: int = 50,
    ) -> Image.Image:
        image = Image.new("RGB", (width, height), color=bg_color)
        draw = ImageDraw.Draw(image)

        max_width = width - 2 * margin
        max_height = height - 2 * margin
        final_text_color = self._random_color(is_text=True) if perturb else text_color

        best_font = self.default_font
        best_lines: list[str] = []
        best_size = self.max_font_size
        min_size = 12

        for size in range(self.max_font_size, min_size - 1, -2):
            font = self._get_font_instance(size)
            lines = self._wrap_text(text, font, max_width, draw)
            line_spacing = max(4, int(size * 0.25))
            total_text_height = len(lines) * (size + line_spacing)
            if total_text_height <= max_height or size <= min_size + 1:
                best_font = font
                best_lines = lines
                best_size = size
                break

        y = margin
        line_spacing = max(4, int(best_size * 0.25))
        for line in best_lines:
            draw.text((margin, y), line, font=best_font, fill=final_text_color)
            y += best_size + line_spacing

        if perturb:
            image = self._add_random_lines(image)
            image = self._add_salt_pepper_noise(image, prob=0.015)

        return image

    @staticmethod
    def create_blank_image(size: tuple[int, int] = (1000, 1000), color: str = "white") -> Image.Image:
        return Image.new("RGB", size, color=color)

    @staticmethod
    def resize_to_height(image: Image.Image, height: int) -> Image.Image:
        if image.height == height:
            return image
        width = max(1, int(image.width * height / image.height))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    @staticmethod
    def resize_to_width(image: Image.Image, width: int) -> Image.Image:
        if image.width == width:
            return image
        height = max(1, int(image.height * width / image.width))
        return image.resize((width, height), Image.Resampling.LANCZOS)

    @classmethod
    def stitch_images(cls, image_list: list[Image.Image | None], direction: str = "horizontal") -> Image.Image | None:
        valid_images = [img.convert("RGB") for img in image_list if img is not None]
        if not valid_images:
            return None
        if len(valid_images) == 1:
            return valid_images[0]

        if direction == "horizontal":
            target_h = min(img.height for img in valid_images)
            resized = [cls.resize_to_height(img, target_h) for img in valid_images]
            canvas = Image.new("RGB", (sum(img.width for img in resized), target_h), color="white")
            x = 0
            for img in resized:
                canvas.paste(img, (x, 0))
                x += img.width
            return canvas

        if direction == "vertical":
            target_w = min(img.width for img in valid_images)
            resized = [cls.resize_to_width(img, target_w) for img in valid_images]
            canvas = Image.new("RGB", (target_w, sum(img.height for img in resized)), color="white")
            y = 0
            for img in resized:
                canvas.paste(img, (0, y))
                y += img.height
            return canvas

        raise ValueError(f"Unknown stitch direction: {direction}")


def save_image(image: Image.Image, path: Path) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    image.save(path)
    return path.resolve()
