from __future__ import annotations

from pathlib import Path

from PIL import Image, ImageFilter


class WelcomeBackgrounds:
    @staticmethod
    def hex_to_rgb(color: str) -> tuple[int, int, int]:
        color = color.strip("#")
        return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

    @classmethod
    def mix_color(cls, c0: str, c1: str, t: float) -> str:
        t = max(0.0, min(1.0, t))
        r0, g0, b0 = cls.hex_to_rgb(c0)
        r1, g1, b1 = cls.hex_to_rgb(c1)
        r = int(r0 + (r1 - r0) * t)
        g = int(g0 + (g1 - g0) * t)
        b = int(b0 + (b1 - b0) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    @classmethod
    def create_gradient_image(cls, width: int, height: int, c0: str, c1: str) -> Image.Image:
        small_img = Image.new("RGB", (2, 2))
        p0 = cls.hex_to_rgb(c0)
        p1 = cls.hex_to_rgb(c1)
        small_img.putpixel((0, 0), p0)
        small_img.putpixel((1, 1), p1)
        small_img.putpixel((0, 1), cls.hex_to_rgb(cls.mix_color(c0, c1, 0.5)))
        small_img.putpixel((1, 0), cls.hex_to_rgb(cls.mix_color(c0, c1, 0.5)))
        return small_img.resize((width, height), Image.Resampling.BILINEAR)

    @staticmethod
    def resize_cover(image: Image.Image, width: int, height: int) -> Image.Image:
        width = max(1, int(width))
        height = max(1, int(height))
        iw, ih = image.size
        scale = max(width / iw, height / ih)
        new_w = max(1, int(iw * scale))
        new_h = max(1, int(ih * scale))
        resized = image.resize((new_w, new_h), Image.Resampling.LANCZOS)
        left = (new_w - width) // 2
        top = (new_h - height) // 2
        return resized.crop((left, top, left + width, top + height))

    @classmethod
    def load_images(cls, bg_dir: Path) -> list[Image.Image]:
        supported = {".png", ".jpg", ".jpeg", ".bmp", ".webp"}
        images: list[Image.Image] = []

        if bg_dir.exists() and bg_dir.is_dir():
            for path in sorted(bg_dir.iterdir()):
                if path.suffix.lower() not in supported:
                    continue
                try:
                    images.append(Image.open(path).convert("RGB"))
                except Exception:
                    continue

        if images:
            return [images[0]]

        palette = [
            ("#0b1020", "#133457"),
            ("#10131c", "#2a3450"),
            ("#111826", "#214866"),
        ]
        return [
            cls.create_gradient_image(1600, 900, c0, c1).filter(ImageFilter.GaussianBlur(radius=4))
            for c0, c1 in palette
        ]
