from __future__ import annotations

from collections.abc import Callable
from pathlib import Path
import tkinter as tk
from tkinter import ttk

from PIL import Image, ImageFilter, ImageTk, ImageDraw

from modules.base_module import FeatureModule


class WelcomeScreen:
    """Animated welcome screen with crossfade background and frameless splash-like UI."""

    # Color constants for animations
    TITLE_COLOR_START = "#505862"
    TITLE_COLOR_END = "#f2f5ff"
    TITLE_STROKE_START = "#080b10"
    TITLE_STROKE_END = "#070a0f"
    DESC_COLOR_START = "#4d5560"
    DESC_COLOR_END = "#c7ced8"
    DESC_STROKE_START = "#05070a"
    DESC_STROKE_END = "#111722"

    # Offset constants for stroke effects
    TITLE_STROKE_OFFSETS = [(-2, 0), (2, 0), (0, -2), (0, 2)]
    DESC_STROKE_OFFSETS = [(-1, 0), (1, 0), (0, -1), (0, 1)]

    def __init__(
        self,
        parent: ttk.Frame,
        on_open_module: Callable[[int], None],
        on_request_close: Callable[[], None] | None = None,
    ) -> None:
        self.parent = parent
        self.on_open_module = on_open_module
        self.on_request_close = on_request_close

        self.frame = tk.Frame(parent, bg="#0b0f14", bd=0, highlightthickness=0)
        self.frame.pack(fill=tk.BOTH, expand=True)

        self.root = self.frame.winfo_toplevel()
        self.is_frameless = False

        self.use_overlay_mask = True
        self.overlay_alpha_top = 172
        self.overlay_alpha_bottom = 124
        self.switch_interval_ms = 2000  # image pause duration in milliseconds
        self.transition_ms = 250        # transition duration in milliseconds
        self.transition_fps = 60        # frame rate for smooth rendering

        self.bg_canvas: tk.Canvas
        self.canvas_image_id: int | None = None
        self.current_photo: ImageTk.PhotoImage | None = None

        self.title_stroke_ids: list[int] = []
        self.title_main_id: int | None = None
        self.desc_stroke_ids: list[int] = []
        self.desc_text_id: int | None = None

        self.close_button: tk.Label | None = None

        self.background_images = self.load_background_images()
        self.current_index = 0
        self.next_index = 1 if len(self.background_images) > 1 else 0

        self.current_resized: Image.Image | None = None
        self.next_resized: Image.Image | None = None
        self.in_transition = False
        self.transition_step = 0
        self.transition_total_steps = max(1, int(self.transition_ms / (1000 / self.transition_fps)))
        self.transition_after_id: str | None = None
        self.switch_after_id: str | None = None

        self.last_w = 1
        self.last_h = 1

        self.title_base_y = 0
        self.desc_base_y = 0

        self.title_after_id: str | None = None
        self.desc_after_id: str | None = None
        self.entry_played = False

        self.drag_start_x = 0
        self.drag_start_y = 0
        self.window_start_x = 0
        self.window_start_y = 0

        self.card_widgets: list[tk.Label] = []

        self.build_ui()

    def build_ui(self) -> None:
        self.bg_canvas = tk.Canvas(self.frame, bg="#0b0f14", highlightthickness=0, bd=0)
        self.bg_canvas.place(relx=0, rely=0, relwidth=1, relheight=1)

        title_text = "Ptychography Toolkit"
        title_font = ("Segoe UI Black", 40)
        for _ in self.TITLE_STROKE_OFFSETS:
            stroke_id = self.bg_canvas.create_text(
                0,
                0,
                text=title_text,
                fill=self.TITLE_STROKE_START,
                font=title_font,
                justify="center",
                anchor="n",
                width=700,
            )
            self.title_stroke_ids.append(stroke_id)

        self.title_main_id = self.bg_canvas.create_text(
            0,
            0,
            text=title_text,
            fill=self.TITLE_COLOR_START,
            font=title_font,
            justify="center",
            anchor="n",
            width=700,
        )

        desc_text = "Portable ptychography utilities with streamlined visual workflows."
        desc_font = ("Segoe UI", 20, "italic")
        for _ in self.DESC_STROKE_OFFSETS:
            desc_stroke_id = self.bg_canvas.create_text(
                0,
                0,
                text=desc_text,
                fill=self.DESC_STROKE_START,
                font=desc_font,
                justify="center",
                anchor="n",
                width=700,
            )
            self.desc_stroke_ids.append(desc_stroke_id)

        self.desc_text_id = self.bg_canvas.create_text(
            0,
            0,
            text=desc_text,
            fill=self.DESC_COLOR_START,
            font=desc_font,
            justify="center",
            anchor="n",
            width=700,
        )

        self.close_button = tk.Label(
            self.frame,
            text="✕",
            bg="#2a3039",
            fg="#f3f6ff",
            font=("Segoe UI", 12, "bold"),
            width=3,
            cursor="hand2",
            bd=0,
            highlightthickness=0,
        )
        close_btn = self.close_button
        close_btn.place(relx=0.985, rely=0.02, anchor="ne")
        close_btn.bind("<Enter>", lambda _: close_btn.configure(bg="#3f4856"))
        close_btn.bind("<Leave>", lambda _: close_btn.configure(bg="#2a3039"))
        close_btn.bind("<ButtonRelease-1>", lambda _: self.request_close())

        self.bg_canvas.bind("<ButtonPress-1>", self.on_drag_start)
        self.bg_canvas.bind("<B1-Motion>", self.on_drag_motion)

        self.frame.bind("<Configure>", self.on_resize)

        self.enable_frameless_mode()
        self.render_background(0.0)
        self.schedule_next_switch()
        self.play_entry_animation()

    def enable_frameless_mode(self) -> None:
        if self.is_frameless:
            return
        self.is_frameless = True
        try:
            self.root.overrideredirect(True)
        except tk.TclError:
            return
        self.root.bind("<Escape>", self.on_escape_close)

    def disable_frameless_mode(self) -> None:
        if not self.is_frameless:
            return
        self.is_frameless = False
        try:
            self.root.overrideredirect(False)
        except tk.TclError:
            return
        self.root.unbind("<Escape>")

    def on_escape_close(self, _: tk.Event) -> None:
        self.request_close()

    def request_close(self) -> None:
        if self.on_request_close is not None:
            self.on_request_close()
            return
        self.root.quit()
        self.root.destroy()

    def on_drag_start(self, event: tk.Event) -> None:
        self.drag_start_x = event.x_root
        self.drag_start_y = event.y_root
        self.window_start_x = self.root.winfo_x()
        self.window_start_y = self.root.winfo_y()

    def on_drag_motion(self, event: tk.Event) -> None:
        if not self.is_frameless:
            return
        dx = event.x_root - self.drag_start_x
        dy = event.y_root - self.drag_start_y
        x = self.window_start_x + dx
        y = self.window_start_y + dy
        self.root.geometry(f"+{x}+{y}")

    def hex_to_rgb(self, color: str) -> tuple[int, int, int]:
        color = color.strip("#")
        return int(color[0:2], 16), int(color[2:4], 16), int(color[4:6], 16)

    def mix_color(self, c0: str, c1: str, t: float) -> str:
        t = max(0.0, min(1.0, t))
        r0, g0, b0 = self.hex_to_rgb(c0)
        r1, g1, b1 = self.hex_to_rgb(c1)
        r = int(r0 + (r1 - r0) * t)
        g = int(g0 + (g1 - g0) * t)
        b = int(b0 + (b1 - b0) * t)
        return f"#{r:02x}{g:02x}{b:02x}"

    def load_background_images(self) -> list[Image.Image]:
        bg_dir = Path(__file__).resolve().parent / "background"
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
            return images

        palette = [
            ("#0b1020", "#133457"),
            ("#10131c", "#2a3450"),
            ("#111826", "#214866"),
        ]
        generated: list[Image.Image] = []
        for c0, c1 in palette:
            generated.append(self.create_gradient_image(1600, 900, c0, c1).filter(ImageFilter.GaussianBlur(radius=4)))
        return generated

    def create_gradient_image(self, width: int, height: int, c0: str, c1: str) -> Image.Image:
        """Generate high-quality gradient by scaling a micro image using bilinear interpolation."""
        small_img = Image.new("RGB", (2, 2))
        p0 = self.hex_to_rgb(c0)
        p1 = self.hex_to_rgb(c1)
        small_img.putpixel((0, 0), p0)
        small_img.putpixel((1, 1), p1)
        small_img.putpixel((0, 1), self.hex_to_rgb(self.mix_color(c0, c1, 0.5)))
        small_img.putpixel((1, 0), self.hex_to_rgb(self.mix_color(c0, c1, 0.5)))
        return small_img.resize((width, height), Image.Resampling.BILINEAR)

    def resize_cover(self, image: Image.Image, width: int, height: int) -> Image.Image:
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

    def prepare_resized_pair(self) -> None:
        if not self.background_images:
            return
        self.current_resized = self.resize_cover(self.background_images[self.current_index], self.last_w, self.last_h)
        self.next_resized = self.resize_cover(self.background_images[self.next_index], self.last_w, self.last_h)

    def apply_overlay(self, image: Image.Image) -> Image.Image:
        if not self.use_overlay_mask:
            return image
        width, height = image.size
        alpha_mask = Image.new("L", (width, height), 0)
        draw = ImageDraw.Draw(alpha_mask)

        a_top = max(0, min(255, int(self.overlay_alpha_top)))
        a_bottom = max(0, min(255, int(self.overlay_alpha_bottom)))

        for y in range(height):
            t = y / max(1, height - 1)
            a = int(a_top + (a_bottom - a_top) * t)
            draw.line((0, y, width, y), fill=a)

        overlay = Image.new("RGBA", (width, height), (0, 0, 0, 0))
        overlay.putalpha(alpha_mask)
        base = image.convert("RGBA")
        return Image.alpha_composite(base, overlay).convert("RGB")

    def set_canvas_image(self, image: Image.Image) -> None:
        display = self.apply_overlay(image)
        self.current_photo = ImageTk.PhotoImage(display)
        if self.canvas_image_id is None:
            self.canvas_image_id = self.bg_canvas.create_image(0, 0, image=self.current_photo, anchor="nw")
        else:
            self.bg_canvas.itemconfig(self.canvas_image_id, image=self.current_photo)

        # Keep text overlays above image after every frame update.
        for stroke_id in self.title_stroke_ids:
            self.bg_canvas.tag_raise(stroke_id)
        if self.title_main_id is not None:
            self.bg_canvas.tag_raise(self.title_main_id)
        for stroke_id in self.desc_stroke_ids:
            self.bg_canvas.tag_raise(stroke_id)
        if self.desc_text_id is not None:
            self.bg_canvas.tag_raise(self.desc_text_id)

    def render_background(self, blend_t: float) -> None:
        if not self.background_images:
            return
        self.prepare_resized_pair()
        if self.current_resized is None or self.next_resized is None:
            return
        if self.current_resized.size != self.next_resized.size:
            self.next_resized = self.next_resized.resize(self.current_resized.size, Image.Resampling.LANCZOS)
        blended = Image.blend(self.current_resized, self.next_resized, max(0.0, min(1.0, blend_t)))
        self.set_canvas_image(blended)

    def schedule_next_switch(self) -> None:
        if self.switch_after_id is not None:
            self.frame.after_cancel(self.switch_after_id)
        self.switch_after_id = self.frame.after(self.switch_interval_ms, self.start_transition)

    def start_transition(self) -> None:
        if len(self.background_images) < 2:
            self.schedule_next_switch()
            return
        self.in_transition = True
        self.transition_step = 0
        self.run_transition_frame()

    def run_transition_frame(self) -> None:
        if not self.in_transition:
            return

        # Apply Cubic Ease-Out easing function for smooth transition
        # Linear progress from 0.0 to 1.0
        t_linear = self.transition_step / max(1, self.transition_total_steps)
        # Eased progress: fast start, slow end for natural deceleration effect
        t_ease = 1 - (1 - t_linear) ** 3 
        
        self.render_background(t_ease)

        if self.transition_step >= self.transition_total_steps:
            self.in_transition = False
            self.current_index = self.next_index
            self.next_index = (self.current_index + 1) % len(self.background_images)
            self.render_background(0.0)
            self.schedule_next_switch()
            return

        self.transition_step += 1
        delay = int(1000 / self.transition_fps)
        self.transition_after_id = self.frame.after(delay, self.run_transition_frame)

    def animate_value(self, duration_ms: int, callback: Callable[[float], None]) -> None:
        delay = int(1000 / self.transition_fps)
        steps = max(1, int(duration_ms / delay))
        
        def step(i: int) -> None:
            t = i / steps
            callback(t)
            if i < steps:
                self.frame.after(delay, lambda: step(i + 1))
        
        step(0)

    def update_title_stepped(self, t: float) -> None:
        """Update title animation state at progress t."""
        if self.title_main_id is None: return
        color = self.mix_color(self.TITLE_COLOR_START, self.TITLE_COLOR_END, t)
        y = int((self.title_base_y + 15) - 15 * t)
        center_x = self.last_w * 0.5
        
        for idx, stroke_id in enumerate(self.title_stroke_ids):
            ox, oy = self.TITLE_STROKE_OFFSETS[idx]
            self.bg_canvas.coords(stroke_id, center_x + ox, y + oy)
        self.bg_canvas.itemconfig(self.title_main_id, fill=color)
        self.bg_canvas.coords(self.title_main_id, center_x, y)

    def update_desc_stepped(self, t: float) -> None:
        """Update description animation state at progress t."""
        if self.desc_text_id is None: return
        color = self.mix_color(self.DESC_COLOR_START, self.DESC_COLOR_END, t)
        center_x = self.last_w * 0.5
        self.bg_canvas.itemconfig(self.desc_text_id, fill=color)
        stroke_color = self.mix_color(self.DESC_STROKE_START, self.DESC_STROKE_END, t)
        for stroke_id in self.desc_stroke_ids:
            self.bg_canvas.itemconfig(stroke_id, fill=stroke_color)

    def play_entry_animation(self) -> None:
        """Play entry animation with eased curves and optimized delays."""
        if self.entry_played:
            return
        self.entry_played = True

        # Cancel any existing animations
        if self.title_after_id is not None:
            self.frame.after_cancel(self.title_after_id)
        if self.desc_after_id is not None:
            self.frame.after_cancel(self.desc_after_id)

        def start_title() -> None:
            self.animate_value(350, lambda t: self.update_title_stepped(1 - (1 - t)**3))

        def start_desc() -> None:
            self.animate_value(300, lambda t: self.update_desc_stepped(1 - (1 - t)**2))

        self.title_after_id = self.frame.after(50, start_title)
        self.desc_after_id = self.frame.after(150, start_desc)

    def clear_cards(self) -> None:
        for widget in self.card_widgets:
            widget.destroy()
        self.card_widgets = []

    def set_card_style(self, card: tk.Label, hover: bool = False, pressed: bool = False) -> None:
        if pressed:
            card.configure(
                bg="#1f2a3a",
                fg="#f3f6ff",
                relief="solid",
                bd=1,
                highlightthickness=1,
                highlightbackground="#a9c3df",
                highlightcolor="#a9c3df",
            )
            return
        if hover:
            card.configure(
                bg="#2d3a4d",
                fg="#f7fbff",
                relief="solid",
                bd=1,
                highlightthickness=1,
                highlightbackground="#b8d0ea",
                highlightcolor="#b8d0ea",
            )
            return
        card.configure(
            bg="#263446",
            fg="#f3f6ff",
            relief="solid",
            bd=1,
            highlightthickness=1,
            highlightbackground="#93abc5",
            highlightcolor="#93abc5",
        )

    def bind_card_events(self, card: tk.Label, command: Callable[[], None]) -> None:
        def on_enter(_: tk.Event) -> None:
            self.set_card_style(card, hover=True)

        def on_leave(_: tk.Event) -> None:
            self.set_card_style(card, hover=False)

        def on_press(_: tk.Event) -> None:
            self.set_card_style(card, pressed=True)

        def on_release(_: tk.Event) -> None:
            self.set_card_style(card, hover=True)
            command()

        card.bind("<Enter>", on_enter)
        card.bind("<Leave>", on_leave)
        card.bind("<ButtonPress-1>", on_press)
        card.bind("<ButtonRelease-1>", on_release)

    def layout_cards(self) -> None:
        if not self.card_widgets:
            return

        cols = 2
        card_w = 0.40
        x_positions = [0.08, 0.52]
        card_h = 0.12
        y_start = 0.60
        y_gap = 0.15

        for i, card in enumerate(self.card_widgets):
            row = i // cols
            col = i % cols
            card.place(relx=x_positions[col], rely=y_start + row * y_gap, relwidth=card_w, relheight=card_h)

    def update_module_buttons(self, modules: list[FeatureModule], default_module_index: int = 0) -> None:
        _ = default_module_index
        self.clear_cards()

        if not modules:
            empty = tk.Label(
                self.frame,
                text="No function module registered yet.",
                bg="#242c37",
                fg="#d5dce8",
                font=("Segoe UI", 13),
                padx=16,
                pady=14,
            )
            empty.place(relx=0.5, rely=0.64, relwidth=0.40, relheight=0.10, anchor="n")
            self.card_widgets.append(empty)
            return

        for index, module in enumerate(modules):
            label = f"Open {module.name}"
            card = tk.Label(
                self.frame,
                text=label,
                bg="#263446",
                fg="#f3f6ff",
                font=("Segoe UI", 20, "bold"),
                padx=24,
                pady=18,
                cursor="hand2",
                justify="center",
                bd=1,
                relief="solid",
                highlightthickness=1,
                highlightbackground="#93abc5",
                highlightcolor="#93abc5",
            )
            self.set_card_style(card)
            self.bind_card_events(card, lambda i=index: self.on_open_module(i))
            self.card_widgets.append(card)

        self.layout_cards()

    def on_resize(self, event: tk.Event) -> None:
        if event.widget is not self.frame:
            return

        w = max(1, int(event.width))
        h = max(1, int(event.height))
        if w == self.last_w and h == self.last_h:
            return

        self.last_w = w
        self.last_h = h

        self.title_base_y = int(h * 0.11)
        self.desc_base_y = int(h * 0.24)

        if self.title_main_id is not None:
            title_width = max(560, int(w * 0.60))
            for stroke_id in self.title_stroke_ids:
                self.bg_canvas.itemconfig(stroke_id, width=title_width)
            self.bg_canvas.itemconfig(self.title_main_id, width=title_width)

            center_x = w * 0.5
            for idx, stroke_id in enumerate(self.title_stroke_ids):
                ox, oy = self.TITLE_STROKE_OFFSETS[idx]
                self.bg_canvas.coords(stroke_id, center_x + ox, self.title_base_y + oy)
            self.bg_canvas.coords(self.title_main_id, center_x, self.title_base_y)

        if self.desc_text_id is not None:
            desc_width = max(540, int(w * 0.62))
            for stroke_id in self.desc_stroke_ids:
                self.bg_canvas.itemconfig(stroke_id, width=desc_width)
            self.bg_canvas.itemconfig(self.desc_text_id, width=desc_width)

            center_x = w * 0.5
            for idx, stroke_id in enumerate(self.desc_stroke_ids):
                ox, oy = self.DESC_STROKE_OFFSETS[idx]
                self.bg_canvas.coords(stroke_id, center_x + ox, self.desc_base_y + oy)
            self.bg_canvas.coords(self.desc_text_id, w * 0.5, self.desc_base_y)

        self.layout_cards()

        if self.in_transition:
            t = self.transition_step / max(1, self.transition_total_steps)
            self.render_background(t)
        else:
            self.render_background(0.0)

    def hide(self) -> None:
        if self.switch_after_id is not None:
            self.frame.after_cancel(self.switch_after_id)
            self.switch_after_id = None
        if self.transition_after_id is not None:
            self.frame.after_cancel(self.transition_after_id)
            self.transition_after_id = None
        if self.title_after_id is not None:
            self.frame.after_cancel(self.title_after_id)
            self.title_after_id = None
        if self.desc_after_id is not None:
            self.frame.after_cancel(self.desc_after_id)
            self.desc_after_id = None

        self.disable_frameless_mode()
        self.frame.pack_forget()

    def show(self) -> None:
        self.frame.pack(fill=tk.BOTH, expand=True)
        self.enable_frameless_mode()
        self.render_background(0.0)
        if self.switch_after_id is None and not self.in_transition:
            self.schedule_next_switch()
