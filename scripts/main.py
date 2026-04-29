import moderngl
import moderngl_window
import numpy as np
import cv2
import torch
from transformers import SegformerImageProcessor, SegformerForSemanticSegmentation


# ----------------------------
# MODEL
# ----------------------------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

model = SegformerForSemanticSegmentation.from_pretrained(
    "../model/retrained"
).to(DEVICE)
model.eval()

processor = SegformerImageProcessor(size=512)


# ----------------------------
# WINDOW CLASS
# ----------------------------
class ARApp(moderngl_window.WindowConfig):
    title = "Road Segmentation AR"
    window_size = (1280, 720)
    aspect_ratio = None
    resizable = True

    def __init__(self, **kwargs):
        super().__init__(**kwargs)

        self.cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

        # fullscreen quad
        self.quad = moderngl_window.geometry.quad_2d(size=(2, 2))

        # shader
        self.program = self.ctx.program(
            vertex_shader="""
                #version 330
                in vec2 in_position;
                out vec2 v_uv;

                void main() {
                    v_uv = in_position * 0.5 + 0.5;
                    gl_Position = vec4(in_position, 0.0, 1.0);
                }
            """,
            fragment_shader=open("shader.glsl").read()
        )

        # textures
        self.cam_tex = self.ctx.texture((1280, 720), 3)
        self.mask_tex = self.ctx.texture((512, 512), 1, dtype="u1")

        self.program["cameraTex"] = 0
        self.program["maskTex"] = 1

    def update(self, time, delta):
        ret, frame = self.cap.read()
        if not ret:
            return

        frame = cv2.resize(frame, (1280, 720))

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        inputs = processor(images=rgb, return_tensors="pt").to(DEVICE)

        with torch.no_grad():
            out = model(**inputs)

        mask = out.logits.argmax(1)[0].cpu().numpy().astype(np.uint8)
        mask = cv2.resize(mask, (512, 512), interpolation=cv2.INTER_NEAREST)

        # upload
        self.cam_tex.write(frame.tobytes())
        self.mask_tex.write(mask.tobytes())

        # 🔥 IMPORTANT: force bind
        self.cam_tex.use(location=0)
        self.mask_tex.use(location=1)

    def render(self, time, frame_time):
        self.ctx.clear(0.0, 0.0, 0.0)
        self.program["cameraTex"].value = 0
        self.program["maskTex"].value = 1
        self.quad.render(self.program)

    def on_render(self, time, frame_time):
        self.render(time, frame_time)


if __name__ == "__main__":
    moderngl_window.run_window_config(ARApp)