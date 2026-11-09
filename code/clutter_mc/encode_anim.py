"""Encode figures/anim_frames/*.png into an animated GIF (plays anywhere) and an animated WebP
(smaller, sharper) with Pillow. usage: python encode_anim.py [fps] [gif_width]"""
import glob, os, sys
from PIL import Image

fps = float(sys.argv[1]) if len(sys.argv) > 1 else 20
gw = int(sys.argv[2]) if len(sys.argv) > 2 else 960
HERE = os.path.dirname(os.path.abspath(__file__))
files = sorted(glob.glob(os.path.join(HERE, "figures", "anim_frames", "frame_*.png")))
frames = [Image.open(f).convert("RGB") for f in files]
dur = int(round(1000 / fps))
out = os.path.join(HERE, "figures")
webp = os.path.join(out, "radar_on_moving_sea.webp")
frames[0].save(webp, save_all=True, append_images=frames[1:], duration=dur, loop=0, quality=70, method=4)
small = [im.resize((gw, round(im.height * gw / im.width)), Image.LANCZOS) for im in frames]
pal = small[len(small) // 2].quantize(colors=255, method=Image.Quantize.MEDIANCUT)
gq = [im.quantize(palette=pal, dither=Image.Dither.NONE) for im in small]
gif = os.path.join(out, "radar_on_moving_sea.gif")
gq[0].save(gif, save_all=True, append_images=gq[1:], duration=dur, loop=0, optimize=False)
for p in (gif, webp):
    print(f"{os.path.basename(p)}: {len(files)} frames, {os.path.getsize(p)/1e6:.1f} MB")
