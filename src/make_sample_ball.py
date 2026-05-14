from pathlib import Path

from PIL import Image, ImageDraw


def main() -> None:
    out = Path("input/sample_ball.png")
    out.parent.mkdir(parents=True, exist_ok=True)

    image = Image.new("RGB", (720, 520), (244, 244, 240))
    draw = ImageDraw.Draw(image)

    bbox = (150, 70, 570, 490)
    draw.ellipse(bbox, fill=(238, 238, 228), outline=(30, 30, 30), width=4)

    draw.pieslice((155, 75, 565, 485), start=210, end=330, fill=(225, 55, 48))
    draw.pieslice((155, 75, 565, 485), start=35, end=120, fill=(40, 118, 210))
    draw.pieslice((155, 75, 565, 485), start=125, end=178, fill=(245, 190, 45))

    draw.arc((205, 120, 515, 440), start=80, end=280, fill=(25, 25, 25), width=5)
    draw.arc((205, 120, 515, 440), start=260, end=80, fill=(25, 25, 25), width=5)
    draw.line((360, 73, 360, 487), fill=(25, 25, 25), width=4)

    draw.ellipse((305, 220, 415, 330), fill=(238, 238, 228), outline=(25, 25, 25), width=4)
    image.save(out)
    print(out)


if __name__ == "__main__":
    main()
