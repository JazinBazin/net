from PIL import Image
from bs4 import BeautifulSoup
from urllib.request import urlopen
import os


def html_url_parser(url, save_dir, show=False, wait=False):
    """
    HTML parser to download images from URL.
    Params:\n
    `url` - Image url\n
    `save_dir` - Directory to save extracted images\n
    `show` - Show downloaded image\n
    `wait` - Press key to continue executing
    """

    website = urlopen(url)
    html = website.read()

    soup = BeautifulSoup(html, "html.parser")

    for image_id, link in enumerate(soup.find_all('a', href=True)):
        if image_id == 0:
            continue

        img_url = link['href']

        try:
            if not os.path.isfile(save_dir + "img-%d.png" % image_id):
                print("[INFO] Downloading image from URL:", link['href'])
                image = Image.open(urlopen(img_url))
                image.save(save_dir + "img-%d.png" % image_id, "PNG")
                if show:
                    image.show()
            else:
                print('skipped')
        except KeyboardInterrupt:
            print("[EXCEPTION] Pressed 'Ctrl+C'")
            break
        except Exception as image_exception:
            print("[EXCEPTION]", image_exception)
            continue

        if wait:
            key = input("[INFO] Press any key to continue ('q' to exit)... ")
            if key.lower() == 'q':
                break


if __name__ == "__main__":
    URL_TRAIN_IMG = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/sat/index.html"
    URL_TRAIN_GT = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/train/map/index.html"

    URL_TEST_IMG = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/sat/index.html"
    URL_TEST_GT = "https://www.cs.toronto.edu/~vmnih/data/mass_roads/valid/map/index.html"

    html_url_parser(url=URL_TRAIN_IMG, save_dir="dataset/training/input/")
    html_url_parser(url=URL_TRAIN_GT, save_dir="dataset/training/output/")

    html_url_parser(url=URL_TEST_IMG, save_dir="dataset/testing/input/")
    html_url_parser(url=URL_TEST_GT, save_dir="dataset/testing/output/")

    print("[INFO] All done!")
