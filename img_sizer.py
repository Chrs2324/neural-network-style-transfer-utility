import PIL
from PIL import Image

class ImageSizer():
    def resize(self, img: Image, width: int) -> Image:
        """Resize while maintaining aspect ratio
        
        Arguments:
            img {Image} -- Original Image
            width {int} -- The desired width
        
        Returns:
            Image -- The resized image
        """
        wpercent = (width / float(img.size[0]))
        hsize = int((float(img.size[1]) * float(wpercent)))
        return img.resize((width, hsize), PIL.Image.ANTIALIAS)

    def matchsize(self, img_to_match: Image, img_to_resize: Image):
        w, h = img_to_match.size[0], img_to_match.size[1]
        return img_to_resize.resize((w, h), PIL.Image.ANTIALIAS)

if __name__ == "__main__":
    print("[ Debug ]")
    img = Image.open("testimg.jpg")
    newimg = ImageSizer.resize(0, img=img, width=200)
    img2 = Image.open("testimg2.JPG")
    newimg2 = ImageSizer.matchsize(0, newimg, img2)
    newimg.save('rimage.jpg')
    newimg2.save('rimage2.jpg')