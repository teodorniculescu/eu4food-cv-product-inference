import numpy as np
import cv2
import torchvision.transforms as transforms


class CutoutNumpy:
    def __init__(self, cutout_size=None, cutout_percent=None, probability=0.5, color=(255, 255, 255)):
        if (cutout_size is None and cutout_percent is None) or (not cutout_size is None and not cutout_percent is None):
            raise Exception('ERROR: Please specify either percent or size')
        self.cutout_size = cutout_size
        self.cutout_percent = cutout_percent
        self.probability = probability
        self.color = color

    def __call__(self, image):
        if np.random.rand(1)[0] > self.probability:
            return image

        image_height, image_width, _ = image.shape

        if self.cutout_size is None:
            cutout_size_h = int(image_height * self.cutout_percent)
            cutout_size_w = int(image_width * self.cutout_percent)
            cutout_size = (cutout_size_h, cutout_size_w)
        else:
            cutout_size = (self.cutout_size, self.cutout_size)
        
        h_pos = np.random.randint(0, image_height - cutout_size[0])
        from_h = h_pos
        to_h = h_pos + cutout_size[0]

        w_pos = np.random.randint(0, image_width - cutout_size[1])
        from_w = h_pos
        to_w = h_pos + cutout_size[1]

        image[from_h:to_h, from_w:to_w, :] = self.color

        return image


if __name__ == '__main__':
    image_size = (224, 224)
    image_path = "dataset/C07_CocaCola_CocaCola/RO/IMG_3992.JPEG"
    image = cv2.imread(image_path)

    transform = transforms.Compose([
        CutoutNumpy(cutout_percent=0.1, probability=1), 
        transforms.ToPILImage(),
        transforms.Resize(image_size),
        transforms.ToTensor(),
    ])

    vertical_list = []
    for i in range(0, 10):
        horizontal_list = []
        for j in range(0, 10):
            transformed_image = transform(image.copy()).numpy()
            transformed_image = (transformed_image * 255).astype(np.uint8)
            transformed_image = np.transpose(transformed_image, (1, 2, 0))

            horizontal_list.append(transformed_image)

        concatenated_image = cv2.hconcat(horizontal_list)
        vertical_list.append(concatenated_image)

    result_image = cv2.vconcat(vertical_list)
    cv2.imwrite('result_image.png', result_image)


