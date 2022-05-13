''' 
Load model and data, run them together.
'''

import Data_Preparation
from models import MLP

data, labels = Data_Preparation.prepare_data()
print(data)

output = MLP.train(data=data, labels=labels)
print(output)


'''
Example of dataloader dataset:
class CatsDogsData(Dataset):
	
	'''kaggle catsdogs dataset.'''
	def __init__(self, image_dir=train_dir, transforms=True, train=True):
		self.image_dir = image_dir
		self.images = os.listdir(image_dir)
		self.transforms = transforms
		self.train = train

	def __len__(self):
		return len(self.images)

	def __getitem__(self, index):
		img_path = os.path.join(self.image_dir, self.images[index])
		image = cv2.imread(img_path, 1) # 1 for color
		image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
		image = cv2.resize(image, (128, 128)) #default 244 for vgg
		image = cv2.normalize(image, image, 0, 255, cv2.NORM_MINMAX) #(src, dst, min, max, norm)
		
		if self.train==True:
			if self.transforms==True:
				transform = A.Compose([ #copied and pasted from tutorial.
					A.CLAHE(),
					A.RandomRotate90(),
					A.Transpose(),
					A.Blur(blur_limit=3),
					A.OpticalDistortion(),
					A.GridDistortion(),
					A.HueSaturationValue(),
				])

				image = transform(image=image)['image']
				image = np.array(image, dtype='float32')
				image = np.moveaxis(image, -1, 0) #move channels first
			

			if self.images[index][0:3] == str('cat'):
				label = 0.
				# label = 'cat'
			elif self.images[index][0:3] == str('dog'):
				label = 1.
				# label = 'dog'
			else:
				raise Exception("label is neither cat nor dog.")

			return (image, label)
		else: #train is false
			image = np.array(image, dtype='float32')
			image = np.moveaxis(image, -1, 0) #move channels first
			return image


'''