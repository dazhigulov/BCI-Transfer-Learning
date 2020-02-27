


# set the matplotlib backend so figures can be saved in the background
import matplotlib
matplotlib.use("Agg")

# import the necessary packages
import keras
import matplotlib.pyplot as plt
import numpy as np
import argparse
import os
import cv2
import random
import shutil
# import Augmentor

#from dataPreprocessor import dataProvider
from train_utils_bci import (NDStandardScaler, subject_specific, leave1out, importseveralsubjects, pad_with_zeros, pad_by_duplicating, generator)
from data_pooler_bci import data_pooler

from keras.preprocessing.image import ImageDataGenerator
from keras.preprocessing.image import save_img
from keras.callbacks import LearningRateScheduler
from keras.optimizers import RMSprop
from keras.utils import np_utils
from keras.models import load_model
from keras import models
from keras import layers
from keras import optimizers
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
from sklearn.metrics import roc_curve
from sklearn.metrics import roc_auc_score
from imutils import paths
from PIL import Image
from keras.optimizers import Adagrad

'''import tensorflow as tf  
from keras.backend.tensorflow_backend import set_session  
config = tf.ConfigProto()  
config.gpu_options.allow_growth = True  # dynamically grow the memory used on the GPU  
config.log_device_placement = True  # to log device placement (on which device the operation ran)  
                                    # (nothing gets printed in Jupyter, only if you run it standalone)
sess = tf.Session(config=config) 
set_session(sess)  # set this TensorFlow session as the default session for Keras  
'''
#FUNCTIONS TO IMPORT DIFFERENT MODELS


def func1(shape):
	from keras.applications import ResNet50
	BS = 16
	conv_base = ResNet50(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func2(shape):
	from keras.applications import ResNet101
	BS = 16
	conv_base = ResNet101(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func3(shape):
	from keras.applications import ResNet152
	BS = 8
	conv_base = ResNet152(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func4(shape):
	from keras.applications import DenseNet121
	BS = 32
	conv_base = DenseNet121(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func5(shape):
	from keras.applications import DenseNet201
	BS = 8
	conv_base = DenseNet201(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func6(shape):
	from keras.applications import Xception
	BS = 32
	conv_base = Xception(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func7(shape):
	from keras.applications import InceptionV3
	BS = 32
	conv_base = InceptionV3(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func8(shape):
	from keras.applications import VGG16
	BS = 16
	conv_base = VGG16(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base

def func9(shape):
	from keras.applications import VGG19
	BS = 16
	conv_base = VGG19(weights = 'imagenet',
                 	include_top = False,
                 	input_shape = (shape[0],shape[1],3))
	return BS,conv_base



i = 1
ii = 0
k = 1    #FOR SEPARATE FIGURES

for ss in range(8): ## FOR EACH SUBJECT SEPARATELY


	NUM_EPOCHS = 50
	#INIT_LR = 1e-2

	model_names = ['Xception']

	model_names1 = [
					'DenseNet121',
					'DenseNet201',
					'InceptionV3',
					'Xception',
					'ResNet50',
					'ResNet101',
					'ResNet152',
					'VGG16',
					'VGG19'
					]
	for name in model_names:
		train_norm, y_train, val_subjects_norm, test_subjects_norm = data_pooler(dataset_name='TenHealthyData', subIndexTest = ss)
		if (name == 'InceptionV4') or (name =='InceptionV3') or (name =='InceptionV2') or (name =='Xception'):
			shape = [299,299]
		else:
			shape = [224,224]

		if (name == 'ResNet50'): BS,conv_base = func1(shape)
		elif (name == 'ResNet101'): BS,conv_base = func2(shape)
		elif (name == 'ResNet152'): BS,conv_base = func3(shape)
		elif (name == 'DenseNet121'): BS,conv_base = func4(shape)
		elif (name == 'DenseNet201'): BS,conv_base = func5(shape)
		elif (name == 'Xception'): BS,conv_base = func6(shape)
		elif (name == 'InceptionV3'): BS,conv_base = func7(shape)
		elif (name == 'VGG16'): BS,conv_base = func8(shape)
		elif (name == 'VGG19'): BS,conv_base = func9(shape)

		#x_train, y_train, x_valid, y_valid, x_test, y_test = dataProvider(dataset_name = 'EPFL', batch_size=BS)
		#print(train_norm.shape)



		train_gen = generator(train_norm,
	                  y_train, 
	                  min_index=0,
	                  max_index=None,
	                  batch_size=BS,
	                  desired_size = shape[0],                 #************WARNING************
	                  #color_mode="grayscale",
	                  shuffle=True) #see what None does

		'''val_gen = generator(val_norm,
                    y_val,
                    min_index=0,
                    max_index=None,
                    batch_size=BS,
                    desired_size = shape[0],                   #************WARNING************
                    #color_mode="grayscale",
                    shuffle=True)'''
		totalVal = 0
		for jj in range(len(val_subjects_norm)):
			totalVal += len(val_subjects_norm[jj]['ytrain'])
			val_gen = generator(val_subjects_norm[jj]['xtrain'],
                         val_subjects_norm[jj]['ytrain'],
                         min_index=0,
                         max_index=None,
                         batch_size=BS,
                         desired_size = shape[0],
                         shuffle = True                 		 #************WARNING************
                         #color_mode="grayscale",
                         )

		totalTrain = len(y_train)
		classWeight = [np.sum(y_train == 0), np.sum(y_train == 1)]




		#Feature extraction with data augmentation
		model = models.Sequential()
		model.add(conv_base)
		model.add(layers.Flatten())
		model.add(layers.Dropout(0.2))
		model.add(layers.Dense(256, activation = 'relu'))
		model.add(layers.Dense(1, activation = 'sigmoid'))

		conv_base.trainable = True
		'''set_trainable = False
		print("\n\n\n"+str(conv_base.layers))'''
		count = 0
		for layer in conv_base.layers:
				#if(count > 39):
					layer.trainable = True
				#count = count + 1
		model.summary()
			
		model.compile(loss = 'binary_crossentropy',
	        		optimizer = optimizers.RMSprop(learning_rate=0.0001, rho=0.9),
	        		metrics = ['acc'])


		#es = EarlyStopping(monitor='val_accuracy', mode='max', min_delta=5)
		mcp_save = keras.callbacks.ModelCheckpoint(str(i)+"Test-subject-"+str(ss)+"_bci_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5",monitor='val_acc', verbose=0, 
			save_best_only=True, save_weights_only=False, mode='max', period=1)

		H = model.fit_generator(
			train_gen,
			steps_per_epoch=totalTrain // BS,
			validation_data=val_gen,
			validation_steps=totalVal // BS,
			class_weight=classWeight,
			epochs=NUM_EPOCHS,
			callbacks = [mcp_save])


		
		# save model and architecture to single file
		#model.save("noaug_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")
		model.save(str(i)+"Test-subject-"+str(ss)+"_BCI_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")
		print("Saved model to disk")

		


		model = load_model(str(i)+"Test-subject-"+str(ss)+"_bci_Best_model#"+str(NUM_EPOCHS)+'_'+ str(name) + ".h5")

		fname1 = str(i)+"Test-subject-"+str(ss)+"_img_AllLayer_modelHistory#"+str(NUM_EPOCHS)+'_'+ str(name) + ".txt"
		f = open(fname1,"w+")

		for jj in range(len(test_subjects_norm)):
			print("hi!")
			totalTest = len(test_subjects_norm[jj]['ytrain'])
			test_gen = generator(test_subjects_norm[jj]['xtrain'],
                         test_subjects_norm[jj]['ytrain'],
                         min_index=0,
                         max_index=None,
                         batch_size=BS,
                         desired_size = shape[0]                  #************WARNING************
                         #color_mode="grayscale",
                         )
			predIdxs = model.predict_generator(test_gen,verbose=2,
						steps=(totalTest // BS))
			# for each image in the testing set we need to find the index of the
			# label with corresponding largest predicted probability
#			predIdxs = np.argmax(predIdxs, axis=1)
			predIdxs = np.where(predIdxs >= 0.5, 1, 0)
			predIdxs = predIdxs.reshape(len(predIdxs),)
			predIdxs = np.float32(predIdxs)
            #print(classification_report(test_gen.classes, predIdxs, target_names=y_test))
			#test_gen.reset()
			auc = roc_auc_score(test_subjects_norm[jj]['ytrain'][0:len(predIdxs)],predIdxs)
			cm = confusion_matrix(test_subjects_norm[jj]['ytrain'][0:len(predIdxs)], predIdxs)                               #CHANGED EVERY testGen to valGen
			total = sum(sum(cm))
			acc = (cm[0, 0] + cm[1, 1]) / total
			sensitivity0 = cm[0, 0] / (cm[0, 0] + cm[0, 1])
			sensitivity1 = cm[1, 1] / (cm[1, 1] + cm[1, 0])
			#sensitivity2 = cm[2, 2] / (cm[2, 2] + cm[2, 0] + cm[2,1] + cm[2,3])
			#sensitivity3 = cm[3, 3] / (cm[3, 3] + cm[3, 0] + cm[3,1] + cm[3,2])
			sensitivity = (sensitivity1+sensitivity0)/2
			f.write("Test subject \n" + str(ss) + "\nValidation acc: " + str(round(H.history['val_acc'][NUM_EPOCHS-1],4)) + "; Training acc: " + str(round(H.history['acc'][NUM_EPOCHS-1],4)) + 
				"\nAUC: " + str(round(auc,4)) + "; Sensitivity (avg): " + str(round(sensitivity,4)) + "; Test acc: " + str(round(acc,4)))
			f.write("\nSensitivity0: {:.4f}".format(sensitivity0))
			f.write("\nSensitivity1: {:.4f}".format(sensitivity1))
			#f.write("\nSensitivity2: {:.4f}".format(sensitivity2))
			#f.write("\nSensitivity3: {:.4f}".format(sensitivity3))
			#f.write("\n\n" + classification_report(testGen.classes, predIdxs,
			#						target_names=testGen.class_indices.keys()))
			f.write("\n" + str(cm))

			del predIdxs
#        e1 = "noaug_modelHistory#"+str(NUM_EPOCHS)+'_'+ str(name) + ".txt"
		f.close()
        

# =============================================================================
# 		print(cm)
# 		print("test acc: {:.4f}".format(acc))
# 		print("sensitivity0: {:.4f}".format(sensitivity0))
# 		print("sensitivity1: {:.4f}".format(sensitivity1))
# 		print("sensitivity2: {:.4f}".format(sensitivity2))
# 		print("sensitivity3: {:.4f}".format(sensitivity3))
# 		print("sensitivity (avg): {:.4f}".format(sensitivity))
# =============================================================================

		plt.figure(k)
		plt.plot(H.history['acc'])
		plt.plot(H.history['val_acc'])
		plt.title('model accuracy')
		plt.ylabel('accuracy')
		plt.xlabel('epoch')
		plt.legend(['train', 'val'], loc='upper left')
		#plt.savefig('noaug_modelHistory#'+str(NUM_EPOCHS)+'_'+ str(name)+'_accuracy.png')
		plt.savefig(str(i)+"Test-subject-"+str(ss)+'_img_AllLayer_modelHistory#'+str(NUM_EPOCHS)+'_'+ str(name)+'_accuracy.png')

		plt.figure(k+1)
		plt.plot(H.history['loss'])
		plt.plot(H.history['val_loss'])
		plt.title('model loss')
		plt.ylabel('loss')
		plt.xlabel('epoch')
		plt.legend(['loss', 'val_loss'], loc='upper left')
		#plt.savefig('noaug_modelHistory#'+str(NUM_EPOCHS)+'_'+ str(name)+'_loss.png')
		plt.savefig(str(i)+"Test-subject-"+str(ss)+'_img_AllLayer_modelHistory#'+str(NUM_EPOCHS)+'_'+ str(name)+'_loss.png')
		
		#k = k + 2
		
	i = i + 1

	ii = ii + 4
	del train_norm, y_train, val_subjects_norm, test_subjects_norm