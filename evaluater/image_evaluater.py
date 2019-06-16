from base.base_evaluater import BaseEvaluater
from keras.preprocessing import image
import os
import cv2
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix

class ImageEvaluater(BaseEvaluater):
    def __init__(self, model, valData, train_labels, config):
        super(ImageEvaluater, self).__init__(model, valData, train_labels, config)
    
        model.load_weights(self.config.evaluater.weights_path)

        # get all the test images paths
        #pathx="/home/stephen/Projects/flower-recognition/dataset/test"
        pathx=self.config.data_loader.validation_data_path        
        
        #test_images = os.listdir(pathx)
#         train_labels =  {'buttercup': 1, 'tigerlily': 14, 'bluebell': 0, 'crocus': 4, 'daisy': 6, 'snowdrop': 12, 'lily_valley': 10,
#          'tulip': 15, 'daffodil': 5, 'iris': 9, 'pansy': 11, 'colts_foot': 2, 'fritillary': 8, 'dandelion': 7,
#          'cowslip': 3, 'windflower': 16, 'sunflower': 13}
        #train_labels= self.valData.class_indices
        #print(train_labels)
        image_size = (self.config.trainer.dim, self.config.trainer.dim)
        # loop through each image in the test data
        y_pred=[]
        class_names=[]
        
        showGUI=True

        for class_label in range(0,len(train_labels)):
            class_name = self.get_name(train_labels, class_label) 
            class_names.append(class_name)          
            if os.path.isdir(pathx + "/" + class_name): 
                for filename in os.listdir(pathx + "/" + class_name):
                    if str(filename).lower().endswith(('.png', '.jpg', '.jpeg')):
                        path=pathx+ "/" +class_name+ "/" +filename
          
                        img = image.load_img(path,target_size=image_size)
                        #(1, height, width, channels), add a dimension because 
                        #the model expects this shape: (batch_size, height, width, channels)
                        # imshow expects values in the range [0, 1]
                        img_tensor  = image.img_to_array(img)
                        img_tensor  = np.expand_dims(img_tensor , axis=0)
                        img_tensor /= 255. 
                         
                        prediction = model.predict(img_tensor)                     
                        percentages = self.create_percentages(prediction)
                        top5 = self.top_five(percentages, train_labels)
                        y_pred.append(train_labels[top5[0][1]])
                        # perform prediction on test image
                        if showGUI==True:
                            print ("I think it is a " + self.format_top_five(top5))
                            img_color = cv2.imread(path, 1)
                            cv2.putText(img_color, str(round(top5[0][0],2))+"% sure it's a " + top5[0][1], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,128), 2)
                            cv2.imshow("test", img_color)
                          
                            key = cv2.waitKey(0) & 0xFF
                            
                            if (key == ord('q')):
                                cv2.destroyAllWindows()
                            elif (key == ord('e')):
                                cv2.destroyAllWindows()
                                cv2.waitKey(330)
                                showGUI=False

        

                
        #y_prob=model.predict_generator(valData)
        #y_pred_fromGen = y_prob.argmax(axis=-1) 
        print("valData",valData.classes)
        print("predictedData",y_pred)
        #print("predictedData",y_pred_fromGen)

        y_true=valData.classes
        print('Classification Report')
        print(classification_report(y_true, y_pred, target_names=train_labels))
        
        print('Confusion Matrix')
        cnf_matrix = confusion_matrix(y_true, y_pred)
        sns.heatmap(cnf_matrix,
            annot=True,
            cmap="Set2",
            yticklabels=class_names)
        
        plt.xlabel('Predicted')
        plt.ylabel('True')
        plt.show()
        
        


#        np.set_printoptions(precision=2)

#         # Plot non-normalized confusion matrix
#         plt.figure()
#         plot_confusion_matrix(cnf_matrix, classes=class_names,
#         title='Confusion matrix, without normalization')
#         
#         # Plot normalized confusion matrix
#         plt.figure()
#         plot_confusion_matrix(cnf_matrix, classes=class_names, normalize=True,
#         title='Normalized confusion matrix')
# 
#         plt.show()
#         print(confusion_matrix(valData.classes, y_pred))
 
    def format_top_five(self,five):
        """
        Format the top five predictions into a more pleasing look
        :param five: list of top five percentages containing tuples (percentage, name)
        :return: Formatted string of the predictions
        """
        result = '\n***** Top Five Predictions *****\n\n'
        result += 'Confidence\t\tFlower Name\n'
        result += '==================================\n\n'
        for pair in five:
            result += str(round(pair[0], 2)) + '%' + '\t\t\t' + pair[1] + '\n'
        return result
    
                    
    def create_percentages(self,probabilities):
        """
        Take a numpy array containing the probabilities of some other input
        data for what appropriate flower class it belongs to.
        :param probabilities: a numpy array of float values which are probabilities
        :return: a numpy array of float values as percentages
        """
        sumProb = np.sum(probabilities)
        percentages = []  # standard python list to contain the percentages
    
        # to calculate the percentage take each independent probability and divide it by the sum of all
        for prob in np.nditer(probabilities):
            percentages.append((prob / sumProb) * 100)
    
        return percentages
     
    def top_five(self, percentages, names):
        """
        Create the top 5 predictions for the given flower and convert them into percentages.
        :param percentages: list of percentages that line up with class labels
        :param names: is the dictionary that contains the class names and their integer labels
        :return: a list of the top five percentages as tuples with (percent, name_of_flower)
        """
        five = []
        loc = 0
        for percent in percentages:
            if len(five) > 0:
                for value in five:
                    if percent > value[0]:
                        five.remove(value)
                        five.append((percent, self.get_name(names, loc)))
                        break
                    elif len(five) < 5:
                        five.append((percent, self.get_name(names, loc)))
                        break
    
            else:
                five.append((percent, self.get_name(names, loc)))
            loc += 1
        five.sort(key=lambda flow_tup: flow_tup[0], reverse=True)
        return five
    
    def get_name(self, names, location):
        """
        Reads in the appropriate dictionary of classes and the location of the class we want
        :param names: dictionary of classes and integer labels
        :param location: integer label of flower
        :return: the name of the flower that lines up with the passes location
        """
        for name in names:
            if names[name] == location:
                return name
        return 'invalid location passed to get_name'
