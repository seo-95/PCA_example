from PIL import Image
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA 
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB

def show_img(img, title=None):
    plt.imshow(np.reshape(img, (227, 227, 3)))
    if title != None :
        plt.title(title)
    plt.show()
    
def compute_pca(data, n_comp):
    #fit_transform() and inverse_transform() must work on the same PCA object
    pca = PCA(n_comp)
    X_t = pca.fit_transform(data)
    return pca.inverse_transform(X_t)

def destandardize(data, mean_v, std_v):
    i = 0
    denormalized = []
    for img in data:
        data_cpy = np.array(img).copy()
        data_cpy = data_cpy*std_v[i] + mean_v[i]
        denormalized.append(np.uint8(data_cpy))
        i += 1
    return np.array(denormalized)

def standardize(data):
    mean_v = []
    std_v = []
    normalized = []
    for img in data:
        img_cpy = np.array(img).copy() 
        currimg_mean = np.mean(img_cpy) 
        currimg_std = np.std(img_cpy) 
        mean_v.append(currimg_mean)
        std_v.append(currimg_std)
        img_std = (img_cpy-currimg_mean)/currimg_std
        normalized.append(img_std)
    return np.array(normalized), np.array(mean_v), np.array(std_v)


def manual_fit(data, from_pc, to_pc=None):
    if to_pc == None :
        #perfoms PCA with all the PCs
        pca = PCA()
        #fit the PCA on the data (computes the PCs on the data)
        pca.fit(data) 
        to_pc = len(pca.components_)
    else:
        to_pc += 1
        pca = PCA(to_pc)
        pca.fit(data)
    # extracts the last N PCs (last N rows.. corrispondant columns are extracted automatically)
    pca.components_ = pca.components_[from_pc : to_pc]
    return pca

#to_print is the image to print as result
def principal_component_analysis(dataset, to_print):
    to_print = to_print % len(dataset)
    show_img(dataset[to_print], 'Original')
    
    #standardization: it centers the data around 0 wit unit variance
    normalized, mean_v, std_v = standardize(dataset)
    
    #the pca are ordered based on the variance (from the highest to the lowest)
    inverse = compute_pca(normalized, 60)
    reconstructed = destandardize(inverse, mean_v, std_v)
    show_img(reconstructed[to_print], 'first 60 PCs:')
    
    
    inverse = compute_pca(normalized, 6)
    reconstructed = destandardize(inverse, mean_v, std_v)
    show_img(reconstructed[to_print], 'first 6 PCs:')
  

    inverse = compute_pca(normalized, 2)
    reconstructed = destandardize(inverse, mean_v, std_v)
    show_img(reconstructed[to_print], 'first 2 PCs:')
    
    
    pca = manual_fit(normalized, -6)
    transformed = np.dot(normalized, pca.components_.T)
    inverse = np.dot(transformed, pca.components_) 
    reconstructed = destandardize(inverse, mean_v, std_v)
    show_img(reconstructed[to_print], 'last 6 PCs')
    
     
    pca = manual_fit(normalized, 1, 1)
    transformed = np.dot(normalized, pca.components_.T)
    inverse = np.dot(transformed, pca.components_) 
    reconstructed = destandardize(inverse, mean_v, std_v)
    show_img(reconstructed[to_print], '1st and 2nd PCs')
    
    
    pca = manual_fit(normalized, 2, 3)
    transformed = np.dot(normalized, pca.components_.T)
    inverse = np.dot(transformed, pca.components_) 
    reconstructed = destandardize(inverse, mean_v, std_v)
    show_img(reconstructed[to_print], '3rd and 4th PCs')
    
    
    pca = manual_fit(normalized, 9, 10)
    transformed = np.dot(normalized, pca.components_.T)
    inverse = np.dot(transformed, pca.components_) 
    reconstructed = destandardize(inverse, mean_v, std_v)
    show_img(reconstructed[to_print], '10th and 11th PCs')  
   
    
def visualize_scatter_plot(dataset) :
    
    normalized_house, mean_v_house, std_v_house = standardize(dataset[0:280])
    normalized_dog, mean_v_dog, std_v_dog = standardize(dataset[280:469])
    normalized_person, mean_v_person, std_v_person = standardize(dataset[469:901])
    normalized_guitar, mean_v_guitar, std_v_guitar = standardize(dataset[901:1086])
    
    
    #scatter plot with the first 2 PCs
    pca = manual_fit(normalized_house, 0, 1)
    transformed = np.dot(normalized_house, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='red', label='houses')
    
    pca = manual_fit(normalized_dog, 0, 1)
    transformed = np.dot(normalized_dog, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='green', label='dogs')
    
    pca = manual_fit(normalized_person, 0, 1)
    transformed = np.dot(normalized_person, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='blue', label='people')
    
    pca = manual_fit(normalized_guitar, 0, 1)
    transformed = np.dot(normalized_guitar, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='black', label='guitars')
    
    plt.title('Scatter plot with the first 2 PCs')
    plt.legend()
    plt.show()
    
    #scatter plot with third and fourth PCs
    pca = manual_fit(normalized_house, 2, 3)
    transformed = np.dot(normalized_house, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='red', label='houses')
    
    pca = manual_fit(normalized_dog, 2, 3)
    transformed = np.dot(normalized_dog, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='green', label='dogs')
    
    pca = manual_fit(normalized_person, 2, 3)
    transformed = np.dot(normalized_person, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='blue', label='people')
    
    pca = manual_fit(normalized_guitar, 2, 3)
    transformed = np.dot(normalized_guitar, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='black', label='guitars')
    
    plt.title('Scatter plot with 3rd and 4th PCs')
    plt.legend()
    plt.show()
    
    #scatter plot with tenth and eleventh PCs
    pca = manual_fit(normalized_house, 9, 10)
    transformed = np.dot(normalized_house, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='red', label='houses')
    
    pca = manual_fit(normalized_dog, 9, 10)
    transformed = np.dot(normalized_dog, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='green', label='dogs')
    
    pca = manual_fit(normalized_person, 9, 10)
    transformed = np.dot(normalized_person, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='blue', label='people')
    
    pca = manual_fit(normalized_guitar, 9, 10)
    transformed = np.dot(normalized_guitar, pca.components_.T)
    plt.scatter(transformed[:, 0], transformed[:, 1], c='black', label='guitars')
    
    plt.title('Scatter plot with 10th and 11th PCs')
    plt.legend()
    plt.show()

   
def classify(dataset, label):
    #the model will be more accurate if the dataset is standardized
    dataset, mean_v, std_v = standardize(dataset)
    X_train, X_test, Y_train, Y_test = train_test_split(dataset, label, test_size=0.33)
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    train_scr = classifier.score(X_train, Y_train)*100
    test_scr = classifier.score(X_test, Y_test)*100
    print('SCORE ON TRAIN SET: %.2f%%' % train_scr)
    print('SCORE ON TEST SET: %.2f%%' % test_scr)
    
def pca_and_classify(dataset, label):
     #the model will be more accurate if the dataset is standardized
    norm_dataset, mean_v, std_v = standardize(dataset)
    
    #classification using 2 PCs
    inverse = compute_pca(norm_dataset, 2)
    X_train, X_test, Y_train, Y_test = train_test_split(inverse, label, test_size=0.33)
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    train_scr = classifier.score(X_train, Y_train)*100
    test_scr = classifier.score(X_test, Y_test)*100
    print('2PCs: SCORE ON TRAIN SET: %.2f%%' % train_scr)
    print('2PCs: SCORE ON TEST SET: %.2f%%' % test_scr)   

    #classification using 3th and 4th PCs
    pca = manual_fit(norm_dataset, 2, 3)
    transformed = np.dot(norm_dataset, pca.components_.T)
    inverse = np.dot(transformed, pca.components_) 
    X_train, X_test, Y_train, Y_test = train_test_split(inverse, label, test_size=0.33)
    classifier = GaussianNB()
    classifier.fit(X_train, Y_train)
    train_scr = classifier.score(X_train, Y_train)*100
    test_scr = classifier.score(X_test, Y_test)*100
    print('3th-4thPCs: SCORE ON TRAIN SET: %.2f%%' % train_scr)
    print('3th-4thPCs: SCORE ON TEST SET: %.2f%%' % test_scr) 
    
                            ##########
                            #  MAIN  #
                            ##########
imgpath = 'PACS_homework/'
dataset = []
label = []

#imgs indexes ranges:
#   houses: 0-279
#   dogs: 280-468
#   people: 469-900
#   guitars: 901-1086

for root, dirs, files in os.walk(imgpath, topdown = False):
    files.sort()
    for file in files:
        imgdata = np.asarray(Image.open(root+'/'+file))
        imgdata = imgdata
        imgdata = imgdata.ravel()
        dataset.append((imgdata)) 
        label.append(root[14:len(root)])

menu = "\t\t\t--Main menu--\n\
            0.PCA on all (intensive)\n\
            1.PCA on houses\n\
            2.PCA on dogs\n\
            3.PCA on people\n\
            4.PCA on guitars\n\
            5.Visualize scatter plot\n\
            6.Classify (Naive-Bayes)\n\
            7.Classify + PCA\n"
while 1 :
    cmd = input(menu)
    if cmd == '0':
        principal_component_analysis(dataset, 0)
    elif cmd == '1' :
        principal_component_analysis(dataset[0:280], 0)
    elif cmd == '2' :
        principal_component_analysis(dataset[280:469], 0)
    elif cmd == '3' :
        principal_component_analysis(dataset[469:901], 0)
    elif cmd == '4' :
        principal_component_analysis(dataset[901:1086], 0)
    elif cmd == '5' :
        visualize_scatter_plot(dataset)
    elif cmd == '6' :
        classify(dataset, label)
    elif cmd == '7' :
        pca_and_classify(dataset, label)
    elif cmd == 'exit' :
        sys.exit(0)
    elif cmd == 'back' or cmd == 'b' :
        break
    else:
        print('Invalid command')



















