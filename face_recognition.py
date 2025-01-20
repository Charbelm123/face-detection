# import cv2
import os
import random
import numpy as np  
# from matplotlib import pyplot as plt
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Input, Layer , Dense
from tensorflow.keras.metrics import Recall, Precision
import tensorflow as tf 
import tarfile   
import uuid
import time



gpus = tf.config.list_physical_devices('GPU')
if gpus:
    print(f"GPUs found: {[gpu.name for gpu in gpus]}")
else:
    print("No GPUs found. Using CPU.")

# umcompress out tar gz file !tar -xf lfw.tgz
tar_file_path = r"face recognition\lfw.tgz"
extraction_path = r"face recognition"
with tarfile.open(tar_file_path, 'r:gz') as tar:
    tar.extractall(path=extraction_path)  


POS_PATH = r"positive"
NEG_PATH = r"negative"
ANC_PATH = r"anchor"

for directory in os.listdir('lfw'):
    for file in os.listdir(os.path.join('lfw', directory)):
        EX_PATH = os.path.join('lfw', directory, file)
        NEW_PATH = os.path.join(NEG_PATH, file)
        os.replace(EX_PATH, NEW_PATH)

print('done')



#establishconnection to webcam
cap = cv2.VideoCapture(0)
while cap.isOpened():
    ret, frame = cap.read()
    #cut down frame to 250x250  
    frame = frame[120:120+250,200:200+250,:]
    if cv2.waitKey(1) & 0xFF == ord('a'):
        #create unique file patrh for image
        imgname = os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1()))
        #save image to path
        cv2.imwrite(imgname, frame)
        print('Anchor image saved') 

    if cv2.waitKey(1) & 0xFF == ord('p'):
        #create unique file patrh for image
        imgname = os.path.join(POS_PATH,'{}.jpg'.format(uuid.uuid1()))
        #save image to path
        cv2.imwrite(imgname, frame)
        print('Positive image saved')

    #show image back to screen 
    cv2.imshow('Image Collection', frame)
    #breaking # when we hit q in breaks
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
#close image show frame
cv2.destroyAllWindows()

#data augmentation
def data_aug(img):
    data = []
    for i in range(10):
        # 
        img = tf.image.stateless_random_brightness(img,max_delta=0.02,seed=(1,2))  
        img = tf.image.stateless_random_contrast(img,lower=0.6,upper=1,seed=(1,2))
        img = tf.image.stateless_random_flip_left_right(img,seed=(np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_jpeg_quality(img,min_jpeg_quality = 90,max_jpeg_quality=100,seed = (np.random.randint(100),np.random.randint(100)))
        img = tf.image.stateless_random_saturation(img,lower=0.9,upper =1,seed=(np.random.randint(100),np.random.randint(100)))
        data.append(img)
    return data


for file_path in os.listdir(ANC_PATH):
    img_path = os.path.join(ANC_PATH,file_path)
    img = cv2.imread(img_path)
    augmented_images = data_aug(img)
    for image in augmented_images:
        cv2.imwrite(os.path.join(ANC_PATH,'{}.jpg'.format(uuid.uuid1())),image.numpy())

print('done augmenting anchor images')

for file_path in os.listdir(POS_PATH):
    img_path = os.path.join(POS_PATH,file_path)
    img = cv2.imread(img_path)
    augmented_images = data_aug(img)
    for image in augmented_images:
        cv2.imwrite(os.path.join(POS_PATH,'{}.jpg'.format(uuid.uuid1())),image.numpy())

print('done augmenting positive images')

for file_path in os.listdir(NEG_PATH):
    img_path = os.path.join(NEG_PATH,file_path)
    img = cv2.imread(img_path)
    augmented_images = data_aug(img)
    for image in augmented_images:
        cv2.imwrite(os.path.join(NEG_PATH,'{}.jpg'.format(uuid.uuid1())),image.numpy())

print('done augmenting negative images')


#load and preprocess images
# get images directories
anchor = tf.data.Dataset.list_files(ANC_PATH + '\*.jpg').take(3000)
positive = tf.data.Dataset.list_files(POS_PATH + '\*.jpg').take(3000)
negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(3000)
# negative = negative.as_numpy_iterator()
# print(negative.next())



def preprocess(file_path):
    #read image from file path
    byte_img = tf.io.read_file(file_path)
    #load in the image
    img = tf.io.decode_jpeg(byte_img)
    #resize image to be 100x100x3 
    img = tf.image.resize(img, (100, 100))
    #scale image to be betweeen 0 and 1
    img = img / 255.0
    return img




#create labelled dataset
#positives in each row it has the path of 2 images which are for the same person and a label of 1
positives = tf.data.Dataset.zip((anchor, positive,tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))    
#negatives in each row it has the path of 2 images which are not for the same person and a label of 0
negatives = tf.data.Dataset.zip((anchor, negative,tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
#combine the two datasets
dataset = tf.data.Dataset.concatenate(positives, negatives)

# # preprocess data in dataset (pos,anchor,label),(neg,anchor,label)
def preprocess_twin(input_image_path,validation_image_path,label):
    return(preprocess(input_image_path),preprocess(validation_image_path),label)

# build dataloader pipeline
# map runs preprocess_twin on all data in dataset
dataset = dataset.map(preprocess_twin)
dataset = dataset.cache()
# shuffle dataset
dataset = dataset.shuffle(buffer_size = 10000)



# training data
train_data = dataset.take(round(len(dataset)*0.7))
train_data = train_data.batch(16)
train_data = train_data.prefetch(8)

# testing data
test_data = dataset.skip(round(len(dataset)*0.7))
test_data = test_data.take(round(len(dataset)*0.3))
test_data = test_data.batch(16)
test_data = test_data.prefetch(8)




# build embedding layer
def make_embeddings():
    inp = Input(shape=(100,100,3),name='input_image')
    c1 = Conv2D(64,(10,10),activation = 'relu')(inp)
    m1 = MaxPooling2D(64,(2,2),padding = 'same')(c1)
    c2 = Conv2D(128,(7,7),activation = 'relu')(m1)
    m2 = MaxPooling2D(64,(2,2),padding = 'same')(c2)
    c3 = Conv2D(128,(4,4),activation = 'relu')(m2)
    m3 = MaxPooling2D(64,(2,2),padding = 'same')(c3)
    c4 = Conv2D(256,(4,4),activation='relu')(m3)
    f1 = Flatten()(c4)
    d1 = Dense(4096,activation='sigmoid')(f1)
    return Model(inputs=inp,outputs=d1,name='embedding')


# # print(embedding.summary())

# build distance layer
# L1 distance class
class L1Dist(Layer):
    # inheritance
    def __init__(self, **kwargs):
        super().__init__()
    
    # similarity calculation
    def call(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)
    

# # embedding = make_embeddings()
# # input_image = Input(name='input_img', shape=(100,100,3))
# # validation_image = Input(name='validation_img', shape=(100,100,3))
# # inp_embedding = embedding(input_image)
# # val_embedding = embedding(validation_image)

# # siamese_layer = L1Dist()
# # distances = siamese_layer(inp_embedding, val_embedding)
# # classifier = Dense(1, activation='sigmoid')(distances)
# # print(classifier)

# # siamese_network = Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')
# # siamese_network.summary()


def build_model():
        
    embedding = make_embeddings()
    input_image = Input(name='input_img', shape=(100,100,3))
    validation_image = Input(name='validation_img', shape=(100,100,3))
    inp_embedding = embedding(input_image)
    val_embedding = embedding(validation_image)

    siamese_layer = L1Dist()
    distances = siamese_layer(inp_embedding, val_embedding)
    classifier = Dense(1, activation='sigmoid')(distances)

    return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

siamese_network = build_model()
# # print(siamese_network.summary())


#training
#setup loss function    
bninary_cross_loss = tf.losses.BinaryCrossentropy()
# #setup optimizer    
optimizer = tf.keras.optimizers.Adam(0.0001)
# #setup checkpoint
checkpoint_dir = r"C:\Users\CharbelMazloum\OneDrive - Growth Technology\Desktop\face recognition\tarining_checkpoints"
checkpoint_prefix = os.path.join(checkpoint_dir, "ckpt")
checkpoint = tf.train.Checkpoint(model=siamese_network, optimizer=optimizer)

# # build training step
@tf.function
def train_step(batch):
    #record operations for automatic differentiation
    with tf.GradientTape() as tape:
        #get the anchor and positive and negative image
        X = batch[:2]
        #get the label
        y= batch[2]
        #forward pass
        y_pred = siamese_network(X, training=True)
        #calculate loss
        loss = bninary_cross_loss(y, y_pred)
    #calculate gradients
    gradients = tape.gradient(loss, siamese_network.trainable_variables)
    #apply gradients,  updating weights
    optimizer.apply_gradients(zip(gradients, siamese_network.trainable_variables))
    return loss


#training loop
def train(train_data, epochs):
    for epoch in range(1,epochs):
        print('\n Epoch {}/{}'.format(epoch, epochs))
        progbar = tf.keras.utils.Progbar(len(train_data))
        r = Recall()
        p = Precision()
        #loop through each batch
        for i, batch in enumerate(train_data):
            #cal the trainig step function
            loss = train_step(batch)
            #predict
            y_pred = siamese_network.predict(batch[:2])
            #update progress bar
            r.update_state(batch[2], y_pred)
            p.update_state(batch[2], y_pred)
            progbar.update(i+1, values=[('loss', loss)])
        #output recall and precision
        print(loss.numpy(), r.result().numpy(), p.result().numpy())
        #save checkpoint
        if epoch % 10 ==0:
            checkpoint.save(file_prefix=checkpoint_prefix)

#train model
epochs = 50
train(train_data, epochs)


#evaluation
#get a batch of test data
# test_input,test_val,y_true = test_data.as_numpy_iterator().next()
# #make predictions
# y_pred = siamese_network.predict([test_input,test_val])
# #post process predictions
# y_pred = [1 if i>0.5 else 0 for i in y_pred]
# print(y_pred)

# #calculate recall and precision
# #create recall object
# recall = Recall()
# #update recall object with true and predicted values
# recall.update_state(y_true, y_pred)
# #convert recall object to numpy
# recall = recall.result().numpy()
# #create precision object
# precision = Precision()
# #update precision object with true and predicted values
# precision.update_state(y_true, y_pred)
# #convert precision object to numpy
# precision = precision.result().numpy()
# #output recall and precision
# print('Recall:', recall)
# print('Precision:', precision)





# #vizualtize results

# #plot input images
# plt.figure(figsize=(18,10))
# #used to plot images side by side
# plt.subplot(1,2,1)
# #plot input image
# plt.imshow(test_input[0])
# plt.title('Input Image')
# #plot validation image
# plt.subplot(1,2,2)
# plt.imshow(test_val[0])
# plt.title('Validation Image')
# plt.show()

#save model
siamese_network.save(r"face_recognition_model.h5")
#load model
model = tf.keras.models.load_model(r"C:\Users\CharbelMazloum\OneDrive - Growth Technology\Desktop\face recognition\face_recognition_model.h5", custom_objects={'L1Dist': L1Dist, 'BinaryCrossentropy': tf.losses.BinaryCrossentropy,})
print('model loaded')
#evaluate model using recall and precision
r = Recall()
p = Precision()
for test_input, test_val, y_true in test_data.as_numpy_iterator():
    print(y_true)

    y_pred = model.predict([test_input,test_val])
    y_pred = [1 if y >0.5 else 0 for y in y_pred]

    r.update_state(y_true,y_pred)
    p.update_state(y_true,y_pred)
    break
print(r.result().numpy(), p.result().numpy())


#model summary
print(model.summary())


#real time face recognition
#detection_threshold: metric above which above a prediction is considered positive
#verification_threshold: proportion of positive predictions / total predictions in verification images
def verify(model,detection_threshold,verification_threshold):
    results = []
    #loop through all images in verification images
    for image in os.listdir(r"C:\Users\CharbelMazloum\OneDrive - Growth Technology\Desktop\face recognition\application_data\verification_images"):
        #preprocess input image from webcam(openCV)
        input_image = preprocess(r"C:\Users\CharbelMazloum\OneDrive - Growth Technology\Desktop\face recognition\application_data\input_image\input_image.jpg")
        #preprocess verification images
        verification_image = preprocess(os.path.join(r"C:\Users\CharbelMazloum\OneDrive - Growth Technology\Desktop\face recognition\application_data\verification_images", image))
        #make prediction
        #expamd_dims is used to add a batch dimension to the input image
        result = model.predict(list(np.expand_dims([input_image,verification_image],axis=1)))
        #append result to results
        results.append(result)
    #turn results into numpy array, then filter out results above detection threshold, then calculate the sum of the results
    detection = np.sum(np.array(results)>detection_threshold)
    #calculate the proportion of positive predictions to total predictions
    verification = detection/len(os.listdir(r"C:\Users\CharbelMazloum\OneDrive - Growth Technology\Desktop\face recognition\application_data\verification_images"))
    #return whether the verification is above the verification threshold and the results
    verified = verification>verification_threshold  
    return verified, results

function that uses opencv to get 50 validation images and send then to a spesific path
def get_validation_images_auto():
    cap = cv2.VideoCapture(0)  # Open the webcam
    count = 0
    capture_interval = 0.2  # Interval in seconds between capturing images

    while cap.isOpened():
        ret, frame = cap.read()  # Capture a frame from the webcam
        frame = frame[120:120+250, 200:200+250, :]  # Crop the frame as needed

        # Save the image automatically without keyboard input
        if cv2.waitKey(1) & 0xFF == ord('a'):

            imgname = os.path.join(r"C:\Users\CharbelMazloum\OneDrive - Growth Technology\Desktop\face recognition\application_data\verification_images", '{}.jpg'.format(uuid.uuid1()))
            cv2.imwrite(imgname, frame)
            print(f"Image {count + 1} saved.")
            count += 1

        # Exit after saving 50 images
        if count == 50:
            break


        # Add a delay between captures
        time.sleep(capture_interval)

        # Press 'q' to exit early
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()  # Release the webcam
    cv2.destroyAllWindows()  # Close any OpenCV windows

# function that get the input image from the webcam and save it to a spesific path
def get_input_image():
    cap = cv2.VideoCapture(0)
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[120:120+250,200:200+250,:]
        if cv2.waitKey(1) & 0xFF == ord('a'):
            imgname = os.path.join(r"C:\Users\CharbelMazloum\OneDrive - Growth Technology\Desktop\face recognition\application_data\input_image",'input_image.jpg')
            cv2.imwrite(imgname, frame)
            break
        cv2.imshow(r"C:\Users\CharbelMazloum\OneDrive - Growth Technology\Desktop\face recognition\application_data\input_image", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


#test verification

# get_validation_images()
get_validation_images_auto()
# get_input_image
get_input_image()
#verify images
verified, results = verify(model,0.5,0.5)
print('verified:', verified)
print('results:', np.squeeze(results)) 








