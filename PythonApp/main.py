import shutil
import os
import numpy as np
import cv2
import uuid
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Layer, Conv2D, Dense, MaxPooling2D, Input, Flatten
from tensorflow.keras.metrics import Precision, Recall
import tensorflow as tf
import time


NEG_PATH = os.path.join('data', 'negative')
# choose the number according to your webcam
VIDEO_NUMBER = 0
# train parameter
BATCH_SIZE = 8
PRE_PATCH = 4
EPOCHS = 30
# choose detection and verification threshold
DETECTION = 0.8
VERIFICATION = 0.8

# configure GPU
gpus = tf.config.experimental.list_physical_devices('GPU')
for gpu in gpus:
    tf.config.experimental.set_memory_growth(gpu, True)
time.sleep(3)

# process img
def preprocess(file_path):
    # Read in image from file path
    byte_img = tf.io.read_file(file_path)
    # Load in the image
    img = tf.io.decode_jpeg(byte_img)

    # Preprocessing steps - resizing the image to be 100x100x3
    img = tf.image.resize(img, (100, 100))
    # Scale image to be between 0 and 1
    img = img / 255.0

    # Return image
    return img


def preprocess_twin(input_img, validation_img, label):
    return(preprocess(input_img), preprocess(validation_img), label)

# create a new account
def create_account(name):
    # create verification and input directories

    POS_PATH = os.path.join('data', name, 'positive')
    ANC_PATH = os.path.join('data', name, 'anchor')

    if os.path.isdir(os.path.join('data', name)):
        print('This account has already exist')
        return False
    os.makedirs(POS_PATH)
    os.makedirs(ANC_PATH)
    return True

# collect data to train model
def collect_data(anc_path, pos_path):
    cap = cv2.VideoCapture(VIDEO_NUMBER)
    print('\npressing a to capture anchor image')
    print('pressing p to capture positive image')
    print('press q to quit')
    # begin capture and store image
    while cap.isOpened():
        ret, frame = cap.read()
        # Cut down frame to 250x250px
        frame = frame[120:120 + 250, 200:200 + 250, :]

        # Collect anchors
        if cv2.waitKey(1) & 0XFF == ord('a'):
            # Create the unique file path
            imgname = os.path.join(anc_path, '{}.jpg'.format(uuid.uuid1()))
            # Write out anchor image
            cv2.imwrite(imgname, frame)

        # Collect positives
        if cv2.waitKey(1) & 0XFF == ord('p'):
            # Create the unique file path
            imgname = os.path.join(pos_path, '{}.jpg'.format(uuid.uuid1()))
            # Write out positive image
            cv2.imwrite(imgname, frame)

        # Show image back to screen
        cv2.imshow('Image Collection', frame)

        # Breaking gracefully
        if cv2.waitKey(1) & 0XFF == ord('q'):
            break

    # Release the webcam
    cap.release()
    # Close the image show frame
    cv2.destroyAllWindows()


# Siamese L1 Distance class
class L1Dist(Layer):
    def __init__(self, **kwargs):
        super().__init__()

    # similarity calculation
    def __call__(self, input_embedding, validation_embedding):
        return tf.math.abs(input_embedding - validation_embedding)


def train_model(name, anc_path, pos_path):
    # Build Embedding Layer
    def make_embedding():
        inp = Input(shape=(100, 100, 3), name='input_image')

        # First block
        c1 = Conv2D(64, (10, 10), activation='relu')(inp)
        m1 = MaxPooling2D(64, (2, 2), padding='same')(c1)

        # Second block
        c2 = Conv2D(128, (7, 7), activation='relu')(m1)
        m2 = MaxPooling2D(64, (2, 2), padding='same')(c2)

        # Third block
        c3 = Conv2D(128, (4, 4), activation='relu')(m2)
        m3 = MaxPooling2D(64, (2, 2), padding='same')(c3)

        # Final embedding block
        c4 = Conv2D(256, (4, 4), activation='relu')(m3)
        f1 = Flatten()(c4)
        d1 = Dense(4096, activation='sigmoid')(f1)

        return Model(inputs=[inp], outputs=[d1], name='embedding')

    # Create the model
    def make_siamese_model():
        # Anchor image input in the network
        input_image = Input(name='input_img', shape=(100, 100, 3))

        # Validation image in the network
        validation_image = Input(name='validation_img', shape=(100, 100, 3))

        # Combine siamese distance components
        siamese_layer = L1Dist()
        siamese_layer._name = 'distance'
        distances = siamese_layer(embedding(input_image), embedding(validation_image))

        # Classification layer
        classifier = Dense(1, activation='sigmoid')(distances)

        return Model(inputs=[input_image, validation_image], outputs=classifier, name='SiameseNetwork')

    anchor = tf.data.Dataset.list_files(anc_path + '\*.jpg').take(150)
    positive = tf.data.Dataset.list_files(pos_path + '\*.jpg').take(150)
    negative = tf.data.Dataset.list_files(NEG_PATH + '\*.jpg').take(150)

    positives = tf.data.Dataset.zip((anchor, positive, tf.data.Dataset.from_tensor_slices(tf.ones(len(anchor)))))
    negatives = tf.data.Dataset.zip((anchor, negative, tf.data.Dataset.from_tensor_slices(tf.zeros(len(anchor)))))
    data = positives.concatenate(negatives)

    data = data.map(preprocess_twin)
    data = data.cache()
    data = data.shuffle(buffer_size=1000)
    # Training partition
    train_data = data.take(len(data))
    train_data = train_data.batch(BATCH_SIZE)
    train_data = train_data.prefetch(PRE_PATCH)

    embedding = make_embedding()
    l1 = L1Dist()
    siamese_model = make_siamese_model()

    binary_cross_loss = tf.losses.BinaryCrossentropy()
    opt = tf.keras.optimizers.Adam(1e-4)  # 0.0001

    @tf.function
    def train_step(batch):
        # Record all of our operations
        with tf.GradientTape() as tape:
            # Get anchor and positive/negative image
            X = batch[:2]
            # Get label
            y = batch[2]

            # Forward pass
            yhat = siamese_model(X, training=True)
            # Calculate loss
            loss = binary_cross_loss(y, yhat)
        print(loss)

        # Calculate gradients
        grad = tape.gradient(loss, siamese_model.trainable_variables)

        # Calculate updated weights and apply to siamese model
        opt.apply_gradients(zip(grad, siamese_model.trainable_variables))

        # Return loss
        return loss

    # train loop
    def train(data, EPOCHS):
        # Loop through epochs
        for epoch in range(1, EPOCHS + 1):
            print('\n Epoch {}/{}'.format(epoch, EPOCHS))
            progbar = tf.keras.utils.Progbar(len(data))

            # Creating a metric object
            r = Recall()
            p = Precision()

            # Loop through each batch
            for idx, batch in enumerate(data):
                # Run train step here
                loss = train_step(batch)
                yhat = siamese_model.predict(batch[:2])
                r.update_state(batch[2], yhat)
                p.update_state(batch[2], yhat)
                progbar.update(idx + 1)
            print(loss.numpy(), r.result().numpy(), p.result().numpy())

    train(train_data, EPOCHS)
    model_path = os.path.join('data', name, 'siamesemodel.h5')
    siamese_model.save(model_path)
    print('OK')



def facial_recognition(name, pos_path):
    # Make verification function
    def verify(model, detection_threshold, verification_threshold):
        # Build results array
        results = []
        input_img = preprocess(os.path.join(INPUT_PATH, 'input_image.jpg'))
        for image in os.listdir(VER_PATH):
            validation_img = preprocess(os.path.join(VER_PATH, image))
            # Make Predictions
            result = model.predict(list(np.expand_dims([input_img, validation_img], axis=1)))
            results.append(result)

        # Detection Threshold: Metric above which a prediction is considered positive
        detection = np.sum(np.array(results) > detection_threshold)
        detected = []
        for x in range(0,8):
            detected.append(np.sum(np.array(results) > (0.5 + x/20)) / len(os.listdir(os.path.join(VER_PATH))))

        # Verification Threshold: Proportion of positive predictions / total positive samples
        verification = detection / len(os.listdir(os.path.join(VER_PATH)))
        verified = verification > verification_threshold

        return detected, verified

    VER_PATH = os.path.join('application_data', 'verification_img')
    INPUT_PATH = os.path.join('application_data', 'input_img')
    # copy verification img from positive img
    if os.path.isdir(VER_PATH):
        shutil.rmtree(VER_PATH)
    shutil.copytree(pos_path, VER_PATH, dirs_exist_ok = True)

    if os.path.isdir(INPUT_PATH):
        shutil.rmtree(INPUT_PATH)
    os.makedirs(INPUT_PATH)

    # load model
    model_path = os.path.join('data', name, 'siamesemodel.h5')
    siamese_model = tf.keras.models.load_model(model_path, custom_objects={'L1Dist':L1Dist, 'BinaryCrossentropy':tf.losses.BinaryCrossentropy})

    cap = cv2.VideoCapture(VIDEO_NUMBER)
    print('pressing v for about 2 seconds to trigger')
    print('press q to quit')
    while cap.isOpened():
        ret, frame = cap.read()
        frame = frame[120:120 + 250, 200:200 + 250, :]

        cv2.imshow('Verification', frame)
        # Verification trigger
        if cv2.waitKey(100) & 0xFF == ord('v'):
            cv2.imwrite(os.path.join(INPUT_PATH, 'input_image.jpg'), frame)
            # Run verification
            result, allow_access = verify(siamese_model, DETECTION, VERIFICATION)
            print(allow_access)
            print(result)
 
        if cv2.waitKey(100) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()


while(True):
    print('Choose 1 of the 5 option below: ')
    print('1. Make a new account.')
    print('2. Update an existing account.')
    print('3. Train model for an existing account.')
    print('4. Use facial recognition.')
    print('5. Exit the program\n\n')
    option = int(input('Enter your option here:'))
    time.sleep(1.5)

    if option == 1:
        valid_name = False
        while(not valid_name):
            name = input('\nEnter the name: ')
            valid_name = create_account(name)

    if option in [2,3,4]:
        while(True):
            name = input('\nEnter the username: ')
            if not os.path.isdir(os.path.join('data', name)):
                print('\nThere no such user, please try again\n')
                time.sleep(1.5)
            break
        POS_PATH = os.path.join('data', name, 'positive')
        ANC_PATH = os.path.join('data', name, 'anchor')

    if option == 2:
        print('Choose 1 to overwrite images or choose 2 to add image')
        option2 = 0
        while(True):
            option2 = int(input('Enter your option: '))
            time.sleep(1.5)
            if option2 not in [1, 2]:
                print('invalid option!!!\n\n')
                continue
            break
        if option2 == 1:
            shutil.rmtree(POS_PATH)
            shutil.rmtree(ANC_PATH)
            os.makedirs(POS_PATH)
            os.makedirs(ANC_PATH)
        collect_data(ANC_PATH, POS_PATH)

    if option == 3:
        train_model(name, ANC_PATH, POS_PATH)

    if option == 4:
        facial_recognition(name, POS_PATH)
    if option == 5:
        break
