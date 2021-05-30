import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import numpy as np
# Import Model
from models.DeepECGNet import DeepECGNet
from models.EEGNet import EEGNet

class Distiller(keras.Model):
    def __init__(self, student, teacher):
        super(Distiller, self).__init__()
        self.teacher = teacher
        self.student = student

    def compile(
        self,
        optimizer,
        metrics,
        student_loss_fn,
        distillation_loss_fn,
        alpha=0.1,
        temperature=3,
    ):
        """ Configure the distiller.

        Args:
            optimizer: Keras optimizer for the student weights
            metrics: Keras metrics for evaluation
            student_loss_fn: Loss function of difference between student
                predictions and ground-truth
            distillation_loss_fn: Loss function of difference between soft
                student predictions and soft teacher predictions
            alpha: weight to student_loss_fn and 1-alpha to distillation_loss_fn
            temperature: Temperature for softening probability distributions.
                Larger temperature gives softer distributions.
        """
        super(Distiller, self).compile(optimizer=optimizer, metrics=metrics)
        self.student_loss_fn = student_loss_fn
        self.distillation_loss_fn = distillation_loss_fn
        self.alpha = alpha
        self.temperature = temperature

    def train_step(self, data):
        # Unpack data
        x, y = data

        # Forward pass of teacher
        teacher_predictions = self.teacher(x, training=False)
        print(teacher_predictions)
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predictions = self.student(x, training=True)

            # Compute losses
            student_loss = self.student_loss_fn(y, student_predictions)
            distillation_loss = self.distillation_loss_fn(
                tf.nn.softmax(teacher_predictions / self.temperature, axis=1),
                tf.nn.softmax(student_predictions / self.temperature, axis=1),
            )
            loss = self.alpha * student_loss + (1 - self.alpha) * distillation_loss

        # Compute gradients
        trainable_vars = self.student.trainable_variables
        gradients = tape.gradient(loss, trainable_vars)

        # Update weights
        self.optimizer.apply_gradients(zip(gradients, trainable_vars))

        # Update the metrics configured in `compile()`.
        self.compiled_metrics.update_state(y, student_predictions)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update(
            {"student_loss": student_loss, "distillation_loss": distillation_loss}
        )
        return results

    def test_step(self, data):
        # Unpack the data
        x, y = data

        # Compute predictions
        y_prediction = self.student(x, training=False)

        # Calculate the loss
        student_loss = self.student_loss_fn(y, y_prediction)

        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)

        # Return a dict of performance
        results = {m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        return results




loadPath = "/Users/kok_ksy/Desktop/dataset/eeg_dataset_ASR_alpha.npz"
data = np.load(loadPath)

x_Train_EEG = data['x_Train']
x_Validate_EEG = data['x_Validate']
x_Test_EEG = data['x_Test']
x_Validate_EEG = data['x_Validate']
y_Train_EEG = data['y_Train']
y_Test_EEG = data['y_Test']
y_Validate_EEG = data['y_Validate']
data.close()

kernels, chans, samples = 1, x_Train_EEG.shape[1], x_Train_EEG.shape[2]

x_Train_EEG = x_Train_EEG.reshape(x_Train_EEG.shape[0], chans, samples, kernels)
x_Validate_EEG = x_Validate_EEG.reshape(x_Validate_EEG.shape[0], chans, samples, kernels)
x_Test_EEG = x_Test_EEG.reshape(x_Test_EEG.shape[0], chans, samples, kernels)

# Load ECG Data numpy format
loadPath = "/Users/kok_ksy/Desktop/dataset/ecg_dataset_128_norm.npz"
data = np.load(loadPath)

x_Train_ECG = data['x_Train']
x_Test_ECG = data['x_Test']
x_Validate_ECG = data['x_Validate']
y_Train_ECG = data['y_Train']
y_Test_ECG = data['y_Test']
y_Validate_ECG = data['y_Validate']

data.close()

x_Train_ECG = x_Train_ECG.reshape(x_Train_ECG.shape[0], samples, 1)
x_Validate_ECG = x_Validate_ECG.reshape(x_Validate_ECG.shape[0], samples, 1)
x_Test_ECG = x_Test_ECG.reshape(x_Test_ECG.shape[0], samples, 1)


# Create the teacher
teacher = EEGNet(nb_classes = 2, Chans = chans, Samples = samples, 
               dropoutRate = 0.1, kernLength = 64, F1 = 8, D = 2, F2 = 16, 
               dropoutType = 'Dropout')
teacher.summary()

# Create the student
student = DeepECGNet(samples, dropoutRate=0.5)

# Clone student for later comparison
student_scratch = keras.models.clone_model(student)


# Train teacher as usual
teacher.compile(
    optimizer=keras.optimizers.Adam(),
    loss=keras.losses.BinaryCrossentropy(from_logits=True),
    metrics=['accuracy']
)

# Train and evaluate teacher on data.
teacher.fit(
    x_Train_EEG, 
    y_Train_EEG, 
    epochs=5,
    batch_size=128)
teacher.evaluate(x_Test_EEG, y_Test_EEG)

# Initialize and compile distiller
distiller = Distiller(student=student, teacher=teacher)
distiller.compile(
    optimizer=keras.optimizers.Adam(),
    student_loss_fn=keras.losses.BinaryCrossentropy(from_logits=True),
    distillation_loss_fn=keras.losses.KLDivergence(),
    alpha=0.1,
    temperature=10,
    metrics=['accuracy']
)

# Distill teacher to student
distiller.fit(
    x_Train_ECG, 
    y_Train_ECG, 
    epochs=3,
    batch_size=128)

# Evaluate student on test dataset
distiller.evaluate(x_Test_ECG, y_Test_ECG)