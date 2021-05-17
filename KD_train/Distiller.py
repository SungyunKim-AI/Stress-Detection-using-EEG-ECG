import tensorflow as tf
from tensorflow import keras

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
        self.temperature = temperature
        self.alpha = alpha
    

    def train_step(self, EEG_data, ECG_data):
        EEG_x, EEG_y = EEG_data
        ECG_x, ECG_y = ECG_data

        
        # Forward pass of teacher
        teacher_prediction = self.teacher(EEG_x, training=False)
        print("Tecaher prediction   ...", teacher_prediction)
        with tf.GradientTape() as tape:
            # Forward pass of student
            student_predcition = self.student(ECG_x, training=True)
            # Compute losses
            student_loss = self.student_loss_fn(ECG_y, student_predcition)
            
            distillation_loss=self.distillation_loss_fn(
            tf.nn.softmax(teacher_prediction/self.temperature, axis=1),
            tf.nn.softmax(student_predcition/self.temperature, axis=1)
            )

            loss= self.alpha* student_loss + (1-self.alpha)* distillation_loss
            print("Loss in distiller :",loss)
            # Compute gradients
            trainable_vars= self.student.trainable_variables
            gradients=tape.gradient(loss, trainable_vars)
            gradients = [gradient * (self.temperature ** 2) for gradient in gradients]
            # Update weights
            self.optimizer.apply_gradients(zip(gradients, trainable_vars))
            
            # Update the metrics configured in `compile()`
            self.compiled_metrics.update_state(ECG_y, student_predcition)
            
            # Return a dict of performance
            results={ m.name: m.result()  for m in self.metrics}
            results.update({"student_loss": student_loss, "distillation_loss": distillation_loss})
            print("Train...", results)
            return results
    

    def test_step(self, data):
        # Unpack the data
        x, y = data
        
        ## Compute predictions
        y_prediction= self.student(x, training=False)
        
        # calculate the loss
        student_loss= self.student_loss_fn(y, y_prediction)
        
        # Update the metrics.
        self.compiled_metrics.update_state(y, y_prediction)
        
        # Return a dict of performance
        results ={m.name: m.result() for m in self.metrics}
        results.update({"student_loss": student_loss})
        print("Test...", results)
        return results