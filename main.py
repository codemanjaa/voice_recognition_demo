import numpy as np

from tensorflow.keras import models

from recording_helper import record_audio, terminate
from tf_helper import preprocess_audiobuffer, get_yamnet_class

# !! Adjust  this in the correct order according to the model
#commands = ['left', 'down', 'stop', 'up', 'right', 'no', 'go', 'yes']
commands = ['no', 'stop', 'up', 'left', 'down', 'yes', 'right', 'go'] # Model V2
#commands = ['left', 'go', 'up', 'no', 'down', 'right', 'stop', 'yes'] #Model V1.1
loaded_model = models.load_model("saved_modellV2")

def predict_mic():
    audio = record_audio()
    yamnet_result = get_yamnet_class(audio)
    spec = preprocess_audiobuffer(audio)
    prediction = loaded_model(spec)
    label_prediction = np.argmax(prediction, axis=1)
    command = commands[label_prediction[0]]
    print("Predicted label:", command)
    return command

if __name__ == "__main__":
    from turtle_helper import move_turtle
    while True:
        command = predict_mic()
        move_turtle(command)
        if command == "stop":
           terminate()
           break